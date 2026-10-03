using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using NInfer.Manager;

if (args is ["--startup-checks"]) { StartupRegistrationTests.Run(); return 0; }
if (args.Length >= 2 && args[0] == "--signal-stop") return NativeProcess.SignalStop(int.Parse(args[1])) ? 0 : 1;
if (args.Length >= 2 && args[0] == "--sleeper")
{
    File.WriteAllText(args[1], Environment.ProcessId.ToString());
    Console.CancelKeyPress += (_, e) => e.Cancel = true;
    await Task.Delay(Timeout.Infinite);
    return 0;
}
if (args.Length >= 1 && args[0] == "--fake")
{
    var port = int.Parse(args[1]);
    var delay = int.Parse(args[2]);
    var control = args[3];
    var stop = new CancellationTokenSource();
    Console.CancelKeyPress += (_, e) => { e.Cancel = true; File.WriteAllText(control + ".graceful", "yes"); if (!File.Exists(control + ".ignore")) stop.Cancel(); };
    Console.WriteLine("ARGS=" + System.Text.Json.JsonSerializer.Serialize(args));
    Console.WriteLine("ENV=" + Environment.GetEnvironmentVariable("NINFER_TEST_VALUE"));
    if (File.Exists(control + ".spawn"))
    {
        var childInfo = new ProcessStartInfo(Environment.ProcessPath!) { UseShellExecute = false, CreateNoWindow = true };
        childInfo.ArgumentList.Add(typeof(NativeProcess).Assembly.Location);
        childInfo.ArgumentList.Add("--sleeper");
        childInfo.ArgumentList.Add(control + ".child-pid");
        Process.Start(childInfo)?.Dispose();
    }
    try
    {
        await Task.Delay(delay, stop.Token);
        using var listener = new TcpListener(IPAddress.Loopback, port);
        listener.Start();
        while (!stop.IsCancellationRequested)
        {
            using var connection = await listener.AcceptTcpClientAsync(stop.Token);
            var stream = connection.GetStream();
            using var reader = new StreamReader(stream, Encoding.ASCII, leaveOpen: true);
            var line = await reader.ReadLineAsync(stop.Token);
            if (line?.Contains("/quit ") == true) return 13;
            while (await reader.ReadLineAsync(stop.Token) is { Length: > 0 }) { }
            var status = File.Exists(control + ".fail") ? "503 Service Unavailable" : "200 OK";
            var response = Encoding.ASCII.GetBytes($"HTTP/1.1 {status}\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK");
            await stream.WriteAsync(response, stop.Token);
        }
    }
    catch (OperationCanceledException) { }
    return 0;
}

var root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "test-package"));
Directory.CreateDirectory(root);
var controlPath = Path.Combine(root, Guid.NewGuid().ToString("N"));
var dotnet = Environment.ProcessPath!;
var testAssembly = typeof(NativeProcess).Assembly.Location;
var pass = 0;
pass += StartupRegistrationTests.Run();
var dataRoot = Path.Combine(root, "user-data");
LaunchSpec Spec(string id, int port = 19981, int delay = 0) => new(id, id, dotnet, root,
    new[] { testAssembly, "--fake", port.ToString(), delay.ToString(), controlPath, "space \"quote\" trailing\\", "" },
    new Dictionary<string, string?> { ["NINFER_TEST_VALUE"] = "quoted value" },
    $"http://127.0.0.1:{port}/v1", "fake-model", Path.Combine(root, id + ".out.log"), Path.Combine(root, id + ".err.log"), Path.Combine(root, id + ".requests.log"));
void Check(bool success, string name) { if (!success) throw new Exception("FAIL " + name); pass++; Console.WriteLine("PASS " + name); }
async Task WaitState(EngineController engine, string state)
{
    var until = DateTimeOffset.UtcNow.AddSeconds(20);
    while (engine.Snapshot.State != state && DateTimeOffset.UtcNow < until) await Task.Delay(100);
    Check(engine.Snapshot.State == state, "state " + state);
}
await using var engine = new EngineController(dataRoot);
await engine.StartAsync(Spec("ready"));
Check(engine.Snapshot.State == "Running", "readiness");
var firstPid = engine.Snapshot.Pid!.Value;
try { await engine.StartAsync(Spec("duplicate")); throw new Exception("Duplicate accepted"); }
catch (InvalidOperationException) { Check(engine.Snapshot.Pid == firstPid, "duplicate start rejected without disturbing engine"); }
await engine.StopAsync();
Check(engine.Snapshot.State == "Stopped", "stop state");
Check(File.Exists(controlPath + ".graceful"), "CTRL_BREAK graceful shutdown");
Check(NativeProcess.ListeningProcessIds(19981).Count == 0, "stop releases API port");
var log = File.ReadAllText(Path.Combine(root, "ready.out.log"));
Check(log.Contains("ENV=quoted value") && log.Contains("trailing") && log.Contains("\"\"]"), "argument/environment/log redirect");

using (var request = new CancellationTokenSource())
{
    await engine.StartAsync(Spec("request-cancel-after-ready"), request.Token);
    request.Cancel();
    await Task.Delay(1100);
    Check(engine.Snapshot.State == "Running", "request cancellation after readiness leaves service running");
    await engine.StopAsync();
}

var starting = engine.StartAsync(Spec("cancel", delay: 60000));
await Task.Delay(500);
await engine.StopAsync();
try { await starting; throw new Exception("Cancelled startup accepted"); }
catch (OperationCanceledException) { Check(engine.Snapshot.State == "Stopped", "stop during startup"); }

using (var request = new CancellationTokenSource(500))
{
    try { await engine.StartAsync(Spec("request-cancel-startup", delay: 60000), request.Token); throw new Exception("Cancelled startup accepted"); }
    catch (OperationCanceledException) { Check(engine.Snapshot.State == "Stopped", "caller cancellation during startup cleans process"); }
}

await engine.StartAsync(Spec("health-failure"));
File.WriteAllText(controlPath + ".fail", "yes");
await WaitState(engine, "Failed");
await engine.StopAsync();
Check(NativeProcess.ListeningProcessIds(19981).Count == 0, "503 cleanup");
File.Delete(controlPath + ".fail");

using (var listener = new TcpListener(IPAddress.Loopback, 19981))
{
    listener.Start();
    try { await engine.StartAsync(Spec("occupied")); throw new Exception("Occupied port accepted"); }
    catch (InvalidOperationException) { Check(NativeProcess.ListeningProcessIds(19981).Contains(Environment.ProcessId), "external listener preserved"); }
}

await engine.StartAsync(Spec("unexpected-exit"));
using (var http = new HttpClient()) { try { await http.GetAsync("http://127.0.0.1:19981/quit"); } catch (HttpRequestException) { } }
await WaitState(engine, "Failed");
await engine.StopAsync();
Check(NativeProcess.ListeningProcessIds(19981).Count == 0, "unexpected-exit cleanup");

File.WriteAllText(controlPath + ".ignore", "yes");
await engine.StartAsync(Spec("forced-stop"));
await engine.StopAsync();
Check(engine.Snapshot.State == "Stopped" && NativeProcess.ListeningProcessIds(19981).Count == 0, "ignored console signal falls back to owned Job termination");
File.Delete(controlPath + ".ignore");

File.WriteAllText(controlPath + ".spawn", "yes");
var child = NativeProcess.Start(Spec("job-close"));
using var childProcess = Process.GetProcessById(child.Id);
var descendantDeadline = DateTimeOffset.UtcNow.AddSeconds(5);
while (!File.Exists(controlPath + ".child-pid") && DateTimeOffset.UtcNow < descendantDeadline) await Task.Delay(100);
using var descendant = Process.GetProcessById(int.Parse(File.ReadAllText(controlPath + ".child-pid")));
child.Dispose();
await childProcess.WaitForExitAsync();
await descendant.WaitForExitAsync();
Check(childProcess.HasExited, "Job close kills owned process");
Check(descendant.HasExited, "Job close kills descendant without PID adoption");
Check(File.Exists(Path.Combine(dataRoot, "runtime", "manager-run.json")), "run state persisted under the selected user data root");
Console.WriteLine($"ALL {pass} CHECKS PASSED");
return 0;
