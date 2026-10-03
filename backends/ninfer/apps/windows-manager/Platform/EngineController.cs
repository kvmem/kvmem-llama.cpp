using System.Diagnostics;
using System.Net;
using System.Text.Json;

namespace NInfer.Manager;

/// <summary>Owns exactly one engine lifetime; no process discovered by PID is ever stopped or adopted.</summary>
public sealed class EngineController : IEngineController
{
    private readonly string statePath;
    private readonly SemaphoreSlim gate = new(1, 1);
    private readonly HttpClient http = new(new HttpClientHandler { UseProxy = false }) { Timeout = TimeSpan.FromSeconds(2) };
    private EngineSnapshot snapshot = new("Stopped");
    private Session? session;
    private bool disposed;
    public EngineSnapshot Snapshot => Volatile.Read(ref snapshot);
    public event Action? Changed;

    public EngineController(string dataRoot)
    {
        statePath = Path.Combine(Path.GetFullPath(dataRoot), "runtime", "manager-run.json");
    }

    public async Task StartAsync(LaunchSpec spec, CancellationToken cancellationToken = default)
    {
        Session started;
        await gate.WaitAsync(cancellationToken);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (session is not null) throw new InvalidOperationException("A model is already running, starting, or stopping.");
            cancellationToken.ThrowIfCancellationRequested();
            var api = new Uri(spec.ApiBase, UriKind.Absolute);
            if (api.Scheme is not "http" and not "https" || !api.IsLoopback)
                throw new ArgumentException("The managed engine API must use a local HTTP address.");
            if (NativeProcess.ListeningProcessIds(api.Port).Count != 0)
                throw new InvalidOperationException($"API port {api.Port} is already occupied. The existing process was not changed.");
            Publish(new EngineSnapshot("Starting", spec.ProfileId, spec.ProfileName,
                ApiBase: spec.ApiBase, ModelId: spec.ModelId, StdoutPath: spec.StdoutPath,
                StderrPath: spec.StderrPath, RequestLogPath: spec.RequestLogPath, Launch: spec));
            try
            {
                var child = NativeProcess.Start(spec);
                started = new Session(spec, child, new Uri(api, "/health"), cancellationToken);
                session = started;
                Publish(Snapshot with { Pid = child.Id, StartedAt = child.CreatedAt });
                started.Completion = Task.Run(() => RunLifecycleAsync(started));
            }
            catch (Exception error)
            {
                Publish(Snapshot with { State = "Failed", Error = error.Message, Pid = null });
                throw;
            }
        }
        finally { gate.Release(); }
        await started.Ready.Task;
    }

    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        Task? completion;
        await gate.WaitAsync(cancellationToken);
        try
        {
            if (session is null)
            {
                if (Snapshot.State != "Stopped") Publish(Snapshot with { State = "Stopped", Pid = null, Error = null });
                return;
            }
            Publish(Snapshot with { State = "Stopping" });
            session.Stop.Cancel();
            completion = session.Completion;
        }
        finally { gate.Release(); }
        // Cancellation only cancels this caller's wait. Once requested, owned-process cleanup always finishes.
        await completion.WaitAsync(cancellationToken);
    }

    private async Task RunLifecycleAsync(Session current)
    {
        Exception? failure = null;
        var wasReady = false;
        try
        {
            var startup = Stopwatch.StartNew();
            while (true)
            {
                current.Stop.Token.ThrowIfCancellationRequested();
                if (current.Child.HasExited) throw new IOException($"Engine exited during startup (code {current.Child.ExitCode}). See stderr log.");
                if (startup.Elapsed > TimeSpan.FromSeconds(300)) throw new TimeoutException("Engine did not become ready within 300 seconds.");
                var health = await GetHealthAsync(current);
                if (health == HttpStatusCode.OK)
                {
                    await gate.WaitAsync();
                    try
                    {
                        current.Stop.Token.ThrowIfCancellationRequested();
                        if (current.Child.HasExited) throw new IOException("Engine exited before readiness was published.");
                        current.DetachCallerCancellation();
                        Publish(Snapshot with { State = "Running", Error = null });
                        wasReady = true;
                        current.Ready.TrySetResult();
                    }
                    finally { gate.Release(); }
                    break;
                }
                await Task.Delay(250, current.Stop.Token);
            }

            var failedHealthChecks = 0;
            while (true)
            {
                await Task.Delay(1000, current.Stop.Token);
                if (current.Child.HasExited) throw new IOException($"Engine exited unexpectedly (code {current.Child.ExitCode}). See stderr log.");
                var health = await GetHealthAsync(current);
                if (health == HttpStatusCode.ServiceUnavailable)
                    throw new IOException("Engine health is unavailable (HTTP 503). The owned engine has been stopped; inspect its log before restarting.");
                if (health == HttpStatusCode.OK) failedHealthChecks = 0;
                else if (++failedHealthChecks >= 3)
                    throw new IOException("Engine health failed three consecutive checks. The owned engine has been stopped.");
            }
        }
        catch (OperationCanceledException) when (current.Stop.IsCancellationRequested) { }
        catch (Exception error) { failure = error; }
        finally
        {
            if (failure is not null)
            {
                await gate.WaitAsync();
                try { Publish(Snapshot with { State = "Failed", Error = failure.Message }); }
                finally { gate.Release(); }
            }
            try { await current.Child.StopAsync(TimeSpan.FromSeconds(10)); }
            catch (Exception error) { failure ??= error; }
            finally { current.Child.Dispose(); }
            await gate.WaitAsync();
            try
            {
                session = null;
                Publish(Snapshot with { State = failure is null ? "Stopped" : "Failed", Pid = null, Error = failure?.Message });
                if (!wasReady)
                {
                    if (failure is not null) current.Ready.TrySetException(failure);
                    else current.Ready.TrySetCanceled();
                }
                current.DetachCallerCancellation();
                current.Stop.Dispose();
            }
            finally { gate.Release(); }
        }
    }

    private async Task<HttpStatusCode?> GetHealthAsync(Session current)
    {
        try
        {
            // A successful HTTP reply on a reused port must not make an unrelated process our engine.
            if (!NativeProcess.ListeningProcessIds(current.HealthUri.Port).Contains(current.Child.Id)) return null;
            using var response = await http.GetAsync(current.HealthUri, HttpCompletionOption.ResponseHeadersRead, current.Stop.Token);
            return response.StatusCode;
        }
        catch (HttpRequestException) { return null; }
        catch (TaskCanceledException) when (!current.Stop.IsCancellationRequested) { return null; }
    }

    private void Publish(EngineSnapshot value)
    {
        Volatile.Write(ref snapshot, value);
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(statePath)!);
            var temporary = statePath + ".tmp";
            File.WriteAllText(temporary, JsonSerializer.Serialize(new
            {
                schemaVersion = 1, managerPid = Environment.ProcessId,
                managerStartedAt = Process.GetCurrentProcess().StartTime.ToUniversalTime(),
                updatedAt = DateTimeOffset.UtcNow, engine = value
            }, new JsonSerializerOptions(JsonSerializerDefaults.Web) { WriteIndented = true }));
            File.Move(temporary, statePath, true);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            Debug.WriteLine($"Could not persist manager run state: {error.Message}");
        }
        if (Changed is not null)
            foreach (Action subscriber in Changed.GetInvocationList())
                try { subscriber(); } catch (Exception error) { Debug.WriteLine($"Engine state listener failed: {error}"); }
    }

    public async ValueTask DisposeAsync()
    {
        await gate.WaitAsync();
        try { if (disposed) return; disposed = true; }
        finally { gate.Release(); }
        await StopAsync();
        http.Dispose();
    }

    private sealed class Session
    {
        private readonly CancellationTokenRegistration callerCancellation;
        public LaunchSpec Spec { get; }
        public NativeProcess Child { get; }
        public Uri HealthUri { get; }
        public CancellationTokenSource Stop { get; }
        public TaskCompletionSource Ready { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public Task Completion { get; set; } = Task.CompletedTask;
        public Session(LaunchSpec spec, NativeProcess child, Uri healthUri, CancellationToken cancellationToken)
        {
            Spec = spec;
            Child = child;
            HealthUri = healthUri;
            Stop = new CancellationTokenSource();
            callerCancellation = cancellationToken.Register(() => Stop.Cancel());
        }
        // Cancelling the HTTP request after readiness must not stop a successfully launched service.
        public void DetachCallerCancellation() => callerCancellation.Dispose();
    }
}
