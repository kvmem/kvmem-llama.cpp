using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using NInfer.Manager;

internal static class EngineTelemetryCheck
{
    public static async Task RunAsync(Action<bool, string> check)
    {
        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Logging.ClearProviders();
        await using var app = builder.Build();
        var authorization = "";
        app.MapGet("/stats", (HttpContext context) =>
        {
            authorization = context.Request.Headers.Authorization.ToString();
            return authorization == "Bearer fixture-monitor-key"
                ? Results.Json(new { fixture = 42 })
                : Results.StatusCode((int)HttpStatusCode.Unauthorized);
        });
        await app.StartAsync();
        var endpoint = new Uri(app.Urls.Single());
        var launch = new LaunchSpec("fixture", "Fixture", "unused.exe", ".",
            ["model.ninfer", "--api-key", "fixture-monitor-key", "--stats-port", endpoint.Port.ToString()],
            new Dictionary<string, string?>(), "http://127.0.0.1:1/v1", "fixture", "", "", "");
        var engine = new EngineSnapshot("Running", "fixture", ApiBase: launch.ApiBase, Launch: launch);
        using var sampler = new TelemetryReader();
        await sampler.SampleAsync(engine);
        var sample = await sampler.ReadAsync(engine, CancellationToken.None);
        check(sample.Error is null && sample.Stats is JsonElement stats && stats.GetProperty("fixture").GetInt32() == 42,
            "monitor reads authenticated stats from the separate stats port");
        check(authorization == "Bearer fixture-monitor-key", "monitor sends the running engine API key");

        var ordinary = engine with { ApiBase = endpoint + "v1", Launch = launch with { Arguments = ["model.ninfer", "--stats-port", "0"] } };
        using var request = EngineTelemetryRequest.Create(ordinary, new Uri(ordinary.ApiBase!));
        check(request.RequestUri?.Port == endpoint.Port && request.RequestUri.AbsolutePath == "/stats",
            "disabled stats port uses the ordinary engine API port");
        check(request.Headers.Authorization is null, "monitor does not reuse another profile's API key");
        await app.StopAsync();

        var now = DateTimeOffset.UtcNow;
        var log = Path.Combine(Path.GetTempPath(), "ninfer-log-reuse-" + Guid.NewGuid().ToString("N") + ".jsonl");
        string Record(int id, DateTimeOffset time) => JsonSerializer.Serialize(new
        {
            @event = "request_done", timestamp_unix_ms = time.ToUnixTimeMilliseconds(),
            request = new { request_id = id }, result = new { prompt_tokens = 10, completion_tokens = 2 }
        });
        var gpu = new GpuTelemetrySample(false, null, 0, 0, 0, null);
        try
        {
            File.WriteAllText(log, Record(1, now.AddMinutes(-1)) + "\n" + Record(2, now.AddSeconds(1)) + "\n");
            using var monitor = new MonitorSession();
            var first = new EngineSnapshot("Running", RequestLogPath: log, StartedAt: now);
            monitor.UseEngine(first); monitor.ReadRequests();
            var previous = monitor.Sample(null, gpu, now.AddSeconds(2));
            check(previous.Counts.Completed == 1, "fixed request log excludes requests from earlier engine runs");
            var next = first with { StartedAt = now.AddSeconds(3) };
            File.AppendAllText(log, Record(3, now.AddSeconds(4)) + "\n");
            monitor.UseEngine(next); monitor.ReadRequests();
            var current = monitor.Sample(null, gpu, now.AddSeconds(5));
            check(current.Counts.Completed == 1 && current.SessionId != previous.SessionId,
                "restarting with the same request log starts a new monitor session");
            monitor.UseEngine(next with { State = "Stopped", Pid = null });
            check(monitor.Sample(null, gpu, now.AddSeconds(6)).Counts.Completed == 1,
                "stopping preserves the current custom-log session counts");
        }
        finally { File.Delete(log); }
    }
}
