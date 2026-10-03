using System.Text;
using System.Text.Json;

namespace NInfer.Manager;

public sealed record MonitorCounts(long Completed, long Errors, long Rejected, bool CatchingUp, long UnreadBytes, long MalformedLines, string? LogError);
public sealed record MonitorMtp(double Accepted, double Drafted, double Rounds, double? AcceptanceRate, double? TokensPerRound);
public sealed record MonitorCachePath(string Path, int Requests, double HitTokens);
public sealed record MonitorCache(double HitTokens, double PromptTokens, double? TokenHitRate, int HitRequests, int Requests, double? RequestHitRate, IReadOnlyList<MonitorCachePath> Paths);
public sealed record MonitorWindow(int Seconds, int RequestLimit, int SampleCount, bool Truncated,
    double? TtftP50Ms, double? TtftP95Ms, double? ProcP50Ms, double? ProcP95Ms,
    double? PrefillP50Tps, double? PrefillP95Tps, double? DecodeP50Tps, double? DecodeP95Tps,
    MonitorMtp Mtp, MonitorCache Cache);
public sealed record MonitorPoint(long T, bool Valid, double? PrefillTps, double? DecodeTps,
    double? TtftP50Ms, double? TtftP95Ms, double? MtpAcceptanceRate,
    double? GpuUtilizationPercent, double? GpuPowerWatts, double? GpuPowerLimitWatts, double? GpuTemperatureC,
    ulong? GpuFreeBytes, double? DedicatedBytes, double? DeviceKvTokens, double? HostKvBytes,
    double? H2dBytes, double? D2hBytes, double? D2dBytes,
    double? Waiting, double? Running, double? Prefilling, double? Materializing);
public sealed record MonitorSnapshot(string? SessionId, DateTimeOffset SampledAt, int IntervalSeconds, int HistorySeconds,
    MonitorCounts Counts, MonitorWindow Window, IReadOnlyList<MonitorPoint> History, MonitorPoint? Latest,
    int HistoryStoredSamples = 0, bool HistoryDownsampled = false, int ThroughputWindowSeconds = ThroughputWindow.Seconds);
public sealed record TelemetrySnapshot(JsonElement? Stats, IReadOnlyList<JsonElement> RecentRequests,
    string LogTail, string? Error, DateTimeOffset SampledAt, GpuTelemetrySample? Gpu, MonitorSnapshot Monitor);

/// <summary>Samples independently of browser activity; only this worker mutates a session.</summary>
internal sealed class TelemetryReader : IDisposable
{
    internal const int IntervalSeconds = 2;
    internal const int HistorySeconds = 6 * 60 * 60;
    private readonly HttpClient client = new(new HttpClientHandler { UseProxy = false }) { Timeout = TimeSpan.FromMilliseconds(1200) };
    private readonly SemaphoreSlim gate = new(1, 1);
    private readonly MonitorSession session = new();
    private readonly CancellationTokenSource stopping = new();
    private Task? worker;
    private int disposed;
    private TelemetrySnapshot cached = new(null, [], "", null, DateTimeOffset.MinValue, null, new(null, DateTimeOffset.MinValue, IntervalSeconds, HistorySeconds,
        new(0, 0, 0, false, 0, 0, null), MonitorSession.EmptyWindow, [], null));

    public void Start(Func<EngineSnapshot> engine, CancellationToken cancellationToken)
    {
        if (worker is not null) throw new InvalidOperationException("The monitor is already sampling.");
        worker = Task.Run(async () =>
        {
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, stopping.Token);
            using var timer = new PeriodicTimer(TimeSpan.FromSeconds(IntervalSeconds));
            try
            {
                do { await SampleAsync(engine(), linked.Token); }
                while (await timer.WaitForNextTickAsync(linked.Token));
            }
            catch (OperationCanceledException) when (linked.IsCancellationRequested) { }
        });
    }

    // HTTP reads never drive sampling or append history. Every browser sees the same immutable sample.
    public Task<TelemetrySnapshot> ReadAsync(EngineSnapshot engine, CancellationToken cancellationToken) => Task.FromResult(Volatile.Read(ref cached));

    internal async Task SampleAsync(EngineSnapshot engine, CancellationToken cancellationToken = default)
    {
        await gate.WaitAsync(cancellationToken);
        try
        {
            session.UseEngine(engine);
            session.ReadRequests();
            JsonElement? stats = null;
            string? error = null;
            if (engine.State == "Running" && Uri.TryCreate(engine.ApiBase, UriKind.Absolute, out var api))
            {
                try
                {
                    using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                    deadline.CancelAfter(TimeSpan.FromMilliseconds(1200));
                    using var request = EngineTelemetryRequest.Create(engine, api);
                    using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, deadline.Token);
                    response.EnsureSuccessStatusCode();
                    await using var stream = await response.Content.ReadAsStreamAsync(deadline.Token);
                    using var bytes = new MemoryStream();
                    var block = new byte[8192];
                    int count;
                    while ((count = await stream.ReadAsync(block, deadline.Token)) != 0)
                    {
                        if (bytes.Length + count > 2 * 1024 * 1024) throw new InvalidDataException("Engine stats exceeded the 2 MiB limit.");
                        bytes.Write(block, 0, count);
                    }
                    using var document = JsonDocument.Parse(bytes.ToArray());
                    if (document.RootElement.ValueKind != JsonValueKind.Object) throw new InvalidDataException("Engine stats must be an object.");
                    stats = document.RootElement.Clone();
                }
                catch (Exception ex) when (ex is HttpRequestException or OperationCanceledException or InvalidDataException or IOException or JsonException)
                {
                    if (cancellationToken.IsCancellationRequested) throw;
                    error = ex.Message;
                }
            }
            var now = DateTimeOffset.UtcNow;
            var gpu = GpuTelemetry.Sample();
            var monitor = session.Sample(stats, gpu, now);
            var tail = ReadTail(engine.StderrPath, 64 * 1024, 120);
            if (string.IsNullOrEmpty(tail)) tail = ReadTail(engine.StdoutPath, 64 * 1024, 120);
            Volatile.Write(ref cached, new(stats, session.RecentRequests, tail, error, now, gpu, monitor));
        }
        finally { gate.Release(); }
    }

    internal static JsonElement NormalizeRequest(JsonElement record)
    {
        var values = record.EnumerateObject().ToDictionary(property => property.Name,
            property => (object?)property.Value.Clone(), StringComparer.Ordinal);
        JsonElement? Field(params string[] path) { var found = MonitorSession.Field(record, path); return found.ValueKind is JsonValueKind.Undefined ? null : found.Clone(); }
        double? Number(params string[] path) => MonitorSession.Number(record, path);
        values["request_id"] = Field("request", "request_id");
        values["timestamp"] = Field("timestamp_unix_ms");
        values["protocol"] = Field("request", "protocol");
        values["prompt_tokens"] = Field("result", "prompt_tokens");
        values["completion_tokens"] = Field("result", "completion_tokens");
        values["reasoning_tokens"] = Field("result", "model_thinking_tokens");
        values["cached_tokens"] = Field("result", "prefix_cache_hit_tokens");
        values["ttft_ms"] = Number("timings_seconds", "ttft") * 1000;
        values["queue_ms"] = Number("engine_timing", "queue_wait_seconds") * 1000;
        var ttft = Number("timings_seconds", "ttft");
        var queue = Number("engine_timing", "queue_wait_seconds");
        values["proc_ms"] = ttft.HasValue && queue is >= 0 && queue <= ttft ? (ttft.Value - queue.Value) * 1000 : null;
        values["prefill_tps"] = MonitorSession.Ratio(Number("result", "computed_prefill_tokens"), Number("timings_seconds", "prefill"));
        var completion = Number("result", "completion_tokens");
        values["decode_tps"] = MonitorSession.Ratio(completion.HasValue ? Math.Max(0, completion.Value - 1) : null, Number("timings_seconds", "decode"));
        values["finish_reason"] = Field("result", "finish_reason");
        values["prefix_reuse_path"] = Field("result", "prefix_reuse_path");
        values["spec_accept"] = Field("speculative", "accepted_tokens");
        values["spec_drafted"] = Field("speculative", "drafted_tokens");
        values["spec_rounds"] = Field("speculative", "rounds");
        values["spec_acceptance_rate"] = MonitorSession.Ratio(Number("speculative", "accepted_tokens"), Number("speculative", "drafted_tokens"));
        values["error_message"] = Field("error", "message");
        return JsonSerializer.SerializeToElement(values, ConfigurationStore.Json);
    }

    internal static string ReadTail(string? path, int maxBytes, int maxLines)
    {
        if (string.IsNullOrEmpty(path)) return "";
        try
        {
            using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            var count = (int)Math.Min(file.Length, maxBytes);
            var offset = Math.Max(0, file.Length - count);
            file.Seek(offset, SeekOrigin.Begin);
            var bytes = new byte[count];
            var read = file.ReadAtLeast(bytes, count, false);
            var text = Encoding.UTF8.GetString(bytes, 0, read);
            if (offset > 0) { var firstBreak = text.IndexOf('\n'); text = firstBreak >= 0 ? text[(firstBreak + 1)..] : ""; }
            return string.Join('\n', text.Split('\n').TakeLast(maxLines));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return ""; }
    }

    public async Task StopAsync()
    {
        if (Volatile.Read(ref disposed) != 0) return;
        stopping.Cancel();
        if (worker is not null) await worker;
        Dispose();
    }
    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0) return;
        stopping.Cancel();
        // The worker runs on the thread pool and never needs the WinForms synchronization context.
        try { worker?.GetAwaiter().GetResult(); }
        finally { session.Dispose(); client.Dispose(); gate.Dispose(); stopping.Dispose(); }
    }
}

/// <summary>A run's bounded request window, lifetime counters, and six-hour sample ring.</summary>
internal sealed class MonitorSession : IDisposable
{
    internal const int WindowSeconds = 3600;
    internal const int RequestLimit = 10000;
    internal static readonly MonitorWindow EmptyWindow = new(WindowSeconds, RequestLimit, 0, false, null, null, null, null, null, null, null, null,
        new(0, 0, 0, null, null), new(0, 0, null, 0, 0, null, []));
    private readonly List<(long Time, JsonElement Record)> window = [];
    private readonly Queue<JsonElement> recent = new();
    private readonly Queue<MonitorPoint> history = new();
    private readonly IncrementalRequestLog reader = new();
    private string? identity;
    private string? requestPath;
    private long completed, errors, rejected, truncatedUntil;
    private JsonElement? previousStats;
    private readonly ThroughputWindow throughput = new();
    private readonly HashSet<string> seen = new(StringComparer.Ordinal);
    private readonly Queue<(string Key, long Time)> seenOrder = new();
    private long replayBefore, latestTimestamp, runStartedAt;
    public IReadOnlyList<JsonElement> RecentRequests => recent.Reverse().ToArray();

    public void UseEngine(EngineSnapshot engine)
    {
        // A custom log can be reused. Process start identifies the run; stopping retains its history.
        if (engine.State == "Starting" && !engine.StartedAt.HasValue) return;
        var next = engine.StartedAt.HasValue ? $"{engine.StartedAt:O}|{engine.RequestLogPath}" : engine.RequestLogPath;
        if (next is null || next == identity) return;
        identity = next;
        requestPath = engine.RequestLogPath;
        window.Clear(); recent.Clear(); history.Clear(); reader.Reset(); seen.Clear(); seenOrder.Clear();
        completed = errors = rejected = truncatedUntil = replayBefore = latestTimestamp = 0;
        runStartedAt = engine.StartedAt?.ToUnixTimeMilliseconds() ?? 0;
        previousStats = null; throughput.Reset();
    }

    public void ReadRequests() => reader.Read(requestPath, record => AddRequest(record, DateTimeOffset.UtcNow), () => replayBefore = latestTimestamp);

    internal void AddRequest(JsonElement record, DateTimeOffset now)
    {
        var kind = Field(record, "event");
        if (kind.ValueKind != JsonValueKind.String || kind.GetString() is not ("request_done" or "request_error" or "request_rejected")) return;
        var timestamp = Number(record, "timestamp_unix_ms") is double stamp ? (long)stamp : now.ToUnixTimeMilliseconds();
        if (timestamp < runStartedAt) return;
        var key = $"{Field(record, "server_instance_id")}|{Field(record, "request", "request_id")}|{kind}|{Number(record, "timestamp_unix_ms")}";
        // On rewind, older records were already counted. Recent boundary keys cover concurrent equal timestamps.
        if (timestamp < replayBefore || !seen.Add(key)) return;
        seenOrder.Enqueue((key, timestamp));
        if (seenOrder.Count > 65536) seen.Remove(seenOrder.Dequeue().Key);
        latestTimestamp = Math.Max(latestTimestamp, timestamp);
        switch (kind.GetString()) { case "request_done": completed++; break; case "request_error": errors++; break; case "request_rejected": rejected++; break; }
        var normalized = TelemetryReader.NormalizeRequest(record);
        recent.Enqueue(normalized);
        while (recent.Count > 40) recent.Dequeue();
        if (kind.GetString() != "request_done" || timestamp < now.AddSeconds(-WindowSeconds).ToUnixTimeMilliseconds()) return;
        window.Add((timestamp, normalized));
        if (window.Count > RequestLimit)
        {
            truncatedUntil = Math.Max(truncatedUntil, window[0].Time + WindowSeconds * 1000L);
            window.RemoveAt(0);
        }
    }

    public MonitorSnapshot Sample(JsonElement? stats, GpuTelemetrySample gpu, DateTimeOffset now)
    {
        var cutoff = now.AddSeconds(-WindowSeconds).ToUnixTimeMilliseconds();
        window.RemoveAll(item => item.Time < cutoff);
        var summary = Summarize(now);
        double? Transfer(string direction)
        {
            var main = Delta(stats, previousStats, "context_cache", "main_kv_transfers", direction, "bytes");
            var backend = Delta(stats, previousStats, "context_cache", "backend_kv_transfers", direction, "bytes");
            return main.HasValue && backend.HasValue ? main + backend : null;
        }
        double? N(params string[] path) => stats.HasValue ? Number(stats.Value, path) : null;
        var rates = throughput.Add(now, N("counters", "computed_prefill_tokens"), N("counters", "committed_decode_tokens"));
        var point = new MonitorPoint(now.ToUnixTimeMilliseconds(), stats.HasValue, rates.Prefill, rates.Decode,
            summary.TtftP50Ms, summary.TtftP95Ms, summary.Mtp.AcceptanceRate,
            gpu.UtilizationPercent, gpu.PowerWatts, gpu.PowerLimitWatts, gpu.TemperatureC, gpu.FreeBytes,
            N("memory", "cuda_residency", "dedicated_bytes"), N("occupancy", "device_main_kv_tokens"), N("occupancy", "host_kv_bytes"),
            Transfer("h2d"), Transfer("d2h"), Transfer("d2d"), N("requests", "waiting"), N("requests", "running"), N("requests", "prefilling"), N("requests", "materializing"));
        // A failed poll breaks rates. The next successful sample becomes a fresh baseline.
        previousStats = stats;
        if (identity is not null)
        {
            history.Enqueue(point);
            while (history.Count > TelemetryReader.HistorySeconds / TelemetryReader.IntervalSeconds || history.TryPeek(out var oldest) && oldest.T < now.AddSeconds(-TelemetryReader.HistorySeconds).ToUnixTimeMilliseconds()) history.Dequeue();
        }
        return new(identity, now, TelemetryReader.IntervalSeconds, TelemetryReader.HistorySeconds,
            new(completed, errors, rejected, reader.CatchingUp, reader.UnreadBytes, reader.MalformedLines, reader.Error), summary,
            CompactHistory(history.ToArray()), point, history.Count, history.Count > 1800);
    }

    // Retain two-second observations; throughput averages non-zero intervals over ten seconds
    // and holds the last valid value during zero-delta samples.
    // Network snapshots retain bucket endpoints and extrema
    // so long sessions stay small without flattening token-rate, GPU power, or memory spikes.
    internal static IReadOnlyList<MonitorPoint> CompactHistory(MonitorPoint[] points)
    {
        if (points.Length <= 1800) return points;
        var output = new List<MonitorPoint>(1800);
        Func<MonitorPoint, double?>[] fields = [point => point.PrefillTps, point => point.DecodeTps,
            point => point.TtftP50Ms, point => point.TtftP95Ms, point => point.MtpAcceptanceRate,
            point => point.GpuPowerWatts, point => point.GpuPowerLimitWatts, point => point.GpuTemperatureC,
            point => point.GpuUtilizationPercent, point => point.GpuFreeBytes, point => point.DedicatedBytes,
            point => point.DeviceKvTokens, point => point.HostKvBytes, point => point.H2dBytes, point => point.D2hBytes, point => point.D2dBytes,
            point => point.Waiting, point => point.Running, point => point.Prefilling, point => point.Materializing];
        var bucketCount = 1800 / (2 * fields.Length + 3);
        var bucketSize = (int)Math.Ceiling(points.Length / (double)bucketCount);
        for (var start = 0; start < points.Length; start += bucketSize)
        {
            var end = Math.Min(start + bucketSize, points.Length);
            var selected = new SortedSet<int> { start, end - 1 };
            foreach (var field in fields)
            {
                int? min = null, max = null;
                for (var i = start; i < end; i++) if (field(points[i]) is double value)
                {
                    if (!min.HasValue || value < field(points[min.Value])) min = i;
                    if (!max.HasValue || value > field(points[max.Value])) max = i;
                }
                if (min.HasValue) selected.Add(min.Value);
                if (max.HasValue) selected.Add(max.Value);
            }
            // Counter resets can break throughput even when the rest of /stats remains valid.
            var gap = Array.FindIndex(points, start, end - start, point => !point.Valid || point.PrefillTps is null || point.DecodeTps is null);
            if (gap >= 0) selected.Add(gap);
            output.AddRange(selected.Select(index => points[index]));
        }
        return output;
    }

    internal MonitorWindow Summarize(DateTimeOffset now)
    {
        var records = window.Select(item => item.Record).ToArray();
        double Sum(string key) => records.Sum(record => Number(record, key) ?? 0);
        double? P(string key, double percentile) => Percentile(records.Select(record => Number(record, key)), percentile);
        var accepted = Sum("spec_accept"); var drafted = Sum("spec_drafted"); var rounds = Sum("spec_rounds");
        var hits = Sum("cached_tokens"); var prompts = Sum("prompt_tokens");
        var hitRequests = records.Count(record => Number(record, "cached_tokens") > 0);
        var paths = records.GroupBy(record => Field(record, "prefix_reuse_path").ValueKind == JsonValueKind.String ? Field(record, "prefix_reuse_path").GetString()! : "unknown")
            .Select(group => new MonitorCachePath(group.Key, group.Count(), group.Sum(record => Number(record, "cached_tokens") ?? 0))).OrderByDescending(path => path.Requests).ToArray();
        return new(WindowSeconds, RequestLimit, records.Length, now.ToUnixTimeMilliseconds() < truncatedUntil,
            P("ttft_ms", .5), P("ttft_ms", .95), P("proc_ms", .5), P("proc_ms", .95), P("prefill_tps", .5), P("prefill_tps", .95), P("decode_tps", .5), P("decode_tps", .95),
            new(accepted, drafted, rounds, Ratio(accepted, drafted), Ratio(accepted, rounds)),
            new(hits, prompts, Ratio(hits, prompts), hitRequests, records.Length, Ratio(hitRequests, records.Length), paths));
    }

    internal static JsonElement Field(JsonElement root, params string[] path)
    {
        foreach (var key in path) if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty(key, out root)) return default;
        return root;
    }
    internal static double? Number(JsonElement root, params string[] path)
    {
        var value = Field(root, path);
        return value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var number) && double.IsFinite(number) ? number : null;
    }
    internal static double? Ratio(double? numerator, double? denominator) => numerator.HasValue && denominator > 0 ? numerator / denominator : null;
    private static double? Delta(JsonElement? current, JsonElement? previous, params string[] path)
    {
        if (!current.HasValue || !previous.HasValue) return null;
        var a = Number(current.Value, path); var b = Number(previous.Value, path);
        return a.HasValue && b.HasValue && a >= b ? a - b : null;
    }
    internal static double? Percentile(IEnumerable<double?> values, double percentile)
    {
        var sorted = values.Where(value => value.HasValue).Select(value => value!.Value).Order().ToArray();
        if (sorted.Length == 0) return null;
        var index = percentile * (sorted.Length - 1); var lo = (int)Math.Floor(index); var hi = (int)Math.Ceiling(index);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo);
    }
    public void Dispose() => reader.Dispose();
}
