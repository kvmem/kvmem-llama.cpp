using NInfer.Manager;
using System.Text.Json;

internal static class ThroughputWindowCheck
{
    internal static void Run(Action<bool, string> check)
    {
        var start = DateTimeOffset.Parse("2026-09-30T00:00:00Z");
        bool Near(double? actual, double expected) => actual.HasValue && Math.Abs(actual.Value - expected) < 1e-8;
        var window = new ThroughputWindow();
        check(window.Add(start, 0, 0) is { Prefill: null, Decode: null }, "throughput begins with a counter baseline rather than lifetime averages");
        var first = window.Add(start.AddSeconds(2), 1000, 100);
        check(first is { Prefill: 500, Decode: 50 }, "first available two-second interval does not get diluted by an unobserved ten-second history");
        var afterBurst = window.Add(start.AddSeconds(4), 1000, 100);
        check(afterBurst is { Prefill: 500, Decode: 50 }, "zero-token polls retain each metric's last effective throughput");
        window.Add(start.AddSeconds(6), 1000, 100);
        window.Add(start.AddSeconds(8), 1000, 100);
        check(window.Add(start.AddSeconds(10), 1000, 100) is { Prefill: 500, Decode: 50 }, "idle intervals neither update the displayed rates nor dilute the averaging denominator");
        check(window.Add(start.AddSeconds(12), 1000, 100) is { Prefill: 500, Decode: 50 }, "expired nonzero samples do not clear the last effective displayed value during idle polls");
        window.Add(start.AddSeconds(14), 1000, 100);
        check(window.Add(start.AddSeconds(16), 1040, 104) is { Prefill: 20, Decode: 2 }, "work after more than ten idle seconds uses only its new interval and excludes expired samples and idle time");

        window.Reset();
        window.Add(start, 0, 0);
        check(window.Add(start.AddSeconds(2), 0, 0) is { Prefill: null, Decode: null }, "an engine with no positive token increments has no invented zero throughput");
        check(window.Add(start.AddSeconds(4), 200, 0) is { Prefill: 100, Decode: null }, "Prefill can update independently before Decode has any effective sample");
        check(window.Add(start.AddSeconds(6), 200, 40) is { Prefill: 100, Decode: 20 }, "a Decode-only interval preserves the preceding Prefill value");
        check(window.Add(start.AddSeconds(8), 400, 40) is { Prefill: 100, Decode: 20 }, "a Prefill-only interval excludes idle Prefill time and preserves Decode");
        check(window.Add(start.AddSeconds(10), 400, 120) is { Prefill: 100, Decode: 30 }, "Decode averages only its own nonzero intervals while Prefill remains held");
        var independent = window.Add(start.AddSeconds(12), 500, 120);
        check(Near(independent.Prefill, 500d / 6) && independent.Decode == 30, "each metric has an independent nonzero-duration denominator");

        window.Reset();
        window.Add(start, 0, 0);
        window.Add(start.AddSeconds(2), 200, 20);
        var uneven = window.Add(start.AddSeconds(6), 1000, 100);
        check(Near(uneven.Prefill, 1000d / 6) && Near(uneven.Decode, 100d / 6), "unequal poll intervals use summed token increments divided by elapsed time, not an average of sample rates");

        window.Reset();
        window.Add(start, 0, 0);
        window.Add(start.AddSeconds(3), 300, 30);
        window.Add(start.AddSeconds(8), 1300, 130);
        var clipped = window.Add(start.AddSeconds(12), 1700, 170);
        check(Near(clipped.Prefill, 150) && Near(clipped.Decode, 15), "an irregular interval crossing the sliding boundary contributes only its overlapping fraction");
        var advanced = window.Add(start.AddSeconds(14), 2100, 190);
        check(Near(advanced.Prefill, 160) && Near(advanced.Decode, 14), "advancing the boundary trims an old segment and incorporates the actual new token increments");

        check(window.Add(start.AddSeconds(16), null, null) is { Prefill: null, Decode: null }, "missing stats show gaps and erase the smoothing baseline");
        check(window.Add(start.AddSeconds(18), 2000, 200) is { Prefill: null, Decode: null }, "the first poll after a gap cannot mix pre-gap tokens into the new rate");
        check(window.Add(start.AddSeconds(20), 2200, 220) is { Prefill: 100, Decode: 10 }, "rates recover using only the fresh post-gap interval");
        check(window.Add(start.AddSeconds(22), 1, 1) is { Prefill: null, Decode: null }, "decreasing engine counters invalidate the entire averaging window");
        check(window.Add(start.AddSeconds(24), 21, 5) is { Prefill: 10, Decode: 2 }, "counter reset recovery excludes tokens from the previous counter generation");
        check(window.Add(start.AddSeconds(26), 41, 2) is { Prefill: null, Decode: null }, "one counter rolling back invalidates both rates at the shared engine boundary");
        check(window.Add(start.AddSeconds(28), 81, 6) is { Prefill: 20, Decode: 2 }, "partial counter rollback also receives a fresh baseline");
        check(window.Add(start.AddSeconds(36), 281, 26) is { Prefill: null, Decode: null }, "a poll delayed beyond three sampling intervals cannot bridge unobserved history");
        check(window.Add(start.AddSeconds(38), 381, 36) is { Prefill: 50, Decode: 5 }, "delayed sampling resumes with its new observed interval");
        check(window.Add(start.AddSeconds(38), 400, 40) is { Prefill: null, Decode: null }, "duplicate timestamps cannot divide by zero or retain a stale average");
        check(window.Add(start.AddSeconds(37), 410, 41) is { Prefill: null, Decode: null }, "a backwards wall clock breaks the window");
        check(window.Add(start.AddSeconds(39), 430, 45) is { Prefill: 10, Decode: 2 }, "the window recovers after a backwards clock using increasing observation times");
        foreach (var invalid in new double?[] { null, double.NaN, double.PositiveInfinity, -1 })
        {
            check(window.Add(start.AddSeconds(41), invalid, 50) is { Prefill: null, Decode: null } &&
                window.Add(start.AddSeconds(43), 500, 60) is { Prefill: null, Decode: null },
                "missing or invalid counters never fabricate throughput: " + (invalid?.ToString() ?? "missing"));
        }

        using var session = new MonitorSession();
        var engine = new EngineSnapshot("Running", Pid: 999, StartedAt: start, RequestLogPath: "fixture-unused.jsonl");
        var gpu = new GpuTelemetrySample(true, "Fixture", 16UL << 30, 14UL << 30, 2UL << 30, null, PowerWatts: 100);
        JsonElement Stats(int prefill, int decode) => JsonSerializer.SerializeToElement(new { counters = new { computed_prefill_tokens = prefill, committed_decode_tokens = decode } });
        session.UseEngine(engine);
        session.Sample(Stats(0, 0), gpu, start);
        session.Sample(Stats(1000, 100), gpu, start.AddSeconds(2));
        var smoothed = session.Sample(Stats(1000, 100), gpu with { PowerWatts = 200 }, start.AddSeconds(4));
        check(smoothed.ThroughputWindowSeconds == 10 && smoothed.IntervalSeconds == 2 &&
            smoothed.Latest is { PrefillTps: 500, DecodeTps: 50, GpuPowerWatts: 200 } &&
            smoothed.History.Last() == smoothed.Latest,
            "monitor cards and history expose identical ten-second rates while GPU stays at the latest two-second sample");
        session.UseEngine(engine with { State = "Stopped", Pid = null });
        var stopped = session.Sample(null, gpu, start.AddSeconds(6));
        check(stopped.Latest is { Valid: false, PrefillTps: null, DecodeTps: null } &&
            stopped.History.Any(point => point.PrefillTps == 500),
            "stopping blanks current throughput while retaining the completed run's smoothed history");
        session.UseEngine(engine with { StartedAt = start.AddMinutes(1), Pid = 1000 });
        var restarted = session.Sample(Stats(5000, 500), gpu, start.AddMinutes(1));
        var restartedRate = session.Sample(Stats(5040, 506), gpu, start.AddMinutes(1).AddSeconds(2));
        check(restarted.History.Count == 1 && restarted.Latest is { PrefillTps: null, DecodeTps: null } &&
            restartedRate.Latest is { PrefillTps: 20, DecodeTps: 3 },
            "a new engine process starts its own throughput window even when its request-log path is reused");
        var longHistory = Enumerable.Range(0, 2000).Select(i => smoothed.Latest! with
        {
            T = i * 2000, PrefillTps = i == 511 ? null : 100, DecodeTps = i == 511 ? null : 10
        }).ToArray();
        var compacted = MonitorSession.CompactHistory(longHistory);
        check(compacted.Count <= 1800 && compacted.Any(point => point.T == 511 * 2000 && point.Valid && point.PrefillTps is null),
            "long-history compaction retains throughput reset gaps even when other engine stats remain valid");
    }
}
