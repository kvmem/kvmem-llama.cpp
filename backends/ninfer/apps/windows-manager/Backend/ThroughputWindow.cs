namespace NInfer.Manager;

/// <summary>Time-weighted throughput of nonzero intervals; idle polls retain each metric's last value.</summary>
internal sealed class ThroughputWindow
{
    internal const int Seconds = 10;
    private static readonly TimeSpan MaximumSampleGap = TimeSpan.FromSeconds(TelemetryReader.IntervalSeconds * 3);
    private readonly Queue<Interval> intervals = new();
    private Sample? previous;
    private double? lastPrefill, lastDecode;
    private sealed record Sample(DateTimeOffset Time, double Prefill, double Decode);
    private sealed record Interval(DateTimeOffset Start, DateTimeOffset End, double Prefill, double Decode);

    internal void Reset() { intervals.Clear(); previous = null; lastPrefill = lastDecode = null; }

    internal (double? Prefill, double? Decode) Add(DateTimeOffset now, double? prefill, double? decode)
    {
        if (prefill is not double p || decode is not double d || !double.IsFinite(p) || !double.IsFinite(d) || p < 0 || d < 0)
        {
            Reset();
            return (null, null);
        }

        var current = new Sample(now, p, d);
        var prior = previous;
        previous = current;
        if (prior is null || now <= prior.Time || now - prior.Time > MaximumSampleGap || p < prior.Prefill || d < prior.Decode)
        {
            // Counter resets, missing/late polls, and clock changes start a new baseline.
            intervals.Clear();
            lastPrefill = lastDecode = null;
            return (null, null);
        }

        var prefillDelta = p - prior.Prefill;
        var decodeDelta = d - prior.Decode;
        if (prefillDelta > 0 || decodeDelta > 0) intervals.Enqueue(new(prior.Time, now, prefillDelta, decodeDelta));
        var cutoff = now.AddSeconds(-Seconds);
        while (intervals.TryPeek(out var oldest) && oldest.End <= cutoff) intervals.Dequeue();
        // Always advance counters and expire old work, including on idle polls. A later burst must
        // not inherit old tokens or the time spent waiting, even though the display holds its value.
        if (prefillDelta == 0 && decodeDelta == 0) return (lastPrefill, lastDecode);
        double prefillTokens = 0, decodeTokens = 0, prefillElapsed = 0, decodeElapsed = 0;
        foreach (var interval in intervals)
        {
            var start = interval.Start > cutoff ? interval.Start : cutoff;
            var seconds = (interval.End - start).TotalSeconds;
            // Counters locate work between polls, not individual tokens in time. Interpolate only
            // the interval crossing the exact window boundary; weight by time, never sample count.
            var fraction = seconds / (interval.End - interval.Start).TotalSeconds;
            if (interval.Prefill > 0) { prefillTokens += interval.Prefill * fraction; prefillElapsed += seconds; }
            if (interval.Decode > 0) { decodeTokens += interval.Decode * fraction; decodeElapsed += seconds; }
        }
        if (prefillDelta > 0 && prefillElapsed > 0) lastPrefill = prefillTokens / prefillElapsed;
        if (decodeDelta > 0 && decodeElapsed > 0) lastDecode = decodeTokens / decodeElapsed;
        return (lastPrefill, lastDecode);
    }
}
