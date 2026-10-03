namespace NInfer.Manager;

public sealed record LaunchProfile
{
    public string Id { get; init; } = Guid.NewGuid().ToString("N");
    public string Name { get; init; } = "New profile";
    public string ModelPath { get; init; } = "";
    public string EnginePath { get; init; } = "engine/ninfer-serve.exe";
    public Dictionary<string, string?> Parameters { get; init; } = new();
    public Dictionary<string, string?> Environment { get; init; } = new();
}

public sealed record ManagerSettings
{
    public int SchemaVersion { get; init; } = 1;
    public int WebPort { get; init; } = 8090;
    public List<string> ModelDirectories { get; init; } = ["model"];
    public string DefaultProfileId { get; init; } = "xxs-160k";
    public bool StartWithWindows { get; init; } = false;
    public bool AutoStartModel { get; init; } = true;
    public string Language { get; init; } = "zh";
}

public sealed record LaunchSpec(string ProfileId, string ProfileName, string Executable,
    string WorkingDirectory, IReadOnlyList<string> Arguments,
    IReadOnlyDictionary<string, string?> Environment, string ApiBase, string ModelId,
    string StdoutPath, string StderrPath, string RequestLogPath);

public sealed record EngineSnapshot(string State, string? ProfileId = null,
    string? ProfileName = null, int? Pid = null, string? ApiBase = null,
    string? ModelId = null, string? Error = null, string? StdoutPath = null,
    string? StderrPath = null, string? RequestLogPath = null,
    DateTimeOffset? StartedAt = null, LaunchSpec? Launch = null);

public interface IEngineController : IAsyncDisposable
{
    EngineSnapshot Snapshot { get; }
    event Action? Changed;
    Task StartAsync(LaunchSpec spec, CancellationToken cancellationToken = default);
    Task StopAsync(CancellationToken cancellationToken = default);
}
