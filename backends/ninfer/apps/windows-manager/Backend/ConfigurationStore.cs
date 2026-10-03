using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace NInfer.Manager;

/// <summary>Versioned on-disk configuration. A failed read never becomes an empty replacement.</summary>
public sealed class ConfigurationStore
{
    public static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web) { WriteIndented = true };
    private readonly object gate = new();
    private readonly string configRoot;
    private ManagerSettings settings;
    private readonly Dictionary<string, LaunchProfile> profiles = new(StringComparer.Ordinal);
    public string PackageRoot { get; }
    public string DataRoot { get; }
    public ManagerSettings Settings { get { lock (gate) return Clone(settings); } }
    public IReadOnlyList<LaunchProfile> Profiles { get { lock (gate) return profiles.Values.OrderBy(p => p.Name, StringComparer.OrdinalIgnoreCase).Select(Clone).ToArray(); } }

    public ConfigurationStore(ManagerPaths paths, string? seedRoot = null)
    {
        PackageRoot = paths.PackageRoot;
        DataRoot = paths.DataRoot;
        configRoot = paths.ConfigRoot;
        ImportPackageConfiguration();
        Directory.CreateDirectory(Path.Combine(configRoot, "profiles"));
        Directory.CreateDirectory(Path.Combine(configRoot, "history"));
        var settingsPath = Path.Combine(configRoot, "settings.json");
        settings = File.Exists(settingsPath) ? Read<ManagerSettings>(settingsPath) : ReadSeed<ManagerSettings>("settings.json", seedRoot);
        ValidateSettings(settings);
        foreach (var path in Directory.EnumerateFiles(Path.Combine(configRoot, "profiles"), "*.json"))
        {
            var profile = ReadProfile(path);
            if (!string.Equals(Path.GetFileNameWithoutExtension(path), profile.Id, StringComparison.Ordinal))
                throw new InvalidDataException($"Profile filename does not match its ID: {path}");
            ValidateProfile(profile);
            if (!profiles.TryAdd(profile.Id, profile)) throw new InvalidDataException($"Duplicate profile ID: {profile.Id}");
        }
        // Import defaults once. Deleting a profile later must not silently resurrect it.
        var initialized = Path.Combine(configRoot, "initialized.json");
        if (!File.Exists(initialized))
        {
            var incoming = SeedProfiles(seedRoot).Select(path => ReadSeed<LaunchProfile>(path, seedRoot)).ToArray();
            foreach (var profile in incoming) ValidateProfile(profile);
            foreach (var profile in incoming)
                if (!profiles.ContainsKey(profile.Id)) { WriteProfile(profile); profiles.Add(profile.Id, profile); }
            AtomicWrite(initialized, new { schemaVersion = 1, createdAt = DateTimeOffset.UtcNow });
        }
        foreach (var file in new[] { "chat_template.jinja", "chat_template.LICENSE", "device-profiles.json" })
        {
            var target = Path.Combine(configRoot, file);
            if (File.Exists(target)) continue;
            using var source = OpenSeed(file, seedRoot);
            using var destination = new FileStream(target, FileMode.CreateNew, FileAccess.Write);
            source.CopyTo(destination);
        }
        if (!File.Exists(settingsPath)) AtomicWrite(settingsPath, settings);
    }

    private void ImportPackageConfiguration()
    {
        var packageConfig = Path.Combine(PackageRoot, "config");
        if (string.Equals(PackageRoot, DataRoot, StringComparison.OrdinalIgnoreCase) ||
            !Directory.Exists(packageConfig) ||
            (Directory.Exists(configRoot) && Directory.EnumerateFiles(configRoot, "*", SearchOption.AllDirectories).Any())) return;

        // The package config is a first-run seed. User data is authoritative after initialization.
        foreach (var source in Directory.EnumerateFiles(packageConfig, "*", SearchOption.AllDirectories))
        {
            var target = Path.Combine(configRoot, Path.GetRelativePath(packageConfig, source));
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.Copy(source, target, false);
            File.SetAttributes(target, File.GetAttributes(target) & ~FileAttributes.ReadOnly);
        }
    }

    private const string SeedPrefix = "NInfer.Manager.Config/";
    private static Stream OpenSeed(string path, string? seedRoot) => seedRoot is not null
        ? File.OpenRead(Path.Combine(seedRoot, path))
        : typeof(ConfigurationStore).Assembly.GetManifestResourceStream(SeedPrefix + path)
            ?? throw new InvalidDataException($"The manager is missing its embedded configuration resource: {path}");
    private static T ReadSeed<T>(string path, string? seedRoot)
    {
        using var stream = OpenSeed(path, seedRoot);
        return JsonSerializer.Deserialize<T>(stream, Json) ?? throw new InvalidDataException($"Empty configuration resource: {path}");
    }
    private static IEnumerable<string> SeedProfiles(string? seedRoot) => seedRoot is not null
        ? Directory.EnumerateFiles(Path.Combine(seedRoot, "profiles"), "*.json").Select(path => "profiles/" + Path.GetFileName(path))
        : typeof(ConfigurationStore).Assembly.GetManifestResourceNames()
            .Where(name => name.StartsWith(SeedPrefix + "profiles/", StringComparison.Ordinal) && name.EndsWith(".json", StringComparison.Ordinal))
            .Select(name => name[SeedPrefix.Length..]);

    public void SaveProfile(LaunchProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ValidateProfile(profile);
        lock (gate)
        {
            var owned = Clone(profile);
            if (owned.Parameters.TryGetValue("--cuda-memory-policy", out var policy))
                owned.Parameters["--cuda-memory-policy"] = CanonicalMemoryPolicy(policy);
            Archive(ProfilePath(owned.Id), "profile-" + owned.Id);
            WriteProfile(owned);
            profiles[owned.Id] = owned;
        }
    }

    public void DeleteProfile(string id)
    {
        ValidateId(id);
        lock (gate)
        {
            if (!profiles.ContainsKey(id)) throw new KeyNotFoundException("Profile does not exist.");
            if (settings.DefaultProfileId == id) throw new ArgumentException("Choose another default profile before deleting this profile.");
            Archive(ProfilePath(id), "deleted-profile-" + id);
            File.Delete(ProfilePath(id));
            profiles.Remove(id);
        }
    }

    public void SaveSettings(ManagerSettings value)
    {
        ArgumentNullException.ThrowIfNull(value);
        ValidateSettings(value);
        lock (gate)
        {
            if (!profiles.ContainsKey(value.DefaultProfileId)) throw new ArgumentException("Default profile does not exist.");
            Archive(Path.Combine(configRoot, "settings.json"), "settings");
            AtomicWrite(Path.Combine(configRoot, "settings.json"), value);
            settings = Clone(value);
        }
    }

    public LaunchSpec BuildLaunchSpec(string profileId)
    {
        LaunchProfile profile;
        lock (gate) profile = profiles.TryGetValue(profileId, out var stored) ? Clone(stored) : throw new KeyNotFoundException("Profile does not exist.");
        ValidateProfile(profile);
        var executable = ResolvePath(profile.EnginePath);
        var modelPath = ResolvePath(profile.ModelPath);
        if (!File.Exists(executable)) throw new FileNotFoundException("Engine executable is missing.", executable);
        var model = ModelCatalog.ReadModel(modelPath, PackageRoot);
        if (!model.IsComplete) throw new InvalidDataException("Model is incomplete or invalid: " + model.Error);
        var context = Integer(profile.Parameters, "--max-context", 1, int.MaxValue, 163840);
        if (model.MaxContext is > 0 && context > model.MaxContext) throw new ArgumentException($"Context {context} exceeds model metadata limit {model.MaxContext}.");
        var port = Integer(profile.Parameters, "--port", 1, 65535, 18081);
        if (port == Settings.WebPort) throw new ArgumentException("Engine and management website must use different ports.");
        if (Integer(profile.Parameters, "--stats-port", 0, 65535, 0) == Settings.WebPort)
            throw new ArgumentException("Engine statistics and management website must use different ports.");
        var host = profile.Parameters.GetValueOrDefault("--host") ?? "127.0.0.1";
        var modelId = profile.Parameters.GetValueOrDefault("--model-id") ?? Path.GetFileNameWithoutExtension(modelPath);
        var label = DateTimeOffset.Now.ToString("yyyyMMdd-HHmmss-fff", CultureInfo.InvariantCulture) + "-" + profile.Id + "-" + Guid.NewGuid().ToString("N")[..6];
        var logs = Path.Combine(DataRoot, "logs");
        Directory.CreateDirectory(logs);
        var requests = profile.Parameters.TryGetValue("--request-log-jsonl", out var requestPath)
            ? ResolveDataPath(requestPath!) : Path.Combine(logs, label + ".requests.jsonl");
        Directory.CreateDirectory(Path.GetDirectoryName(requests)!);
        var arguments = new List<string> { modelPath };
        var parameters = new Dictionary<string, string?>(profile.Parameters, StringComparer.Ordinal)
        {
            ["--host"] = host, ["--port"] = port.ToString(CultureInfo.InvariantCulture),
            ["--model-id"] = modelId, ["--request-log-jsonl"] = requests,
            ["--max-context"] = context.ToString(CultureInfo.InvariantCulture)
        };
        foreach (var pair in parameters)
        {
            arguments.Add(pair.Key);
            if (pair.Value is null) continue;
            var value = pair.Value;
            if (pair.Key is "--chat-template" or "--context-cost-presets")
            {
                value = ResolveResourcePath(value);
                if (!File.Exists(value)) throw new FileNotFoundException("Profile resource is missing.", value);
            }
            else if (pair.Key == "--device-profile-path")
                value = ResolveResourcePath(value);
            else if (pair.Key is "--prefix-cache-file" or "--disk-kv-path")
                value = ResolveDataPath(value);
            arguments.Add(value);
        }
        return new LaunchSpec(profile.Id, profile.Name, executable, PackageRoot, arguments,
            new Dictionary<string, string?>(profile.Environment), $"http://{(host == "localhost" ? "127.0.0.1" : host)}:{port}/v1", modelId,
            Path.Combine(logs, label + ".stdout.log"), Path.Combine(logs, label + ".stderr.log"), requests);
    }

    public string ResolvePath(string path) => Path.GetFullPath(Path.IsPathRooted(path) ? path : Path.Combine(PackageRoot, path));
    private string ResolveDataPath(string path) => Path.GetFullPath(Path.IsPathRooted(path) ? path : Path.Combine(DataRoot, path));
    private string ResolveResourcePath(string path)
    {
        if (Path.IsPathRooted(path)) return Path.GetFullPath(path);
        var relative = Path.GetRelativePath(PackageRoot, ResolvePath(path));
        return relative.StartsWith("config" + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)
            ? Path.GetFullPath(Path.Combine(DataRoot, relative)) : ResolvePath(path);
    }
    private string ProfilePath(string id) => Path.Combine(configRoot, "profiles", id + ".json");
    private void WriteProfile(LaunchProfile profile) => AtomicWrite(ProfilePath(profile.Id), new ProfileDocument(1, DateTimeOffset.UtcNow, profile));
    private sealed record ProfileDocument(int SchemaVersion, DateTimeOffset SavedAt, LaunchProfile Profile);
    private static T Clone<T>(T value) => JsonSerializer.Deserialize<T>(JsonSerializer.Serialize(value, Json), Json)!;
    private static T Read<T>(string path)
    {
        try { return JsonSerializer.Deserialize<T>(File.ReadAllText(path), Json) ?? throw new JsonException("Empty JSON value."); }
        catch (Exception ex) when (ex is JsonException or IOException) { throw new InvalidDataException($"Cannot read configuration {path}; it was preserved. {ex.Message}", ex); }
    }
    private static LaunchProfile ReadProfile(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        if (!document.RootElement.TryGetProperty("schemaVersion", out var schema) || schema.GetInt32() != 1)
            throw new InvalidDataException($"Unsupported profile schema: {path}");
        return document.RootElement.GetProperty("profile").Deserialize<LaunchProfile>(Json)
            ?? throw new InvalidDataException($"Empty profile: {path}");
    }
    private void Archive(string source, string prefix)
    {
        if (!File.Exists(source)) return;
        var name = $"{DateTimeOffset.UtcNow:yyyyMMddTHHmmssfffffffZ}-{prefix}-{Guid.NewGuid():N}.json";
        File.Copy(source, Path.Combine(configRoot, "history", name), false);
    }
    private static void AtomicWrite<T>(string path, T value)
    {
        var temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            using (var file = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096, FileOptions.WriteThrough))
            {
                JsonSerializer.Serialize(file, value, Json);
                file.Flush(true);
            }
            if (File.Exists(path)) File.Replace(temporary, path, null);
            else File.Move(temporary, path);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }
    private static void ValidateId(string id)
    {
        if (string.IsNullOrWhiteSpace(id) || !Regex.IsMatch(id, "^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$", RegexOptions.CultureInvariant))
            throw new ArgumentException("Profile ID must be 1–80 letters, digits, underscores or hyphens.");
    }
    private static void ValidateSettings(ManagerSettings value)
    {
        if (value.SchemaVersion != 1) throw new ArgumentException("Unsupported settings schema version.");
        if (value.Language is not ("zh" or "en")) throw new ArgumentException("Language must be zh or en.");
        if (value.WebPort is < 1024 or > 65535) throw new ArgumentException("Management port must be between 1024 and 65535.");
        ValidateId(value.DefaultProfileId);
        if (value.ModelDirectories is null || value.ModelDirectories.Count is < 1 or > 32 || value.ModelDirectories.Any(string.IsNullOrWhiteSpace))
            throw new ArgumentException("Supply between 1 and 32 model directories.");
        foreach (var directory in value.ModelDirectories) _ = Path.GetFullPath(directory);
    }
    public static void ValidateProfile(LaunchProfile value)
    {
        ValidateId(value.Id);
        if (string.IsNullOrWhiteSpace(value.Name) || value.Name.Length > 200) throw new ArgumentException("Profile name is required and limited to 200 characters.");
        if (string.IsNullOrWhiteSpace(value.ModelPath) || !value.ModelPath.EndsWith(".ninfer", StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("Choose a .ninfer model entry file.");
        if (string.IsNullOrWhiteSpace(value.EnginePath) || !value.EnginePath.EndsWith(".exe", StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("Engine path must point to an executable.");
        if (value.Parameters is null || value.Environment is null || value.Parameters.Count > 256 || value.Environment.Count > 64) throw new ArgumentException("Invalid parameter or environment dictionary.");
        foreach (var pair in value.Parameters)
        {
            if (!Regex.IsMatch(pair.Key, "^--[A-Za-z][A-Za-z0-9_-]*$", RegexOptions.CultureInvariant)) throw new ArgumentException($"Invalid argument name: {pair.Key}");
            if (pair.Value?.Contains('\0') == true || pair.Value?.Length > 32768) throw new ArgumentException($"Invalid value for {pair.Key}.");
        }
        foreach (var pair in value.Environment)
            if (string.IsNullOrWhiteSpace(pair.Key) || pair.Key.Contains('=') || pair.Key.Contains('\0') || pair.Value?.Contains('\0') == true) throw new ArgumentException("Invalid environment variable.");
        var p = value.Parameters;
        foreach (var option in new[] { "--chat-template", "--device-profile-path", "--context-cost-presets", "--prefix-cache-file", "--disk-kv-path", "--request-log-jsonl" })
            if (p.TryGetValue(option, out var path) && string.IsNullOrWhiteSpace(path))
                throw new ArgumentException($"{option} requires a non-empty path.");
        var memoryPolicy = p.TryGetValue("--cuda-memory-policy", out var policy) ? CanonicalMemoryPolicy(policy) : "default";
        if (memoryPolicy != "default")
        {
            foreach (var headroom in new[] { "--kv-headroom-mib", "--vram-headroom-mib" })
                if (p.ContainsKey(headroom)) throw new ArgumentException($"{headroom} is available only with --cuda-memory-policy default.");
            if (p.ContainsKey("--vision") || p.ContainsKey("--vision-cpu") || p.ContainsKey("--wddm-evictable-budget") ||
                (p.TryGetValue("--devices", out var devices) && devices?.Split(',').Length > 1))
                throw new ArgumentException("--cuda-memory-policy mixed/strict requires one GPU, text-only inference and ordinary CUDA allocation without --wddm-evictable-budget.");
        }
        var strict = memoryPolicy == "strict" || memoryPolicy.StartsWith("strict-", StringComparison.Ordinal);
        var hybrid = strict || p.ContainsKey("--use-alt-prefix-caching");
        if (hybrid && (p.ContainsKey("--use-original-prefix-caching") || p.ContainsKey("--no-prefix-reuse")))
            throw new ArgumentException("Hybrid prefix caching cannot be combined with --use-original-prefix-caching or --no-prefix-reuse; strict enables the hybrid cache automatically.");
        foreach (var option in new[] { "--device-snapshot-slots", "--cache-taps-per-request", "--cache-tap-ladder", "--cache-tap-min-gap", "--prefix-cache-file" })
            if (!hybrid && p.ContainsKey(option)) throw new ArgumentException($"{option} requires --use-alt-prefix-caching with default/mixed, or --cuda-memory-policy strict.");
        if (p.ContainsKey("--device-snapshot-slots")) Integer(p, "--device-snapshot-slots", 1, 64, 1);
        if (p.TryGetValue("--host", out var host) && host is not "127.0.0.1" and not "localhost") throw new ArgumentException("The manager supports a local engine host (127.0.0.1 or localhost).");
        var port = Integer(p, "--port", 1, 65535, 18081);
        var statsPort = Integer(p, "--stats-port", 0, 65535, 0);
        if (statsPort == port) throw new ArgumentException("--stats-port must differ from --port; 0 disables the separate statistics listener.");
        if (p.TryGetValue("--api-key", out var apiKey) && (apiKey is null || apiKey.Contains('\r') || apiKey.Contains('\n')))
            throw new ArgumentException("--api-key must be a string without line breaks; an empty string disables authentication.");
        var context = Integer(p, "--max-context", 1, int.MaxValue, 163840);
        var output = Integer(p, "--default-max-tokens", 0, int.MaxValue, 0);
        if (output > context) throw new ArgumentException("Default output tokens must not exceed the context limit; 0 means engine default without a fixed cap.");
        if (p.TryGetValue("--kv-capacity", out var kv) && kv != "auto" && Integer(p, "--kv-capacity", 1, int.MaxValue, context) < context) throw new ArgumentException("KV capacity must cover the requested context.");
        Integer(p, "--prefill-chunk", 1, int.MaxValue, 640);
        var concurrency = Integer(p, "--max-concurrency", 1, 8, 1);
        Integer(p, "--cuda-graph-allowance-mib", 0, int.MaxValue, 72);
        Integer(p, "--host-cache-mib", 0, int.MaxValue, 6144);
        ValidateSpeculativeOptions(p, concurrency);
        ValidateVisionOptions(p);
        Integer(p, "--top-k", 0, int.MaxValue, 20);
        if (p.TryGetValue("--model-id", out var modelId) && (string.IsNullOrWhiteSpace(modelId) || modelId.Length > 256)) throw new ArgumentException("Model ID is required and limited to 256 characters.");
        if (p.TryGetValue("--default-reasoning-effort", out var effort) && effort is not "none" and not "minimal" and not "low" and not "medium" and not "high" and not "xhigh" and not "max") throw new ArgumentException("Unknown reasoning effort.");
        foreach (var key in new[] { "--temperature", "--top-p", "--min-p", "--presence-penalty", "--frequency-penalty" })
            if (p.TryGetValue(key, out var number) && (!double.TryParse(number, NumberStyles.Float, CultureInfo.InvariantCulture, out var numeric) || !double.IsFinite(numeric))) throw new ArgumentException($"{key} must be a finite number.");
    }
    private static void ValidateSpeculativeOptions(IReadOnlyDictionary<string, string?> p, int concurrency)
    {
        var enabled = p.TryGetValue("--spec", out var backend);
        if (enabled && backend is not ("mtp" or "dflash" or "dflash2"))
            throw new ArgumentException("--spec must be mtp, dflash or dflash2; omit it to disable speculative decoding.");
        var drafts = Integer(p, "--draft-tokens", 0, 15, 0);
        var proposal = Flag(p, "--lm-head-draft");
        var adaptive = Flag(p, "--adaptive-mtp");
        if (enabled && drafts == 0)
            throw new ArgumentException("--spec requires --draft-tokens between 1 and 15.");
        if (!enabled && (drafts != 0 || proposal))
            throw new ArgumentException("--draft-tokens and --lm-head-draft require --spec mtp, dflash or dflash2.");
        if (adaptive && backend != "mtp")
            throw new ArgumentException("--adaptive-mtp requires --spec mtp.");
        var window = Integer(p, "--mtp-attention-window", 0, int.MaxValue, 0);
        if (window != 0 && (backend != "mtp" || window < drafts + 1))
            throw new ArgumentException("--mtp-attention-window requires --spec mtp and at least --draft-tokens + 1 keys; 0 uses the whole history.");
        var ngram = Integer(p, "--ngram-draft-tokens", 0, 63, enabled ? 15 : 0);
        Integer(p, "--ngram-min-match", 4, 64, 12);
        if (ngram != 0 && !enabled)
            throw new ArgumentException("--ngram-draft-tokens above 0 requires --spec mtp, dflash or dflash2.");
        if (ngram > 15 && concurrency != 1)
            throw new ArgumentException("--ngram-draft-tokens above 15 requires --max-concurrency 1.");
        var archive = Mebibytes(p, "--ngram-archive-mib", 0, 0);
        var session = Mebibytes(p, "--ngram-session-mib", 0, 128);
        if (archive > 0 && (ngram == 0 || session == 0 || session > archive))
            throw new ArgumentException("--ngram-archive-mib above 0 requires ngram drafting and --ngram-session-mib between 1 and the total archive MiB.");
        if (Flag(p, "--ngram-native-sessions") && archive == 0)
            throw new ArgumentException("--ngram-native-sessions requires --ngram-archive-mib above 0.");
        // The server permits a dormant lookup value without --spec; retain it for later use.
        Integer(p, "--lookup-ngram", 0, int.MaxValue, 0);
    }
    private static void ValidateVisionOptions(IReadOnlyDictionary<string, string?> p)
    {
        var enabled = Flag(p, "--vision");
        var cpu = Flag(p, "--vision-cpu");
        var residency = p.TryGetValue("--vision-residency", out var selected) ? selected : "resident";
        if (residency is not ("resident" or "overlay" or "cpu"))
            throw new ArgumentException("--vision-residency must be resident, overlay or cpu.");
        if (cpu && p.ContainsKey("--vision-residency") && residency != "cpu")
            throw new ArgumentException("--vision-cpu conflicts with a --vision-residency other than cpu.");
        if (cpu) residency = "cpu";
        if (p.TryGetValue("--vision-offload", out var offload))
        {
            if (offload is not ("on" or "off")) throw new ArgumentException("--vision-offload must be on or off.");
            var offloadResidency = offload == "on" ? "overlay" : "resident";
            if ((cpu || p.ContainsKey("--vision-residency")) && residency != offloadResidency)
                throw new ArgumentException("--vision-offload conflicts with the selected Vision residency.");
            residency = offloadResidency;
        }
        if (residency != "resident" && !enabled && !cpu)
            throw new ArgumentException("--vision-residency overlay or cpu requires --vision or --vision-cpu.");
        Integer(p, "--vision-max-merged", 64, 16384, residency == "cpu" ? 256 : 16384);
        Mebibytes(p, "--media-cache-mib", 0, 1024);
        Mebibytes(p, "--media-live-mib", 1, 2048);
        Integer(p, "--media-preprocess-threads", 0, 64, 0);
    }
    private static bool Flag(IReadOnlyDictionary<string, string?> p, string option)
    {
        if (!p.TryGetValue(option, out var value)) return false;
        if (value is not null) throw new ArgumentException($"{option} is a switch and does not accept a value.");
        return true;
    }
    private static ulong Mebibytes(IReadOnlyDictionary<string, string?> p, string option, ulong minimum, ulong fallback)
    {
        if (!p.TryGetValue(option, out var raw)) return fallback;
        const ulong maximum = ulong.MaxValue / (1024UL * 1024UL);
        if (!ulong.TryParse(raw, NumberStyles.None, CultureInfo.InvariantCulture, out var value) || value < minimum || value > maximum)
            throw new ArgumentException($"{option} must be an integer between {minimum} and {maximum} MiB.");
        return value;
    }
    private static string CanonicalMemoryPolicy(string? policy)
    {
        if (policy is "default" or "mixed" or "strict") return policy;
        var match = Regex.Match(policy ?? "", "^strict-([0-9]+)-([0-9]+)\\z", RegexOptions.CultureInvariant);
        if (!match.Success ||
            !ulong.TryParse(match.Groups[1].Value, NumberStyles.None, CultureInfo.InvariantCulture, out var reserve) ||
            reserve > ulong.MaxValue / (1024UL * 1024UL) ||
            !uint.TryParse(match.Groups[2].Value, NumberStyles.None, CultureInfo.InvariantCulture, out var step) ||
            step is < 1 or > 16384)
            throw new ArgumentException("--cuda-memory-policy must be default, mixed, strict, or strict-RESERVE-STEP; RESERVE is 0..17592186044415 MiB and STEP is 1..16384 MiB. strict uses 64/128 MiB.");
        return reserve == 64 && step == 128 ? "strict" :
            "strict-" + reserve.ToString(CultureInfo.InvariantCulture) + "-" + step.ToString(CultureInfo.InvariantCulture);
    }
    private static int Integer(IReadOnlyDictionary<string, string?> parameters, string name, int minimum, int maximum, int fallback)
    {
        if (!parameters.TryGetValue(name, out var raw)) return fallback;
        if (!int.TryParse(raw, NumberStyles.None, CultureInfo.InvariantCulture, out var value) || value < minimum || value > maximum)
            throw new ArgumentException($"{name} must be an integer between {minimum} and {maximum}.");
        return value;
    }
}
