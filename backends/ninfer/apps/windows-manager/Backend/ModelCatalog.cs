using System.Buffers.Binary;
using System.Text.Json;

namespace NInfer.Manager;

public sealed record ModelEntry(string Path, string Name, string? ArtifactId, string? Architecture,
    long? MaxContext, IReadOnlyList<string> Components, IReadOnlyList<string> ResourcePaths,
    long FileBytes, int PartCount, bool IsComplete, string? Error);

/// <summary>Reads only the bounded v3 directory and volume headers; never materializes tensor data.</summary>
public sealed class ModelCatalog(ConfigurationStore store)
{
    private readonly SemaphoreSlim scanGate = new(1, 1);
    private IReadOnlyList<ModelEntry> models = [];
    public IReadOnlyList<ModelEntry> Models => Volatile.Read(ref models);
    public Task<IReadOnlyList<ModelEntry>> ScanAsync(CancellationToken cancellationToken = default) => Task.Run(async () =>
    {
        await scanGate.WaitAsync(cancellationToken);
        try
        {
            var found = new List<ModelEntry>();
            var visited = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var options = new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = true, AttributesToSkip = FileAttributes.ReparsePoint | FileAttributes.System };
            foreach (var root in store.Settings.ModelDirectories.Select(store.ResolvePath))
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!Directory.Exists(root))
                {
                    found.Add(new(root, System.IO.Path.GetFileName(root), null, null, null, [], [], 0, 0, false, "Model directory does not exist."));
                    continue;
                }
                try
                {
                    foreach (var path in Directory.EnumerateFiles(root, "*.ninfer", options))
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        if (!visited.Add(System.IO.Path.GetFullPath(path))) continue;
                        if (visited.Count > 10000) throw new InvalidDataException("Model scan exceeded 10,000 entries. Choose a narrower directory.");
                        found.Add(ReadModel(path, store.PackageRoot));
                    }
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    found.Add(new(root, System.IO.Path.GetFileName(root), null, null, null, [], [], 0, 0, false, ex.Message));
                }
            }
            var result = found.OrderBy(m => m.Name, StringComparer.OrdinalIgnoreCase).ToArray();
            Volatile.Write(ref models, result);
            return (IReadOnlyList<ModelEntry>)result;
        }
        finally { scanGate.Release(); }
    }, cancellationToken);

    public static ModelEntry ReadModel(string path, string packageRoot)
    {
        path = System.IO.Path.GetFullPath(path);
        var relative = System.IO.Path.GetRelativePath(packageRoot, path);
        var displayPath = relative.StartsWith(".." + System.IO.Path.DirectorySeparatorChar, StringComparison.Ordinal) || System.IO.Path.IsPathRooted(relative) ? path : relative.Replace('\\', '/');
        string name = System.IO.Path.GetFileNameWithoutExtension(path);
        string? artifactId = null, architecture = null;
        long? maxContext = null;
        long fileBytes = 0;
        var components = new List<string>();
        var resources = new List<string>();
        int partCount = 0;
        try
        {
            using var entry = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete, 4096, FileOptions.RandomAccess);
            Span<byte> header = stackalloc byte[32];
            entry.ReadExactly(header);
            if (!header[..8].SequenceEqual("NINFER\0\x03"u8)) throw new InvalidDataException("Not a NInfer v3 entry file.");
            var jsonBytes = BinaryPrimitives.ReadUInt64LittleEndian(header[8..16]);
            if (jsonBytes is 0 or > 64 * 1024 * 1024 || jsonBytes > (ulong)Math.Max(0, entry.Length - 32)) throw new InvalidDataException("Invalid or oversized directory JSON (limit 64 MiB).");
            artifactId = Convert.ToHexString(header[16..32]).ToLowerInvariant();
            var artifactBytes = header[16..32].ToArray();
            var directoryBytes = new byte[(int)jsonBytes];
            entry.ReadExactly(directoryBytes);
            using var document = JsonDocument.Parse(directoryBytes, new JsonDocumentOptions { MaxDepth = 64 });
            var root = document.RootElement;
            if (root.TryGetProperty("metadata", out var metadata) && metadata.TryGetProperty("name", out var metadataName)) name = metadataName.GetString() ?? name;
            var componentMap = root.GetProperty("components");
            if (!componentMap.TryGetProperty("text", out var text)) throw new InvalidDataException("Missing text component.");
            foreach (var component in componentMap.EnumerateObject())
            {
                components.Add(component.Name);
                if (component.Value.TryGetProperty("resources", out var map))
                    foreach (var resource in map.EnumerateObject())
                    {
                        var objectId = resource.Value.GetString() ?? throw new InvalidDataException("Invalid resource ID.");
                        resources.Add(objectId);
                    }
            }
            var config = text.GetProperty("config");
            if (config.TryGetProperty("architectures", out var architectures) && architectures.ValueKind == JsonValueKind.Array)
                architecture = string.Join(", ", architectures.EnumerateArray().Select(a => a.GetString()));
            if (config.TryGetProperty("max_position_embeddings", out var context) && context.TryGetInt64(out var contextValue)) maxContext = contextValue;
            var files = root.GetProperty("files");
            if (files.ValueKind != JsonValueKind.Array || files.GetArrayLength() is < 1 or > 10000) throw new InvalidDataException("Invalid volume directory.");
            long totalPayload = 0;
            var siblingNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { System.IO.Path.GetFileName(path) };
            var parent = System.IO.Path.GetDirectoryName(path)!;
            foreach (var file in files.EnumerateArray())
            {
                var payload = file.GetProperty("payload_bytes").GetInt64();
                if (payload <= 0) throw new InvalidDataException("Invalid volume payload length.");
                totalPayload = checked(totalPayload + payload);
                if (partCount == 0)
                {
                    if (file.GetProperty("path").ValueKind != JsonValueKind.Null) throw new InvalidDataException("The first volume path must be null.");
                    var aligned = checked(((32L + (long)jsonBytes + 4095) / 4096) * 4096);
                    if (entry.Length != checked(aligned + payload)) throw new InvalidDataException("Entry file length does not match its directory; conversion or copy may be incomplete.");
                    fileBytes = entry.Length;
                }
                else
                {
                    var sibling = file.GetProperty("path").GetString();
                    if (string.IsNullOrWhiteSpace(sibling) || sibling.IndexOfAny(['/', '\\', ':']) >= 0 || sibling is "." or ".." || !siblingNames.Add(sibling)) throw new InvalidDataException("Invalid or duplicate continuation filename.");
                    var siblingPath = System.IO.Path.Combine(parent, sibling);
                    using var part = new FileStream(siblingPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete, 4096, FileOptions.RandomAccess);
                    part.ReadExactly(header);
                    if (!header[..8].SequenceEqual("NINPRT\0\x03"u8) || BinaryPrimitives.ReadUInt64LittleEndian(header[8..16]) != (ulong)partCount || !header[16..32].SequenceEqual(artifactBytes)) throw new InvalidDataException($"Continuation header does not belong to this model: {sibling}");
                    if (part.Length != checked(4096 + payload)) throw new InvalidDataException($"Continuation length does not match its directory: {sibling}");
                    fileBytes = checked(fileBytes + part.Length);
                }
                partCount++;
            }
            var availableResources = new HashSet<string>(StringComparer.Ordinal);
            foreach (var obj in root.GetProperty("objects").EnumerateArray())
            {
                var offset = obj.GetProperty("offset").GetInt64();
                var bytes = obj.GetProperty("bytes").GetInt64();
                if (offset < 0 || bytes < 0 || checked(offset + bytes) > totalPayload) throw new InvalidDataException("Object extends outside declared payload.");
                if (obj.GetProperty("kind").GetString() == "resource") availableResources.Add(obj.GetProperty("id").GetString()!);
            }
            foreach (var resource in resources)
                if (!availableResources.Contains(resource)) throw new InvalidDataException($"Missing declared resource object: {resource}");
            return new(displayPath, name, artifactId, architecture, maxContext, components, resources, fileBytes, partCount, true, null);
        }
        catch (Exception ex) when (ex is InvalidDataException or IOException or UnauthorizedAccessException or JsonException or KeyNotFoundException or InvalidOperationException or OverflowException or FormatException or ArgumentException)
        {
            return new(displayPath, name, artifactId, architecture, maxContext, components, resources, fileBytes, partCount, false, ex.Message);
        }
    }
}
