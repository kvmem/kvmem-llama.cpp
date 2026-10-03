using System.Security.Cryptography;
using System.Text;

namespace NInfer.Manager;

/// <summary>Separates read-only package files from this installation's writable user data.</summary>
public sealed record ManagerPaths(string PackageRoot, string DataRoot, string InstallationId)
{
    public string ConfigRoot => Path.Combine(DataRoot, "config");
    public string RuntimeRoot => Path.Combine(DataRoot, "runtime");
    public string LogsRoot => Path.Combine(DataRoot, "logs");
    public bool IsPortable => string.Equals(PackageRoot, DataRoot, StringComparison.OrdinalIgnoreCase);

    public static ManagerPaths Select(string packageRoot, string? localAppDataRoot = null)
    {
        var package = Path.TrimEndingDirectorySeparator(Path.GetFullPath(packageRoot));
        var id = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(package.ToLowerInvariant())))[..24];
        var local = localAppDataRoot ?? Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var failures = new List<string>();
        if (!string.IsNullOrWhiteSpace(local))
        {
            var candidate = Path.Combine(local, "NInferManager", id);
            if (TryPrepare(candidate, out var error)) return new(package, Path.GetFullPath(candidate), id);
            failures.Add(candidate + ": " + error);
        }
        else failures.Add("LocalApplicationData is unavailable.");

        if (TryPrepare(package, out var packageError)) return new(package, package, id);
        failures.Add(package + ": " + packageError);
        throw new IOException("无法创建可写的数据目录。请检查本地应用数据目录或程序目录的写入权限。\n" +
            "Cannot create writable manager data. Check LocalAppData or package directory permissions.\n" + string.Join("\n", failures));
    }

    private static bool TryPrepare(string root, out string? error)
    {
        try
        {
            // Validate every directory that the running manager writes, not just its parent.
            foreach (var relative in new[] { "", "config", "config/profiles", "config/history", "runtime", "logs" })
            {
                var directory = Path.Combine(root, relative);
                Directory.CreateDirectory(directory);
                var probe = Path.Combine(directory, ".write-probe-" + Guid.NewGuid().ToString("N"));
                using var file = new FileStream(probe, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1, FileOptions.DeleteOnClose);
                file.WriteByte(0);
            }
            error = null;
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.Security.SecurityException or ArgumentException or NotSupportedException)
        {
            error = ex.Message;
            return false;
        }
    }
}
