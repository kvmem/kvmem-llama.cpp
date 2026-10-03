using System.Runtime.InteropServices;
using Microsoft.Win32;

namespace NInfer.Manager;

public readonly record struct StartupRegistrationStatus(bool IsRegistered, bool DisabledByWindows)
{
    public bool IsEnabled => IsRegistered && !DisabledByWindows;
}

public static class StartupRegistration
{
    private static StartupRegistry Open() => new(Registry.CurrentUser,
        @"Software\Microsoft\Windows\CurrentVersion\Run",
        @"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run");

    public static StartupRegistrationStatus GetStatus(string exe, string root) => Open().GetStatus(exe, root);
    public static bool IsEnabled(string exe, string root) => GetStatus(exe, root).IsEnabled;
    public static void SetEnabled(string exe, string root, bool enabled) => Open().SetEnabled(exe, root, enabled);
}

// The registry root and paths are injected so tests never touch Windows startup entries.
internal sealed class StartupRegistry(RegistryKey hive, string runPath, string approvedPath)
{
    internal const string ValueName = "NInferManager";
    private const string Prefix = ValueName + "-";

    public StartupRegistrationStatus GetStatus(string exe, string root)
    {
        using var key = hive.OpenSubKey(runPath);
        using var approved = hive.OpenSubKey(approvedPath);
        return new(Matches(key?.GetValue(ValueName) as string, exe, root),
            IsDisabled(approved?.GetValue(ValueName) as byte[]));
    }

    public void SetEnabled(string exe, string root, bool enabled)
    {
        var fullExe = Path.GetFullPath(exe);
        var fullRoot = NormalizeRoot(root);
        if (enabled && !File.Exists(fullExe))
            throw new FileNotFoundException("Startup executable does not exist.", fullExe);

        // Ownership checks and writes must be one operation across installation copies.
        using var mutex = new Mutex(false, @"Local\NInferManager-StartupRegistration");
        try { mutex.WaitOne(); }
        catch (AbandonedMutexException) { /* The abandoned mutex is now owned by us. */ }
        try
        {
            using var key = hive.CreateSubKey(runPath, true)
                ?? throw new IOException("Could not open the current-user startup registry key.");
            if (!enabled)
            {
                if (Matches(key.GetValue(ValueName) as string, fullExe, fullRoot))
                    key.DeleteValue(ValueName, false);
                return;
            }

            var oldNames = key.GetValueNames().Where(name =>
                name.StartsWith(Prefix, StringComparison.OrdinalIgnoreCase) &&
                TryReadCommand(key.GetValue(name) as string, out var target, out _) &&
                string.Equals(Path.GetFileName(target), "NInferManager.exe", StringComparison.OrdinalIgnoreCase)).ToArray();
            using var approved = hive.OpenSubKey(approvedPath);
            // Changing the registration name must not evade a Windows startup-disable choice.
            if (!IsDisabled(approved?.GetValue(ValueName) as byte[]))
            {
                var disabled = oldNames.Select(name => approved?.GetValue(name) as byte[]).FirstOrDefault(IsDisabled);
                if (disabled is not null)
                {
                    using var writableApproved = hive.CreateSubKey(approvedPath, true)
                        ?? throw new IOException("Could not preserve the Windows startup-disabled state.");
                    writableApproved.SetValue(ValueName, disabled, RegistryValueKind.Binary);
                }
            }
            key.SetValue(ValueName, BuildCommand(fullExe, fullRoot), RegistryValueKind.String);
            foreach (var name in oldNames) key.DeleteValue(name, false);
            // Keep StartupApproved values, including disabled records, under Windows' control.
        }
        finally { mutex.ReleaseMutex(); }
    }

    internal static string BuildCommand(string exe, string root) =>
        $"{NativeProcess.QuoteArgument(Path.GetFullPath(exe))} --root {NativeProcess.QuoteArgument(NormalizeRoot(root))} --autostart";

    private static string NormalizeRoot(string root) => Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));

    private static bool Matches(string? command, string exe, string root) =>
        TryReadCommand(command, out var target, out var package) &&
        string.Equals(target, Path.GetFullPath(exe), StringComparison.OrdinalIgnoreCase) &&
        string.Equals(package, NormalizeRoot(root), StringComparison.OrdinalIgnoreCase);

    private static bool IsDisabled(byte[]? value) =>
        value is { Length: >= 4 } && BitConverter.ToUInt32(value, 0) is 3 or 7;

    private static bool TryReadCommand(string? command, out string exe, out string root)
    {
        exe = root = "";
        if (string.IsNullOrWhiteSpace(command)) return false;
        var arguments = CommandLineToArgvW(command, out var count);
        if (arguments == IntPtr.Zero) return false;
        try
        {
            if (count != 4) return false;
            string Arg(int index) => Marshal.PtrToStringUni(Marshal.ReadIntPtr(arguments, index * IntPtr.Size)) ?? "";
            if (Arg(1) != "--root" || Arg(3) != "--autostart" || Arg(0).Length == 0 || Arg(2).Length == 0) return false;
            try
            {
                exe = Path.GetFullPath(Arg(0));
                root = NormalizeRoot(Arg(2));
                return true;
            }
            catch (Exception error) when (error is ArgumentException or NotSupportedException or PathTooLongException)
            { return false; }
        }
        finally { LocalFree(arguments); }
    }

    [DllImport("shell32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CommandLineToArgvW(string commandLine, out int argumentCount);
    [DllImport("kernel32.dll")]
    private static extern IntPtr LocalFree(IntPtr memory);
}
