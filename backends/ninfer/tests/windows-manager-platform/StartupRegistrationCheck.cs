using Microsoft.Win32;
using NInfer.Manager;

internal static class StartupRegistrationTests
{
    public static int Run()
    {
        var id = Guid.NewGuid().ToString("N");
        var registryPath = @"Software\NInfer\Tests\Startup-" + id;
        var temporaryRoot = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "ninfer-startup-" + id));
        var firstRoot = Path.Combine(temporaryRoot, "First install 中文");
        var secondRoot = Path.Combine(temporaryRoot, "Second install");
        var firstExe = Path.Combine(firstRoot, "NInferManager.exe");
        var secondExe = Path.Combine(secondRoot, "NInferManager.exe");
        Directory.CreateDirectory(firstRoot);
        Directory.CreateDirectory(secondRoot);
        File.WriteAllBytes(firstExe, []);
        File.WriteAllBytes(secondExe, []);
        var passed = 0;
        void Check(bool success, string name)
        {
            if (!success) throw new Exception("FAIL startup " + name);
            passed++;
            Console.WriteLine("PASS startup " + name);
        }
        try
        {
            using var hive = Registry.CurrentUser.CreateSubKey(registryPath, true)!;
            using var run = hive.CreateSubKey("Run", true)!;
            using var approved = hive.CreateSubKey("Approved", true)!;
            var registration = new StartupRegistry(hive, "Run", "Approved");
            const string fixedName = StartupRegistry.ValueName;
            Check(!registration.GetStatus(firstExe, firstRoot).IsRegistered, "absent registration is not owned");

            run.SetValue("OtherApplication", "unrelated.exe --start");
            run.SetValue("NInferManager-not-ours", "other.exe --root elsewhere --autostart");
            run.SetValue("NInferManager-malformed", "NInferManager.exe --unexpected");
            run.SetValue("NInferManager-first", StartupRegistry.BuildCommand(firstExe, firstRoot));
            run.SetValue("NInferManager-second", StartupRegistry.BuildCommand(secondExe, secondRoot));
            registration.SetEnabled(firstExe, firstRoot + Path.DirectorySeparatorChar, true);
            Check(registration.GetStatus(firstExe, firstRoot).IsEnabled, "quoted paths and trailing separator round trip");
            Check(registration.GetStatus(firstExe.ToUpperInvariant(), firstRoot.ToUpperInvariant()).IsRegistered,
                "ownership uses case-insensitive Windows paths");
            Check(run.GetValue("NInferManager-first") is null && run.GetValue("NInferManager-second") is null,
                "registration removes previous manager startup commands");
            Check((string?)run.GetValue("OtherApplication") == "unrelated.exe --start" &&
                run.GetValue("NInferManager-not-ours") is not null && run.GetValue("NInferManager-malformed") is not null,
                "registration preserves other applications and unrecognized commands");
            Check(!registration.GetStatus(firstExe, secondRoot).IsRegistered &&
                !registration.GetStatus(secondExe, firstRoot).IsRegistered, "ownership checks both executable and package root");

            registration.SetEnabled(secondExe, secondRoot, true);
            Check(registration.GetStatus(secondExe, secondRoot).IsEnabled &&
                !registration.GetStatus(firstExe, firstRoot).IsRegistered, "last registration takes ownership");
            registration.SetEnabled(firstExe, firstRoot, false);
            Check(registration.GetStatus(secondExe, secondRoot).IsRegistered,
                "older installation cannot unregister the newer installation");
            registration.SetEnabled(secondExe, secondRoot, false);
            Check(run.GetValue(fixedName) is null, "current owner can unregister itself");

            var disabled = new byte[] { 3, 0, 0, 0, 11, 12, 13, 14, 15, 16, 17, 18 };
            approved.SetValue(fixedName, disabled, RegistryValueKind.Binary);
            registration.SetEnabled(firstExe, firstRoot, true);
            var status = registration.GetStatus(firstExe, firstRoot);
            Check(status.IsRegistered && status.DisabledByWindows && !status.IsEnabled,
                "Windows-disabled registration retains ownership but is not enabled");
            Check(((byte[])approved.GetValue(fixedName)!).SequenceEqual(disabled), "registration preserves Windows disable bytes");
            registration.SetEnabled(secondExe, secondRoot, true);
            Check(registration.GetStatus(secondExe, secondRoot).DisabledByWindows &&
                ((byte[])approved.GetValue(fixedName)!).SequenceEqual(disabled), "changing installation does not bypass Windows disable");
            registration.SetEnabled(secondExe, secondRoot, false);
            Check(((byte[])approved.GetValue(fixedName)!).SequenceEqual(disabled), "unregister leaves Windows approval choice intact");

            approved.DeleteValue(fixedName, false);
            run.SetValue("NInferManager-disabled", StartupRegistry.BuildCommand(firstExe, firstRoot));
            approved.SetValue("NInferManager-disabled", disabled, RegistryValueKind.Binary);
            registration.SetEnabled(secondExe, secondRoot, true);
            Check(registration.GetStatus(secondExe, secondRoot).DisabledByWindows &&
                ((byte[])approved.GetValue(fixedName)!).SequenceEqual(disabled), "single registration preserves prior Windows disable choice");
            Check(run.GetValue("NInferManager-disabled") is null &&
                ((byte[])approved.GetValue("NInferManager-disabled")!).SequenceEqual(disabled), "cleanup only removes manager Run values");

            approved.SetValue(fixedName, new byte[] { 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, RegistryValueKind.Binary);
            Check(registration.GetStatus(secondExe, secondRoot).DisabledByWindows, "alternate Windows disabled state is shown");
            approved.SetValue(fixedName, new byte[] { 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, RegistryValueKind.Binary);
            Check(registration.GetStatus(secondExe, secondRoot).IsEnabled, "Windows enable choice is reflected on next read");
            var before = (string)run.GetValue(fixedName)!;
            try
            {
                registration.SetEnabled(Path.Combine(temporaryRoot, "missing.exe"), firstRoot, true);
                throw new Exception("Missing startup executable accepted");
            }
            catch (FileNotFoundException)
            { Check((string?)run.GetValue(fixedName) == before, "failed registration leaves current owner intact"); }

            var driveRoot = Path.GetPathRoot(firstRoot)!;
            registration.SetEnabled(firstExe, driveRoot, true);
            Check(registration.GetStatus(firstExe, driveRoot).IsEnabled, "drive-root trailing backslash parses correctly");
        }
        finally
        {
            Registry.CurrentUser.DeleteSubKeyTree(registryPath, false);
            var parent = Path.TrimEndingDirectorySeparator(Path.GetFullPath(Path.GetTempPath())) + Path.DirectorySeparatorChar;
            if (!temporaryRoot.StartsWith(parent, StringComparison.OrdinalIgnoreCase) || Path.GetFileName(temporaryRoot) != "ninfer-startup-" + id)
                throw new InvalidOperationException("Unexpected startup test cleanup path.");
            Directory.Delete(temporaryRoot, true);
        }
        Console.WriteLine($"ALL {passed} STARTUP CHECKS PASSED (isolated registry key)");
        return passed;
    }
}
