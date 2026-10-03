using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace NInfer.Manager;

/// <summary>A suspended child assigned to an owned, non-inherited kill-on-close Job before it runs.</summary>
public sealed class NativeProcess : IDisposable
{
    private SafeKernelHandle? job;
    private SafeKernelHandle? process;
    public int Id { get; }
    public DateTimeOffset CreatedAt { get; }

    private NativeProcess(SafeKernelHandle job, SafeKernelHandle process, int id)
    {
        this.job = job;
        this.process = process;
        Id = id;
        CreatedAt = DateTimeOffset.UtcNow;
    }

    public bool HasExited => process is null || WaitForSingleObject(process, 0) == 0;
    public uint ExitCode
    {
        get
        {
            if (process is null || !GetExitCodeProcess(process, out var code)) return uint.MaxValue;
            return code;
        }
    }

    public static NativeProcess Start(LaunchSpec spec)
    {
        var executable = Path.GetFullPath(spec.Executable);
        if (!File.Exists(executable)) throw new FileNotFoundException("Engine executable does not exist.", executable);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(spec.StdoutPath))!);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(spec.StderrPath))!);
        var security = new SecurityAttributes { Length = Marshal.SizeOf<SecurityAttributes>(), InheritHandle = true };
        using var output = OpenLog(spec.StdoutPath, ref security);
        using var error = OpenLog(spec.StderrPath, ref security);
        using var input = CreateFile("NUL", 0x80000000, 3, ref security, 3, 0x80, IntPtr.Zero);
        if (input.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error(), "Could not open child standard input.");

        var ownedJob = CreateJobObject(IntPtr.Zero, null);
        if (ownedJob.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error(), "Could not create engine Job.");
        SafeKernelHandle? ownedProcess = null;
        SafeKernelHandle? thread = null;
        IntPtr attributeList = IntPtr.Zero, handles = IntPtr.Zero, environment = IntPtr.Zero;
        var attributesInitialized = false;
        try
        {
            var limits = new JobExtendedLimitInformation();
            limits.BasicLimitInformation.LimitFlags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if (!SetInformationJobObject(ownedJob, 9, ref limits, (uint)Marshal.SizeOf<JobExtendedLimitInformation>()))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Could not configure engine Job.");

            nuint attributeBytes = 0;
            InitializeProcThreadAttributeList(IntPtr.Zero, 1, 0, ref attributeBytes);
            attributeList = Marshal.AllocHGlobal(checked((nint)attributeBytes));
            if (!InitializeProcThreadAttributeList(attributeList, 1, 0, ref attributeBytes))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            attributesInitialized = true;
            handles = Marshal.AllocHGlobal(IntPtr.Size * 3);
            Marshal.WriteIntPtr(handles, 0, input.DangerousGetHandle());
            Marshal.WriteIntPtr(handles, IntPtr.Size, output.DangerousGetHandle());
            Marshal.WriteIntPtr(handles, IntPtr.Size * 2, error.DangerousGetHandle());
            if (!UpdateProcThreadAttribute(attributeList, 0, (nuint)0x20002, handles,
                    (nuint)(IntPtr.Size * 3), IntPtr.Zero, IntPtr.Zero))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            var startup = new StartupInfoEx
            {
                StartupInfo = new StartupInfo
                {
                    Cb = Marshal.SizeOf<StartupInfoEx>(), Flags = 0x101, ShowWindow = 0,
                    StdInput = input.DangerousGetHandle(), StdOutput = output.DangerousGetHandle(),
                    StdError = error.DangerousGetHandle()
                },
                AttributeList = attributeList
            };
            var variables = new SortedDictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (System.Collections.DictionaryEntry entry in System.Environment.GetEnvironmentVariables())
                variables[(string)entry.Key] = (string)entry.Value!;
            foreach (var (key, value) in spec.Environment)
            {
                if (key.Contains('=') || key.Contains('\0') || value?.Contains('\0') == true)
                    throw new ArgumentException("Invalid child environment entry.");
                if (value is null) variables.Remove(key); else variables[key] = value;
            }
            environment = Marshal.StringToHGlobalUni(string.Join('\0', variables.Select(x => $"{x.Key}={x.Value}")) + "\0\0");
            var command = new StringBuilder(string.Join(' ', new[] { executable }.Concat(spec.Arguments).Select(QuoteArgument)));
            // A separate hidden console is required for CTRL_BREAK without affecting the tray or other applications.
            const uint flags = 0x4 | 0x10 | 0x400 | 0x80000; // SUSPENDED | NEW_CONSOLE | UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT
            if (!CreateProcess(executable, command, IntPtr.Zero, IntPtr.Zero, true, flags,
                    environment, spec.WorkingDirectory, ref startup, out var info))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Could not start engine.");
            ownedProcess = new SafeKernelHandle(info.Process, true);
            thread = new SafeKernelHandle(info.Thread, true);
            if (!AssignProcessToJobObject(ownedJob, ownedProcess))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Could not assign engine to owned Job.");
            if (ResumeThread(thread) == uint.MaxValue)
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Could not resume engine.");
            var result = new NativeProcess(ownedJob, ownedProcess, checked((int)info.ProcessId));
            ownedJob = null!;
            ownedProcess = null;
            return result;
        }
        finally
        {
            if (ownedProcess is not null)
            {
                TerminateProcess(ownedProcess, 1); // Still our suspended/owned native handle, never a PID lookup.
                ownedProcess.Dispose();
            }
            ownedJob?.Dispose();
            thread?.Dispose();
            if (attributeList != IntPtr.Zero)
            {
                if (attributesInitialized) DeleteProcThreadAttributeList(attributeList);
                Marshal.FreeHGlobal(attributeList);
            }
            if (handles != IntPtr.Zero) Marshal.FreeHGlobal(handles);
            if (environment != IntPtr.Zero) Marshal.FreeHGlobal(environment);
        }
    }

    private static SafeFileHandle OpenLog(string path, ref SecurityAttributes security)
    {
        var handle = CreateFile(Path.GetFullPath(path), 0x40000000, 3, ref security, 2, 0x80, IntPtr.Zero);
        if (handle.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error(), $"Could not open log: {path}");
        return handle;
    }

    // Implements the CommandLineToArgvW / Microsoft C runtime quoting convention.
    internal static string QuoteArgument(string value)
    {
        if (value.Contains('\0')) throw new ArgumentException("Command arguments cannot contain NUL.");
        var result = new StringBuilder("\"");
        var slashes = 0;
        foreach (var c in value)
        {
            if (c == '\\') { slashes++; continue; }
            result.Append('\\', c == '"' ? slashes * 2 + 1 : slashes);
            result.Append(c);
            slashes = 0;
        }
        return result.Append('\\', slashes * 2).Append('"').ToString();
    }

    public async Task StopAsync(TimeSpan gracefulTimeout)
    {
        if (!HasExited)
        {
            await SendStopHelperAsync();
            var deadline = DateTimeOffset.UtcNow + gracefulTimeout;
            while (!HasExited && DateTimeOffset.UtcNow < deadline) await Task.Delay(100);
            if (!HasExited && job is not null && !TerminateJobObject(job, 1))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Could not terminate owned engine Job.");
            var killDeadline = DateTimeOffset.UtcNow.AddSeconds(5);
            while (!HasExited && DateTimeOffset.UtcNow < killDeadline) await Task.Delay(50);
            if (!HasExited) throw new IOException("Owned engine did not exit after Job termination.");
        }
        // Closing the Job also removes descendants if the top-level process exited first.
        job?.Dispose();
        job = null;
    }

    private async Task SendStopHelperAsync()
    {
        try
        {
            var info = new ProcessStartInfo(System.Environment.ProcessPath!)
            { UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = AppContext.BaseDirectory };
            if (string.Equals(Path.GetFileNameWithoutExtension(info.FileName), "dotnet", StringComparison.OrdinalIgnoreCase))
                info.ArgumentList.Add(Path.Combine(AppContext.BaseDirectory, typeof(NativeProcess).Assembly.GetName().Name + ".dll"));
            info.ArgumentList.Add("--signal-stop");
            info.ArgumentList.Add(Id.ToString(System.Globalization.CultureInfo.InvariantCulture));
            using var helper = Process.Start(info);
            if (helper is null) return;
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            try { await helper.WaitForExitAsync(timeout.Token); }
            catch (OperationCanceledException) { if (!helper.HasExited) helper.Kill(); }
        }
        catch (Exception exception) when (exception is Win32Exception or IOException or InvalidOperationException)
        {
            // The Job remains authoritative; an unavailable console signal falls back to its bounded termination.
        }
    }

    private static readonly ConsoleControlHandler IgnoreControl = _ => true;

    /// <summary>Run only in a short-lived --signal-stop helper, before any UI or single-instance initialization.</summary>
    public static bool SignalStop(int pid)
    {
        if (pid <= 0) return false;
        FreeConsole();
        if (!AttachConsole((uint)pid)) return false;
        try
        {
            if (!SetConsoleCtrlHandler(IgnoreControl, true)) return false;
            var sent = GenerateConsoleCtrlEvent(1, 0); // CTRL_BREAK_EVENT to this dedicated console.
            if (sent) Thread.Sleep(100);
            return sent;
        }
        finally { FreeConsole(); }
    }

    public static IReadOnlySet<int> ListeningProcessIds(int port)
    {
        var ids = new HashSet<int>();
        foreach (var family in new[] { 2, 23 })
        {
            var bytes = 0;
            var status = GetExtendedTcpTable(IntPtr.Zero, ref bytes, false, family, 3, 0);
            if (status is not 0 and not 122) throw new Win32Exception((int)status, "Could not inspect API listener ownership.");
            var buffer = Marshal.AllocHGlobal(Math.Max(bytes, 4));
            try
            {
                for (var attempt = 0; attempt < 3; attempt++)
                {
                    status = GetExtendedTcpTable(buffer, ref bytes, false, family, 3, 0);
                    if (status != 122) break;
                    buffer = Marshal.ReAllocHGlobal(buffer, Math.Max(bytes, 4));
                }
                if (status != 0) throw new Win32Exception((int)status, "Could not inspect API listener ownership.");
                var count = Marshal.ReadInt32(buffer);
                var rowSize = family == 2 ? 24 : 56;
                var portOffset = family == 2 ? 8 : 20;
                var pidOffset = family == 2 ? 20 : 52;
                if (count < 0 || (long)count * rowSize + 4 > bytes) throw new IOException("Invalid TCP listener table.");
                for (var i = 0; i < count; i++)
                {
                    var row = IntPtr.Add(buffer, 4 + i * rowSize);
                    var rawPort = (uint)Marshal.ReadInt32(row, portOffset);
                    var hostPort = (int)(((rawPort & 0xff) << 8) | ((rawPort >> 8) & 0xff));
                    if (hostPort == port) ids.Add(Marshal.ReadInt32(row, pidOffset));
                }
            }
            finally { Marshal.FreeHGlobal(buffer); }
        }
        return ids;
    }

    public void Dispose()
    {
        job?.Dispose(); // Kill-on-close protects against manager exceptions and abrupt shutdown.
        job = null;
        process?.Dispose();
        process = null;
    }

    private sealed class SafeKernelHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        public SafeKernelHandle() : base(true) { }
        public SafeKernelHandle(IntPtr handle, bool ownsHandle) : base(ownsHandle) => SetHandle(handle);
        protected override bool ReleaseHandle() => CloseHandle(handle);
    }
    [StructLayout(LayoutKind.Sequential)] private struct SecurityAttributes { public int Length; public IntPtr Descriptor; [MarshalAs(UnmanagedType.Bool)] public bool InheritHandle; }
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)] private struct StartupInfo
    {
        public int Cb; public string? Reserved; public string? Desktop; public string? Title;
        public uint X, Y, XSize, YSize, XCountChars, YCountChars, FillAttribute, Flags;
        public ushort ShowWindow, Reserved2Size; public IntPtr Reserved2, StdInput, StdOutput, StdError;
    }
    [StructLayout(LayoutKind.Sequential)] private struct StartupInfoEx { public StartupInfo StartupInfo; public IntPtr AttributeList; }
    [StructLayout(LayoutKind.Sequential)] private struct ProcessInformation { public IntPtr Process, Thread; public uint ProcessId, ThreadId; }
    [StructLayout(LayoutKind.Sequential)] private struct JobBasicLimitInformation
    {
        public long PerProcessUserTimeLimit, PerJobUserTimeLimit; public uint LimitFlags;
        public nuint MinimumWorkingSetSize, MaximumWorkingSetSize; public uint ActiveProcessLimit;
        public nuint Affinity; public uint PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)] private struct IoCounters { public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount, ReadTransferCount, WriteTransferCount, OtherTransferCount; }
    [StructLayout(LayoutKind.Sequential)] private struct JobExtendedLimitInformation
    {
        public JobBasicLimitInformation BasicLimitInformation; public IoCounters IoInfo;
        public nuint ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
    }
    private delegate bool ConsoleControlHandler(uint controlType);
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] private static extern SafeFileHandle CreateFile(string fileName, uint access, uint share, ref SecurityAttributes security, uint creation, uint flags, IntPtr template);
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] private static extern SafeKernelHandle CreateJobObject(IntPtr security, string? name);
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool SetInformationJobObject(SafeKernelHandle job, int infoClass, ref JobExtendedLimitInformation info, uint length);
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool AssignProcessToJobObject(SafeKernelHandle job, SafeKernelHandle process);
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool InitializeProcThreadAttributeList(IntPtr list, int count, uint flags, ref nuint size);
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool UpdateProcThreadAttribute(IntPtr list, uint flags, nuint attribute, IntPtr value, nuint size, IntPtr previous, IntPtr returnedSize);
    [DllImport("kernel32.dll")] private static extern void DeleteProcThreadAttributeList(IntPtr list);
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool CreateProcess(string application, StringBuilder command, IntPtr processSecurity, IntPtr threadSecurity, [MarshalAs(UnmanagedType.Bool)] bool inherit, uint flags, IntPtr environment, string directory, ref StartupInfoEx startup, out ProcessInformation info);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern uint ResumeThread(SafeKernelHandle thread);
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool TerminateProcess(SafeKernelHandle process, uint exitCode);
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool TerminateJobObject(SafeKernelHandle job, uint exitCode);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern uint WaitForSingleObject(SafeKernelHandle handle, uint milliseconds);
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool GetExitCodeProcess(SafeKernelHandle process, out uint code);
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool FreeConsole();
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool AttachConsole(uint pid);
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool SetConsoleCtrlHandler(ConsoleControlHandler handler, [MarshalAs(UnmanagedType.Bool)] bool add);
    [DllImport("kernel32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool GenerateConsoleCtrlEvent(uint controlEvent, uint group);
    [DllImport("iphlpapi.dll", SetLastError = true)] private static extern uint GetExtendedTcpTable(IntPtr table, ref int bytes, [MarshalAs(UnmanagedType.Bool)] bool order, int family, int tableClass, uint reserved);
}
