using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Serialization;

namespace NInfer.Manager;

public sealed record GpuTelemetrySample(
    [property: JsonPropertyName("available")] bool Available,
    [property: JsonPropertyName("name")] string? Name,
    [property: JsonPropertyName("totalBytes")] ulong? TotalBytes,
    [property: JsonPropertyName("usedBytes")] ulong? UsedBytes,
    [property: JsonPropertyName("freeBytes")] ulong? FreeBytes,
    [property: JsonPropertyName("error")] string? Error,
    [property: JsonPropertyName("utilizationPercent")] double? UtilizationPercent = null,
    [property: JsonPropertyName("powerWatts")] double? PowerWatts = null,
    [property: JsonPropertyName("powerLimitWatts")] double? PowerLimitWatts = null,
    [property: JsonPropertyName("temperatureC")] double? TemperatureC = null);

/// <summary>Read-only, whole-device NVML telemetry. This never creates a CUDA context.</summary>
public static class GpuTelemetry
{
    private static readonly object Gate = new();
    private static readonly TimeSpan CacheLifetime = TimeSpan.FromSeconds(2);
    private static NativeNvml? native;
    private static GpuTelemetrySample? cached;
    private static long sampledAt;

    static GpuTelemetry()
    {
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            lock (Gate)
            {
                native?.Dispose();
                native = null;
            }
        };
    }

    public static GpuTelemetrySample Sample()
    {
        lock (Gate)
        {
            if (cached is not null && Stopwatch.GetElapsedTime(sampledAt) < CacheLifetime) return cached;
            try
            {
                native ??= new NativeNvml();
                cached = native.Read();
            }
            catch (Exception error)
            {
                native?.Dispose();
                native = null;
                cached = new(false, null, null, null, null, error.Message);
            }
            sampledAt = Stopwatch.GetTimestamp();
            return cached;
        }
    }

    private sealed class NativeNvml : IDisposable
    {
        private IntPtr library;
        private IntPtr device;
        private bool initialized;
        private Shutdown shutdown = null!;
        private GetMemory getMemory = null!;
        private GetUtilization? getUtilization;
        private GetUnsignedValue? getPower, getPowerLimit;
        private GetTemperature? getTemperature;
        private ErrorString errorString = null!;
        private string name = "";

        public NativeNvml()
        {
            if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("GPU telemetry requires the Windows NVIDIA driver.");
            // Only trusted driver installation locations: never search the working directory or PATH.
            var candidates = new[]
            {
                Path.Combine(Environment.SystemDirectory, "nvml.dll"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "NVIDIA Corporation", "NVSMI", "nvml.dll")
            };
            foreach (var candidate in candidates)
                if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out library)) break;
            if (library == IntPtr.Zero) throw new DllNotFoundException("The NVIDIA driver NVML library is unavailable.");
            try
            {
                var init = Export<Initialize>("nvmlInit_v2");
                shutdown = Export<Shutdown>("nvmlShutdown");
                var getHandle = Export<GetHandle>("nvmlDeviceGetHandleByIndex_v2");
                getMemory = Export<GetMemory>("nvmlDeviceGetMemoryInfo");
                getUtilization = OptionalExport<GetUtilization>("nvmlDeviceGetUtilizationRates");
                getPower = OptionalExport<GetUnsignedValue>("nvmlDeviceGetPowerUsage");
                getPowerLimit = OptionalExport<GetUnsignedValue>("nvmlDeviceGetPowerManagementLimit");
                getTemperature = OptionalExport<GetTemperature>("nvmlDeviceGetTemperature");
                var getName = Export<GetName>("nvmlDeviceGetName");
                errorString = Export<ErrorString>("nvmlErrorString");
                Check(init(), "nvmlInit_v2");
                initialized = true;
                Check(getHandle(0, out device), "nvmlDeviceGetHandleByIndex_v2");
                var nameBuffer = new byte[256];
                Check(getName(device, nameBuffer, (uint)nameBuffer.Length), "nvmlDeviceGetName");
                var terminator = Array.IndexOf(nameBuffer, (byte)0);
                name = Encoding.UTF8.GetString(nameBuffer, 0, terminator < 0 ? nameBuffer.Length : terminator);
            }
            catch { Dispose(); throw; }
        }

        public GpuTelemetrySample Read()
        {
            Check(getMemory(device, out var memory), "nvmlDeviceGetMemoryInfo");
            // Unsupported counters stay unknown; they must not hide otherwise valid VRAM data.
            double? utilization = getUtilization is not null && getUtilization(device, out var load) == 0
                ? load.Gpu : null;
            double? power = getPower is not null && getPower(device, out var milliwatts) == 0
                ? milliwatts / 1000.0 : null;
            double? powerLimit = getPowerLimit is not null && getPowerLimit(device, out var limit) == 0
                ? limit / 1000.0 : null;
            double? temperature = getTemperature is not null && getTemperature(device, 0, out var degrees) == 0
                ? degrees : null;
            return new(true, name, memory.Total, memory.Used, memory.Free, null,
                utilization, power, powerLimit, temperature);
        }

        private T Export<T>(string symbol) where T : Delegate =>
            Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(library, symbol));

        private T? OptionalExport<T>(string symbol) where T : Delegate =>
            NativeLibrary.TryGetExport(library, symbol, out var address)
                ? Marshal.GetDelegateForFunctionPointer<T>(address) : null;

        private void Check(int status, string operation)
        {
            if (status == 0) return;
            var description = Marshal.PtrToStringAnsi(errorString(status)) ?? $"NVML error {status}";
            throw new InvalidOperationException($"{operation}: {description} ({status}).");
        }

        public void Dispose()
        {
            if (initialized)
            {
                shutdown();
                initialized = false;
            }
            if (library != IntPtr.Zero)
            {
                NativeLibrary.Free(library);
                library = IntPtr.Zero;
            }
            device = IntPtr.Zero;
        }

        // nvmlMemory_t v1 contains three unsigned long long fields in total/free/used order.
        [StructLayout(LayoutKind.Sequential)] private struct Memory { public ulong Total, Free, Used; }
        [StructLayout(LayoutKind.Sequential)] private struct Utilization { public uint Gpu, Memory; }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Initialize();
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Shutdown();
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int GetHandle(uint index, out IntPtr handle);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int GetMemory(IntPtr handle, out Memory memory);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int GetUtilization(IntPtr handle, out Utilization utilization);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int GetUnsignedValue(IntPtr handle, out uint value);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int GetTemperature(IntPtr handle, uint sensor, out uint temperature);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int GetName(IntPtr handle, [Out] byte[] buffer, uint length);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate IntPtr ErrorString(int status);
    }
}
