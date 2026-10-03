using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.Win32.SafeHandles;

namespace NInfer.Manager;

/// <summary>Read complete JSONL records incrementally. Each tick reads at most 1 MiB plus file identity probes.</summary>
internal sealed class IncrementalRequestLog : IDisposable
{
    internal const int ReadBudget = 1024 * 1024;
    internal const int MaxLineBytes = 256 * 1024;
    private FileStream? file;
    private string? fileIdentity;
    private readonly MemoryStream partial = new();
    private bool discardLine;
    private byte[] anchor = [];
    private bool disposed;
    public long UnreadBytes { get; private set; }
    public bool CatchingUp => UnreadBytes > 0;
    public long MalformedLines { get; private set; }
    public string? Error { get; private set; }

    public void Read(string? path, Action<JsonElement> consume, Action replay)
    {
        if (string.IsNullOrWhiteSpace(path)) return;
        FileStream? current = null;
        try
        {
            Error = null;
            current = Open(path);
            var currentIdentity = Identity(current);
            if (file is null)
            {
                file = current; current = null; fileIdentity = currentIdentity;
            }
            // Rotation renames the file. Keep the old handle until its remaining complete lines are read.
            var replacement = current is not null && currentIdentity != fileIdentity;
            if (!replacement && (file.Length < file.Position || !AnchorMatches(file)))
            {
                file.Position = 0; partial.SetLength(0); discardLine = false; anchor = [];
                replay();
            }
            var budget = ReadBudget;
            Drain(file, ref budget, consume);
            if (replacement && file.Position == file.Length)
            {
                file.Dispose(); file = current; current = null; fileIdentity = currentIdentity;
                partial.SetLength(0); discardLine = false; anchor = [];
                replay();
                Drain(file!, ref budget, consume);
            }
            UnreadBytes = Math.Max(0, file!.Length - file.Position) + (replacement && current is not null ? current.Length : 0);
            SaveAnchor(file);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or Win32Exception)
        {
            Error = ex.Message;
        }
        finally { current?.Dispose(); }
    }

    private void Drain(FileStream source, ref int budget, Action<JsonElement> consume)
    {
        var block = new byte[32 * 1024];
        while (budget > 0)
        {
            var count = source.Read(block, 0, Math.Min(block.Length, budget));
            if (count == 0) break;
            budget -= count;
            for (var i = 0; i < count; i++)
            {
                if (block[i] == '\n')
                {
                    if (!discardLine && partial.Length > 0)
                    {
                        try
                        {
                            using var document = JsonDocument.Parse(partial.GetBuffer().AsMemory(0, (int)partial.Length));
                            if (document.RootElement.ValueKind == JsonValueKind.Object) consume(document.RootElement);
                            else MalformedLines++;
                        }
                        catch (JsonException) { MalformedLines++; }
                    }
                    partial.SetLength(0); discardLine = false;
                }
                else if (!discardLine)
                {
                    if (partial.Length == MaxLineBytes)
                    {
                        partial.SetLength(0); discardLine = true; MalformedLines++;
                    }
                    else partial.WriteByte(block[i]);
                }
            }
        }
    }

    private bool AnchorMatches(FileStream source)
    {
        if (anchor.Length == 0) return true;
        var position = source.Position;
        source.Position = position - anchor.Length;
        var check = new byte[anchor.Length];
        var read = source.ReadAtLeast(check, check.Length, false);
        source.Position = position;
        return read == anchor.Length && check.AsSpan().SequenceEqual(anchor);
    }

    private void SaveAnchor(FileStream source)
    {
        var position = source.Position;
        anchor = new byte[(int)Math.Min(128, position)];
        source.Position = position - anchor.Length;
        var read = source.ReadAtLeast(anchor, anchor.Length, false);
        if (read != anchor.Length) anchor = [];
        source.Position = position;
    }

    private static FileStream Open(string path) => new(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete, 1, FileOptions.SequentialScan);
    private static string Identity(FileStream source)
    {
        if (!GetFileInformationByHandle(source.SafeFileHandle, out var info)) throw new Win32Exception(Marshal.GetLastWin32Error());
        return $"{info.VolumeSerialNumber}:{info.FileIndexHigh}:{info.FileIndexLow}";
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct FileInformation
    {
        public uint Attributes;
        public System.Runtime.InteropServices.ComTypes.FILETIME CreationTime, LastAccessTime, LastWriteTime;
        public uint VolumeSerialNumber, FileSizeHigh, FileSizeLow, NumberOfLinks, FileIndexHigh, FileIndexLow;
    }
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetFileInformationByHandle(SafeFileHandle handle, out FileInformation info);

    public void Reset()
    {
        file?.Dispose(); file = null; fileIdentity = null;
        partial.SetLength(0); discardLine = false; anchor = [];
        UnreadBytes = MalformedLines = 0; Error = null;
    }
    public void Dispose()
    {
        if (disposed) return;
        Reset(); partial.Dispose(); disposed = true;
    }
}
