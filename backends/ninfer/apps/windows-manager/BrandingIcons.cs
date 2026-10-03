namespace NInfer.Manager;

/// <summary>Owned, multi-resolution tray artwork. The engine state remains explicit in the menu.</summary>
internal sealed class BrandingIcons : IDisposable
{
    public Icon Running { get; } = Load("ninfer-active.ico");
    public Icon Stopped { get; } = Load("ninfer-inactive.ico");

    private static Icon Load(string name)
    {
        using var stream = typeof(BrandingIcons).Assembly.GetManifestResourceStream("NInfer.Manager.Assets." + name)
            ?? throw new InvalidOperationException("Missing NInfer icon resource: " + name);
        using var icon = new Icon(stream, SystemInformation.SmallIconSize);
        return (Icon)icon.Clone();
    }

    public void Dispose() { Running.Dispose(); Stopped.Dispose(); }
}
