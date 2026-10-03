namespace NInfer.Manager;

internal sealed class TrayContext : ApplicationContext
{
    private readonly ConfigurationStore store;
    private readonly IEngineController engine;
    private readonly string origin, token;
    private readonly Control dispatcher = new();
    private readonly NotifyIcon icon;
    private readonly BrandingIcons artwork = new();
    private readonly ContextMenuStrip menu = new();
    private readonly System.Windows.Forms.Timer timer = new() { Interval = 700 };
    private bool exiting;
    private string lastKey = "";

    public TrayContext(ConfigurationStore store, IEngineController engine, string origin, string token)
    {
        this.store = store; this.engine = engine; this.origin = origin; this.token = token;
        _ = dispatcher.Handle;
        icon = new NotifyIcon { Icon = artwork.Stopped, Text = "NInfer", ContextMenuStrip = menu, Visible = true };
        icon.DoubleClick += (_, _) => OpenPage("/");
        timer.Tick += (_, _) => RefreshMenu();
        timer.Start();
        RefreshMenu();
        Microsoft.Win32.SystemEvents.SessionEnding += SessionEnding;
    }

    public void OpenPage(string path) => Program.OpenBrowser(origin + path + "?launchToken=" + token);
    public void RefreshSoon() { if (!dispatcher.IsDisposed) dispatcher.BeginInvoke(() => { lastKey = ""; RefreshMenu(); }); }
    public void RequestExit() { if (!dispatcher.IsDisposed) dispatcher.BeginInvoke(async () => await ExitAsync()); }
    public void StartDefault() => dispatcher.BeginInvoke(async () => await StartProfile(store.Settings.DefaultProfileId));
    private string T(string zh, string en) => store.Settings.Language == "en" ? en : zh;

    private void RefreshMenu()
    {
        var state = engine.Snapshot;
        var startup = StartupRegistration.GetStatus(Environment.ProcessPath!, store.PackageRoot);
        var key = state.ToString() + string.Join('|', store.Profiles.Select(p => p.Id + p.Name)) + store.Settings.ToString() + startup;
        if (key == lastKey || menu.Visible) return;
        lastKey = key;
        string label = state.State switch { "Starting" => T("正在加载", "Loading"), "Running" => T("运行中", "Running"), "Stopping" => T("正在停止", "Stopping"), "Failed" => T("异常", "Failed"), _ => T("已停止", "Stopped") };
        icon.Text = ("NInfer · " + label + " " + state.ProfileName).Trim()[..Math.Min(63, ("NInfer · " + label + " " + state.ProfileName).Trim().Length)];
        icon.Icon = state.State == "Running" ? artwork.Running : artwork.Stopped;
        foreach (var item in menu.Items.Cast<ToolStripItem>().ToArray()) item.Dispose();
        menu.Items.Clear();
        Add("NInfer · " + label + (state.ProfileName is null ? "" : "：" + state.ProfileName), null, false);
        menu.Items.Add(new ToolStripSeparator());
        Add(T("打开监控页面", "Open monitor"), () => OpenPage("/"));
        Add(T("管理模型…", "Manage models…"), () => OpenPage("/models"));
        var start = new ToolStripMenuItem(T("启动模型", "Start model")) { Enabled = state.State is "Stopped" or "Failed" };
        foreach (var profile in store.Profiles)
        {
            var id = profile.Id;
            var item = new ToolStripMenuItem(profile.Name);
            item.Enabled = File.Exists(Path.GetFullPath(Path.Combine(store.PackageRoot, profile.ModelPath)));
            item.Click += async (_, _) => await StartProfile(id);
            start.DropDownItems.Add(item);
        }
        menu.Items.Add(start);
        var stop = Add(T("停止服务", "Stop service"), null, state.State is "Running" or "Starting" or "Failed");
        stop.Click += async (_, _) => { try { await engine.StopAsync(); } catch (Exception ex) { Error(ex); } };
        if (state.State == "Running")
        {
            menu.Items.Add(new ToolStripSeparator());
            Add("API Base：" + state.ApiBase, () => Clipboard.SetText(state.ApiBase!));
            Add(T("模型 ID：", "Model ID: ") + state.ModelId, () => Clipboard.SetText(state.ModelId!));
        }
        menu.Items.Add(new ToolStripSeparator());
        var auto = Add(T("随 Windows 登录启动", "Start with Windows sign-in") +
            (startup.IsRegistered && startup.DisabledByWindows ? T("（Windows 已禁用）", " (disabled by Windows)") : ""), null);
        auto.Checked = startup.IsRegistered;
        auto.Click += (_, _) =>
        {
            try
            {
                // Follow the displayed action; ownership may change while this menu is open.
                bool enabled = !auto.Checked;
                StartupRegistration.SetEnabled(Environment.ProcessPath!, store.PackageRoot, enabled);
                store.SaveSettings(store.Settings with { StartWithWindows = enabled });
                lastKey = "";
                if (enabled && StartupRegistration.GetStatus(Environment.ProcessPath!, store.PackageRoot).DisabledByWindows)
                    icon.ShowBalloonTip(7000, "NInfer", T("Windows 已禁用此启动项。请在任务管理器的启动应用中启用 NInferManager。",
                        "Windows has disabled this startup entry. Enable NInferManager in Task Manager's Startup apps."), ToolTipIcon.Info);
            }
            catch (Exception ex) { Error(ex); }
        };
        var load = Add(T("登录后自动加载默认模型", "Load default model automatically"), null);
        load.Checked = store.Settings.AutoStartModel;
        load.Click += (_, _) => { try { store.SaveSettings(store.Settings with { AutoStartModel = !store.Settings.AutoStartModel }); lastKey = ""; } catch (Exception ex) { Error(ex); } };
        var languages = new ToolStripMenuItem(T("语言 / Language", "Language / 语言"));
        foreach (var (code, caption) in new[] { ("zh", "简体中文"), ("en", "English") })
        {
            var item = new ToolStripMenuItem(caption) { Checked = store.Settings.Language == code };
            item.Click += (_, _) => { try { store.SaveSettings(store.Settings with { Language = code }); lastKey = ""; } catch (Exception ex) { Error(ex); } };
            languages.DropDownItems.Add(item);
        }
        menu.Items.Add(languages);
        var exit = Add(T("退出（停止模型与监控）", "Exit (stop model and monitor)"), null);
        exit.Click += async (_, _) => await ExitAsync();
    }

    private ToolStripMenuItem Add(string text, Action? action, bool enabled = true)
    {
        var item = new ToolStripMenuItem(text) { Enabled = enabled };
        if (action is not null) item.Click += (_, _) => { try { action(); } catch (Exception ex) { Error(ex); } };
        menu.Items.Add(item);
        return item;
    }

    private async Task StartProfile(string id)
    {
        try { await engine.StartAsync(store.BuildLaunchSpec(id)); }
        catch (Exception ex) { Error(ex); }
    }

    private void Error(Exception ex)
    {
        icon.ShowBalloonTip(7000, "NInfer", ex.Message, ToolTipIcon.Error);
        File.AppendAllText(Path.Combine(store.DataRoot, "logs", "manager-errors.log"), $"{DateTimeOffset.Now:o} {ex}\n");
    }

    private async Task ExitAsync()
    {
        if (exiting) return;
        exiting = true;
        try { await engine.StopAsync(); }
        catch (Exception ex) { Error(ex); }
        finally { icon.Visible = false; ExitThread(); }
    }

    private void SessionEnding(object sender, Microsoft.Win32.SessionEndingEventArgs args) => RequestExit();

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            Microsoft.Win32.SystemEvents.SessionEnding -= SessionEnding;
            timer.Dispose(); icon.Dispose(); artwork.Dispose(); menu.Dispose(); dispatcher.Dispose();
        }
        base.Dispose(disposing);
    }
}
