using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;

namespace NInfer.Manager;

public static class ManagerApi
{
    public static void Map(WebApplication app, ConfigurationStore store, IEngineController engine,
        Action requestExit, Action? settingsChanged = null)
    {
        var catalog = new ModelCatalog(store);
        var telemetry = new TelemetryReader();
        telemetry.Start(() => engine.Snapshot, app.Lifetime.ApplicationStopping);
        _ = catalog.ScanAsync().ContinueWith(task =>
        {
            if (task.Exception is not null) app.Logger.LogError(task.Exception, "Initial model scan failed");
        }, TaskScheduler.Default);
        _ = Task.Run(async () =>
        {
            using var interval = new PeriodicTimer(TimeSpan.FromSeconds(15));
            var stopping = app.Lifetime.ApplicationStopping;
            try
            {
                while (await interval.WaitForNextTickAsync(stopping))
                {
                    try { await catalog.ScanAsync(stopping); }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidDataException)
                    { app.Logger.LogWarning(ex, "Model directory refresh failed"); }
                }
            }
            catch (OperationCanceledException) when (stopping.IsCancellationRequested) { }
        });

        app.MapGet("/api/state", async (HttpContext context) =>
        {
            var snapshot = engine.Snapshot;
            var metrics = await telemetry.ReadAsync(snapshot, context.RequestAborted);
            return Results.Json(new { engine = snapshot, settings = store.Settings, profiles = store.Profiles,
                models = catalog.Models, stats = metrics.Stats, statsError = metrics.Error, gpu = metrics.Gpu, monitor = metrics.Monitor,
                recentRequests = metrics.RecentRequests, logTail = metrics.LogTail, sampledAt = metrics.SampledAt });
        });
        app.MapGet("/api/metrics", async (HttpContext context) => Results.Json(await telemetry.ReadAsync(engine.Snapshot, context.RequestAborted)));
        app.MapGet("/api/profiles", () => Results.Json(store.Profiles));
        app.MapGet("/api/models", () => Results.Json(catalog.Models));
        app.MapPut("/api/language", async (HttpContext context) =>
        {
            try
            {
                using var body = await JsonDocument.ParseAsync(context.Request.Body, cancellationToken: context.RequestAborted);
                var language = body.RootElement.ValueKind == JsonValueKind.Object && body.RootElement.TryGetProperty("language", out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
                if (language is not ("zh" or "en")) throw new ArgumentException("Language must be zh or en.");
                store.SaveSettings(store.Settings with { Language = language });
                return Results.Json(new { language });
            }
            catch (Exception ex) when (IsUserError(ex)) { return Problem(ex); }
        });
        app.MapPost("/api/models/scan", async (HttpContext context) =>
        {
            try { return Results.Json(await catalog.ScanAsync(context.RequestAborted)); }
            catch (Exception ex) when (IsUserError(ex)) { return Problem(ex); }
        });
        app.MapPut("/api/profiles/{id}", async (string id, HttpContext context) =>
        {
            try
            {
                var profile = await context.Request.ReadFromJsonAsync<LaunchProfile>(ConfigurationStore.Json, context.RequestAborted)
                    ?? throw new ArgumentException("Profile JSON is required.");
                if (!string.Equals(id, profile.Id, StringComparison.Ordinal)) throw new ArgumentException("URL and profile IDs must match.");
                store.SaveProfile(profile);
                return Results.Json(profile);
            }
            catch (Exception ex) when (IsUserError(ex)) { return Problem(ex); }
        });
        app.MapDelete("/api/profiles/{id}", (string id) =>
        {
            try
            {
                store.DeleteProfile(id);
                return Results.Json(new { deleted = id });
            }
            catch (Exception ex) when (IsUserError(ex)) { return Problem(ex); }
        });
        app.MapPut("/api/settings", async (HttpContext context) =>
        {
            try
            {
                var settings = await context.Request.ReadFromJsonAsync<ManagerSettings>(ConfigurationStore.Json, context.RequestAborted)
                    ?? throw new ArgumentException("Settings JSON is required.");
                var previous = store.Settings;
                store.SaveSettings(settings);
                try { settingsChanged?.Invoke(); }
                catch (Exception applyError)
                {
                    try
                    {
                        store.SaveSettings(previous);
                        settingsChanged?.Invoke();
                    }
                    catch (Exception rollbackError)
                    {
                        throw new InvalidOperationException($"Settings application failed: {applyError.Message}. Restoring the previous settings also failed: {rollbackError.Message}", applyError);
                    }
                    throw new InvalidOperationException($"Settings application failed; the previous settings were restored. {applyError.Message}", applyError);
                }
                if (!previous.ModelDirectories.SequenceEqual(settings.ModelDirectories, StringComparer.OrdinalIgnoreCase))
                    _ = catalog.ScanAsync();
                return Results.Json(new { settings = store.Settings, restartRequired = previous.WebPort != settings.WebPort });
            }
            catch (Exception ex) when (IsUserError(ex)) { return Problem(ex); }
        });
        app.MapPost("/api/start/{id}", (string id) =>
        {
            try
            {
                if (engine.Snapshot.State is "Starting" or "Running" or "Stopping") return Results.Conflict(new { error = "A model is already active. Stop it before starting another." });
                var spec = store.BuildLaunchSpec(id);
                // StartAsync publishes Starting synchronously; HTTP never waits through CUDA loading.
                var pending = engine.StartAsync(spec);
                // Port/ownership validation can fail before the first asynchronous startup wait.
                // Return that failure to the caller instead of accepting a launch that never began.
                if (pending.IsCompleted) pending.GetAwaiter().GetResult();
                _ = pending.ContinueWith(task =>
                {
                    if (task.Exception is not null) app.Logger.LogError(task.Exception, "Model startup failed for {ProfileId}", id);
                }, TaskScheduler.Default);
                return Results.Json(new { accepted = true, profileId = id }, statusCode: StatusCodes.Status202Accepted);
            }
            catch (Exception ex) when (IsUserError(ex)) { return Problem(ex); }
        });
        app.MapPost("/api/stop", async () =>
        {
            try { await engine.StopAsync(); return Results.Json(engine.Snapshot); }
            catch (Exception ex) when (IsUserError(ex)) { return Problem(ex); }
        });
        app.MapPost("/api/exit", (HttpContext context) =>
        {
            // Dispatch only after this response has flushed; UI shutdown owns engine cleanup.
            context.Response.OnCompleted(() => { requestExit(); return Task.CompletedTask; });
            return Results.Json(new { accepted = true }, statusCode: StatusCodes.Status202Accepted);
        });
        app.Lifetime.ApplicationStopped.Register(() => telemetry.StopAsync().GetAwaiter().GetResult());
    }

    private static bool IsUserError(Exception ex) => ex is ArgumentException or InvalidDataException or IOException or UnauthorizedAccessException or KeyNotFoundException or JsonException or InvalidOperationException;
    private static IResult Problem(Exception ex) => Results.Json(new { error = ex.Message }, statusCode: ex is KeyNotFoundException ? 404 : ex is InvalidOperationException ? 409 : 400);
}
