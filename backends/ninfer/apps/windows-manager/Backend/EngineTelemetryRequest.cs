using System.Globalization;
using System.Net.Http.Headers;

namespace NInfer.Manager;

internal static class EngineTelemetryRequest
{
    public static HttpRequestMessage Create(EngineSnapshot engine, Uri api)
    {
        var endpoint = new UriBuilder(api) { Path = "/stats", Query = "", Fragment = "" };
        var arguments = engine.Launch?.Arguments;
        if (int.TryParse(Value(arguments, "--stats-port"), NumberStyles.None, CultureInfo.InvariantCulture,
                out var port) && port is >= 1 and <= 65535)
            endpoint.Port = port;
        var request = new HttpRequestMessage(HttpMethod.Get, endpoint.Uri);
        var key = Value(arguments, "--api-key");
        if (!string.IsNullOrEmpty(key)) request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", key);
        return request;
    }

    private static string? Value(IReadOnlyList<string>? arguments, string option)
    {
        if (arguments is null) return null;
        for (var i = 1; i + 1 < arguments.Count; i++)
            if (arguments[i] == option) return arguments[i + 1];
        return null;
    }
}
