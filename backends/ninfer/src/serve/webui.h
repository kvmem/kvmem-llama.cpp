#pragma once

// Static WebUI files compiled into ninfer-serve with -DNINFER_WEBUI_DIR. A build without it has none,
// and the server then answers no WebUI route.

#include <cstddef>
#include <span>
#include <string_view>

namespace ninfer::serve {

struct WebUiAsset {
    std::string_view path;         // relative to the WebUI root, "index.html" for the entry page
    std::string_view content_type; // of the file as served, after any Content-Encoding is removed
    std::string_view encoding;     // "gzip" for a precompressed file, empty otherwise
    std::string_view etag;
    std::span<const unsigned char> bytes;
};

[[nodiscard]] std::span<const WebUiAsset> webui_assets() noexcept;

// The file served for `path` (without its leading '/'). A precompressed copy is preferred when the
// client accepts gzip, and is the fallback when it is the only copy.
[[nodiscard]] const WebUiAsset* find_webui_asset(std::string_view path, bool accepts_gzip) noexcept;

} // namespace ninfer::serve
