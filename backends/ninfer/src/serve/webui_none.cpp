#include "serve/webui.h"

namespace ninfer::serve {

std::span<const WebUiAsset> webui_assets() noexcept { return {}; }

const WebUiAsset* find_webui_asset(std::string_view, bool) noexcept { return nullptr; }

} // namespace ninfer::serve
