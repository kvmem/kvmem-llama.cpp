// Build-time generator for the WebUI files of ninfer-serve: `embed <out.cpp> <asset-dir>` writes one
// translation unit that defines webui_assets() and find_webui_asset() over every file below the
// directory. A file ending in .gz is served as its inner name with Content-Encoding: gzip.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct Asset {
    std::string path;
    std::string content_type;
    std::string encoding;
    std::vector<unsigned char> bytes;
};

std::string content_type_for(const std::string& name) {
    const auto dot       = name.rfind('.');
    const std::string ext = dot == std::string::npos ? std::string() : name.substr(dot);
    if (ext == ".html" || ext == ".htm") { return "text/html; charset=utf-8"; }
    if (ext == ".js" || ext == ".mjs") { return "text/javascript; charset=utf-8"; }
    if (ext == ".css") { return "text/css; charset=utf-8"; }
    if (ext == ".json" || ext == ".map") { return "application/json"; }
    if (ext == ".webmanifest") { return "application/manifest+json"; }
    if (ext == ".svg") { return "image/svg+xml"; }
    if (ext == ".png") { return "image/png"; }
    if (ext == ".jpg" || ext == ".jpeg") { return "image/jpeg"; }
    if (ext == ".webp") { return "image/webp"; }
    if (ext == ".gif") { return "image/gif"; }
    if (ext == ".ico") { return "image/x-icon"; }
    if (ext == ".woff2") { return "font/woff2"; }
    if (ext == ".woff") { return "font/woff"; }
    if (ext == ".ttf") { return "font/ttf"; }
    if (ext == ".wasm") { return "application/wasm"; }
    if (ext == ".txt") { return "text/plain; charset=utf-8"; }
    return "application/octet-stream";
}

std::uint64_t fnv1a(const std::vector<unsigned char>& bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : bytes) { hash = (hash ^ byte) * 1099511628211ULL; }
    return hash;
}

std::string escaped(const std::string& text) {
    std::string out;
    for (const char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; }
        out += c;
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <out.cpp> <asset-dir>\n", argv[0]);
        return 2;
    }
    const fs::path root = argv[2];
    std::vector<Asset> assets;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) { continue; }
        Asset asset;
        asset.path = entry.path().lexically_relative(root).generic_string();
        if (asset.path.size() > 3 && asset.path.ends_with(".gz")) {
            asset.path.resize(asset.path.size() - 3);
            asset.encoding = "gzip";
        }
        asset.content_type = content_type_for(asset.path);
        std::ifstream in(entry.path(), std::ios::binary);
        asset.bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        if (!in.good() && !in.eof()) {
            std::fprintf(stderr, "embed: cannot read %s\n", entry.path().string().c_str());
            return 1;
        }
        assets.push_back(std::move(asset));
    }
    std::sort(assets.begin(), assets.end(), [](const Asset& a, const Asset& b) {
        return a.path != b.path ? a.path < b.path : a.encoding < b.encoding;
    });
    const bool has_index = std::any_of(assets.begin(), assets.end(),
                                       [](const Asset& a) { return a.path == "index.html"; });
    if (!has_index) {
        std::fprintf(stderr, "embed: %s has no index.html\n", root.string().c_str());
        return 1;
    }

    std::string out = "#include \"serve/webui.h\"\n\n#include <array>\n\nnamespace ninfer::serve {\n"
                      "namespace {\n\n";
    char buffer[64];
    for (std::size_t i = 0; i < assets.size(); ++i) {
        std::snprintf(buffer, sizeof(buffer), "alignas(16) constexpr unsigned char asset_%zu[] = {", i);
        out += buffer;
        const auto& bytes = assets[i].bytes;
        for (std::size_t b = 0; b < bytes.size(); ++b) {
            if (b % 24 == 0) { out += "\n    "; }
            std::snprintf(buffer, sizeof(buffer), "%u,", static_cast<unsigned>(bytes[b]));
            out += buffer;
        }
        if (bytes.empty()) { out += "0"; }
        out += "\n};\n";
    }
    std::snprintf(buffer, sizeof(buffer), "\nconst std::array<WebUiAsset, %zu> kAssets{{\n",
                  assets.size());
    out += buffer;
    for (std::size_t i = 0; i < assets.size(); ++i) {
        const Asset& asset = assets[i];
        std::snprintf(buffer, sizeof(buffer), "\\\"%016llx\\\"",
                      static_cast<unsigned long long>(fnv1a(asset.bytes)));
        out += "    {\"" + escaped(asset.path) + "\", \"" + asset.content_type + "\", \"" +
               asset.encoding + "\", \"" + buffer + "\", {asset_" + std::to_string(i) + ", " +
               std::to_string(asset.bytes.size()) + "}},\n";
    }
    out += "}};\n\n} // namespace\n\n"
           "std::span<const WebUiAsset> webui_assets() noexcept { return kAssets; }\n\n"
           "const WebUiAsset* find_webui_asset(std::string_view path, bool accepts_gzip) noexcept {\n"
           "    const WebUiAsset* plain      = nullptr;\n"
           "    const WebUiAsset* compressed = nullptr;\n"
           "    for (const WebUiAsset& asset : kAssets) {\n"
           "        if (asset.path != path) { continue; }\n"
           "        (asset.encoding.empty() ? plain : compressed) = &asset;\n"
           "    }\n"
           "    if (compressed != nullptr && (accepts_gzip || plain == nullptr)) { return compressed; }\n"
           "    return plain;\n"
           "}\n\n} // namespace ninfer::serve\n";

    std::ifstream previous(argv[1], std::ios::binary);
    const std::string existing((std::istreambuf_iterator<char>(previous)),
                               std::istreambuf_iterator<char>());
    if (existing == out) { return 0; }
    std::ofstream file(argv[1], std::ios::binary | std::ios::trunc);
    file << out;
    return file.good() ? 0 : 1;
}
