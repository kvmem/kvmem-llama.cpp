#pragma once

#include <cstdint>
#include <span>

#include "media/decode/decode.h"

namespace ninfer::media::decode {

// True when the first 8 bytes are the PNG file signature.
bool is_png(std::span<const std::uint8_t> bytes);

// Display dimensions of a PNG from its IHDR chunk. Enforces the decoded-pixel
// budget and throws std::invalid_argument for a non-PNG or truncated header.
ImageInfo png_image_info(std::span<const std::uint8_t> bytes, std::uint64_t max_decoded_pixels);

// Decode a non-interlaced PNG into 24-bit RGB. Supports color types 0,2,3,4,6
// (gray, RGB, palette, gray+alpha, RGBA) at bit depths 1,2,4,8,16. Alpha is
// composited over white, matching the FFmpeg image path. Throws
// std::invalid_argument for malformed input and Error(ErrorKind::BudgetExceeded)
// when the pixel budget is exceeded.
Image decode_png(std::span<const std::uint8_t> bytes, std::uint64_t max_decoded_pixels);

} // namespace ninfer::media::decode