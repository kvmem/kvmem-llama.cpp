#include "media/decode/png_decode.h"

extern "C" {
#include <zlib.h>
}

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::media::decode {
namespace {

constexpr std::array<std::uint8_t, 8> kSignature = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};

[[nodiscard]] std::uint32_t be32(const std::uint8_t* p) noexcept {
    return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) | static_cast<std::uint32_t>(p[3]);
}

struct Header {
    int width = 0;
    int height = 0;
    int bit_depth = 0;
    int color_type = 0;
};

struct Palette {
    std::vector<std::array<std::uint8_t, 3>> rgb;
    std::vector<std::uint8_t> alpha; // populated when has_alpha
    bool has_alpha = false;
};

struct Parse {
    Header header;
    std::vector<std::uint8_t> idat;
    Palette palette;
    bool have_ihdr = false;
};

// Channels per pixel for each color type (0 for an unsupported type).
[[nodiscard]] int samples_per_pixel(int color_type) {
    switch (color_type) {
        case 0: return 1; // gray
        case 2: return 3; // rgb
        case 3: return 1; // palette
        case 4: return 2; // gray + alpha
        case 6: return 4; // rgba
        default: return 0;
    }
}

// Walk the chunk stream, collecting IHDR, PLTE, tRNS, and the concatenated IDAT.
void parse_chunks(std::span<const std::uint8_t> bytes, Parse& out) {
    std::size_t offset = 8;
    while (offset + 8 <= bytes.size()) {
        const std::uint32_t length = be32(bytes.data() + offset);
        offset += 4;
        if (offset + 4 + static_cast<std::size_t>(length) + 4 > bytes.size()) {
            throw std::invalid_argument("png chunk extends past end of stream");
        }
        const std::uint8_t* tag  = bytes.data() + offset;
        offset                  += 4;
        const std::uint8_t* data = bytes.data() + offset;
        offset += length + 4; // payload + CRC
        if (std::memcmp(tag, "IHDR", 4) == 0) {
            if (out.have_ihdr || length < 13) {
                throw std::invalid_argument("png has a malformed IHDR chunk");
            }
            out.have_ihdr         = true;
            out.header.width      = static_cast<int>(be32(data));
            out.header.height     = static_cast<int>(be32(data + 4));
            out.header.bit_depth  = data[8];
            out.header.color_type = data[9];
            // compression=0, filter=0, interlace=0 are the only supported methods.
            if (data[10] != 0 || data[11] != 0 || data[12] != 0) {
                throw std::invalid_argument(
                    "png uses an unsupported compression, filter, or interlace method");
            }
            if (out.header.width <= 0 || out.header.height <= 0) {
                throw std::invalid_argument("png dimensions are invalid");
            }
        } else if (std::memcmp(tag, "PLTE", 4) == 0) {
            if (length % 3 != 0 || length == 0) {
                throw std::invalid_argument("png has a malformed PLTE chunk");
            }
            const std::size_t entries = length / 3;
            out.palette.rgb.resize(entries);
            for (std::size_t i = 0; i < entries; ++i) {
                out.palette.rgb[i] = {data[3 * i], data[3 * i + 1], data[3 * i + 2]};
            }
        } else if (std::memcmp(tag, "tRNS", 4) == 0) {
            if (out.header.color_type == 3) {
                if (out.palette.rgb.empty() || length > out.palette.rgb.size()) {
                    throw std::invalid_argument("png tRNS chunk is malformed");
                }
                out.palette.has_alpha = true;
                out.palette.alpha.assign(out.palette.rgb.size(), 255);
                for (std::size_t i = 0; i < length; ++i) { out.palette.alpha[i] = data[i]; }
            }
            // tRNS for gray/rgb (a single transparent color) is treated as opaque.
        } else if (std::memcmp(tag, "IDAT", 4) == 0) {
            out.idat.insert(out.idat.end(), data, data + length);
        }
    }
    if (!out.have_ihdr) { throw std::invalid_argument("png is missing its IHDR chunk"); }
}

[[nodiscard]] std::vector<std::uint8_t> inflate_zlib(std::span<const std::uint8_t> data,
                                                     std::size_t expected) {
    z_stream strm{};
    if (inflateInit2(&strm, 15 + 32) != Z_OK) {
        throw std::invalid_argument("png: failed to initialize decompressor");
    }
    if (data.size() > std::numeric_limits<uInt>::max()) {
        throw std::invalid_argument("png IDAT stream exceeds the zlib input limit");
    }
    strm.next_in  = const_cast<std::uint8_t*>(data.data());
    strm.avail_in = static_cast<uInt>(data.size());
    std::vector<std::uint8_t> out;
    out.reserve(std::min<std::size_t>(expected, 65536));
    std::vector<std::uint8_t> chunk(65536);
    // Decompress up to `expected + 1` bytes. Producing `expected + 1` proves the stream is
    // longer than the image; reaching Z_STREAM_END at exactly `expected` proves it matches.
    // This keeps the output bounded (DoS-safe) while distinguishing exact / too-short /
    // too-long streams. (A stream that is exactly `expected` still has zlib's trailing
    // end-marker block unprocessed, so the final inflate returns Z_OK, not Z_STREAM_END —
    // comparing total produced bytes avoids misreading that as "longer".)
    const std::size_t cap = expected + 1;
    while (out.size() < cap) {
        const std::size_t want = std::min(chunk.size(), cap - out.size());
        strm.next_out = chunk.data();
        strm.avail_out = static_cast<uInt>(want);
        const int status = inflate(&strm, Z_NO_FLUSH);
        const std::size_t produced = want - strm.avail_out;
        out.insert(out.end(), chunk.data(), chunk.data() + produced);
        if (status == Z_STREAM_END) { break; }
        if (status != Z_OK) {
            inflateEnd(&strm);
            throw std::invalid_argument("png: IDAT stream failed to decompress");
        }
    }
    inflateEnd(&strm);
    if (out.size() > expected) {
        throw std::invalid_argument("png: IDAT stream is longer than the image");
    }
    if (out.size() != expected) {
        throw std::invalid_argument("png: IDAT stream length does not match the image");
    }
    return out;
}

// Reconstruct the raw (unfiltered) scanlines from the compressed stream. The encoder
// stores F = raw - predictor (mod 256), so reconstruction ADDS the predictor back:
// raw = F + predictor (wrapping). Predictors are pixel-aligned (libpng applies the filter
// at sample granularity): left = recon(x-bpp), up = prev(x), up-left = prev(x-bpp), where
// bpp is the number of bytes per pixel (bytes_per_sample * samples_per_pixel).
[[nodiscard]] std::vector<std::uint8_t> unfilter(const std::vector<std::uint8_t>& compressed,
                                                 int height, int row_bytes, int bpp) {
    std::vector<std::uint8_t> recon(static_cast<std::size_t>(height) * row_bytes);
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* src      = compressed.data() + static_cast<std::size_t>(y) * (row_bytes + 1);
        const std::uint8_t filter    = src[0];
        const std::uint8_t* filtered = src + 1;
        std::uint8_t* cur            = recon.data() + static_cast<std::size_t>(y) * row_bytes;
        const std::uint8_t* up       = y > 0 ? cur - row_bytes : nullptr;
        for (int x = 0; x < row_bytes; ++x) {
            const std::uint8_t left   = x >= bpp ? cur[x - bpp] : 0;
            const std::uint8_t upb    = up != nullptr ? up[x] : 0;
            const std::uint8_t upleft = (up != nullptr && x >= bpp) ? up[x - bpp] : 0;
            std::uint8_t v = filtered[x];
            switch (filter) {
                case 0: break;
                case 1: v = static_cast<std::uint8_t>(v + left); break;
                case 2: v = static_cast<std::uint8_t>(v + upb); break;
                case 3: v = static_cast<std::uint8_t>(v + ((left + upb) >> 1)); break;
                case 4: {
                    const int pa = std::abs(static_cast<int>(upb) - static_cast<int>(upleft));
                    const int pb = std::abs(static_cast<int>(left) - static_cast<int>(upleft));
                    const int pc = std::abs(static_cast<int>(left) + static_cast<int>(upb) -
                                            2 * static_cast<int>(upleft));
                    const std::uint8_t c = (pa <= pb && pa <= pc) ? left : (pb <= pc ? upb : upleft);
                    v = static_cast<std::uint8_t>(v + c);
                    break;
                }
                default: throw std::invalid_argument("png: unknown scanline filter type");
            }
            cur[x] = v;
        }
    }
    return recon;
}

// Reduce a full-precision sample to 8 bits.
[[nodiscard]] std::uint8_t to8(std::uint32_t v, int bit_depth) {
    if (bit_depth == 16) { return static_cast<std::uint8_t>(v >> 8); }
    if (bit_depth == 8) { return static_cast<std::uint8_t>(v); }
    const int maxv = (1 << bit_depth) - 1;
    return static_cast<std::uint8_t>((static_cast<int>(v) * 255 + maxv / 2) / maxv);
}

// Composite a channel over white by an 8-bit alpha (matches the FFmpeg image path).
[[nodiscard]] std::uint8_t composite_white(std::uint8_t channel, std::uint8_t alpha) {
    if (alpha >= 255) { return channel; }
    return static_cast<std::uint8_t>(((255 - alpha) * 255 + alpha * static_cast<int>(channel) + 127) /
                                     255);
}

} // namespace

bool is_png(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < kSignature.size()) { return false; }
    return std::memcmp(bytes.data(), kSignature.data(), kSignature.size()) == 0;
}

ImageInfo png_image_info(std::span<const std::uint8_t> bytes, std::uint64_t max_decoded_pixels) {
    if (!is_png(bytes)) { throw std::invalid_argument("not a png image"); }
    Parse parse;
    parse_chunks(bytes, parse);
    const std::uint64_t pixels =
        static_cast<std::uint64_t>(parse.header.width) * static_cast<std::uint64_t>(parse.header.height);
    if (pixels > max_decoded_pixels) {
        throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
    }
    return ImageInfo{.width = parse.header.width, .height = parse.header.height};
}

Image decode_png(std::span<const std::uint8_t> bytes, std::uint64_t max_decoded_pixels) {
    if (!is_png(bytes)) { throw std::invalid_argument("not a png image"); }
    Parse parse;
    parse_chunks(bytes, parse);
    const int width      = parse.header.width;
    const int height     = parse.header.height;
    const int bit_depth  = parse.header.bit_depth;
    const int color_type = parse.header.color_type;
    const int samples    = samples_per_pixel(color_type);
    if (samples == 0) { throw std::invalid_argument("png has an unsupported color type"); }
    if (color_type == 3 && parse.palette.rgb.empty()) {
        throw std::invalid_argument("png palette image is missing its PLTE chunk");
    }
    const std::uint64_t pixels = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    if (pixels > max_decoded_pixels) {
        throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
    }
    const int bytes_per_sample = bit_depth >= 8 ? bit_depth / 8 : 1;
    // Row geometry feeds 32-bit `int` offsets below, so bound the row width in 64-bit first.
    // A crafted 33,554,432x2 RGBA-16 image passes the 64-bit pixel budget with exactly this
    // product at 2^31 bits and would otherwise wrap `row_bytes` negative.
    const std::uint64_t row_bits = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(samples) *
                                   static_cast<std::uint64_t>(bit_depth);
    if (row_bits == 0) { throw std::invalid_argument("png row width is invalid"); }
    if (row_bits > 32ULL * 1024ULL * 1024ULL) {
        throw Error(ErrorKind::BudgetExceeded, "png row width exceeds the decode buffer limit");
    }
    const int row_bytes = static_cast<int>((row_bits + 7) / 8);
    const std::size_t raw_size = static_cast<std::size_t>(height) * static_cast<std::size_t>(row_bytes + 1);
    const std::vector<std::uint8_t> compressed = inflate_zlib(parse.idat, raw_size);
    // The Sub/Avg/Paeth filters reference the previous sample (pixel-aligned, x-bpp); the
    // reconstruction adds the predictor back. This is correct for every bit depth.
    const std::vector<std::uint8_t> recon =
        unfilter(compressed, height, row_bytes, std::max(1, samples * bytes_per_sample));

    // Full-precision sample (0 .. (1<<bit_depth)-1) at pixel (x,y), channel s.
    auto raw_value = [&](int x, int y, int s) -> std::uint32_t {
        if (bit_depth >= 8) {
            const std::uint8_t* base =
                recon.data() + static_cast<std::size_t>(y) * row_bytes +
                static_cast<std::size_t>(x) * static_cast<std::size_t>(samples) *
                    bytes_per_sample +
                static_cast<std::size_t>(s) * bytes_per_sample;
            if (bit_depth == 16) {
                return (static_cast<std::uint32_t>(base[0]) << 8) | static_cast<std::uint32_t>(base[1]);
            }
            return base[0];
        }
        const int bitpos = (x * samples + s) * bit_depth;
        const int byte   = bitpos >> 3;
        const int shift  = 7 - (bitpos & 7) - (bit_depth - 1);
        const std::uint8_t b = recon[static_cast<std::size_t>(y) * row_bytes + byte];
        return (static_cast<int>(b) >> shift) & ((1 << bit_depth) - 1);
    };

    Image out;
    out.width  = width;
    out.height = height;
    out.rgb.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
    std::uint8_t* px = out.rgb.data();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::uint8_t r = 0, g = 0, b = 0;
            switch (color_type) {
                case 0: {
                    const std::uint8_t v = to8(raw_value(x, y, 0), bit_depth);
                    r = g = b = v;
                    break;
                }
                case 2:
                    r = to8(raw_value(x, y, 0), bit_depth);
                    g = to8(raw_value(x, y, 1), bit_depth);
                    b = to8(raw_value(x, y, 2), bit_depth);
                    break;
                case 3: {
                    const std::uint32_t idx = raw_value(x, y, 0);
                    if (idx >= parse.palette.rgb.size()) {
                        throw std::invalid_argument("png palette index is out of range");
                    }
                    r = parse.palette.rgb[idx][0];
                    g = parse.palette.rgb[idx][1];
                    b = parse.palette.rgb[idx][2];
                    const std::uint8_t a =
                        parse.palette.has_alpha ? parse.palette.alpha[idx] : 255;
                    if (a < 255) {
                        r = composite_white(r, a);
                        g = composite_white(g, a);
                        b = composite_white(b, a);
                    }
                    break;
                }
                case 4: {
                    const std::uint8_t v = to8(raw_value(x, y, 0), bit_depth);
                    const std::uint8_t a = to8(raw_value(x, y, 1), bit_depth);
                    const std::uint8_t c = composite_white(v, a);
                    r = g = b = c;
                    break;
                }
                case 6: {
                    const std::uint8_t a = to8(raw_value(x, y, 3), bit_depth);
                    r = composite_white(to8(raw_value(x, y, 0), bit_depth), a);
                    g = composite_white(to8(raw_value(x, y, 1), bit_depth), a);
                    b = composite_white(to8(raw_value(x, y, 2), bit_depth), a);
                    break;
                }
            }
            px[0] = r;
            px[1] = g;
            px[2] = b;
            px += 3;
        }
    }
    return out;
}

} // namespace ninfer::media::decode