#include "media/decode/decode.h"

#ifdef NINFER_MEDIA_NATIVE_PNG
#include "media/decode/png_decode.h"
#include "png_fixtures_generated.h"
#endif

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Neroued/ninfer#20: this 300x200 yuvj420p JPEG selected an swscale SIMD path that overwrote a
// width-tight RGB24 destination. Keep the exact public reproducer so the test does not depend on
// an encoder or external files.
constexpr std::string_view issue_20_jpeg_base64 = R"jpeg(
/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAx
NDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy
MjIyMjIyMjL/wAARCADIASwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUF
BAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW
V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi
4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAEC
AxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVm
Z2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq
8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDFooor7E+RCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK
KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKA
CiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiii
gAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA7TQtC0290W3uLi23yvu3NvYZwxHY+1aP/AAjGj/8APn/5
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
z/nP/IyxH+Of/pTP0rAf7rS/wx/JBRRRXmnWFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXDeM/+QxD/
ANe6/wDoTV3NcN4z/wCQxD/17r/6E1fXcE/8jZf4ZHicQf7k/VHOUUUV+yHwoUUUUAFFFFABRRRQAUUUUAFFFFAHeeHdQsoNBto5
ry3jcbsq8oBHzHsTWr/aunf8/wDa/wDf5f8AGvLqK+HxfA+HxOIqV3Vac23surufQUeIatKlGmoLRJfceo/2rp3/AD/2v/f5f8aP
7V07/n/tf+/y/wCNeXUVz/8AEP8ADf8AP6X3I1/1lq/8+197PUf7V07/AJ/7X/v8v+NH9q6d/wA/9r/3+X/GvLqKP+If4b/n9L7k
H+stX/n2vvZ6j/aunf8AP/a/9/l/xo/tXTv+f+1/7/L/AI15dRR/xD/Df8/pfcg/1lq/8+197PUf7V07/n/tf+/y/wCNH9q6d/z/
ANr/AN/l/wAa8uoo/wCIf4b/AJ/S+5B/rLV/59r72eo/2rp3/P8A2v8A3+X/ABo/tXTv+f8Atf8Av8v+NeXUUf8AEP8ADf8AP6X3
IP8AWWr/AM+197PUf7V07/n/ALX/AL/L/jR/aunf8/8Aa/8Af5f8a8uoo/4h/hv+f0vuQf6y1f8An2vvZ6j/AGrp3/P/AGv/AH+X
/Gj+1dO/5/7X/v8AL/jXl1FH/EP8N/z+l9yD/WWr/wA+197PUf7V07/n/tf+/wAv+NH9q6d/z/2v/f5f8a8uoo/4h/hv+f0vuQf6
y1f+fa+9nqP9q6d/z/2v/f5f8aP7V07/AJ/7X/v8v+NeXUUf8Q/w3/P6X3IP9Zav/Ptfez1H+1dO/wCf+1/7/L/jR/aunf8AP/a/
9/l/xry6ij/iH+G/5/S+5B/rLV/59r72eo/2rp3/AD/2v/f5f8aP7V07/n/tf+/y/wCNeXUUf8Q/w3/P6X3IP9Zav/Ptfez1H+1d
O/5/7X/v8v8AjXGeLbiG51WJ4Jo5VEABaNgwzubjisGivUyfhOjlmJWJhUcnZqzS6nHjs6qYuj7KUUgooor6w8YKKKKACiiigAoo
ooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK
KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKA
CiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiii
gD//2Q==
)jpeg";

std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
    std::array<int, 256> values;
    values.fill(-1);
    constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (std::size_t i = 0; i < alphabet.size(); ++i) {
        values[static_cast<unsigned char>(alphabet[i])] = static_cast<int>(i);
    }

    std::vector<std::uint8_t> out;
    out.reserve(encoded.size() * 3 / 4);
    unsigned accumulator = 0;
    int bits             = 0;
    for (const unsigned char byte : encoded) {
        if (std::isspace(byte)) { continue; }
        if (byte == '=') { break; }
        const int value = values[byte];
        if (value < 0) { throw std::runtime_error("invalid base64 fixture"); }
        accumulator = (accumulator << 6U) | static_cast<unsigned>(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>(accumulator >> bits));
            accumulator &= (1U << bits) - 1U;
        }
    }
    return out;
}

void expect_pixel(const ninfer::media::decode::Image& image, int x, int y,
                  std::array<int, 3> expected, int tolerance) {
    const std::size_t offset = (static_cast<std::size_t>(y) * image.width + x) * 3;
    for (int channel = 0; channel < 3; ++channel) {
        const int actual = image.rgb[offset + static_cast<std::size_t>(channel)];
        if (actual < expected[static_cast<std::size_t>(channel)] - tolerance ||
            actual > expected[static_cast<std::size_t>(channel)] + tolerance) {
            throw std::runtime_error("decoded JPEG pixel mismatch");
        }
    }
}

void test_issue_20_unaligned_jpeg() {
    const std::vector<std::uint8_t> encoded     = decode_base64(issue_20_jpeg_base64);
    const ninfer::media::decode::ImageInfo info = ninfer::media::decode::inspect_image(encoded, {});
    const ninfer::media::decode::Image image    = ninfer::media::decode::decode_image(encoded, {});
    if (info.width != image.width || info.height != image.height || image.width != 300 ||
        image.height != 200 || image.rgb.size() != 300U * 200U * 3U) {
        throw std::runtime_error("decoded JPEG dimensions mismatch");
    }

    constexpr std::array<int, 3> blue   = {29, 120, 199};
    constexpr std::array<int, 3> yellow = {255, 200, 1};
    expect_pixel(image, 0, 0, blue, 3);
    expect_pixel(image, 299, 0, blue, 3);
    expect_pixel(image, 0, 199, blue, 3);
    expect_pixel(image, 299, 199, blue, 3);
    expect_pixel(image, 150, 100, yellow, 4);

    const ninfer::media::decode::VideoInfo video_info =
        ninfer::media::decode::inspect_video(encoded, {}, 2.0, 4, 16);
    const ninfer::media::decode::Video video =
        ninfer::media::decode::decode_video(encoded, {}, 2.0, 4, 16);
    if (video_info.width != video.width || video_info.height != video.height ||
        video_info.sampled_frames != static_cast<int>(video.frames.size()) ||
        video_info.indices != video.indices) {
        throw std::runtime_error("inspected video geometry differs from decoded video");
    }
}

#ifdef NINFER_MEDIA_NATIVE_PNG

// span over a generated C array fixture.
std::span<const std::uint8_t> sp(const std::uint8_t* arr, std::size_t n) {
    return std::span<const std::uint8_t>(arr, n);
}

// Exact RGB check (tolerance 0) against a decoded image.
void expect_exact(const ninfer::media::decode::Image& image, int x, int y,
                  std::array<int, 3> expected) {
    expect_pixel(image, x, y, expected, 0);
}

void expect_throws_invalid(std::function<void()> fn, const char* what) {
    try {
        fn();
        throw std::runtime_error(std::string("expected invalid_argument, got success: ") + what);
    } catch (const std::invalid_argument&) {
        // expected
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("expected invalid_argument, got: ") + e.what());
    }
}

void test_png_decoders() {
    using namespace ninfer::media::decode;

    // 8-bit RGB, 4x2.
    {
        auto s = sp(png_rgb8, std::size(png_rgb8));
        const ImageInfo info = inspect_image(s, {});
        if (info.width != 4 || info.height != 2) { throw std::runtime_error("png rgb8 dims"); }
        const Image image = decode_image(s, {});
        expect_exact(image, 0, 0, {255, 0, 0});
        expect_exact(image, 1, 0, {0, 255, 0});
        expect_exact(image, 2, 0, {0, 0, 255});
        expect_exact(image, 3, 0, {128, 128, 128});
        expect_exact(image, 0, 1, {1, 2, 3});
        expect_exact(image, 2, 1, {7, 8, 9});
        expect_exact(image, 3, 1, {200, 150, 100});
    }

    // 8-bit gray, 3x2 (triples share the value).
    {
        auto s = sp(png_gray8, std::size(png_gray8));
        const Image image = decode_image(s, {});
        expect_exact(image, 0, 0, {0, 0, 0});
        expect_exact(image, 1, 0, {120, 120, 120});
        expect_exact(image, 2, 0, {255, 255, 255});
        expect_exact(image, 1, 1, {64, 64, 64});
    }

    // 8-bit palette, 3x1 (indices 0,2,1 into PLTE).
    {
        auto s = sp(png_palette, std::size(png_palette));
        const Image image = decode_image(s, {});
        expect_exact(image, 0, 0, {10, 20, 30});
        expect_exact(image, 1, 0, {70, 80, 90});
        expect_exact(image, 2, 0, {40, 50, 60});
    }

    // 16-bit gray, 2x1 (0 and 65535 -> 0 and 255).
    {
        auto s = sp(png_gray16, std::size(png_gray16));
        const Image image = decode_image(s, {});
        expect_exact(image, 0, 0, {0, 0, 0});
        expect_exact(image, 1, 0, {255, 255, 255});
    }

    // 1-bit gray, 2x2 (1,1 / 0,0 -> 255,255 / 0,0).
    {
        auto s = sp(png_gray1, std::size(png_gray1));
        const Image image = decode_image(s, {});
        expect_exact(image, 0, 0, {255, 255, 255});
        expect_exact(image, 1, 0, {255, 255, 255});
        expect_exact(image, 0, 1, {0, 0, 0});
        expect_exact(image, 1, 1, {0, 0, 0});
    }

    // 8-bit RGB, 4x4, one of each filter (1,2,3,4) per row.
    {
        auto s = sp(png_filters, std::size(png_filters));
        const Image image = decode_image(s, {});
        const std::array<std::array<int, 3>, 16> rows = {{
            {10, 20, 30}, {40, 50, 60}, {70, 80, 90}, {1, 2, 3},
            {100, 110, 120}, {130, 140, 150}, {160, 170, 180}, {200, 210, 220},
            {200, 210, 220}, {230, 240, 250}, {0, 1, 2}, {100, 101, 102},
            {5, 6, 7}, {8, 9, 10}, {11, 12, 13}, {14, 15, 16},
        }};
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) { expect_exact(image, x, y, rows[y * 4 + x]); }
        }
    }

    // 8-bit RGBA, 2x1. Pixel 0 alpha 128 composited over white; pixel 1 opaque.
    {
        auto s = sp(png_rgba8, std::size(png_rgba8));
        const Image image = decode_image(s, {});
        expect_exact(image, 0, 0, {255, 127, 127});
        expect_exact(image, 1, 0, {0, 255, 0});
    }

    // palette + tRNS, 3x1. Middle entry alpha 128 composited over white.
    {
        auto s = sp(png_palette_trns, std::size(png_palette_trns));
        const Image image = decode_image(s, {});
        expect_exact(image, 0, 0, {10, 20, 30});
        expect_exact(image, 1, 0, {147, 152, 157});
        expect_exact(image, 2, 0, {70, 80, 90});
    }

    // Pixel budget is enforced before decode (throws Error, a runtime_error).
    {
        auto s = sp(png_rgb8, std::size(png_rgb8));
        Policy small;
        small.max_decoded_pixels = 7; // < 4*2
        try {
            inspect_image(s, small);
            throw std::runtime_error("expected budget error for png rgb8, got success");
        } catch (const Error&) {
            // expected
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("expected budget error, got: ") + e.what());
        }
    }

    // Non-PNG bytes are rejected by the PNG path (and routed to FFmpeg normally).
    {
        std::vector<std::uint8_t> jpeg{0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10};
        expect_throws_invalid([&] { decode_png(jpeg, 64); }, "non-png png_decode");
        if (is_png(jpeg)) { throw std::runtime_error("is_png misdetected jpeg"); }
    }

    // Truncated PNG (cut into IDAT) is rejected.
    {
        const std::size_t keep = 30; // < full; cuts the IDAT stream short
        if (keep >= std::size(png_rgb8)) {
            throw std::runtime_error("png truncation fixture unexpectedly too short");
        }
        const auto it = std::span<const std::uint8_t>(png_rgb8, std::size(png_rgb8)).first(keep);
        expect_throws_invalid([&] { decode_png(it, 64); }, "truncated png");
    }

    // Filtered low-bit-depth scanlines decode correctly (byte-aligned predictors).
    // 4-bit gray, 2x1, samples [1, 15] (Sub-filtered): 1 -> 17, 15 -> 255 in 8-bit.
    {
        auto s = sp(png_lowbit_filtered, std::size(png_lowbit_filtered));
        const Image image = decode_png(s, 64);
        expect_exact(image, 0, 0, {17, 17, 17});
        expect_exact(image, 1, 0, {255, 255, 255});
    }

    // Overlong IDAT (stream decompresses past the image size) is rejected.
    {
        auto s = sp(png_overlong, std::size(png_overlong));
        expect_throws_invalid([&] { decode_png(s, 64); }, "overlong idat png");
    }
}

// Row-width guard regression: a crafted image whose pixel count is within budget
// but whose per-row bit width overflows the 32-bit `row_bytes` arithmetic must be
// rejected with a BudgetExceeded Error, not a wrapped-negative allocation. A
// 33,554,432 x 2 RGBA-16 image has pixels = 67,108,864 (within a generous pixel
// budget) but row_bits = 2^31, which would wrap `row_bytes` negative before the
// 64-bit guard. CRCs are not validated by parse_chunks and the guard fires before
// inflate, so the IDAT payload is irrelevant.
void test_png_row_width_guard() {
    using namespace ninfer::media::decode;
    std::vector<std::uint8_t> png = {
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // signature
        0x00, 0x00, 0x00, 0x0d, // IHDR length = 13
        0x49, 0x48, 0x44, 0x52, // "IHDR"
        0x02, 0x00, 0x00, 0x00, // width  = 33,554,432 (2^25)
        0x00, 0x00, 0x00, 0x02, // height = 2
        0x10, // bit depth = 16
        0x06, // color type = 6 (RGBA)
        0x00, 0x00, 0x00, // compression, filter, interlace
        0x00, 0x00, 0x00, 0x00, // IHDR CRC (not validated)
        0x00, 0x00, 0x00, 0x00, // IDAT length = 0
        0x49, 0x44, 0x41, 0x54, // "IDAT"
        0x00, 0x00, 0x00, 0x00, // IDAT CRC (not validated)
        0x00, 0x00, 0x00, 0x00, // IEND length = 0
        0x49, 0x45, 0x4e, 0x44, // "IEND"
        0x00, 0x00, 0x00, 0x00, // IEND CRC (not validated)
    };
    const std::uint64_t generous_pixel_budget = 128ULL * 1024ULL * 1024ULL;
    try {
        decode_png(png, generous_pixel_budget);
        throw std::runtime_error("expected BudgetExceeded for oversized png row, got success");
    } catch (const Error& e) {
        if (e.kind() != ErrorKind::BudgetExceeded) {
            throw std::runtime_error(std::string("expected BudgetExceeded, got: ") + e.what());
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("expected BudgetExceeded Error, got: ") + e.what());
    }
}

#endif // NINFER_MEDIA_NATIVE_PNG

} // namespace

int main() {
    try {
        test_issue_20_unaligned_jpeg();
#ifdef NINFER_MEDIA_NATIVE_PNG
        test_png_decoders();
        test_png_row_width_guard();
#endif
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
