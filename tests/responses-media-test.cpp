// Use upstream's common_json in its own translation unit. The Responses bridge
// crosses to the server's nlohmann::json through a string boundary.
#include "kvmem-responses.h"
#include "server-common.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

void test_image_media_parser() {
    const std::string request = R"({"input": [{"role": "user", "content": [
        {"type": "input_text", "text": "Describe this"},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}
    ]}]})";
    const std::string converted = kvmem_responses_to_chatcmpl(request);
    server_chat_params params{};
    params.allow_image = true;
    std::vector<raw_buffer> files;
    json body = json::parse(converted);
    oaicompat_chat_process_media(body, params, files);
    CHECK(files.size() == 1);
    CHECK(files[0].size() > 8);
    CHECK(files[0][0] == 0x89 && files[0][1] == 'P' && files[0][2] == 'N' && files[0][3] == 'G');
    CHECK(body["messages"][0]["content"][0]["text"] == "Describe this");
    CHECK(body["messages"][0]["content"][1]["type"] == "media_marker");

    params.allow_image = false;
    body = json::parse(converted);
    files.clear();
    bool rejected = false;
    try {
        oaicompat_chat_process_media(body, params, files);
    } catch (const std::runtime_error & e) {
        rejected = std::string(e.what()).find("mmproj") != std::string::npos;
    }
    CHECK(rejected);
}
