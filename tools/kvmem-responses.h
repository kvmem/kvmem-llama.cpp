#pragma once

#include <string>

// Convert an OpenAI Responses request body into a Chat Completions request
// body. Reuses llama.cpp's server_chat_convert_responses_to_chatcmpl verbatim;
// this header only exposes a string-in/string-out seam so the upstream
// common_json types never meet the server's nlohmann json in one translation
// unit (server-common.h sets `using json = common_json`, the server sets
// `using json = nlohmann::json`).
//
// Throws std::exception (common_json_error or std::invalid_argument) when the
// input is not a valid Responses request.
std::string kvmem_responses_to_chatcmpl(const std::string & body);
