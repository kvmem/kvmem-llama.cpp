#pragma once

#include "kvmem-api-anthropic.h"
#include "kvmem-chat-id.h"

namespace kvmem_anthropic {

inline bool route(const std::string &path) {
    return path == "/v1/messages" || path == "/v1/messages/count_tokens";
}

inline bool authentication_error(const httplib::Request &req, httplib::Response &res) {
    if (!route(req.path)) return false;
    response::error(res, 401, "Invalid API Key");
    return true;
}

template <class Count, class Generate>
void install_routes(httplib::Server &server, const std::string &model, int generation_limit,
                    const std::map<std::string, std::string> &template_defaults, const json &sampling_defaults,
                    Count count, Generate generate) {
    auto handle = [=](const httplib::Request &req, httplib::Response &res) {
        res.set_header("request-id", "req_" + kvmem_chat_request_id());
        try {
            version(req);
            const bool count_only = req.path == "/v1/messages/count_tokens";
            auto parsed =
                parse(json::parse(req.body), model, generation_limit, template_defaults, sampling_defaults, count_only);
            if (count_only) {
                res.set_content(json{{"input_tokens", count(parsed.chat)}}.dump(), "application/json");
            } else {
                response encoder(parsed);
                generate(req, res, std::move(parsed.chat), std::move(encoder));
            }
        } catch (const request_error &e) {
            response::error(res, e.status, e.what());
        } catch (const json::exception &e) {
            response::error(res, 400, e.what());
        } catch (const std::exception &e) {
            response::error(res, 500, e.what());
        }
    };
    server.Post("/v1/messages", handle);
    server.Post("/v1/messages/count_tokens", handle);
}

} // namespace kvmem_anthropic
