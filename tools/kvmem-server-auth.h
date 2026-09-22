#pragma once

#include "httplib.h"
#include <algorithm>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

// A single gate covers API routes before JSON parsing or slot acquisition.
// Only health checks, preflights and the mounted UI's exact asset paths are public.
inline void kvmem_install_auth(httplib::Server & server, std::vector<std::string> keys,
                                std::unordered_set<std::string> ui_paths,
                                std::function<bool(const httplib::Request &, httplib::Response &)> failure_handler = {}) {
    server.set_pre_routing_handler([keys = std::move(keys), ui_paths = std::move(ui_paths), failure_handler = std::move(failure_handler)](
            const httplib::Request & req, httplib::Response & res) {
        using Result = httplib::Server::HandlerResponse;
        if (keys.empty() || req.method == "OPTIONS") return Result::Unhandled;
        if (req.path == "/health" || req.path == "/v1/health") return Result::Unhandled;
        const bool api = req.path == "/props" || req.path == "/models" ||
                         req.path == "/chat/completions" || req.path.compare(0, 4, "/v1/") == 0;
        if (!api && (req.method == "GET" || req.method == "HEAD") && ui_paths.count(req.path))
            return Result::Unhandled;
        std::string key = req.get_header_value("Authorization");
        if (key.empty()) key = req.get_header_value("X-Api-Key");
        if (key.compare(0, 7, "Bearer ") == 0) key.erase(0, 7);
        if (std::find(keys.begin(), keys.end(), key) != keys.end()) return Result::Unhandled;
        res.status = 401;
        res.set_header("Cache-Control", "no-store");
        if (failure_handler && failure_handler(req, res)) return Result::Handled;
        res.set_content(R"({"error":{"message":"Invalid API Key","type":"authentication_error","code":401}})",
                        "application/json; charset=utf-8");
        return Result::Handled;
    });
}
