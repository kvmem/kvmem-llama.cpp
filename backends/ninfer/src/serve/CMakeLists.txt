add_library(ninfer_serve STATIC
  http_transport.cpp
  console_log.cpp
  request_validation.cpp
  structured_output.cpp
  openai_chat_request.cpp
  openai_chat_response.cpp
  openai_chat_http.cpp
  openai_common.cpp
  openai_responses_request.cpp
  openai_responses_state.cpp
  openai_responses_response.cpp
  anthropic_messages_request.cpp
  anthropic_messages_response.cpp
  anthropic_messages_http.cpp
  openai_responses_store.cpp
  translate.cpp
  generation_service.cpp
  request_events.cpp
  console_stats.cpp
  operational_log.cpp
  request_log.cpp
  http_server.cpp
  mcp_proxy.cpp
  load_report.cpp
  serve_metrics.cpp
  openai_responses_http.cpp
  serve_options.cpp
)
ninfer_internal_includes(ninfer_serve)
target_link_libraries(ninfer_serve
  PUBLIC ninfer_engine Threads::Threads ninfer::json ninfer::httplib
  PRIVATE ninfer_media_acquire ninfer_product_logging ninfer_product_prompt_input
          ${NINFER_CUDART_TARGET})

# A WebUI is compiled in only from a local directory, never downloaded at configure time, so offline
# and hermetic builds are unaffected. scripts/fetch-webui.sh unpacks llama.cpp's released WebUI into
# such a directory.
set(NINFER_WEBUI_DIR "" CACHE PATH
    "Directory of built WebUI files (with index.html) to serve from ninfer-serve; empty embeds none")
if(NINFER_WEBUI_DIR)
  if(NOT EXISTS "${NINFER_WEBUI_DIR}/index.html" AND NOT EXISTS "${NINFER_WEBUI_DIR}/index.html.gz")
    message(FATAL_ERROR "NINFER_WEBUI_DIR has no index.html: ${NINFER_WEBUI_DIR}")
  endif()
  add_executable(ninfer_webui_embed webui/embed.cpp)
  file(GLOB_RECURSE NINFER_WEBUI_FILES CONFIGURE_DEPENDS "${NINFER_WEBUI_DIR}/*")
  set(NINFER_WEBUI_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/webui_assets.cpp")
  add_custom_command(
    OUTPUT "${NINFER_WEBUI_SOURCE}"
    COMMAND ninfer_webui_embed "${NINFER_WEBUI_SOURCE}" "${NINFER_WEBUI_DIR}"
    DEPENDS ninfer_webui_embed ${NINFER_WEBUI_FILES}
    COMMENT "Embedding the WebUI from ${NINFER_WEBUI_DIR}"
    VERBATIM)
  target_sources(ninfer_serve PRIVATE "${NINFER_WEBUI_SOURCE}")
else()
  target_sources(ninfer_serve PRIVATE webui_none.cpp)
endif()
