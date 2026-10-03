#!/usr/bin/env bash
set -euo pipefail

# Unpacks llama.cpp's released WebUI into a directory for -DNINFER_WEBUI_DIR. The build itself never
# downloads anything, so offline and hermetic builds stay that way unless this is run first.
#
#   fetch-webui.sh [tag] [dir]
#
# tag defaults to the release below; dir defaults to third_party/webui/<tag> (ignored by git).

tag=${1:-b11165}
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dir=${2:-$root/third_party/webui/$tag}
url="https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-ui.tar.gz"

tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
curl -fL --retry 3 -o "$tmp/ui.tar.gz" "$url"
mkdir -p "$dir"
tar -xzf "$tmp/ui.tar.gz" -C "$dir"
index="$(find "$dir" \( -name index.html -o -name index.html.gz \) -print | head -n 1)"
if [[ -z "$index" ]]; then
  printf 'no index.html in %s\n' "$url" >&2
  exit 1
fi
printf 'configure with -DNINFER_WEBUI_DIR=%s\n' "$(dirname -- "$index")"
