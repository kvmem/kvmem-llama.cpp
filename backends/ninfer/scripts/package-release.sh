#!/usr/bin/env bash
# Build the Linux release archive for the version in the repository's VERSION file.
#
#   NINFER_BUILD_ROOT=~/ninfer/build-linux scripts/package-release.sh
#   scripts/build.sh --package         configure + build, then this
#
# There is one packager, not one per release. Everything version-specific is derived from VERSION,
# whose content is the full release tag (for example 0.10.0-rtx3090): the text before the first `-`
# names RELEASE_NOTES_<version>.md and the checksum file, and the whole tag names the archive. To cut
# a release, bump VERSION and write its release notes; there is nothing here to copy and edit.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f "$repo_root/VERSION" ]] || { printf 'Missing %s\n' "$repo_root/VERSION" >&2; exit 1; }
release_tag="$(tr -d '[:space:]' < "$repo_root/VERSION")"
[[ -n "$release_tag" ]] || { printf '%s is empty\n' "$repo_root/VERSION" >&2; exit 1; }
release_version="${release_tag%%-*}"
release_notes="RELEASE_NOTES_$release_version.md"
# Checked before anything is deleted or built into dist, so a forgotten release-notes file costs
# nothing rather than a finished archive that has to be thrown away.
[[ -f "$repo_root/$release_notes" ]] || {
  printf 'Missing %s: write the release notes for %s before packaging it\n' \
    "$repo_root/$release_notes" "$release_tag" >&2
  exit 1
}

build_root="${NINFER_BUILD_ROOT:-$repo_root/build-linux}"
dist_root="$repo_root/dist"
product_name="ninfer-rtx3090-linux-x64-$release_tag"
product_root="$dist_root/$product_name"
archive_name="$product_name.tar.gz"
archive_path="$dist_root/$archive_name"
checksum_path="$dist_root/SHA256SUMS-v$release_version-linux.txt"

mkdir -p -- "$dist_root"
case "$product_root" in "$dist_root/$product_name") ;; *) exit 1 ;; esac
rm -rf -- "$product_root"
rm -f -- "$archive_path"
# The checksum goes too, and before anything can fail. It is written last, so leaving a previous
# run's copy in place means a mid-run failure can leave `dist` holding a new archive beside a
# checksum for the old one -- and publishing that pair is worse than publishing neither.
rm -f -- "$checksum_path"
mkdir -- "$product_root"

for product in 'apps/ninfer:ninfer' 'apps/ninfer-serve:ninfer-serve' 'bench/ninfer_bench:ninfer_bench'; do
  source_path="$build_root/${product%%:*}"
  destination="$product_root/${product#*:}"
  [[ -f "$source_path" ]] || { printf 'Missing release product: %s\n' "$source_path" >&2; exit 1; }
  cp -- "$source_path" "$destination"
done
cp -- "$repo_root/VERSION" "$repo_root/LICENSE" "$repo_root/$release_notes" "$product_root/"
# The archive README must describe the archive. docs/rtx-3090-linux.md is a guide to *building* from
# source and points at ./scripts/download-model.sh; the packager copies that to the archive root, so
# a user following it from inside the archive got a missing-file error.
cp -- "$repo_root/docs/release-archive-linux.md" "$product_root/README.md"
# Explicit rather than globbed, so a new script is shipped only once someone has decided it belongs
# in the archive.
cp -- "$repo_root"/scripts/{run.sh,download-model.sh} "$product_root/"
(
  cd -- "$product_root"
  mapfile -d '' files < <(find . -maxdepth 1 -type f ! -name SHA256SUMS.txt -print0 | LC_ALL=C sort -z)
  sha256sum -- "${files[@]}" > SHA256SUMS.txt
)
tar -C "$dist_root" -czf "$archive_path" "$product_name"
(
  cd -- "$dist_root"
  sha256sum -- "$archive_name" > "$(basename -- "$checksum_path")"
)
du -h -- "$archive_path" "$checksum_path"
