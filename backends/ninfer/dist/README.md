# RTX 3090 release bundles

Run the Windows packaging script from the repository root after the verified native build exists.
It reads the release from `VERSION` and needs the matching `RELEASE_NOTES_<version>.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package-release.ps1
```

It creates a versioned directory and archive under `dist/`:

- `ninfer-rtx3090-windows-x64-*`: native Windows CLI, server, benchmark, and vcpkg DLLs;
- `SHA256SUMS-v<version>-windows.txt`: archive hash for release verification.

Generated binaries and archives are ignored by Git because GitHub source repositories should not
contain build products. Upload the `.zip` and versioned checksum file as GitHub Release assets.
The packaging guide itself is tracked.

Model artifacts are not included. Download either the 16.29 GiB `qwen3_6_27b.ninfer` or 20.84 GiB
`qwen3_6_35b_a3b.ninfer` artifact from the repositories linked in the project README and verify its
published SHA-256 separately. The compact 35B artifact is text-only; leave `--vision` disabled.

The Windows bundle includes its FFmpeg/curl/zlib DLLs and requires the NVIDIA driver and Microsoft
Visual C++ 2022 runtime.
