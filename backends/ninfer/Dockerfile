# syntax=docker/dockerfile:1

FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ccache \
        cmake \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
        rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Keep source mtimes stable only when contents match; restored/older checkouts must
# still rebuild changed inputs. Packages, the Dockerfile and CMake scripts identify
# the configuration: removed flags and changed defaults must not reuse CMakeCache.txt.
# Lock the mutable Ninja tree and copy deliverables out of the transient cache mount.
RUN --mount=type=cache,id=ninfer-build,target=/build,sharing=locked \
    --mount=type=cache,target=/ccache \
    export CCACHE_DIR=/ccache CCACHE_MAXSIZE=20G \
    && find . -type f \( -path ./Dockerfile -o -name CMakeLists.txt -o -name '*.cmake' \) \
        -exec sha256sum {} + > /build/configuration \
    && dpkg-query -W >> /build/configuration \
    && LC_ALL=C sort -o /build/configuration /build/configuration \
    && build_dir="/build/$(sha256sum /build/configuration | cut -d ' ' -f 1)" \
    && rsync --recursive --links --checksum --delete /src/ /build/src/ \
    && cmake -S /build/src -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build "$build_dir" --parallel --target ninfer ninfer-serve \
    && mkdir -p /out \
    && cp "$build_dir/apps/ninfer" "$build_dir/apps/ninfer-serve" /out/ \
    && ccache --show-stats

FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/*

# The CUDA runtime image ships forward-compatibility libraries in
# /usr/local/cuda*/compat (a newer libcuda.so than the host driver). Forward
# compatibility is supported only on datacenter GPUs; on any GeForce card the
# loader picks these up and every CUDA call fails at startup with
#   cudaErrorCompatNotSupportedOnDevice: forward compatibility was attempted
#   on non supported HW
# Removing them lets the container use the host driver through ordinary CUDA
# minor-version compatibility, which is what an RTX 3090/3090 Ti needs.
RUN rm -rf /usr/local/cuda-13.1/compat /usr/local/cuda-13/compat /usr/local/cuda/compat

COPY --from=build /out/ninfer /usr/local/bin/ninfer
COPY --from=build /out/ninfer-serve /usr/local/bin/ninfer-serve

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

CMD ["ninfer-serve", "--help"]
