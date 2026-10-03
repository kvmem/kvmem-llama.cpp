// The existing independent byte-layout oracle also checks stale generations,
// fragmented physical mappings, transfer groups, and split-plane restoration.
#define main existing_paged_kv_test_main
#include "../test_kv_cache.cpp"
#undef main

int main() {
    try {
        int failures = existing_paged_kv_test_main();
        if (failures) { return failures; }
        ninfer::DeviceContext context(0);
        for (bool quantized : {false, true}) {
            ninfer::KVPageGeometry geometry;
            geometry.device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor;
            for (int layer = 0; layer < 3; ++layer) {
                geometry.planes.push_back({quantized ? ninfer::DType::I8 : ninfer::DType::BF16, 256, 4, 256});
                geometry.planes.push_back({quantized ? ninfer::DType::I8 : ninfer::DType::BF16, 256, 4, 256});
                if (quantized) {
                    geometry.planes.push_back({ninfer::DType::FP16, 4, 4, 256});
                    geometry.planes.push_back({ninfer::DType::FP16, 4, 4, 256});
                }
            }
            failures += exercise_layout_and_transfer(context, geometry,
                quantized ? "P1 3-layer INT8 K/V/scales" : "P1 3-layer BF16 K/V");
        }
        ninfer::KVPageGeometry rotated;
        rotated.device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor;
        for (int layer = 0; layer < 3; ++layer) {
            rotated.planes.push_back({ninfer::DType::I8, 256, 4, 256});
            rotated.planes.push_back({ninfer::DType::U8, 128, 4, 256});
            rotated.planes.push_back({ninfer::DType::FP16, 4, 4, 256});
            rotated.planes.push_back({ninfer::DType::FP16, 8, 4, 256});
        }
        failures += exercise_layout_and_transfer(context, rotated, "P10 3-layer RK8V4 K/V/scales");
        std::cout << (failures ? "P1_PAYLOAD FAIL\n" : "P1_PAYLOAD PASS\n");
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "P1_PAYLOAD FAIL " << e.what() << '\n';
        return 1;
    }
}
