#define main existing_rmsnorm_rope_main
#include "../ops/test_rmsnorm_rope.cpp"
#undef main

int main() {
    if (cuda_unavailable()) { return 77; }
    int failures = 0;
    // The relevant model geometry at page boundaries and at original positions
    // well beyond the 256-token working set. run_text_case uses an FP64 oracle.
    for (int width : {1, 63, 64, 65, 128}) {
        for (int original_position : {1023, 262000}) {
            failures += run_text_case(24, 4, width, original_position, 9173U + width);
        }
    }
    failures += run_text_case(24, 4, 64, 262000, 9113U, true);
    std::cout << (failures ? "P1_ROPE FAIL\n" : "P1_ROPE PASS\n");
    return failures ? 1 : 0;
}
