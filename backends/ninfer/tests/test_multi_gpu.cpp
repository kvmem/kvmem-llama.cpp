// The pieces of the multi-GPU pipeline that need no model, in one executable: the stage plan and its
// solver, the KV pool and block tables across ranks, and the boundary transfer between stages. The GPU
// ones run with two or three ranks on device 0 and report SKIP when there is no CUDA device.

int run_stage_plan_test();
int run_kv_cache_ranks_test();
int run_stage_link_test();

namespace {

// 77 is a suite that found no CUDA device; the others still run.
int failed(int status) { return status != 0 && status != 77 ? 1 : 0; }

} // namespace

int main() {
    int failures = failed(run_stage_plan_test());
    failures += failed(run_kv_cache_ranks_test());
    failures += failed(run_stage_link_test());
    return failures == 0 ? 0 : 1;
}
