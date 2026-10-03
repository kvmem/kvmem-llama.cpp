// Reuse the maintained represented-value codecs and guarded buffers, but select
// original logical keys independently of the production compact page table.
#include "ops/softmax_attention/causal_cache.cpp"

int sparse_case(const CachePlan& plan, const std::vector<int>& prefix, int offset,
                int width, bool graph) {
    const Geometry geometry = kGeometries[0]; // Qwen3.8-27B D256/H24/KV4
    const int original_base = 6 * 64 + offset;
    const int compact_base = static_cast<int>(prefix.size()) * 64 + offset;
    std::vector<int> selected = prefix;
    selected.insert(selected.end(), {6, 7, 8});
    const auto initial = make_cache(geometry, plan, 640, 7103);
    auto expected = initial;
    auto q = make_bf16_values(256ULL * geometry.q_heads * width, 7111, -0.25f, 0.25f);
    auto k = make_bf16_values(256ULL * geometry.kv_heads * width, 7112, -0.25f, 0.25f);
    auto v = make_bf16_values(256ULL * geometry.kv_heads * width, 7113, -1.0f, 1.0f);
    std::vector<std::int32_t> original_positions, compact_positions;
    for (int i = 0; i < width; ++i) {
        original_positions.push_back(original_base + i);
        compact_positions.push_back(compact_base + i);
    }
    append_cache(expected, k, v, original_positions);
    std::vector<double> reference(q.size());
    const int keys = compact_base + width;
    const auto logical_key = [&](int key) { return selected.at(key / 64) * 64 + key % 64; };
    naive_dense_softmax_attention(op_geometry(geometry), width, keys, 0.0625,
        [&](int d, int h, int t) { return double(q[q_index(geometry, h, d, t)]); },
        [&](int d, int h, int j) { return cache_value(expected, true, h, logical_key(j), d); },
        [&](int d, int h, int j) { return cache_value(expected, false, h, logical_key(j), d); },
        [&](int t, int j) { return logical_key(j) <= original_positions[t]; },
        [&](int d, int h, int t, double x) { reference[q_index(geometry, h, d, t)] = x; });

    DeviceCache cache(initial, MappingPattern::Fragmented);
    auto view = cache.batch_view();
    std::vector<std::int32_t> native(view.block_tables.ne[0]);
    cuda_check(cudaMemcpy(native.data(), view.block_tables.data, native.size() * 4,
                          cudaMemcpyDeviceToHost), "read physical table");
    std::vector<std::int32_t> mapping;
    for (auto page : selected) { mapping.push_back(native.at(page)); }
    GuardedDeviceBuffer table(mapping.size() * 4), positions(width * 4), rows(4);
    table.copy_from_host(mapping.data(), mapping.size() * 4);
    positions.copy_from_host(compact_positions.data(), width * 4);
    const std::int32_t row = 0;
    rows.copy_from_host(&row, 4);
    view.block_tables = Tensor(table.data(), DType::I32, {static_cast<int>(mapping.size()), 1});
    auto qb = to_bf16_bits(q), kb = to_bf16_bits(k), vb = to_bf16_bits(v);
    GuardedDeviceBuffer dq(qb.size() * 2), dk(kb.size() * 2), dv(vb.size() * 2), dout(qb.size() * 2);
    dq.copy_from_host(qb.data(), qb.size() * 2);
    dk.copy_from_host(kb.data(), kb.size() * 2);
    dv.copy_from_host(vb.data(), vb.size() * 2);
    Tensor tq(dq.data(), DType::BF16, {256, geometry.q_heads, width});
    Tensor tk(dk.data(), DType::BF16, {256, geometry.kv_heads, width});
    Tensor tv(dv.data(), DType::BF16, {256, geometry.kv_heads, width});
    Tensor tp(positions.data(), DType::I32, {width});
    Tensor tr(rows.data(), DType::I32, {1});
    Tensor output(dout.data(), DType::BF16, {256, geometry.q_heads, width});
    const ops::CausalAttentionExecutionEnvelope envelope{
        .min_visible_keys = static_cast<unsigned>(keys),
        .max_visible_keys = static_cast<unsigned>(keys), .small_prefill = true};
    const auto bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), cache_plan_storage(plan), envelope, 1, width, width);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(bytes, 256));
    WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
    launch_attention_case([&](cudaStream_t stream) {
        ops::causal_softmax_attention(tq, tk, tv, tp, Tensor{}, tr, op_geometry(geometry),
            0.0625F, view, envelope, workspace, output, stream);
    }, graph);
    const auto actual = bf16_bits_to_double(copy_from_guarded<std::uint16_t>(dout, qb.size()));
    const auto label = std::string("P1 sparse ") + cache_name(plan) +
        " prefix=" + std::to_string(prefix.size()) + " offset=" + std::to_string(offset) +
        " width=" + std::to_string(width) + " graph=" + std::to_string(graph);
    int failures = verify_attention(label, actual, reference, attention_criterion(plan));
    failures += verify_cache(label, cache.snapshot(), expected, plan.dtype == DType::BF16);
    failures += cache.verify_guards(label);
    failures += table.verify_guards("compact table");
    failures += scratch.verify_guards("workspace");
    failures += dout.verify_guards("output");
    std::cout << label << (failures ? " FAIL\n" : " PASS\n");
    return failures;
}

int main(int argc, char** argv) {
    try {
        int failures = 0;
        const auto plans = argc == 2 && std::string(argv[1]) == "rk8v4"
            ? std::vector{kPlanRk8v4} : std::vector{kPlanBf16, kPlanInt8};
        for (const auto plan : plans) {
            for (const std::vector<int> pages : {std::vector<int>{0,1,2,3,4,5},
                                               std::vector<int>{0,2,4},
                                               std::vector<int>{4,0,2}}) {
                for (int offset : {0, 63}) {
                    for (int width : {1, 63, 64, 65}) {
                        failures += sparse_case(plan, pages, offset, width, false);
                    }
                }
                failures += sparse_case(plan, pages, 63, 65, true);
            }
        }
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "P1_SPARSE FAIL " << e.what() << '\n';
        return 1;
    }
}
