#include "core/device.h"
#include "ninfer/ops/paged_kv_window.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct Expected {
    std::vector<std::int32_t> tables;
    std::vector<std::int32_t> rows;
    std::vector<std::int32_t> positions;
};

// The documented mapping, row by row.
Expected reference(const std::vector<std::int32_t>& tables, std::int32_t pages,
                   const std::vector<std::int32_t>& table_rows,
                   const std::vector<std::int32_t>& positions, std::int32_t width,
                   const std::vector<std::int32_t>* valid, std::int32_t sink, std::int32_t window) {
    const auto batch = static_cast<std::int32_t>(table_rows.size());
    Expected out;
    out.tables.resize(static_cast<std::size_t>(pages) * batch);
    out.positions.resize(positions.size());
    for (std::int32_t b = 0; b < batch; ++b) {
        const std::int32_t live  = valid != nullptr ? (*valid)[static_cast<std::size_t>(b)] : width;
        const std::int32_t p0    = live > 0 ? positions[static_cast<std::size_t>(b) * width] : 0;
        const std::int32_t first = std::max(0, p0 - window) / 64;
        const std::int32_t shift = std::max(0, first - sink);
        const std::int32_t row   = table_rows[static_cast<std::size_t>(b)];
        for (std::int32_t j = 0; j < pages; ++j) {
            const std::int32_t source = j < sink ? j : j + shift;
            out.tables[static_cast<std::size_t>(b) * pages + j] =
                tables[static_cast<std::size_t>(row) * pages + (source < pages ? source : j)];
        }
        for (std::int32_t c = 0; c < width; ++c) {
            const std::int32_t value = positions[static_cast<std::size_t>(b) * width + c];
            out.positions[static_cast<std::size_t>(b) * width + c] =
                value >= shift * 64 ? value - shift * 64 : value;
        }
        out.rows.push_back(b);
    }
    return out;
}

int window_case(const char* name, std::int32_t pages, std::int32_t table_count,
                const std::vector<std::int32_t>& table_rows,
                const std::vector<std::int32_t>& positions, std::int32_t width,
                const std::vector<std::int32_t>* valid, std::int32_t sink, std::int32_t window) {
    const auto batch = static_cast<std::int32_t>(table_rows.size());
    std::vector<std::int32_t> tables(static_cast<std::size_t>(pages) * table_count);
    for (std::size_t i = 0; i < tables.size(); ++i) {
        tables[i] = static_cast<std::int32_t>(1000 + (i * 7919) % 50021);
    }
    const Expected expected =
        reference(tables, pages, table_rows, positions, width, valid, sink, window);

    GuardedDeviceBuffer d_tables(tables.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_rows(table_rows.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_positions(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_valid(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    GuardedDeviceBuffer out_tables(expected.tables.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer out_rows(expected.rows.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer out_positions(expected.positions.size() * sizeof(std::int32_t));
    d_tables.copy_from_host(tables.data(), tables.size() * sizeof(std::int32_t));
    d_rows.copy_from_host(table_rows.data(), table_rows.size() * sizeof(std::int32_t));
    d_positions.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    if (valid != nullptr) {
        d_valid.copy_from_host(valid->data(), valid->size() * sizeof(std::int32_t));
    }
    out_tables.fill(0xcd);
    out_rows.fill(0xcd);
    out_positions.fill(0xcd);

    Tensor tables_t(d_tables.data(), DType::I32, {pages, table_count});
    Tensor rows_t(d_rows.data(), DType::I32, {batch});
    Tensor positions_t(d_positions.data(), DType::I32, {width, batch});
    Tensor valid_t = valid != nullptr ? Tensor(d_valid.data(), DType::I32, {batch}) : Tensor{};
    Tensor out_tables_t(out_tables.data(), DType::I32, {pages, batch});
    Tensor out_rows_t(out_rows.data(), DType::I32, {batch});
    Tensor out_positions_t(out_positions.data(), DType::I32, {width, batch});
    ops::paged_kv_window_rows(tables_t, rows_t, positions_t, valid_t, sink, window, out_tables_t,
                              out_rows_t, out_positions_t, nullptr);
    cuda_synchronize();

    const std::string label = std::string("paged_kv_window_rows ") + name;
    int failures            = 0;
    failures += verify_exact((label + " tables").c_str(),
                             from_device<std::int32_t>(out_tables.data(), expected.tables.size()),
                             expected.tables);
    failures += verify_exact((label + " rows").c_str(),
                             from_device<std::int32_t>(out_rows.data(), expected.rows.size()),
                             expected.rows);
    failures +=
        verify_exact((label + " positions").c_str(),
                     from_device<std::int32_t>(out_positions.data(), expected.positions.size()),
                     expected.positions);
    failures += verify_exact((label + " preserves tables").c_str(),
                             from_device<std::int32_t>(d_tables.data(), tables.size()), tables);
    for (GuardedDeviceBuffer* buffer : {&out_tables, &out_rows, &out_positions}) {
        failures += buffer->verify_guards(label.c_str());
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    // One row deep into a long history: the sink page, then the pages from the window on.
    failures += window_case("deep single row", 4096, 3, {2}, {200000}, 1, nullptr, 1, 4096);
    // A drafting row of width 6 whose window starts inside its sink: identity.
    failures +=
        window_case("shallow row", 64, 2, {1}, {100, 101, 102, 103, 104, 105}, 6, nullptr, 1, 4096);
    // A masked batch: live rows at different depths, an empty row (zero positions), and a row
    // whose inert tail repeats its last live position.
    const std::vector<std::int32_t> valid{3, 0, 1, 2};
    failures += window_case("masked batch", 512, 6, {5, 0, 3, 1},
                            {20000, 20001, 20002, 0, 0, 0, 30000, 30000, 30000, 70, 71, 71}, 3,
                            &valid, 1, 256);
    // A window of exactly the draft width at a page boundary, and a sink of two pages.
    failures +=
        window_case("tight window", 1024, 1, {0}, {64 * 700, 64 * 700 + 1}, 2, nullptr, 2, 2);
    if (failures == 0) { std::cout << "paged_kv_window_rows passed\n"; }
    return failures == 0 ? 0 : 1;
}
