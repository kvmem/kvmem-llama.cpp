#include "kvmem/kv_content.hpp"
#include <stdexcept>
#include <cstdio>
using namespace kvmem;
static void check(bool yes) { if (!yes) throw std::runtime_error("content assertion failed"); }
template<class F> static void rejects(F f) { try { f(); } catch (const MemoryError &) { return; } throw std::runtime_error("accepted invalid operation"); }
int main() {
    KvContent a(32), b(32);
    check(!(a.stamp().session == b.stamp().session));
    a.reserve(77);
    check(a.stamp().frontier == 0 && a.reserved() == 77);
    rejects([&] { a.block(0); });
    rejects([&] { a.complete(32, 64); });
    a.complete(0, 13);
    const auto prefix13 = a.prefix(13);
    const auto prefix8 = a.prefix(8);
    check(a.contains(prefix13) && !b.contains(prefix13));
    rejects([&] { a.prefix(14); });
    const auto first = a.block(0);
    check(first.valid_tokens == 13);
    MemoryDescriptor descriptor;
    descriptor.identity = {"test", "1", "model", "attention"};
    descriptor.layout = {"rows", 1, 32, 64, 1,
        {{0, PayloadPlaneKind::Key, 0, 32, 1, 1}, {0, PayloadPlaneKind::Value, 32, 32, 1, 1}}};
    descriptor.capabilities.page_tokens = 32;
    descriptor.capabilities.max_position = 1024;
    descriptor.capabilities.native_payload = descriptor.capabilities.sparse_view =
        descriptor.capabilities.original_rope_positions = true;
    MemorySnapshot snapshot{a.stamp(), {{first, 0}}, {0}};
    const MemoryBudget budget{32, 64, 96, 64};
    const auto retained = plan_working_set(descriptor, snapshot, {0}, {}, budget);
    check(retained.next.visible_tokens == 13 && retained.device_tokens_required == 96);
    rejects([&] { plan_working_set(descriptor, snapshot, {1}, {}, budget); });
    const auto evicted = plan_working_set(descriptor, snapshot, {}, {}, budget);
    check(evicted.spill.size() == 1 && evicted.spill[0].valid_tokens == 13);
    a.complete(13, 64);
    check(a.block(0).content_version > first.content_version);
    rejects([&] { a.block(2); });
    a.complete(64, 77);
    const auto full = a.stamp();
    a.truncate(65); // MTP rejection keeps one evaluated row of the final block
    check(a.stamp().frontier == 65 && a.block(2).valid_tokens == 1);
    auto next = full; ++next.view_version;
    rejects([&] { a.publish(full, next); });
    const auto before_replay = a.block(0);
    a.change_execution(); a.complete(4, 12);
    check(a.contains(prefix13)); // execution replay does not change logical history
    check(a.stamp().frontier == 65 && a.block(0).content_version != before_replay.content_version);
    auto expected = a.stamp(); next = expected; ++next.view_version;
    a.publish(expected, next); check(a.stamp() == next);
    const auto old = a.stamp().session;
    a.invalidate(); rejects([&] { a.reserve(66); });
    check(!a.contains(prefix13));
    a.reset(); check(a.stamp().valid && !(a.stamp().session == old) && a.stamp().frontier == 0);
    a.reserve(20); a.complete(0, 8); a.truncate(8);
    check(!a.contains(prefix8)); // reset/regrowth must not resurrect an old lease
    const auto kept8 = a.prefix(8);
    check(a.reserved() == 8 && a.stamp().frontier == 8);
    a.reserve(13); a.complete(8, 13, false);
    check(a.stamp().frontier == 8 && a.evaluated() == 13 && a.block(0).valid_tokens == 8);
    rejects([&] { a.accept(14); });
    a.accept(10); a.truncate(10);
    const auto revoked10 = a.prefix(10);
    a.truncate(8);
    check(a.contains(kept8) && !a.contains(revoked10));
    a.reserve(10); a.complete(8, 10);
    check(a.contains(kept8) && !a.contains(revoked10));
    check(a.stamp().frontier == 10 && a.evaluated() == 10 && a.reserved() == 10);
    std::puts("PASS reservation, completion, partial block, replay, MTP tail, session identity and publication");
}
