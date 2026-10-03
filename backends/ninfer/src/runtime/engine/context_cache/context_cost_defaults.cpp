#include "runtime/engine/context_cache/context_cost.h"

#include <array>
#include <vector>

namespace ninfer::runtime {
namespace {

constexpr ContextPrefillCost kGroupwise27bPrefill{
    .chunk_ns              = 40'813'570,
    .token_ns_q32          = 1'012'273'154'411'951,
    .attention_pair_ns_q32 = 30'497'396'515,
    .vision_item_ns        = 5'986'585,
    .vision_patch_ns_q32   = 23'710'212'854'694,
};
constexpr ContextPrefillCost kNvfp4Fp8Prefill{
    .chunk_ns              = 14'672'989,
    .token_ns_q32          = 375'800'765'711'778,
    .attention_pair_ns_q32 = 8'200'474'657,
    .vision_item_ns        = 5'860'255,
    .vision_patch_ns_q32   = 23'832'529'381'413,
};

// RTX 3090 (sm_86) fits, measured 2026-09-14 on the fork host (Windows 11, CUDA 12.8, 315 W cap)
// with ninfer_context_cost_bench at --max-context 65536 for each groupwise-int artifact. The
// generic prefill model predicted 26.0 s for a 55,000-token Qwen3.8-27B prefill that took 57 s on
// this card; this fit predicts 58.1 s.
constexpr ContextPrefillCost kGroupwise27bPrefillSm86{
    .chunk_ns              = 42'827'781,
    .token_ns_q32          = 3'084'986'452'624'619,
    .attention_pair_ns_q32 = 46'293'818'703,
    .vision_item_ns        = 6'744'019,
    .vision_patch_ns_q32   = 109'358'098'255'328,
};
constexpr ContextPrefillCost kGroupwise35bA3bPrefillSm86{
    .chunk_ns              = 32'306'259,
    .token_ns_q32          = 671'808'709'554'317,
    .attention_pair_ns_q32 = 17'853'060'462,
    .vision_item_ns        = 6'937'447,
    .vision_patch_ns_q32   = 106'078'200'821'712,
};
constexpr const char* kGroupwise35bA3bSignatureText   = "ec5569f8250032ddd23568d599e4abf7e369d7a84971b2de3bd7fedfb9714dd1";
constexpr const char* kGroupwise35bA3bSignatureVision = "f4a7fddf7c517236401d5d7f085879d7735de10d49e19d125c496139f47f6533";

// RTX PRO 6000 Blackwell Workstation Edition (600 W) fits, measured 2026-09-26 with
// ninfer_context_cost_bench at --max-context 65536 on the default 120a build, Vision enabled.
constexpr ContextPrefillCost kGroupwise27bPrefillRtxPro6000{
    .chunk_ns              = 36'345'984,
    .token_ns_q32          = 643'113'761'669'684,
    .attention_pair_ns_q32 = 5'872'308'995,
    .vision_item_ns        = 5'202'185,
    .vision_patch_ns_q32   = 20'368'500'587'416,
};
constexpr ContextPrefillCost kGroupwise35bA3bPrefillRtxPro6000{
    .chunk_ns              = 16'889'725,
    .token_ns_q32          = 109'713'142'774'825,
    .attention_pair_ns_q32 = 4'386'704'449,
    .vision_item_ns        = 5'188'750,
    .vision_patch_ns_q32   = 20'218'782'611'123,
};
constexpr ContextPrefillCost kNvfp4Fp8PrefillRtxPro6000{
    .chunk_ns              = 12'680'294,
    .token_ns_q32          = 296'072'649'706'517,
    .attention_pair_ns_q32 = 9'550'099'381,
    .vision_item_ns        = 5'186'626,
    .vision_patch_ns_q32   = 20'366'821'546'842,
};

} // namespace

const std::array<ContextTransferCost, 3>& generic_context_transfer_cost() {
    // Conservative but useful on an unmeasured machine. These values preserve numerical ranking;
    // they are not a switch that disables the cost model.
    static constexpr std::array<ContextTransferCost, 3> value{
        ContextTransferCost{
            .batch_ns = 100'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 50'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 25'000, .operation_ns = 12'000, .ns_per_byte_q32 = 42'949'673},
    };
    return value;
}

const ContextPrefillCost& generic_context_prefill_cost() {
    // The slower measured 27B configuration is the conservative generic estimate. An unknown model
    // still receives continuous recomputation costs instead of falling back to a discrete tuple.
    return kGroupwise27bPrefill;
}

// Accepted project defaults live only in this table and are compiled into the binary. Runtime JSON
// presets are independent local-machine overrides and never become a build dependency.
const std::vector<ContextCostMachinePreset>& compiled_context_cost_defaults() {
    static const std::vector<ContextCostMachinePreset> defaults{
        ContextCostMachinePreset{
            .hardware_class = "nvidia-geforce-rtx-5090-sm120",
            // Unified from the measured 2026-08-24 transfer corpus. A fresh machine-only transfer
            // calibration can replace this whole array without loading a model.
            .transfer =
                std::array{
                    ContextTransferCost{
                        .batch_ns = 48'655, .operation_ns = 7'433, .ns_per_byte_q32 = 91'692'315},
                    ContextTransferCost{
                        .batch_ns = 0, .operation_ns = 8'457, .ns_per_byte_q32 = 83'354'284},
                    ContextTransferCost{
                        .batch_ns = 3'343, .operation_ns = 9'520, .ns_per_byte_q32 = 2'658'314},
                },
            // Existing measurements apply to these Text bindings with Vision disabled/enabled.
            // The signature excludes trained values and the selected speculative backend; it
            // describes the primary reconstruction work priced by ContextPrefillCost.
            .prefill =
                {
                    {"200f57efee7b0fe1172dfd4a06b1e6e0b2dbc36dfcd6a242fea35338bb5ff0d2",
                     kGroupwise27bPrefill},
                    {"badf2271162e72c8c51a91da02cbb4343d4b7574d037a014ca6862ff2862638b",
                     kGroupwise27bPrefill},
                    {"e6eae48276e11c15c932cb90d258b51b81e144dc13fb461e7ffa202d2caa440a",
                     kNvfp4Fp8Prefill},
                    {"953e22d9b9a6639c09f7389d49084b059dea07720bc0121dfd9c154002459915",
                     kNvfp4Fp8Prefill},
                },
        },
        ContextCostMachinePreset{
            .hardware_class = "nvidia-geforce-rtx-3090-sm86",
            // Measured 2026-09-14 on the fork host (Windows 11, CUDA 12.8, 315 W cap) with
            // ninfer_context_cost_bench, prefill at --max-context 65536 for each groupwise-int
            // artifact. The generic prefill model predicted 26.0 s for a 55,000-token Qwen3.8-27B
            // prefill that took 57 s on this card; this fit predicts 58.1 s.
            .transfer =
                std::array{
                    ContextTransferCost{
                        .batch_ns = 54'593, .operation_ns = 6'305, .ns_per_byte_q32 = 185'859'373},
                    ContextTransferCost{
                        .batch_ns = 48'436, .operation_ns = 6'878, .ns_per_byte_q32 = 172'624'952},
                    ContextTransferCost{
                        .batch_ns = 338, .operation_ns = 6'742, .ns_per_byte_q32 = 10'830'495},
                },
            // Keys are v3 prefill signatures (see models/qwen3_5/measurement.cpp); Qwen3.6-27B and
            // Qwen3.8-27B groupwise-int share one binding signature, and their two measured fits
            // differ by under 1%, so the Qwen3.8 fit serves both. Each pair is Vision
            // disabled/enabled.
            .prefill =
                {
                    {"200f57efee7b0fe1172dfd4a06b1e6e0b2dbc36dfcd6a242fea35338bb5ff0d2",
                     kGroupwise27bPrefillSm86},
                    {"badf2271162e72c8c51a91da02cbb4343d4b7574d037a014ca6862ff2862638b",
                     kGroupwise27bPrefillSm86},
                    // Qwen3.8-27B groupwise-int (with DFlash2 bundle) upgraded from the pinned
                    // v2 release, Vision disabled/enabled, as reported on this card.
                    {"cf336425f495069a4a56f254679b3ebfb82a8ba86d5b3cca4d36b38bf422ec16",
                     kGroupwise27bPrefillSm86},
                    {"e490f4a150c8573657ca0242b4ab9a0922e6986d5e295a03114fcdc4263b7fcc",
                     kGroupwise27bPrefillSm86},
                    {kGroupwise35bA3bSignatureText, kGroupwise35bA3bPrefillSm86},
                    {kGroupwise35bA3bSignatureVision, kGroupwise35bA3bPrefillSm86},
                },
        },
        ContextCostMachinePreset{
            .hardware_class = "nvidia-rtx-pro-6000-blackwell-workstation-edition-sm120",
            .transfer =
                std::array{
                    ContextTransferCost{
                        .batch_ns = 3'277, .operation_ns = 1'441, .ns_per_byte_q32 = 82'421'673},
                    ContextTransferCost{
                        .batch_ns = 6'962, .operation_ns = 1'526, .ns_per_byte_q32 = 85'338'968},
                    ContextTransferCost{
                        .batch_ns = 2'535, .operation_ns = 2'037, .ns_per_byte_q32 = 1'557'550},
                },
            // The Qwen3.8 fit serves the groupwise-int 27B signatures as on the RTX 3090, and the
            // NVFP4/FP8 pair is the one the RTX 5090 entry names. Each pair is Vision
            // disabled/enabled.
            .prefill =
                {
                    {"200f57efee7b0fe1172dfd4a06b1e6e0b2dbc36dfcd6a242fea35338bb5ff0d2",
                     kGroupwise27bPrefillRtxPro6000},
                    {"badf2271162e72c8c51a91da02cbb4343d4b7574d037a014ca6862ff2862638b",
                     kGroupwise27bPrefillRtxPro6000},
                    {"cf336425f495069a4a56f254679b3ebfb82a8ba86d5b3cca4d36b38bf422ec16",
                     kGroupwise27bPrefillRtxPro6000},
                    {"e490f4a150c8573657ca0242b4ab9a0922e6986d5e295a03114fcdc4263b7fcc",
                     kGroupwise27bPrefillRtxPro6000},
                    {kGroupwise35bA3bSignatureText, kGroupwise35bA3bPrefillRtxPro6000},
                    {kGroupwise35bA3bSignatureVision, kGroupwise35bA3bPrefillRtxPro6000},
                    {"e6eae48276e11c15c932cb90d258b51b81e144dc13fb461e7ffa202d2caa440a",
                     kNvfp4Fp8PrefillRtxPro6000},
                    {"953e22d9b9a6639c09f7389d49084b059dea07720bc0121dfd9c154002459915",
                     kNvfp4Fp8PrefillRtxPro6000},
                },
        },
    };
    return defaults;
}

} // namespace ninfer::runtime
