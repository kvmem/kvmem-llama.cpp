#pragma once

// Log probabilities of one next-token distribution, computed on the host from BF16 logits.

#include "ninfer/types.h"

#include <cstdint>
#include <span>

namespace ninfer {

// The log-softmax of `logits` (BF16 bit patterns over the public token domain) at `selected`,
// and the `top` most likely tokens, most likely first with the lower id breaking ties. A NaN logit
// counts as impossible.
[[nodiscard]] FirstTokenLogprobs token_logprobs_from_bf16(std::span<const std::uint16_t> logits,
                                                          TokenId selected, std::uint32_t top);

} // namespace ninfer
