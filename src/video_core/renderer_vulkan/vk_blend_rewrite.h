// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/amdgpu/pixel_format.h"
#include "video_core/amdgpu/regs_color.h"

namespace Vulkan {

struct BlendEquation {
    AmdGpu::BlendControl::BlendFactor src_factor;
    AmdGpu::BlendControl::BlendFunc function;
    AmdGpu::BlendControl::BlendFactor dst_factor;
};

enum class BlendRewriteResult {
    Native,
    Exact,
    Unsupported,
};

[[nodiscard]] constexpr bool IsUnsignedNormalizedBlendTarget(const AmdGpu::NumberFormat format) {
    using Format = AmdGpu::NumberFormat;
    return format == Format::Unorm || format == Format::Srgb;
}

/**
 * Vulkan ignores blend factors for MIN/MAX, while GNM applies them before the operation. If
 * either contribution is identically zero, the GNM equation can be represented exactly with ADD
 * on an unsigned normalized target: MIN(x, 0) clamps to zero and MAX(x, 0) clamps exactly as x.
 */
[[nodiscard]] constexpr BlendRewriteResult RewriteScaledMinMaxBlend(
    BlendEquation& equation, const AmdGpu::NumberFormat target_format) {
    using Factor = AmdGpu::BlendControl::BlendFactor;
    using Function = AmdGpu::BlendControl::BlendFunc;

    if (equation.function != Function::Min && equation.function != Function::Max) {
        return BlendRewriteResult::Native;
    }
    if (equation.src_factor == Factor::One && equation.dst_factor == Factor::One) {
        return BlendRewriteResult::Native;
    }
    if (!IsUnsignedNormalizedBlendTarget(target_format) ||
        (equation.src_factor != Factor::Zero && equation.dst_factor != Factor::Zero)) {
        return BlendRewriteResult::Unsupported;
    }

    if (equation.function == Function::Min) {
        equation.src_factor = Factor::Zero;
        equation.dst_factor = Factor::Zero;
    }
    equation.function = Function::Add;
    return BlendRewriteResult::Exact;
}

} // namespace Vulkan
