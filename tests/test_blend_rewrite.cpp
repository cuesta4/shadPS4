// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_blend_rewrite.h"

namespace {

using Factor = AmdGpu::BlendControl::BlendFactor;
using Function = AmdGpu::BlendControl::BlendFunc;

TEST(BlendRewriteTest, KeepsNativeMinMaxWithUnitFactors) {
    Vulkan::BlendEquation equation{Factor::One, Function::Min, Factor::One};

    EXPECT_EQ(Vulkan::RewriteScaledMinMaxBlend(equation, AmdGpu::NumberFormat::Unorm),
              Vulkan::BlendRewriteResult::Native);
    EXPECT_EQ(equation.function, Function::Min);
}

TEST(BlendRewriteTest, ReducesUnsignedMinWithZeroContributionToZero) {
    Vulkan::BlendEquation equation{Factor::One, Function::Min, Factor::Zero};

    EXPECT_EQ(Vulkan::RewriteScaledMinMaxBlend(equation, AmdGpu::NumberFormat::Srgb),
              Vulkan::BlendRewriteResult::Exact);
    EXPECT_EQ(equation.src_factor, Factor::Zero);
    EXPECT_EQ(equation.dst_factor, Factor::Zero);
    EXPECT_EQ(equation.function, Function::Add);
}

TEST(BlendRewriteTest, ReducesUnsignedMaxWithZeroContributionToOtherTerm) {
    Vulkan::BlendEquation equation{Factor::Zero, Function::Max, Factor::DstColor};

    EXPECT_EQ(Vulkan::RewriteScaledMinMaxBlend(equation, AmdGpu::NumberFormat::Unorm),
              Vulkan::BlendRewriteResult::Exact);
    EXPECT_EQ(equation.src_factor, Factor::Zero);
    EXPECT_EQ(equation.dst_factor, Factor::DstColor);
    EXPECT_EQ(equation.function, Function::Add);
}

TEST(BlendRewriteTest, RejectsEquationsThatNeedDestinationShaderAccess) {
    Vulkan::BlendEquation equation{Factor::SrcColor, Function::Min, Factor::DstColor};

    EXPECT_EQ(Vulkan::RewriteScaledMinMaxBlend(equation, AmdGpu::NumberFormat::Unorm),
              Vulkan::BlendRewriteResult::Unsupported);
    EXPECT_EQ(equation.function, Function::Min);
}

TEST(BlendRewriteTest, DoesNotUseUnsignedReductionForSignedTargets) {
    Vulkan::BlendEquation equation{Factor::One, Function::Min, Factor::Zero};

    EXPECT_EQ(Vulkan::RewriteScaledMinMaxBlend(equation, AmdGpu::NumberFormat::Snorm),
              Vulkan::BlendRewriteResult::Unsupported);
}

} // namespace
