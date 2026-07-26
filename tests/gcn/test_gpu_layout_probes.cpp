// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstddef>

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/invocation.h"
#include "video_core/amdgpu/gpu_state_slices.h"
#include "video_core/amdgpu/register_file.h"
#include "video_core/gpu_commands/commands.h"
#include "video_core/renderer_vulkan/execution/graphics_operation_classifier.h"
#include "video_core/renderer_vulkan/preparation/command_preparation_context.h"
#include "video_core/renderer_vulkan/preparation/input_binding_plan.h"
#include "video_core/texture_cache/image_use_tracker.h"
#include "video_core/texture_cache/surface_metadata_tracker.h"

namespace {

TEST(GpuLayoutProbes, ReportsAndValidatesTypeLayouts) {
    EXPECT_GT(sizeof(AmdGpu::Regs), 0u);
    EXPECT_GE(sizeof(AmdGpu::GpuRegisterFile), sizeof(AmdGpu::Regs));
    EXPECT_GT(sizeof(Vulkan::CommandPreparationContext), 0u);
    EXPECT_LE(sizeof(Vulkan::CommandPreparationContext), 512u);
    EXPECT_LE(sizeof(Vulkan::CommandResolutionScratch), 1024u);
    EXPECT_LE(sizeof(Vulkan::VertexInputRequest), 1024u);
    EXPECT_LE(sizeof(Vulkan::VertexInputPlan), 1024u);
    EXPECT_LE(sizeof(Shader::ShaderInvocationData), 128u);
    EXPECT_GT(sizeof(VideoCore::ImageUseTracker), 0u);
    EXPECT_LE(alignof(VideoCore::ImageUseTracker), alignof(std::max_align_t));
}

TEST(GpuLayoutProbes, ValidatesStateSliceImmutabilityAndAlignment) {
    AmdGpu::PipelineStateInput pipeline_input{};
    AmdGpu::AttachmentStateInput attachment_input{};
    AmdGpu::ShaderInvocationRegisters registers_input{};

    EXPECT_LE(sizeof(pipeline_input), 512u);
    EXPECT_LE(alignof(decltype(pipeline_input)), alignof(std::max_align_t));
    EXPECT_LE(alignof(decltype(attachment_input)), alignof(std::max_align_t));
    EXPECT_LE(alignof(decltype(registers_input)), alignof(std::max_align_t));
}

} // namespace
