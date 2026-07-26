// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstddef>
#include <cstring>

#include <gtest/gtest.h>

#include "shader_recompiler/invocation.h"
#include "video_core/gpu_commands/commands.h"
#include "video_core/renderer_vulkan/preparation/command_preparation_context.h"
#include "video_core/renderer_vulkan/preparation/input_binding_plan.h"

TEST(GpuRefactorLayout, PreparationContextRemainsCompact) {
    EXPECT_LE(sizeof(Vulkan::CommandPreparationContext), 512u);
    EXPECT_LE(sizeof(Vulkan::CommandResolutionScratch), 1024u);
    EXPECT_LE(sizeof(Vulkan::VertexInputRequest), 1024u);
    EXPECT_LE(sizeof(Vulkan::VertexInputPlan), 1024u);
    EXPECT_LE(alignof(Vulkan::CommandPreparationContext), alignof(std::max_align_t));
}

TEST(GpuRefactorLayout, RuntimeInvocationRemainsCompact) {
    EXPECT_LE(sizeof(Shader::ShaderInvocationData), 128u);
}

TEST(GpuRefactorLayout, DescriptorPlansRemainValidAcrossStorageGrowth) {
    Vulkan::CommandResolutionScratch scratch{};
    Vulkan::CommandPreparationContext context{};
    context.Reset(scratch);
    for (u32 index = 0; index < 40; ++index) {
        context.buffer_infos.emplace_back(vk::Buffer{}, index * 16, 16);
        context.descriptor_writes.push_back({
            .binding = index,
            .descriptor_count = 1,
            .info_index = index,
            .type = vk::DescriptorType::eUniformBuffer,
            .info_kind = Vulkan::DescriptorInfoKind::Buffer,
        });
    }
    context.Seal();

    ASSERT_EQ(context.descriptor_writes.size(), context.buffer_infos.size());
    EXPECT_EQ(context.descriptor_writes.front().info_index, 0u);
    EXPECT_EQ(context.descriptor_writes.back().info_index, 39u);
    EXPECT_EQ(context.buffer_infos.back().offset, 39u * 16u);
    EXPECT_EQ(context.resolution, nullptr);
}

TEST(GpuRefactorLayout, CapturedCommandsOwnTheirState) {
    VideoCore::DrawCommand first{};
    VideoCore::DrawCommand second{};
    first.state.draw.num_indices = 11;
    second.state.draw.num_indices = 23;

    EXPECT_EQ(first.state.draw.num_indices, 11U);
    EXPECT_EQ(second.state.draw.num_indices, 23U);
    EXPECT_NE(&first.state, &second.state);
}

TEST(GpuRefactorLayout, ShaderInvocationsDoNotAlias) {
    Shader::ShaderInvocationData first{};
    Shader::ShaderInvocationData second{};
    first.user_data[0] = 0x11111111;
    second.user_data[0] = 0x22222222;
    first.flattened_ud_buf.resize(4);
    second.flattened_ud_buf.resize(4);
    std::memset(first.flattened_ud_buf.data(), 0x11,
                first.flattened_ud_buf.size() * sizeof(u32));
    std::memset(second.flattened_ud_buf.data(), 0x22,
                second.flattened_ud_buf.size() * sizeof(u32));

    EXPECT_EQ(first.user_data[0], 0x11111111U);
    EXPECT_EQ(second.user_data[0], 0x22222222U);
    EXPECT_NE(first.flattened_ud_buf.data(), second.flattened_ud_buf.data());
}
