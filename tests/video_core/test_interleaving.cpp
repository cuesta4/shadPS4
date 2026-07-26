// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "shader_recompiler/invocation.h"
#include "shader_recompiler/shader_program_analysis.h"
#include "video_core/amdgpu/gpu_state_slices.h"
#include "video_core/gpu_commands/commands.h"
#include "video_core/renderer_vulkan/execution/resource_hazard_planner.h"
#include "video_core/renderer_vulkan/preparation/graphics_request_builder.h"

namespace {

TEST(GpuInterleaving, ConsecutiveCommandsWithSameShaderDoNotAlias) {
    Shader::Info shared_analysis{};
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data_a{};
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data_b{};
    std::array<u32, 4> code_a{};
    std::array<u32, 4> code_b{};
    user_data_a[0] = 0xAAAA'AAAA;
    user_data_b[0] = 0xBBBB'BBBB;
    const Shader::ShaderParams params_a{
        .user_data = user_data_a,
        .code = code_a,
        .hash = 1,
    };
    const Shader::ShaderParams params_b{
        .user_data = user_data_b,
        .code = code_b,
        .hash = 1,
    };

    const auto invocation_a = Shader::ShaderInvocationData::FromParams(shared_analysis, params_a);
    const auto invocation_b = Shader::ShaderInvocationData::FromParams(shared_analysis, params_b);

    EXPECT_NE(invocation_a.user_data[0], invocation_b.user_data[0]);
    EXPECT_NE(invocation_a.pgm_base, invocation_b.pgm_base);
    EXPECT_EQ(invocation_a.user_data[0], 0xAAAA'AAAAU);
    EXPECT_EQ(invocation_b.user_data[0], 0xBBBB'BBBBU);
}

TEST(GpuInterleaving, HazardPlansAreDeterministicAndIndependent) {
    VideoCore::BufferAccessStateStore store{};
    std::vector<VideoCore::ResolvedBufferRef> buffers_a;
    std::vector<VideoCore::ResolvedBufferRef> buffers_b;

    buffers_a.push_back({
        .id = VideoCore::BufferId{1},
        .generation = 1,
        .offset = 0,
        .size = 1024,
    });
    buffers_b.push_back({
        .id = VideoCore::BufferId{1},
        .generation = 1,
        .offset = 0,
        .size = 1024,
    });

    const auto plan_a = Vulkan::ResourceHazardPlanner::PlanBufferHazards(
        store, buffers_a, vk::PipelineStageFlagBits2::eVertexShader,
        vk::AccessFlagBits2::eShaderRead);

    EXPECT_TRUE(Vulkan::ResourceHazardPlanner::CommitBufferHazards(store, plan_a));

    const auto plan_b = Vulkan::ResourceHazardPlanner::PlanBufferHazards(
        store, buffers_b, vk::PipelineStageFlagBits2::eVertexShader,
        vk::AccessFlagBits2::eShaderRead);

    EXPECT_FALSE(plan_a.Empty());
    EXPECT_TRUE(plan_b.Empty());
}

} // namespace
