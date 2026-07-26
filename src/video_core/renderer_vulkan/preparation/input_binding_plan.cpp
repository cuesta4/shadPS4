// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/preparation/input_binding_plan.h"

#include "shader_recompiler/invocation.h"
#include "video_core/gpu_commands/captured_state.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

VertexInputRequest InputRequestBuilder::BuildVertexRequest(
    const GraphicsPipeline& pipeline, const Shader::ShaderInvocationData& invocation,
    const VideoCore::GraphicsDrawState& draw_state, const Instance& instance) {
    VertexInputRequest request{};
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    pipeline.GetVertexInputs(request.attributes, request.bindings, divisors,
                             request.guest_buffers,
                             draw_state.vgt_instance_step_rate_0,
                             draw_state.vgt_instance_step_rate_1, invocation);
    request.use_dynamic_vertex_input = instance.IsVertexInputDynamicState();
    return request;
}

IndexInputRequest InputRequestBuilder::BuildIndexRequest(
    const VideoCore::GraphicsDrawState& draw_state, u32 index_offset) {
    const bool is_index16 =
        draw_state.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    return {
        .address =
            draw_state.index_base_address.Address<VAddr>() + index_offset * index_size,
        .size = draw_state.num_indices * index_size,
        .type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32,
    };
}

} // namespace Vulkan
