// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/execution/dynamic_state_emitter.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

void DynamicStateEmitter::Emit(const DynamicStatePlan& plan, DynamicState& state,
                               vk::CommandBuffer command_buffer) const {
    state.SetViewports(plan.viewports);
    state.SetScissors(plan.scissors);
    state.SetDepthTestEnabled(plan.depth_test_enabled);
    if (plan.emit_depth_state) {
        state.SetDepthWriteEnabled(plan.depth_write_enabled);
        state.SetDepthCompareOp(plan.depth_compare_op);
    }
    state.SetDepthBoundsTestEnabled(plan.depth_bounds_test_enabled);
    if (plan.depth_bounds_test_enabled) {
        state.SetDepthBounds(plan.depth_bounds_min, plan.depth_bounds_max);
    }
    state.SetDepthBiasEnabled(plan.depth_bias_enabled);
    if (plan.depth_bias_enabled) {
        state.SetDepthBias(plan.depth_bias_constant, plan.depth_bias_clamp,
                           plan.depth_bias_slope);
    }
    state.SetStencilTestEnabled(plan.stencil_test_enabled);
    if (plan.stencil_test_enabled) {
        state.SetStencilOps(plan.stencil_front_ops, plan.stencil_back_ops);
        state.SetStencilReferences(plan.stencil_front_reference, plan.stencil_back_reference);
        state.SetStencilWriteMasks(plan.stencil_front_write_mask,
                                   plan.stencil_back_write_mask);
        state.SetStencilCompareMasks(plan.stencil_front_compare_mask,
                                     plan.stencil_back_compare_mask);
    }
    state.SetPrimitiveRestartEnabled(plan.primitive_restart_enabled);
    state.SetRasterizerDiscardEnabled(plan.rasterizer_discard_enabled);
    state.SetCullMode(plan.cull_mode);
    state.SetFrontFace(plan.front_face);
    state.SetLineWidth(plan.line_width);
    state.SetBlendConstants(plan.blend_constants);
    state.SetColorWriteMasks(plan.color_write_masks);
    state.SetAttachmentFeedbackLoopEnabled(plan.attachment_feedback_loop);
    state.Commit(instance, command_buffer);
}

} // namespace Vulkan
