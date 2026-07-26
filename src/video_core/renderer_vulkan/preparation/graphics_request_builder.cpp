// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "common/assert.h"
#include "common/debug.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/preparation/graphics_request_builder.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

DynamicStatePlan GraphicsRequestBuilder::BuildDynamicStatePlan(
    const VideoCore::CapturedGraphicsState& state, const GraphicsPipeline& pipeline,
    const Instance& instance, const bool is_indexed,
    const bool attachment_feedback_loop) {
    const auto& regs = state.dynamic;
    const auto& pipeline_state = state.pipeline;
    const auto& attachments = state.attachments;
    DynamicStatePlan plan{};

    const auto combined_scissor_value_tl = [](s16 screen, s16 window, s16 generic,
                                               s16 window_offset) {
        return std::max({screen, s16(window + window_offset), s16(generic + window_offset)});
    };
    const auto combined_scissor_value_br = [](s16 screen, s16 window, s16 generic,
                                               s16 window_offset) {
        return std::min({screen, s16(window + window_offset), s16(generic + window_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;
    AmdGpu::Scissor combined_scissor{};
    combined_scissor.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    combined_scissor.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    combined_scissor.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    combined_scissor.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    if (pipeline_state.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& viewport_control = regs.viewport_control;
    for (u32 index = 0; index < AmdGpu::NUM_VIEWPORTS; ++index) {
        const auto& viewport_regs = regs.viewports[index];
        if (viewport_regs.xscale == 0) {
            continue;
        }

        const float zoffset =
            viewport_control.zoffset_enable ? viewport_regs.zoffset : 0.0f;
        const float zscale = viewport_control.zscale_enable ? viewport_regs.zscale : 1.0f;
        vk::Viewport viewport{};
        if (pipeline_state.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }
        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            viewport.minDepth = std::max(viewport.minDepth, 0.0f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.0f);
        }

        if (pipeline_state.IsClipDisabled()) {
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const float xoffset =
                viewport_control.xoffset_enable ? viewport_regs.xoffset : 0.0f;
            const float xscale =
                viewport_control.xscale_enable ? viewport_regs.xscale : 1.0f;
            const float yoffset =
                viewport_control.yoffset_enable ? viewport_regs.yoffset : 0.0f;
            const float yscale =
                viewport_control.yscale_enable ? viewport_regs.yscale : 1.0f;
            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
        }
        plan.viewports.push_back(viewport);

        auto viewport_scissor = combined_scissor;
        if (regs.mode_control.vport_scissor_enable) {
            viewport_scissor.top_left_x =
                std::max(viewport_scissor.top_left_x,
                         s16(regs.viewport_scissors[index].top_left_x));
            viewport_scissor.top_left_y =
                std::max(viewport_scissor.top_left_y,
                         s16(regs.viewport_scissors[index].top_left_y));
            viewport_scissor.bottom_right_x =
                std::min(AmdGpu::Scissor::Clamp(viewport_scissor.bottom_right_x),
                         regs.viewport_scissors[index].bottom_right_x);
            viewport_scissor.bottom_right_y =
                std::min(AmdGpu::Scissor::Clamp(viewport_scissor.bottom_right_y),
                         regs.viewport_scissors[index].bottom_right_y);
        }
        plan.scissors.push_back({
            .offset = {viewport_scissor.top_left_x, viewport_scissor.top_left_y},
            .extent = {viewport_scissor.GetWidth(), viewport_scissor.GetHeight()},
        });
    }

    if (plan.viewports.empty()) {
        plan.viewports.push_back({
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        plan.scissors.push_back({
            .offset = {0, 0},
            .extent = {1, 1},
        });
    }

    plan.depth_test_enabled =
        attachments.depth_control.depth_enable && attachments.depth_buffer.DepthValid();
    plan.emit_depth_state = plan.depth_test_enabled;
    plan.depth_write_enabled =
        plan.depth_test_enabled && attachments.depth_control.depth_write_enable &&
        !attachments.depth_render_control.depth_clear_enable;
    plan.depth_compare_op =
        LiverpoolToVK::CompareOp(attachments.depth_control.depth_func);
    plan.depth_bounds_test_enabled = attachments.depth_control.depth_bounds_enable;
    plan.depth_bounds_min = regs.depth_bounds_min;
    plan.depth_bounds_max = regs.depth_bounds_max;
    plan.depth_bias_enabled = pipeline_state.polygon_control.NeedsBias();
    const bool front_bias = pipeline_state.polygon_control.enable_polygon_offset_front;
    plan.depth_bias_constant =
        front_bias ? regs.poly_offset.front_offset : regs.poly_offset.back_offset;
    plan.depth_bias_clamp = regs.poly_offset.depth_bias;
    plan.depth_bias_slope =
        (front_bias ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.0f;

    plan.stencil_test_enabled =
        attachments.depth_control.stencil_enable && attachments.depth_buffer.StencilValid();
    if (plan.stencil_test_enabled) {
        plan.stencil_front_ops = {
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op =
                LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op =
                LiverpoolToVK::CompareOp(attachments.depth_control.stencil_ref_func),
        };
        plan.stencil_back_ops = attachments.depth_control.backface_enable
                                    ? StencilOps{
                                          .fail_op = LiverpoolToVK::StencilOp(
                                              regs.stencil_control.stencil_fail_back),
                                          .pass_op = LiverpoolToVK::StencilOp(
                                              regs.stencil_control.stencil_zpass_back),
                                          .depth_fail_op = LiverpoolToVK::StencilOp(
                                              regs.stencil_control.stencil_zfail_back),
                                          .compare_op = LiverpoolToVK::CompareOp(
                                              attachments.depth_control.stencil_bf_func),
                                      }
                                    : plan.stencil_front_ops;
        const auto front = regs.stencil_ref_front;
        const auto back = attachments.depth_control.backface_enable
                              ? regs.stencil_ref_back
                              : regs.stencil_ref_front;
        plan.stencil_front_reference = front.stencil_test_val;
        plan.stencil_back_reference = back.stencil_test_val;
        const bool stencil_clear =
            attachments.depth_render_control.stencil_clear_enable;
        plan.stencil_front_write_mask = stencil_clear ? 0U : front.stencil_write_mask;
        plan.stencil_back_write_mask = stencil_clear ? 0U : back.stencil_write_mask;
        plan.stencil_front_compare_mask = front.stencil_mask;
        plan.stencil_back_compare_mask = back.stencil_mask;
    }

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList ||
               type == AmdGpu::PrimitiveType::RectList;
    };
    plan.primitive_restart_enabled =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() ||
         !is_list_topology(pipeline_state.primitive_type)) &&
        (instance.IsPatchListRestartSupported() ||
         !is_patch_list_topology(pipeline_state.primitive_type));
    ASSERT_MSG(!is_indexed || !plan.primitive_restart_enabled ||
                   regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");
    plan.rasterizer_discard_enabled =
        pipeline_state.clipper_control.dx_rasterization_kill;
    plan.cull_mode = LiverpoolToVK::IsPrimitiveCulled(pipeline_state.primitive_type)
                         ? LiverpoolToVK::CullMode(
                               pipeline_state.polygon_control.CullingMode())
                         : vk::CullModeFlagBits::eNone;
    plan.front_face =
        LiverpoolToVK::FrontFace(pipeline_state.polygon_control.front_face);
    plan.line_width = regs.line_control.Width();
    plan.blend_constants = regs.blend_constants;
    plan.color_write_masks = pipeline.GetGraphicsKey().write_masks;
    plan.attachment_feedback_loop = attachment_feedback_loop;
    return plan;
}

} // namespace Vulkan
