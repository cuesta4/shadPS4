// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <boost/container/small_vector.hpp>

#include "video_core/gpu_commands/captured_state.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

class GraphicsPipeline;
class Instance;

/**
 * Fully materialized Vulkan dynamic-state values for one graphics command.
 *
 * This is intentionally data-only. Building it reads only captured command state, immutable
 * pipeline metadata, and immutable device capabilities.
 */
struct DynamicStatePlan {
    boost::container::small_vector<vk::Viewport, 1> viewports;
    boost::container::small_vector<vk::Rect2D, 1> scissors;

    bool depth_test_enabled{};
    bool emit_depth_state{};
    bool depth_write_enabled{};
    vk::CompareOp depth_compare_op{};
    bool depth_bounds_test_enabled{};
    float depth_bounds_min{};
    float depth_bounds_max{};
    bool depth_bias_enabled{};
    float depth_bias_constant{};
    float depth_bias_clamp{};
    float depth_bias_slope{};

    bool stencil_test_enabled{};
    StencilOps stencil_front_ops{};
    StencilOps stencil_back_ops{};
    u32 stencil_front_reference{};
    u32 stencil_back_reference{};
    u32 stencil_front_write_mask{};
    u32 stencil_back_write_mask{};
    u32 stencil_front_compare_mask{};
    u32 stencil_back_compare_mask{};

    bool primitive_restart_enabled{};
    bool rasterizer_discard_enabled{};
    vk::CullModeFlags cull_mode{};
    vk::FrontFace front_face{};
    float line_width{1.0f};
    std::array<float, 4> blend_constants{};
    ColorWriteMasks color_write_masks{};
    bool attachment_feedback_loop{};
};

class GraphicsRequestBuilder {
public:
    [[nodiscard]] static DynamicStatePlan BuildDynamicStatePlan(
        const VideoCore::CapturedGraphicsState& state, const GraphicsPipeline& pipeline,
        const Instance& instance, bool is_indexed, bool attachment_feedback_loop);
};

} // namespace Vulkan
