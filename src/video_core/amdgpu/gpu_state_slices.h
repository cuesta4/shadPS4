// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include "common/types.h"
#include "video_core/gpu_commands/captured_state.h"

namespace AmdGpu {

using PipelineStateInput = VideoCore::GraphicsPipelineState;
using AttachmentStateInput = VideoCore::GraphicsAttachmentState;
using DynamicStateInput = VideoCore::GraphicsDynamicState;
using DrawStateInput = VideoCore::GraphicsDrawState;
using ResourceStateInput = VideoCore::GraphicsResourceState;
using GpuCommandProvenance = VideoCore::CommandProvenance;

struct ShaderInvocationRegisters {
    VAddr program_base{};
    std::array<u32, 16> user_data{};
    u32 valid_user_data_count{};
};

struct RasterStateInput {
    AmdGpu::PolygonControl polygon_control{};
    AmdGpu::ClipperControl clipper_control{};
    AmdGpu::PrimitiveType primitive_type{};
    AmdGpu::ModeControl mode_control{};
    AmdGpu::LineControl line_control{};
};

struct DepthStencilStateInput {
    AmdGpu::DepthControl depth_control{};
    AmdGpu::DepthRenderControl depth_render_control{};
    AmdGpu::DepthBuffer depth_buffer{};
    float depth_bounds_min{};
    float depth_bounds_max{};
    AmdGpu::StencilControl stencil_control{};
    AmdGpu::StencilRefMask stencil_ref_front{};
    AmdGpu::StencilRefMask stencil_ref_back{};
};

struct BlendStateInput {
    std::array<AmdGpu::BlendControl, AmdGpu::NUM_COLOR_BUFFERS> blend_control{};
    AmdGpu::ColorBufferMask color_target_mask{};
    AmdGpu::ColorControl color_control{};
    AmdGpu::BlendConstants blend_constants{};
};

struct ViewportScissorInput {
    AmdGpu::ViewportControl viewport_control{};
    std::array<AmdGpu::ViewportBounds, AmdGpu::NUM_VIEWPORTS> viewports{};
    std::array<AmdGpu::ViewportScissor, AmdGpu::NUM_VIEWPORTS> viewport_scissors{};
    AmdGpu::Scissor screen_scissor{};
    AmdGpu::ViewportScissor generic_scissor{};
    AmdGpu::WindowOffset window_offset{};
    AmdGpu::ViewportScissor window_scissor{};
};

struct VertexInputState {
    u32 num_indices{};
    AmdGpu::VgtNumInstances num_instances{};
    u32 index_offset{};
    AmdGpu::IndexBufferBase index_base_address{};
    AmdGpu::IndexBufferType index_buffer_type{};
    u32 vgt_instance_step_rate_0{};
    u32 vgt_instance_step_rate_1{};
};

} // namespace AmdGpu
