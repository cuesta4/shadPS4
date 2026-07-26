// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/execution/graphics_operation_classifier.h"

namespace Vulkan {

GraphicsOperationClassification ClassifyGraphicsOperation(
    const VideoCore::CapturedGraphicsState& state) noexcept {
    const auto& pipeline = state.pipeline;
    const auto& attachments = state.attachments;
    using Mode = AmdGpu::ColorControl::OperationMode;
    switch (pipeline.color_control.mode) {
    case Mode::EliminateFastClear:
        return {.operation = GraphicsOperation::EliminateFastClear};
    case Mode::FmaskDecompress:
        return {.operation = GraphicsOperation::FmaskDecompress};
    case Mode::Resolve:
        return {.operation = GraphicsOperation::Resolve};
    default:
        break;
    }

    if (pipeline.primitive_type == AmdGpu::PrimitiveType::None) {
        return {.operation = GraphicsOperation::SkipPrimitiveNone};
    }

    const bool color_disabled = pipeline.color_control.mode == Mode::Disable;
    const bool copy_depth =
        pipeline.depth_render_override.force_z_dirty &&
        pipeline.depth_render_override.force_z_valid && attachments.depth_buffer.DepthValid() &&
        attachments.depth_buffer.DepthWriteValid() &&
        attachments.depth_buffer.DepthAddress() != attachments.depth_buffer.DepthWriteAddress();
    const bool copy_stencil =
        pipeline.depth_render_override.force_stencil_dirty &&
        pipeline.depth_render_override.force_stencil_valid && attachments.depth_buffer.StencilValid() &&
        attachments.depth_buffer.StencilWriteValid() &&
        attachments.depth_buffer.StencilAddress() !=
            attachments.depth_buffer.StencilWriteAddress();
    if (color_disabled && (copy_depth || copy_stencil)) {
        return {
            .operation = GraphicsOperation::DepthStencilCopy,
            .copy_depth = copy_depth,
            .copy_stencil = copy_stencil,
        };
    }
    return {};
}

} // namespace Vulkan
