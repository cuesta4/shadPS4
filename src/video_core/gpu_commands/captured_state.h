// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>

#include "common/types.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"

namespace VideoCore {

struct CommandProvenance {
    u64 sequence{};
    u64 submission{};
    VAddr packet_address{};
    u32 queue_id{};
    u32 packet_offset_dw{};
    u32 opcode{};
};

using GpuCommandProvenance = CommandProvenance;

struct DrawArguments {
    bool indexed{};
    u32 index_buffer_offset{};
    u32 index_or_vertex_count{};
    u32 instance_count{};
};

struct DispatchArguments {
    u32 group_count_x{1};
    u32 group_count_y{1};
    u32 group_count_z{1};
    u32 shared_memory_bytes{};
};

struct GraphicsPipelineState {
    AmdGpu::ShaderProgram ps_program{};
    AmdGpu::ShaderProgram vs_program{};
    AmdGpu::ShaderProgram gs_program{};
    AmdGpu::ShaderProgram es_program{};
    AmdGpu::ShaderProgram hs_program{};
    AmdGpu::ShaderProgram ls_program{};

    AmdGpu::ShaderStageEnable stage_enable{};
    AmdGpu::LsHsConfig ls_hs_config{};
    AmdGpu::TessellationConfig tess_config{};
    AmdGpu::VsOutputControl vs_output_control{};
    AmdGpu::ClipperControl clipper_control{};
    AmdGpu::PolygonControl polygon_control{};
    AmdGpu::PrimitiveType primitive_type{};

    AmdGpu::PsInput ps_input_ena{};
    AmdGpu::PsInput ps_input_addr{};
    std::array<AmdGpu::PsInputControl, 32> ps_inputs{};
    u32 num_interp{};
    AmdGpu::ShaderExportFormat z_export_format{};
    AmdGpu::ColorExportFormat color_export_format{};
    AmdGpu::DepthShaderControl depth_shader_control{};

    std::array<AmdGpu::BlendControl, AmdGpu::NUM_COLOR_BUFFERS> blend_control{};
    AmdGpu::ColorBufferMask color_target_mask{};
    AmdGpu::ColorBufferMask color_shader_mask{};
    AmdGpu::ColorControl color_control{};
    AmdGpu::DepthRenderOverride depth_render_override{};

    u32 vgt_esgs_ring_itemsize{};
    u32 vgt_gs_max_vert_out{};
    std::array<u32, 4> vgt_gs_vert_itemsize{};
    AmdGpu::GsInstances vgt_gs_instance_cnt{};
    AmdGpu::GsMode vgt_gs_mode{};
    AmdGpu::GsOutPrimitiveType vgt_gs_out_prim_type{};
    AmdGpu::StreamOutConfig vgt_strmout_config{};

    [[nodiscard]] const AmdGpu::ShaderProgram* ProgramForStage(u32 index) const noexcept {
        switch (index) {
        case 0:
            return &ps_program;
        case 1:
            return &vs_program;
        case 2:
            return &gs_program;
        case 3:
            return &es_program;
        case 4:
            return &hs_program;
        case 5:
            return &ls_program;
        default:
            return nullptr;
        }
    }

    [[nodiscard]] bool IsClipDisabled() const noexcept {
        return clipper_control.clip_disable || primitive_type == AmdGpu::PrimitiveType::RectList;
    }
};

struct GraphicsAttachmentState {
    std::array<AmdGpu::ColorBuffer, AmdGpu::NUM_COLOR_BUFFERS> color_buffers{};
    std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS> color_extent_hints{};

    AmdGpu::DepthBuffer depth_buffer{};
    AmdGpu::DepthView depth_view{};
    AmdGpu::DepthControl depth_control{};
    AmdGpu::DepthRenderControl depth_render_control{};
    AmdGpu::Address depth_htile_data_base{};
    AmdGpu::CbDbExtent depth_extent_hint{};
    float depth_clear{};
    u32 stencil_clear{};
};

struct GraphicsDynamicState {
    AmdGpu::ViewportControl viewport_control{};
    std::array<AmdGpu::ViewportBounds, AmdGpu::NUM_VIEWPORTS> viewports{};
    std::array<AmdGpu::ViewportScissor, AmdGpu::NUM_VIEWPORTS> viewport_scissors{};
    AmdGpu::Scissor screen_scissor{};
    AmdGpu::ViewportScissor generic_scissor{};
    AmdGpu::WindowOffset window_offset{};
    AmdGpu::ViewportScissor window_scissor{};

    float depth_bounds_min{};
    float depth_bounds_max{};
    AmdGpu::PolygonOffset poly_offset{};
    AmdGpu::StencilControl stencil_control{};
    AmdGpu::StencilRefMask stencil_ref_front{};
    AmdGpu::StencilRefMask stencil_ref_back{};

    u32 enable_primitive_restart{};
    u32 primitive_restart_index{};
    AmdGpu::LineControl line_control{};
    AmdGpu::BlendConstants blend_constants{};
    AmdGpu::ModeControl mode_control{};

};

struct GraphicsDrawState {
    u32 num_indices{};
    AmdGpu::VgtNumInstances num_instances{};
    u32 index_offset{};
    AmdGpu::IndexBufferBase index_base_address{};
    AmdGpu::IndexBufferType index_buffer_type{};
    u32 vgt_instance_step_rate_0{};
    u32 vgt_instance_step_rate_1{};
};

struct GraphicsResourceState {
    AmdGpu::BorderColorBuffer border_color_base{};
};

struct CapturedGraphicsState {
    GraphicsPipelineState pipeline{};
    GraphicsAttachmentState attachments{};
    GraphicsDynamicState dynamic{};
    GraphicsDrawState draw{};
    GraphicsResourceState resources{};
};

struct ComputeDispatchArguments {
    u32 dim_x{1};
    u32 dim_y{1};
    u32 dim_z{1};
    u32 shared_memory_size{};
};

struct CapturedComputeState {
    AmdGpu::ComputeProgram program{};
    ComputeDispatchArguments dispatch{};
    GraphicsResourceState resources{};
};

} // namespace VideoCore
