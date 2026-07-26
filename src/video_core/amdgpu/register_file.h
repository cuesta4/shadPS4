// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <span>

#include "common/assert.h"
#include "common/types.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/gpu_commands/captured_state.h"

namespace AmdGpu {

struct AttachmentExtentHints {
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> color{};
    CbDbExtent depth{};
};

class GpuRegisterFile {
public:
    Regs& Raw() noexcept {
        return regs;
    }

    const Regs& Raw() const noexcept {
        return regs;
    }

    [[nodiscard]] VideoCore::CapturedGraphicsState CaptureGraphics() const noexcept {
        VideoCore::CapturedGraphicsState state{};

        auto& pipeline = state.pipeline;
        pipeline.ps_program = regs.ps_program;
        pipeline.vs_program = regs.vs_program;
        pipeline.gs_program = regs.gs_program;
        pipeline.es_program = regs.es_program;
        pipeline.hs_program = regs.hs_program;
        pipeline.ls_program = regs.ls_program;
        pipeline.stage_enable = regs.stage_enable;
        pipeline.ls_hs_config = regs.ls_hs_config;
        pipeline.tess_config = regs.tess_config;
        pipeline.vs_output_control = regs.vs_output_control;
        pipeline.clipper_control = regs.clipper_control;
        pipeline.polygon_control = regs.polygon_control;
        pipeline.primitive_type = regs.primitive_type;
        pipeline.ps_input_ena = regs.ps_input_ena;
        pipeline.ps_input_addr = regs.ps_input_addr;
        pipeline.ps_inputs = regs.ps_inputs;
        pipeline.num_interp = regs.num_interp;
        pipeline.z_export_format = regs.z_export_format;
        pipeline.color_export_format = regs.color_export_format;
        pipeline.depth_shader_control = regs.depth_shader_control;
        pipeline.blend_control = regs.blend_control;
        pipeline.color_target_mask = regs.color_target_mask;
        pipeline.color_shader_mask = regs.color_shader_mask;
        pipeline.color_control = regs.color_control;
        pipeline.depth_render_override = regs.depth_render_override;
        pipeline.vgt_esgs_ring_itemsize = regs.vgt_esgs_ring_itemsize;
        pipeline.vgt_gs_max_vert_out = regs.vgt_gs_max_vert_out;
        std::copy_n(regs.vgt_gs_vert_itemsize, 4, pipeline.vgt_gs_vert_itemsize.begin());
        pipeline.vgt_gs_instance_cnt = regs.vgt_gs_instance_cnt;
        pipeline.vgt_gs_mode = regs.vgt_gs_mode;
        pipeline.vgt_gs_out_prim_type = regs.vgt_gs_out_prim_type;
        pipeline.vgt_strmout_config = regs.vgt_strmout_config;

        auto& attachments = state.attachments;
        std::copy_n(regs.color_buffers, NUM_COLOR_BUFFERS, attachments.color_buffers.begin());
        attachments.color_extent_hints = extent_hints.color;
        attachments.depth_buffer = regs.depth_buffer;
        attachments.depth_view = regs.depth_view;
        attachments.depth_control = regs.depth_control;
        attachments.depth_render_control = regs.depth_render_control;
        attachments.depth_htile_data_base = regs.depth_htile_data_base;
        attachments.depth_extent_hint = extent_hints.depth;
        attachments.depth_clear = regs.depth_clear;
        attachments.stencil_clear = regs.stencil_clear;

        auto& dynamic = state.dynamic;
        dynamic.viewport_control = regs.viewport_control;
        dynamic.viewports = regs.viewports;
        dynamic.viewport_scissors = regs.viewport_scissors;
        dynamic.screen_scissor = regs.screen_scissor;
        dynamic.generic_scissor = regs.generic_scissor;
        dynamic.window_offset = regs.window_offset;
        dynamic.window_scissor = regs.window_scissor;
        dynamic.depth_bounds_min = regs.depth_bounds_min;
        dynamic.depth_bounds_max = regs.depth_bounds_max;
        dynamic.poly_offset = regs.poly_offset;
        dynamic.stencil_control = regs.stencil_control;
        dynamic.stencil_ref_front = regs.stencil_ref_front;
        dynamic.stencil_ref_back = regs.stencil_ref_back;
        dynamic.enable_primitive_restart = regs.enable_primitive_restart;
        dynamic.primitive_restart_index = regs.primitive_restart_index;
        dynamic.line_control = regs.line_control;
        dynamic.blend_constants = regs.blend_constants;
        dynamic.mode_control = regs.mode_control;

        auto& draw = state.draw;
        draw.num_indices = regs.num_indices;
        draw.num_instances = regs.num_instances;
        draw.index_offset = regs.index_offset;
        draw.index_base_address = regs.index_base_address;
        draw.index_buffer_type = regs.index_buffer_type;
        draw.vgt_instance_step_rate_0 = regs.vgt_instance_step_rate_0;
        draw.vgt_instance_step_rate_1 = regs.vgt_instance_step_rate_1;

        state.resources.border_color_base = regs.ta_bc_base;
        return state;
    }

    [[nodiscard]] VideoCore::CapturedComputeState CaptureCompute(
        const ComputeProgram& program) const noexcept {
        return {
            .program = program,
            .resources = {.border_color_base = regs.ta_bc_base},
        };
    }

    AttachmentExtentHints& ExtentHints() noexcept {
        return extent_hints;
    }

    const AttachmentExtentHints& ExtentHints() const noexcept {
        return extent_hints;
    }

    void WriteConfig(u32 offset, std::span<const u32> values) {
        WriteWords(Regs::ConfigRegWordOffset + offset, values);
    }

    void WriteContext(u32 offset, std::span<const u32> values) {
        WriteWords(Regs::ContextRegWordOffset + offset, values);
    }

    void WriteShader(u32 offset, std::span<const u32> values) {
        WriteWords(Regs::ShRegWordOffset + offset, values);
    }

    void WriteUconfig(u32 offset, std::span<const u32> values) {
        WriteWords(Regs::UconfigRegWordOffset + offset, values);
    }

private:
    void WriteWords(u32 word_offset, std::span<const u32> values) {
        ASSERT(word_offset + values.size() <= std::size(regs.reg_array));
        std::memcpy(&regs.reg_array[word_offset], values.data(), values.size_bytes());
    }

    Regs regs{};
    AttachmentExtentHints extent_hints{};
};

} // namespace AmdGpu
