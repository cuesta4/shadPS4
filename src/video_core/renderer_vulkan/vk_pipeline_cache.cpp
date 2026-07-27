// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>

#include <boost/container/static_vector.hpp>

#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

using Shader::LogicalStage;
using Shader::Output;
using Shader::Stage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

void ResolvedStageResources::Bind(Shader::Info& info) const noexcept {
    info.resolved_buffers = std::span<const AmdGpu::Buffer>{buffers.data(), buffers.size()};
    info.resolved_images = std::span<const AmdGpu::Image>{images.data(), images.size()};
    info.resolved_samplers = std::span<const AmdGpu::Sampler>{samplers.data(), samplers.size()};
    info.resolved_fmasks = std::span<const AmdGpu::Image>{fmasks.data(), fmasks.size()};
    info.resolved_vertex_buffers =
        std::span<const AmdGpu::Buffer>{vertex_buffers.data(), vertex_buffers.size()};
}

namespace {

template <typename DescriptorList, typename ResourceList>
void ResolveResourceList(const DescriptorList& descriptors, ResourceList& resources,
                         const Shader::Info& info) {
    resources.resize(descriptors.size(), boost::container::default_init);
    for (size_t index = 0; index < descriptors.size(); ++index) {
        resources[index] = descriptors[index].GetSharp(info);
    }
}

void ResolveStageResources(Shader::Info& info,
                           const std::optional<Shader::Gcn::FetchShaderData>* fetch_shader,
                           ResolvedStageResources& resolved) {
    ResolveResourceList(info.buffers, resolved.buffers, info);
    ResolveResourceList(info.images, resolved.images, info);
    ResolveResourceList(info.samplers, resolved.samplers, info);
    ResolveResourceList(info.fmasks, resolved.fmasks, info);
    if (fetch_shader && fetch_shader->has_value()) {
        ResolveResourceList((*fetch_shader)->attributes, resolved.vertex_buffers, info);
    } else {
        resolved.vertex_buffers.clear();
    }
    resolved.Bind(info);
}

template <typename T, std::size_t Capacity>
[[nodiscard]] bool EqualResourceList(
    const boost::container::static_vector<T, Capacity>& lhs,
    const boost::container::static_vector<T, Capacity>& rhs) noexcept {
    return lhs.size() == rhs.size() &&
           (lhs.empty() || std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(T)) == 0);
}

[[nodiscard]] bool EqualResolvedResources(const ResolvedStageResources& lhs,
                                          const ResolvedStageResources& rhs) noexcept {
    return EqualResourceList(lhs.buffers, rhs.buffers) &&
           EqualResourceList(lhs.images, rhs.images) &&
           EqualResourceList(lhs.samplers, rhs.samplers) &&
           EqualResourceList(lhs.fmasks, rhs.fmasks) &&
           EqualResourceList(lhs.vertex_buffers, rhs.vertex_buffers);
}

bool BuildSpecializationPlan(Program& program) {
    if (program.specialization_plan_ready) {
        return program.specialization_plan_cacheable;
    }
    program.specialization_plan_ready = true;
    const auto& info = program.info;
    program.specialization_plan_cacheable = !info.srt_info.walker_func &&
                                            info.l_stage != LogicalStage::TessellationControl &&
                                            info.l_stage != LogicalStage::TessellationEval;
    const auto valid_range = [](u32 first, u32 dwords) {
        return first + dwords <= Shader::ShaderParams::NumShaderUserData;
    };
    if (program.specialization_plan_cacheable) {
        for (const auto& resource : info.buffers) {
            program.specialization_plan_cacheable &=
                static_cast<bool>(resource.inline_cbuf) ||
                valid_range(resource.sharp_idx, sizeof(AmdGpu::Buffer) / sizeof(u32));
        }
        for (const auto& resource : info.images) {
            program.specialization_plan_cacheable &=
                valid_range(resource.sharp_idx, sizeof(AmdGpu::Image) / sizeof(u32));
        }
        for (const auto& resource : info.samplers) {
            program.specialization_plan_cacheable &=
                resource.is_inline_sampler ||
                valid_range(resource.sharp_idx, sizeof(AmdGpu::Sampler) / sizeof(u32));
        }
        for (const auto& resource : info.fmasks) {
            program.specialization_plan_cacheable &=
                valid_range(resource.sharp_idx, sizeof(AmdGpu::Image) / sizeof(u32));
        }
        if (info.has_fetch_shader) {
            program.specialization_plan_cacheable &= valid_range(info.fetch_shader_sgpr_base, 2);
        }
    }
    return program.specialization_plan_cacheable;
}

void RefreshDynamicProgramData(
    Shader::Info& info, VAddr program_base,
    std::span<const u32, Shader::ShaderParams::NumShaderUserData> user_data) {
    info.pgm_base = program_base;
    info.user_data = user_data;
    info.RefreshFlatBuf();
}

void RefreshDynamicProgramData(Shader::Info& info, const Shader::ShaderParams& params) {
    RefreshDynamicProgramData(info, params.Base(), params.user_data);
}

struct CachedFetchShader {
    const std::optional<Shader::Gcn::FetchShaderData>* parsed{};
    u64 revision{};

    [[nodiscard]] bool IsUsable(const Shader::Info& info) const noexcept {
        return parsed && (!info.has_fetch_shader || parsed->has_value());
    }
};

[[nodiscard]] CachedFetchShader GetCachedFetchShader(Program& program) {
    static const std::optional<Shader::Gcn::FetchShaderData> EmptyFetchShader{};
    auto& info = program.info;
    if (!info.has_fetch_shader) {
        return {.parsed = &EmptyFetchShader, .revision = 0};
    }

    const u32* code = Shader::Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
    if (!code) {
        return {.parsed = &EmptyFetchShader, .revision = 0};
    }

    for (auto& entry : program.fetch_shader_cache) {
        if (entry.address != code || !entry.parsed) {
            continue;
        }
        ASSERT(entry.parsed->size % sizeof(u32) == 0);
        const size_t word_count = entry.parsed->size / sizeof(u32);
        if (entry.code.size() == word_count &&
            std::equal(entry.code.begin(), entry.code.end(), code)) {
            return {.parsed = &entry.parsed, .revision = entry.revision};
        }
    }

    auto parsed = Shader::Gcn::ParseFetchShader(info);
    if (!parsed) {
        return {.parsed = &EmptyFetchShader, .revision = 0};
    }
    ASSERT(parsed->size % sizeof(u32) == 0);
    const size_t word_count = parsed->size / sizeof(u32);

    u8 slot{};
    if (program.fetch_shader_cache.size() < Program::MaxFetchShaderCacheEntries) {
        slot = static_cast<u8>(program.fetch_shader_cache.size());
        program.fetch_shader_cache.emplace_back();
    } else {
        slot = program.next_fetch_shader_slot;
        program.next_fetch_shader_slot = static_cast<u8>((program.next_fetch_shader_slot + 1) %
                                                         Program::MaxFetchShaderCacheEntries);
    }

    auto& entry = program.fetch_shader_cache[slot];
    entry.address = code;
    entry.code.assign(code, code + word_count);
    entry.parsed = std::move(parsed);
    entry.revision = program.next_fetch_shader_revision++;
    if (program.next_fetch_shader_revision == 0) {
        program.next_fetch_shader_revision = 1;
        program.fetch_shader_cache.clear();
        return GetCachedFetchShader(program);
    }
    return {.parsed = &entry.parsed, .revision = entry.revision};
}

struct StageRawDependencyKey {
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data{};
    u64 fetch_shader_revision{};

    bool operator==(const StageRawDependencyKey&) const noexcept = default;
};

[[nodiscard]] StageRawDependencyKey MakeRawDependencyKey(
    std::span<const u32, Shader::ShaderParams::NumShaderUserData> user_data,
    u64 fetch_shader_revision) {
    StageRawDependencyKey key{};
    std::ranges::copy(user_data, key.user_data.begin());
    key.fetch_shader_revision = fetch_shader_revision;
    return key;
}

struct GraphicsDependencyKey {
    u64 fixed_generation{};
    std::array<StageRawDependencyKey, MaxShaderStages> stage_keys{};
    std::array<VAddr, MaxShaderStages> stage_bases{};
    u32 active_mask{};

    bool operator==(const GraphicsDependencyKey&) const noexcept = default;
};

struct StageOptimizationEntry {
    bool valid{};
    Stage stage{};
    LogicalStage logical_stage{};
    u64 program_hash{};
    VAddr program_base{};
    StageRawDependencyKey raw_dependency{};
    Shader::RuntimeInfo runtime_info{};
    Shader::Backend::Bindings start{};
    PipelineCache::Result result{};
    ResolvedStageResources resolved{};
};

constexpr u8 InvalidStageOptimizationSlot = std::numeric_limits<u8>::max();

struct StageOptimizationSet {
    std::array<StageOptimizationEntry, Program::MaxPermutations> entries{};
    u8 next{};
    u8 current{InvalidStageOptimizationSlot};
};

} // namespace

struct PipelineCache::OptimizationState {
    GraphicsDependencyKey graphics_dependency{};
    GraphicsPipelineKey graphics_key{};
    const GraphicsPipeline* graphics_pipeline{};
    bool graphics_valid{};
    bool graphics_cacheable{};
    std::array<StageOptimizationSet, MaxShaderStages> stage_sets{};
};

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(Stage stage, LogicalStage l_stage) {
    auto& info = runtime_infos[u32(l_stage)];
    const auto& regs = liverpool->regs;
    const auto BuildCommon = [&](const auto& program) {
        info.num_user_data = program.settings.num_user_regs;
        info.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.num_allocated_vgprs = program.NumVgprs();
        info.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.fp_round_mode32 = program.settings.fp_round_mode32;
        info.fp_round_mode16_64 = program.settings.fp_round_mode64;
    };
    info.Initialize(stage);
    switch (stage) {
    case Stage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
        const auto* hull_info = infos[u32(Shader::LogicalStage::TessellationControl)];
        hull_info->ReadTessConstantBuffer(tess_constants);
        info.ls_info.ls_stride = tess_constants.ls_stride;
        break;
    }
    case Stage::Hull: {
        BuildCommon(regs.hs_program);
        info.hs_info.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.hs_info.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.hs_info.tess_type = regs.tess_config.type;
        info.hs_info.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;

        // We need to initialize most hs_info fields after finding the V# with tess constants
        break;
    }
    case Stage::Export: {
        BuildCommon(regs.es_program);
        info.es_info.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Vertex: {
        BuildCommon(regs.vs_program);
        info.vs_info.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        info.vs_info.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.vs_info.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.vs_info.num_outputs = MapOutputs(info.vs_info.outputs, regs.vs_output_control);
        info.vs_info.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        info.vs_info.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
        info.vs_info.clip_disable = regs.IsClipDisabled();
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Geometry: {
        BuildCommon(regs.gs_program);
        auto& gs_info = info.gs_info;
        gs_info.num_outputs = MapOutputs(gs_info.outputs, regs.vs_output_control);
        gs_info.output_vertices = regs.vgt_gs_max_vert_out;
        gs_info.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            gs_info.in_primitive = [&]() {
                switch (regs.tess_config.topology) {
                case AmdGpu::TessellationTopology::Point:
                    return AmdGpu::PrimitiveType::PointList;
                case AmdGpu::TessellationTopology::Line:
                    return AmdGpu::PrimitiveType::LineList;
                case AmdGpu::TessellationTopology::TriangleCw:
                case AmdGpu::TessellationTopology::TriangleCcw:
                    return AmdGpu::PrimitiveType::TriangleList;
                default:
                    UNREACHABLE();
                }
            }();
        } else {
            gs_info.in_primitive = regs.primitive_type;
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            gs_info.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        gs_info.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        gs_info.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        gs_info.mode = regs.vgt_gs_mode.mode;
        const auto params_vc = AmdGpu::GetParams(regs.vs_program);
        gs_info.vs_copy = params_vc.code;
        gs_info.vs_copy_hash = params_vc.hash;
        DumpShader(gs_info.vs_copy, gs_info.vs_copy_hash, Shader::Stage::Vertex, 0, "copy.bin");
        break;
    }
    case Stage::Fragment: {
        BuildCommon(regs.ps_program);
        info.fs_info.en_flags = regs.ps_input_ena;
        info.fs_info.addr_flags = regs.ps_input_addr;
        info.fs_info.num_inputs = regs.num_interp;
        info.fs_info.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.fs_info.mrtz_mask = regs.depth_shader_control.z_export_enable |
                                 (stencil_ref_export_enable << 1) |
                                 (regs.depth_shader_control.mask_export_enable << 2) |
                                 (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.fs_info.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.fs_info.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.fs_info.dual_source_blending = false;
        }
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.fs_info.inputs[i] = {
                .param_index = u8(ps_inputs[i].input_offset),
                .is_default = bool(ps_inputs[i].use_default),
                .is_flat = bool(ps_inputs[i].flat_shade),
                .default_value = u8(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.fs_info.color_buffers[i] = graphics_key.color_buffers[i];
        }
        // Lowered user clip planes ride the same emulation path as guest-exported distances, so
        // the fragment side arms whenever the hardware vertex stage lowers them, keeping its input
        // locations in sync with the shifted vertex outputs.
        const bool lowers_user_clip_planes =
            regs.clipper_control.user_clip_plane_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Geometry));
        info.fs_info.clip_distance_emulation =
            ((regs.vs_output_control.clip_distance_enable &&
              !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Local))) ||
             lowers_user_clip_planes) &&
            profile.needs_clip_distance_emulation;
        break;
    }
    case Stage::Compute: {
        const auto& cs_pgm = liverpool->GetCsRegs();
        info.num_user_data = cs_pgm.settings.num_user_regs;
        info.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.cs_info.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                       cs_pgm.num_thread_z.full};
        info.cs_info.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                    cs_pgm.IsTgidEnabled(2)};
        info.cs_info.shared_memory_size = cs_pgm.SharedMemSize();
        break;
    }
    default:
        break;
    }
    return info;
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetMasterSemaphore(), DescriptorHeapSizes},
      optimization{std::make_unique<OptimizationState>()} {
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
    };
    WarmUp();

    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
}

PipelineCache::~PipelineCache() = default;

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline() {
    auto& opt = *optimization;

    const auto build_dependency = [&]() -> std::optional<GraphicsDependencyKey> {
        GraphicsDependencyKey dependency{};
        dependency.fixed_generation = liverpool->GraphicsPipelineGeneration();
        for (u32 logical_index = 0; logical_index < MaxShaderStages; ++logical_index) {
            if (!infos[logical_index]) {
                continue;
            }
            const auto& stage_set = opt.stage_sets[logical_index];
            if (stage_set.current == InvalidStageOptimizationSlot) {
                return std::nullopt;
            }
            const auto& stage_cache = stage_set.entries[stage_set.current];
            if (!stage_cache.valid) {
                return std::nullopt;
            }
            const auto* shader_program =
                liverpool->regs.ProgramForStage(static_cast<u32>(stage_cache.stage));
            if (!shader_program || !shader_program->Address<u32*>()) {
                return std::nullopt;
            }
            const VAddr program_base = shader_program->Address<VAddr>();
            const auto program_it = program_cache.find(stage_cache.program_hash);
            if (program_it == program_cache.end() ||
                !BuildSpecializationPlan(*program_it.value()) ||
                program_base != stage_cache.program_base) {
                return std::nullopt;
            }

            auto& program = *program_it.value();
            RefreshDynamicProgramData(program.info, program_base, shader_program->user_data);
            const auto cached_fetch_shader = GetCachedFetchShader(program);
            if (!cached_fetch_shader.IsUsable(program.info)) {
                return std::nullopt;
            }
            ResolveStageResources(program.info, cached_fetch_shader.parsed,
                                  program.resolved_resources);
            if (!EqualResolvedResources(stage_cache.resolved, program.resolved_resources)) {
                return std::nullopt;
            }

            dependency.active_mask |= 1U << logical_index;
            dependency.stage_bases[logical_index] = program_base;
            dependency.stage_keys[logical_index] =
                MakeRawDependencyKey(shader_program->user_data, cached_fetch_shader.revision);
        }
        return dependency;
    };

    if (opt.graphics_valid && opt.graphics_cacheable && opt.graphics_pipeline &&
        liverpool->GraphicsPipelineGeneration() == opt.graphics_dependency.fixed_generation) {
        const auto dependency = build_dependency();
        if (dependency && *dependency == opt.graphics_dependency) {
            return opt.graphics_pipeline;
        }
    }

    const auto* pipeline = ResolveGraphicsPipelineSlow();
    opt.graphics_pipeline = pipeline;
    opt.graphics_key = graphics_key;
    opt.graphics_valid = pipeline != nullptr;
    opt.graphics_cacheable = pipeline && CanReuseGraphicsPipeline();
    if (opt.graphics_cacheable) {
        if (const auto dependency = build_dependency()) {
            opt.graphics_dependency = *dependency;
        } else {
            opt.graphics_cacheable = false;
        }
    }
    return pipeline;
}

const GraphicsPipeline* PipelineCache::ResolveGraphicsPipelineSlow() {
    if (!RefreshGraphicsKey()) {
        return nullptr;
    }
    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);

        GraphicsPipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos, fetch_shader, modules, sdata, false);

        RegisterPipelineData(graphics_key, pipeline_hash, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage]) {
                    auto& m = modules[stage];
                    module_related_pipelines[m].emplace_back(graphics_key);
                }
            }
        }
        fetch_shader.reset();
    }
    return it->second.get();
}

bool PipelineCache::CanReuseGraphicsPipeline() const {
    for (const auto* info : infos) {
        if (!info) {
            continue;
        }
        const auto program_it = program_cache.find(info->pgm_hash);
        if (program_it == program_cache.end() || !program_it.value()->specialization_plan_ready ||
            !program_it.value()->specialization_plan_cacheable) {
            return false;
        }
    }
    return true;
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false);
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey() {
    std::memset(&graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;

    const bool db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // First pass to fill render target information needed by shader recompiler
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        // Fill color target information
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();
    }

    // Compile and bind shader stages
    if (!RefreshGraphicsStages()) {
        return false;
    }

    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

bool PipelineCache::RefreshGraphicsStages() {
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;
    fetch_shader = std::nullopt;

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](Shader::Stage stage_in, Shader::LogicalStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader_;
        std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_shader_,
                 key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        if (fetch_shader_) {
            fetch_shader = fetch_shader_;
        }
        return true;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);

    bind_stage(Stage::Fragment, LogicalStage::Fragment);

    const auto* fs_info = infos[static_cast<u32>(LogicalStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(Stage::Export, LogicalStage::Vertex)) {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(Stage::Vertex, LogicalStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(Stage::Export, LogicalStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        bind_stage(Stage::Vertex, LogicalStage::Vertex);
        break;
    default:
        UNREACHABLE_MSG("unhandled stage_en: {}", (u32)regs.stage_enable.raw);
    }

    const auto* vs_info = infos[static_cast<u32>(Shader::LogicalStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        ASSERT_MSG(vs_info->resolved_vertex_buffers.size() == fetch_shader->attributes.size(),
                   "Resolved vertex buffer count does not match fetch shader attributes: {} != {}",
                   vs_info->resolved_vertex_buffers.size(), fetch_shader->attributes.size());
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            const auto buffer = vs_info->resolved_vertex_buffers[vertex_binding];
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey() {
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    std::tie(infos[0], modules[0], fetch_shader, compute_key.value) =
        GetProgram(Shader::Stage::Compute, LogicalStage::Compute, cs_params, binding);
    return true;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.stage, perm_idx, "bin");

    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    DumpShader(spv, info.pgm_hash, info.stage, perm_idx, "spv");

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }

    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);

    const auto name = GetShaderName(info.stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.l_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(Stage stage, LogicalStage l_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    const auto runtime_info = BuildRuntimeInfo(stage, l_stage);
    auto& opt = *optimization;
    const u32 stage_index = static_cast<u32>(l_stage);
    auto& cache_set = opt.stage_sets[stage_index];
    cache_set.current = InvalidStageOptimizationSlot;
    const auto start = binding;

    auto program_it = program_cache.find(params.hash);
    const bool cacheable =
        program_it != program_cache.end() && BuildSpecializationPlan(*program_it.value());

    std::optional<StageRawDependencyKey> raw_dependency;
    CachedFetchShader cached_fetch_shader{};
    if (cacheable) {
        auto& program = *program_it.value();
        RefreshDynamicProgramData(program.info, params);
        cached_fetch_shader = GetCachedFetchShader(program);
        if (cached_fetch_shader.IsUsable(program.info)) {
            ResolveStageResources(program.info, cached_fetch_shader.parsed,
                                  program.resolved_resources);
            raw_dependency = MakeRawDependencyKey(params.user_data, cached_fetch_shader.revision);
            for (u8 slot = 0; slot < cache_set.entries.size(); ++slot) {
                auto& candidate = cache_set.entries[slot];
                const bool basic_key_matches =
                    candidate.valid && candidate.stage == stage &&
                    candidate.logical_stage == l_stage && candidate.program_hash == params.hash &&
                    candidate.program_base == params.Base() && candidate.start == start &&
                    candidate.runtime_info == runtime_info;
                if (basic_key_matches && candidate.raw_dependency == *raw_dependency &&
                    EqualResolvedResources(candidate.resolved, program.resolved_resources)) {
                    program.info.AddBindings(binding);
                    cache_set.current = slot;
                    return candidate.result;
                }
            }
        }
    }

    const auto result = GetProgramSlow(stage, l_stage, params, runtime_info, binding);
    program_it = program_cache.find(params.hash);
    const bool can_store =
        program_it != program_cache.end() && BuildSpecializationPlan(*program_it.value());
    if (!can_store) {
        return result;
    }

    auto& program = *program_it.value();
    RefreshDynamicProgramData(program.info, params);
    cached_fetch_shader = GetCachedFetchShader(program);
    if (!cached_fetch_shader.IsUsable(program.info)) {
        return result;
    }
    raw_dependency = MakeRawDependencyKey(params.user_data, cached_fetch_shader.revision);

    u8 target_slot = InvalidStageOptimizationSlot;
    for (u8 slot = 0; slot < cache_set.entries.size(); ++slot) {
        if (!cache_set.entries[slot].valid) {
            target_slot = slot;
            break;
        }
    }
    if (target_slot == InvalidStageOptimizationSlot) {
        target_slot = cache_set.next;
        cache_set.next = static_cast<u8>((cache_set.next + 1) % cache_set.entries.size());
    }

    auto& target = cache_set.entries[target_slot];
    target = StageOptimizationEntry{
        .valid = true,
        .stage = stage,
        .logical_stage = l_stage,
        .program_hash = params.hash,
        .program_base = params.Base(),
        .raw_dependency = *raw_dependency,
        .runtime_info = runtime_info,
        .start = start,
        .result = result,
        .resolved = program.resolved_resources,
    };
    program.resolved_resources.Bind(program.info);
    cache_set.current = target_slot;
    return result;
}

PipelineCache::Result PipelineCache::GetProgramSlow(Stage stage, LogicalStage l_stage,
                                                    const Shader::ShaderParams& params,
                                                    Shader::RuntimeInfo runtime_info,
                                                    Shader::Backend::Bindings& binding) {
    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        it_pgm.value() = std::make_unique<Program>(stage, l_stage, params);
        auto& program = it_pgm.value();
        auto start = binding;
        const auto module = CompileModule(program->info, runtime_info, params.code, 0, binding);
        RefreshDynamicProgramData(program->info, params);
        const auto cached_fetch_shader = GetCachedFetchShader(*program);
        ResolveStageResources(program->info, cached_fetch_shader.parsed,
                              program->resolved_resources);
        auto spec = Shader::StageSpecialization(program->info, runtime_info, profile, start,
                                                cached_fetch_shader.parsed);
        const auto perm_hash = HashCombine(params.hash, 0);

        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(module, std::move(spec));
        return std::make_tuple(&program->info, module, program->modules[0].spec.fetch_shader_data,
                               perm_hash);
    }

    auto& program = it_pgm.value();
    auto& info = program->info;
    RefreshDynamicProgramData(info, params);
    const auto cached_fetch_shader = GetCachedFetchShader(*program);
    ResolveStageResources(info, cached_fetch_shader.parsed, program->resolved_resources);
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding,
                                            cached_fetch_shader.parsed);

    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    const auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    if (it == program->modules.end()) {
        auto new_info = Shader::Info(stage, l_stage, params);
        module = CompileModule(new_info, runtime_info, params.code, perm_idx, binding);

        RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
        program->AddPermut(module, std::move(spec));
    } else {
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        perm_hash = HashCombine(params.hash, perm_idx);
    }
    return std::make_tuple(&program->info, module,
                           program->modules[perm_idx].spec.fetch_shader_data, perm_hash);
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    optimization->graphics_valid = false;
    optimization->graphics_pipeline = nullptr;
    for (auto& stage_set : optimization->stage_sets) {
        stage_set.current = InvalidStageOptimizationSlot;
        for (auto& entry : stage_set.entries) {
            entry.valid = false;
        }
    }
    std::optional<vk::ShaderModule> new_module{};
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                const auto& d = instance.GetDevice();
                d.destroyShaderModule(m.module);
                m.module = CompileSPV(spv_code, d);
                new_module = m.module;
            }
        }
    }
    if (module_related_pipelines.contains(module)) {
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
            }
        }
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::Stage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::Stage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}
} // namespace Vulkan
