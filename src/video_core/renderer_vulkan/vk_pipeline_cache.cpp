// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <ranges>
#include <type_traits>

#include "common/assert.h"
#include "common/elf_info.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/thread.h"
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

struct ShaderCompileResult {
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data{};
    std::vector<u32> code;
    std::vector<u32> geometry_copy_code;
    Shader::Info info;
    Shader::RuntimeInfo runtime_info{};
    Shader::Backend::Bindings binding_start{};
    Shader::Backend::Bindings bindings{};
    Shader::StageSpecialization specialization{};
    std::vector<u32> debug_spv;
    std::vector<u32> debug_patch;
    vk::ShaderModule module{};
    size_t permutation_index{};
    u64 permutation_hash{};
    bool initial_program{};
    bool collect_shader{};
    bool is_patched{};
};

namespace {

constexpr std::array<u8, 8> NativePipelineCacheMagic{'S', 'H', 'A', 'D', 'V', 'K', 'P', 'C'};
constexpr u32 NativePipelineCacheVersion = 1;
constexpr u64 MaxNativePipelineCacheSize = 512ULL * 1024 * 1024;

struct NativePipelineCacheHeader {
    std::array<u8, 8> magic;
    u32 version;
    u32 pipeline_key_version;
    u32 driver_version;
    u64 payload_size;
    Shader::Profile profile;
};

struct VulkanPipelineCacheHeader {
    u32 header_size;
    u32 header_version;
    u32 vendor_id;
    u32 device_id;
    std::array<u8, VK_UUID_SIZE> uuid;
};

struct GraphicsPipelineBuild {
    GraphicsPipelineKey key;
    std::array<std::array<u32, Shader::ShaderParams::NumShaderUserData>, MaxShaderStages>
        user_data{};
    std::array<std::optional<Shader::Info>, MaxShaderStages> compile_stage_storage;
    std::array<const Shader::Info*, MaxShaderStages> runtime_stages{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader;

    [[nodiscard]] std::array<const Shader::Info*, MaxShaderStages> BindCompileStages() {
        std::array<const Shader::Info*, MaxShaderStages> compile_stages{};
        for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
            auto& info = compile_stage_storage[stage];
            if (info) {
                info->user_data = user_data[stage];
                compile_stages[stage] = &*info;
            }
        }
        return compile_stages;
    }
};

static_assert(std::is_trivially_copyable_v<NativePipelineCacheHeader>);
static_assert(sizeof(VulkanPipelineCacheHeader) == 32);

[[nodiscard]] std::filesystem::path GetNativePipelineCachePath() {
    return Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / "vulkan" /
           Common::ElfInfo::Instance().GameSerial() / "pipeline_cache.bin";
}

[[nodiscard]] bool ValidateNativePipelineCacheData(std::span<const u8> data,
                                                   const Instance& instance) {
    if (data.size() < sizeof(VulkanPipelineCacheHeader)) {
        return false;
    }
    VulkanPipelineCacheHeader header{};
    std::memcpy(&header, data.data(), sizeof(header));
    return header.header_size >= sizeof(header) && header.header_size <= data.size() &&
           header.header_version == static_cast<u32>(VK_PIPELINE_CACHE_HEADER_VERSION_ONE) &&
           header.vendor_id == instance.GetVendorID() &&
           header.device_id == instance.GetDeviceID() &&
           header.uuid == instance.GetPipelineCacheUUID();
}

} // Anonymous namespace

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

std::vector<u8> PipelineCache::LoadNativePipelineCache() const {
    using namespace Common::FS;
    const auto path = GetNativePipelineCachePath();
    const IOFile file{path, FileAccessMode::Read};
    if (!file.IsOpen()) {
        return {};
    }

    const u64 file_size = file.GetSize();
    if (file_size < sizeof(NativePipelineCacheHeader) ||
        file_size > sizeof(NativePipelineCacheHeader) + MaxNativePipelineCacheSize) {
        LOG_WARNING(Render_Vulkan, "Ignoring invalid native Vulkan pipeline cache {}",
                    path.string());
        return {};
    }

    NativePipelineCacheHeader header{};
    if (file.Read(header) != 1 || header.magic != NativePipelineCacheMagic ||
        header.version != NativePipelineCacheVersion ||
        header.pipeline_key_version != Serialization::PipelineKeyVersion ||
        header.driver_version != instance.GetDriverVersion() || header.profile != profile ||
        header.payload_size != file_size - sizeof(header)) {
        LOG_INFO(Render_Vulkan, "Native Vulkan pipeline cache is stale; rebuilding it");
        return {};
    }

    std::vector<u8> data(header.payload_size);
    if (file.Read(data) != data.size() || !ValidateNativePipelineCacheData(data, instance)) {
        LOG_WARNING(Render_Vulkan, "Ignoring incompatible native Vulkan pipeline cache {}",
                    path.string());
        return {};
    }
    LOG_INFO(Render_Vulkan, "Loaded {} KiB native Vulkan pipeline cache", data.size() / 1024);
    return data;
}

void PipelineCache::SaveNativePipelineCache() {
    if (!pipeline_cache || !native_pipeline_cache_dirty.load(std::memory_order_acquire)) {
        return;
    }

    std::scoped_lock lock{native_pipeline_cache_mutex};
    if (!native_pipeline_cache_dirty.exchange(false, std::memory_order_acq_rel)) {
        return;
    }

    auto [result, data] = instance.GetDevice().getPipelineCacheData(*pipeline_cache);
    if (result != vk::Result::eSuccess || data.empty() || data.size() > MaxNativePipelineCacheSize ||
        !ValidateNativePipelineCacheData(data, instance)) {
        native_pipeline_cache_dirty.store(true, std::memory_order_release);
        LOG_WARNING(Render_Vulkan, "Failed to retrieve native Vulkan pipeline cache: {}",
                    vk::to_string(result));
        return;
    }

    const auto path = GetNativePipelineCachePath();
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        native_pipeline_cache_dirty.store(true, std::memory_order_release);
        LOG_WARNING(Render_Vulkan, "Failed to create native pipeline cache directory: {}",
                    ec.message());
        return;
    }

    const NativePipelineCacheHeader header{
        .magic = NativePipelineCacheMagic,
        .version = NativePipelineCacheVersion,
        .pipeline_key_version = Serialization::PipelineKeyVersion,
        .driver_version = instance.GetDriverVersion(),
        .payload_size = data.size(),
        .profile = profile,
    };
    const Common::FS::IOFile file{path, Common::FS::FileAccessMode::Create};
    if (!file.IsOpen() || file.Write(header) != 1 || file.Write(data) != data.size() ||
        !file.Flush()) {
        native_pipeline_cache_dirty.store(true, std::memory_order_release);
        LOG_WARNING(Render_Vulkan, "Failed to persist native Vulkan pipeline cache {}",
                    path.string());
    }
}

void PipelineCache::StartGraphicsPipelineCompiler() {
    for (u32 index = 0; index < graphics_pipeline_workers.size(); ++index) {
        graphics_pipeline_workers[index] =
            std::jthread{[this, index] { GraphicsPipelineCompilerThread(index); }};
    }
}

void PipelineCache::StopGraphicsPipelineCompiler() {
    {
        std::scoped_lock lock{graphics_pipeline_tasks_mutex};
        graphics_pipeline_compiler_stopping = true;
    }
    graphics_pipeline_tasks_cv.notify_all();
    for (auto& worker : graphics_pipeline_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void PipelineCache::WaitForGraphicsPipelineCompiler() {
    std::unique_lock lock{graphics_pipeline_tasks_mutex};
    graphics_pipeline_tasks_cv.wait(lock, [this] {
        return shader_module_tasks.empty() && graphics_pipeline_tasks.empty() &&
               graphics_pipeline_tasks_in_flight == 0;
    });
}

void PipelineCache::GraphicsPipelineCompilerThread(u32 worker_index) {
    const auto name = fmt::format("shadPS4:ShaderCompiler{}", worker_index);
    Common::SetCurrentThreadName(name.c_str());
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);

    while (true) {
        std::packaged_task<void()> task;
        {
            std::unique_lock lock{graphics_pipeline_tasks_mutex};
            graphics_pipeline_tasks_cv.wait(lock, [this] {
                return graphics_pipeline_compiler_stopping || !shader_module_tasks.empty() ||
                       !graphics_pipeline_tasks.empty();
            });
            if (shader_module_tasks.empty() && graphics_pipeline_tasks.empty()) {
                if (graphics_pipeline_compiler_stopping) {
                    return;
                }
                continue;
            }
            const bool take_shader_module =
                !shader_module_tasks.empty() &&
                (worker_index < NumShaderModulePreferredWorkers ||
                 graphics_pipeline_tasks.empty());
            if (take_shader_module) {
                task = std::move(shader_module_tasks.front());
                shader_module_tasks.pop_front();
            } else {
                task = std::move(graphics_pipeline_tasks.front());
                graphics_pipeline_tasks.pop_front();
            }
            ++graphics_pipeline_tasks_in_flight;
        }
        task();
        if (native_pipeline_cache_save_requested.exchange(false, std::memory_order_acq_rel)) {
            SaveNativePipelineCache();
        }
        {
            std::scoped_lock lock{graphics_pipeline_tasks_mutex};
            ASSERT(graphics_pipeline_tasks_in_flight != 0);
            --graphics_pipeline_tasks_in_flight;
        }
        graphics_pipeline_tasks_cv.notify_all();
    }
}

void PipelineCache::QueueGraphicsPipelineTask(std::packaged_task<void()>&& task) {
    {
        std::scoped_lock lock{graphics_pipeline_tasks_mutex};
        ASSERT(!graphics_pipeline_compiler_stopping);
        graphics_pipeline_tasks.emplace_back(std::move(task));
    }
    graphics_pipeline_tasks_cv.notify_one();
}

void PipelineCache::QueueShaderModuleTask(std::packaged_task<void()>&& task) {
    {
        std::scoped_lock lock{graphics_pipeline_tasks_mutex};
        ASSERT(!graphics_pipeline_compiler_stopping);
        shader_module_tasks.emplace_back(std::move(task));
    }
    graphics_pipeline_tasks_cv.notify_one();
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetMasterSemaphore(), DescriptorHeapSizes},
      async_shader_recompiling{EmulatorSettings.IsAsyncShaderRecompiling()} {
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
    const auto initial_data = LoadNativePipelineCache();
    const vk::PipelineCacheCreateInfo cache_info{
        .initialDataSize = initial_data.size(),
        .pInitialData = initial_data.empty() ? nullptr : initial_data.data(),
    };
    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique(cache_info);
    if (cache_result != vk::Result::eSuccess && !initial_data.empty()) {
        LOG_WARNING(Render_Vulkan, "Driver rejected native Vulkan pipeline cache: {}",
                    vk::to_string(cache_result));
        auto fallback = instance.GetDevice().createPipelineCacheUnique({});
        cache_result = fallback.result;
        cache = std::move(fallback.value);
    }
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
    Shader::InitializeSrtWalker();
    WarmUp();
    SaveNativePipelineCache();
    StartGraphicsPipelineCompiler();
}

PipelineCache::~PipelineCache() {
    StopGraphicsPipelineCompiler();
    PublishPendingProgramCompilations();
    SaveNativePipelineCache();
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline() {
    if (!RefreshGraphicsKey()) {
        return nullptr;
    }
    const auto it = graphics_pipelines.find(graphics_key);
    if (it != graphics_pipelines.end()) {
        return it.value().get();
    }
    const auto pending_it = pending_graphics_pipelines.find(graphics_key);
    if (pending_it != pending_graphics_pipelines.end()) {
        if (pending_it.value().wait_for(std::chrono::seconds::zero()) !=
            std::future_status::ready) {
            return nullptr;
        }
        return PublishGraphicsPipeline(pending_it);
    }
    return CreateGraphicsPipeline();
}

const GraphicsPipeline* PipelineCache::PublishGraphicsPipeline(
    tsl::robin_map<GraphicsPipelineKey,
                   std::future<std::unique_ptr<GraphicsPipeline>>>::iterator pending_it) {
    const auto key = pending_it.key();
    auto pipeline = pending_it.value().get();
    pending_graphics_pipelines.erase(pending_it);
    ASSERT(pipeline != nullptr);
    const auto [it, is_new] = graphics_pipelines.try_emplace(key, std::move(pipeline));
    ASSERT(is_new);
    return it.value().get();
}

const GraphicsPipeline* PipelineCache::CreateGraphicsPipeline() {
    const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
    auto build = std::make_unique<GraphicsPipelineBuild>();
    build->key = graphics_key;
    std::ranges::copy(runtime_infos, build->runtime_infos.begin());
    std::ranges::copy(modules, build->modules.begin());
    build->fetch_shader = fetch_shader;
    for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
        const auto* info = infos[stage];
        build->runtime_stages[stage] = info;
        if (!info) {
            continue;
        }
        std::ranges::copy(info->user_data, build->user_data[stage].begin());
        build->compile_stage_storage[stage].emplace(*info);
    }

    const bool compile_async = async_shader_recompiling;
    std::packaged_task<std::unique_ptr<GraphicsPipeline>()> build_task{
        [this, build = std::move(build), pipeline_hash, compile_async]() mutable {
            if (compile_async) {
                LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x} asynchronously",
                         pipeline_hash);
            } else {
                LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);
            }
            auto compile_stages = build->BindCompileStages();
            GraphicsPipeline::SerializationSupport sdata{};
            auto pipeline = std::make_unique<GraphicsPipeline>(
                instance, scheduler, desc_heap, profile, build->key, *pipeline_cache,
                compile_stages, build->runtime_stages, build->runtime_infos,
                std::move(build->fetch_shader), build->modules, sdata, false);
            RegisterPipelineData(build->key, pipeline_hash, sdata);
            native_pipeline_cache_dirty.store(true, std::memory_order_release);
            const u32 updates =
                native_pipeline_cache_updates.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (updates % NativePipelineCacheSaveBatch == 0) {
                native_pipeline_cache_save_requested.store(true, std::memory_order_release);
            }
            return pipeline;
        }};
    auto future = build_task.get_future();
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

    if (!compile_async) {
        build_task();
        auto pipeline = future.get();
        const auto [it, is_new] =
            graphics_pipelines.try_emplace(graphics_key, std::move(pipeline));
        ASSERT(is_new);
        return it.value().get();
    }

    const bool is_new =
        pending_graphics_pipelines.try_emplace(graphics_key, std::move(future)).second;
    ASSERT(is_new);
    QueueGraphicsPipelineTask(std::packaged_task<void()>{
        [task = std::move(build_task)]() mutable { task(); }});
    return nullptr;
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
        native_pipeline_cache_dirty.store(true, std::memory_order_release);
        const u32 updates =
            native_pipeline_cache_updates.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (updates % NativePipelineCacheSaveBatch == 0) {
            native_pipeline_cache_save_requested.store(true, std::memory_order_release);
            QueueGraphicsPipelineTask(std::packaged_task<void()>{[] {}});
        }
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

    enum class BindResult {
        Inactive,
        Ready,
        Pending,
    };

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](Shader::Stage stage_in,
                                Shader::LogicalStage stage_out) -> BindResult {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            modules[stage_out_idx] = nullptr;
            return BindResult::Inactive;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            modules[stage_out_idx] = nullptr;
            return BindResult::Inactive;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        const auto result = GetProgram(stage_in, stage_out, params, binding);
        if (!result) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            modules[stage_out_idx] = nullptr;
            return BindResult::Pending;
        }
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader_;
        std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_shader_,
                 key.stage_hashes[stage_out_idx]) =
            *result;
        if (fetch_shader_) {
            fetch_shader = fetch_shader_;
        }
        return BindResult::Ready;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);

    if (bind_stage(Stage::Fragment, LogicalStage::Fragment) == BindResult::Pending) {
        return false;
    }

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
        if (bind_stage(Stage::Export, LogicalStage::Vertex) != BindResult::Ready) {
            return false;
        }
        if (bind_stage(Stage::Geometry, LogicalStage::Geometry) != BindResult::Ready) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (bind_stage(Stage::Hull, LogicalStage::TessellationControl) != BindResult::Ready) {
            return false;
        }
        if (bind_stage(Stage::Vertex, LogicalStage::TessellationEval) != BindResult::Ready) {
            return false;
        }
        if (bind_stage(Stage::Local, LogicalStage::Vertex) != BindResult::Ready) {
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
        if (bind_stage(Stage::Hull, LogicalStage::TessellationControl) != BindResult::Ready) {
            return false;
        }
        if (bind_stage(Stage::Export, LogicalStage::TessellationEval) != BindResult::Ready) {
            return false;
        }
        if (bind_stage(Stage::Local, LogicalStage::Vertex) != BindResult::Ready) {
            return false;
        }
        if (bind_stage(Stage::Geometry, LogicalStage::Geometry) != BindResult::Ready) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        if (bind_stage(Stage::Vertex, LogicalStage::Vertex) == BindResult::Pending) {
            return false;
        }
        break;
    default:
        UNREACHABLE_MSG("unhandled stage_en: {}", (u32)regs.stage_enable.raw);
    }

    const auto* vs_info = infos[static_cast<u32>(Shader::LogicalStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
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
    const auto result =
        GetProgram(Shader::Stage::Compute, LogicalStage::Compute, cs_params, binding);
    ASSERT(result.has_value());
    std::tie(infos[0], modules[0], fetch_shader, compute_key.value) =
        *result;
    return true;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding,
                                              ShaderCompileResult* async_result) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.stage, perm_idx, "bin");

    thread_local Shader::Pools pools;
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

    const auto name = GetShaderName(info.stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    const bool collect_shader =
        async_result ? async_result->collect_shader : EmulatorSettings.IsShaderCollect();
    if (collect_shader && async_result) {
        async_result->debug_spv = spv;
        if (patch) {
            async_result->debug_patch = *patch;
        }
        async_result->is_patched = is_patched;
    } else if (collect_shader) {
        DebugState.CollectShader(name, info.l_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);
    return module;
}

void PipelineCache::QueueProgramCompilation(
    Program& program, Stage stage, LogicalStage l_stage, const Shader::ShaderParams& params,
    const Shader::RuntimeInfo& runtime_info, Shader::Backend::Bindings binding,
    std::optional<Shader::StageSpecialization> specialization, size_t permutation_index,
    u64 permutation_hash, bool initial_program) {
    ASSERT(stage != Stage::Compute);
    ASSERT(!program.pending_compilation);
    ASSERT(initial_program != specialization.has_value());

    auto result = std::make_shared<ShaderCompileResult>();
    std::ranges::copy(params.user_data, result->user_data.begin());
    result->code.assign(params.code.begin(), params.code.end());
    result->info = Shader::Info(stage, l_stage, params);
    result->info.user_data = result->user_data;
    result->runtime_info = runtime_info;
    if (stage == Stage::Geometry && !runtime_info.gs_info.vs_copy.empty()) {
        result->geometry_copy_code.assign(runtime_info.gs_info.vs_copy.begin(),
                                          runtime_info.gs_info.vs_copy.end());
        result->runtime_info.gs_info.vs_copy = result->geometry_copy_code;
    }
    result->binding_start = binding;
    result->bindings = binding;
    if (specialization) {
        result->specialization = std::move(*specialization);
    }
    result->permutation_index = permutation_index;
    result->permutation_hash = permutation_hash;
    result->initial_program = initial_program;
    result->collect_shader = EmulatorSettings.IsShaderCollect();

    std::packaged_task<void()> compile_task{[this, result] {
        result->module =
            CompileModule(result->info, result->runtime_info, result->code,
                          result->permutation_index, result->bindings, result.get());
        if (result->initial_program) {
            result->specialization = Shader::StageSpecialization(
                result->info, result->runtime_info, profile, result->binding_start);
        }
    }};
    auto completion = compile_task.get_future();
    program.pending_compilation.emplace(
        Program::PendingCompilation{.result = result, .completion = std::move(completion)});
    QueueShaderModuleTask(std::move(compile_task));
}

bool PipelineCache::PublishProgramCompilation(Program& program) {
    if (!program.pending_compilation) {
        return true;
    }
    auto& pending = *program.pending_compilation;
    if (pending.completion.wait_for(std::chrono::seconds::zero()) != std::future_status::ready) {
        return false;
    }

    auto result = std::move(pending.result);
    auto completion = std::move(pending.completion);
    program.pending_compilation.reset();
    completion.get();
    ASSERT(result->module);
    ASSERT(program.modules.size() == result->permutation_index);

    if (result->initial_program) {
        ASSERT(program.modules.empty());
        program.info = std::move(result->info);
    }
    result->specialization.info = &program.info;
    RegisterShaderMeta(program.info, result->specialization.fetch_shader_data,
                       result->specialization, result->permutation_hash,
                       result->permutation_index);
    program.AddPermut(result->module, std::move(result->specialization));
    result->module = nullptr;

    if (result->collect_shader) {
        const auto name = GetShaderName(program.info.stage, program.info.pgm_hash,
                                        result->permutation_index);
        DebugState.CollectShader(name, program.info.l_stage, program.modules.back().module,
                                 result->debug_spv, result->code, result->debug_patch,
                                 result->is_patched);
    }
    if (result->initial_program) {
        program.info.user_data = {};
    }
    return true;
}

void PipelineCache::PublishPendingProgramCompilations() {
    for (auto& [_, program] : program_cache) {
        if (program->pending_compilation) {
            const bool published = PublishProgramCompilation(*program);
            ASSERT(published);
        }
    }
}

std::optional<PipelineCache::Result> PipelineCache::GetProgram(
    Stage stage, LogicalStage l_stage, const Shader::ShaderParams& params,
    Shader::Backend::Bindings& binding) {
    auto runtime_info = BuildRuntimeInfo(stage, l_stage);
    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        it_pgm.value() = std::make_unique<Program>(stage, l_stage, params);
        auto& program = it_pgm.value();
        const auto perm_hash = HashCombine(params.hash, 0);
        if (stage != Stage::Compute && async_shader_recompiling) {
            QueueProgramCompilation(*program, stage, l_stage, params, runtime_info, binding,
                                    std::nullopt, 0, perm_hash, true);
            return std::nullopt;
        }

        auto start = binding;
        const auto module = CompileModule(program->info, runtime_info, params.code, 0, binding);
        auto spec = Shader::StageSpecialization(program->info, runtime_info, profile, start);

        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(module, std::move(spec));
        return Result{&program->info, module, program->modules[0].spec.fetch_shader_data,
                      perm_hash};
    }

    auto& program = it_pgm.value();
    const bool compilation_ready = PublishProgramCompilation(*program);
    if (!compilation_ready && program->modules.empty()) {
        return std::nullopt;
    }
    auto& info = program->info;
    info.pgm_base = params.Base(); // Needs to be actualized for inline cbuffer address fixup
    info.user_data = params.user_data;
    info.RefreshFlatBuf();
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);

    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    const auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    if (it == program->modules.end()) {
        if (!compilation_ready) {
            return std::nullopt;
        }
        if (stage != Stage::Compute && async_shader_recompiling) {
            QueueProgramCompilation(*program, stage, l_stage, params, runtime_info, binding,
                                    std::move(spec), perm_idx, perm_hash, false);
            return std::nullopt;
        }
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
    return Result{&program->info, module, program->modules[perm_idx].spec.fetch_shader_data,
                  perm_hash};
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    WaitForGraphicsPipelineCompiler();
    PublishPendingProgramCompilations();
    for (auto& [_, future] : pending_graphics_pipelines) {
        future.wait();
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
                pending_graphics_pipelines.erase(graphics_key);
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
