// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <optional>
#include <variant>
#include <vector>
#include <tsl/robin_map.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
}

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    struct FastPath {
        bool valid{};
        Shader::RuntimeInfo runtime_info{};
        Shader::Backend::Bindings start{};
        std::vector<u32> flattened_ud_buf{};
        vk::ShaderModule module{};
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader_data{};
        u64 perm_hash{};
        size_t perm_idx{};
    };

    struct FetchShaderCacheEntry {
        bool valid{};
        const u32* code{};
        std::vector<u32> code_snapshot{};
        std::optional<Shader::Gcn::FetchShaderData> data{};
    };

    static constexpr size_t FetchShaderCacheSize = 8;

    Shader::Info info;
    ModuleList modules{};
    FastPath fast_path{};
    std::array<FetchShaderCacheEntry, FetchShaderCacheSize> fetch_shader_cache{};
    size_t next_fetch_shader_cache{};

    Program() = default;
    Program(Shader::Stage stage, Shader::LogicalStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        modules.emplace_back(module, std::move(spec));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        modules[perm_idx] = {module, std::move(spec)};
    }
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool);
    ~PipelineCache();

    void WarmUp();
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    const GraphicsPipeline* GetGraphicsPipeline();

    void InvalidateGraphicsPipelineFastPath() noexcept {
        graphics_pipeline_l1_valid = false;
    }

    const ComputePipeline* GetComputePipeline();

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule,
                              std::optional<Shader::Gcn::FetchShaderData>, u64>;
    Result GetProgram(Shader::Stage stage, Shader::LogicalStage l_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::Stage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool TryRefreshGraphicsStagesForDynamicUserData(const GraphicsPipeline* pipeline);
    bool RefreshComputeKey();

    void DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::Stage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::Stage stage, Shader::LogicalStage l_stage);

    std::optional<Result> TryReuseGraphicsProgram(Shader::Stage stage, Shader::LogicalStage l_stage,
                                                  const Shader::ShaderParams& params,
                                                  Shader::Backend::Bindings& binding);
    const std::optional<Shader::Gcn::FetchShaderData>* GetCachedFetchShaderData(
        Program& program, Shader::Info& info, Shader::Stage stage);
    void RememberGraphicsProgram(Shader::LogicalStage l_stage, size_t program_hash,
                                 Program* program, size_t perm_idx);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

    const GraphicsPipeline* GetGraphicsPipelineSlow();

private:
    const Instance& instance;
    Scheduler& scheduler;
    const bool high_draw_call_optimization;
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start
    // Conservative graphics-pipeline reuse for high draw-call workloads.
    static constexpr u64 GraphicsPipelineRevisionValidationPeriod = 4096;
    bool graphics_pipeline_l1_valid{};
    bool graphics_pipeline_l1_disabled_due_to_mismatch{};
    bool graphics_pipeline_dynamic_ud_disabled_due_to_mismatch{};
    u64 graphics_pipeline_l1_revision{};
    u64 graphics_pipeline_l1_structural_revision{};
    const GraphicsPipeline* graphics_pipeline_l1_pipeline{};
    u64 graphics_pipeline_l1_validation_sequence{};
    u64 graphics_pipeline_dynamic_ud_validation_sequence{};

    struct GraphicsStageReuseEntry {
        bool valid{};
        size_t program_hash{};
        Program* program{};
        size_t perm_idx{};
    };
    std::array<GraphicsStageReuseEntry, MaxShaderStages> graphics_stage_reuse{};

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
};

} // namespace Vulkan
