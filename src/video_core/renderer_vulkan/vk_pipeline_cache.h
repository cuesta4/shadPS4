// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <tsl/robin_map.h>
#include "common/assert.h"
#include "shader_recompiler/invocation.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/shader_program_analysis.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/gpu_commands/captured_state.h"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

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

    std::unique_ptr<const Shader::ShaderProgramAnalysis> analysis;
    ModuleList modules{};

    explicit Program(Shader::Info&& completed_analysis)
        : analysis{std::make_unique<const Shader::ShaderProgramAnalysis>(
              std::move(completed_analysis))} {}

    [[nodiscard]] const Shader::Info& Metadata() const noexcept {
        ASSERT(analysis != nullptr);
        return analysis->Metadata();
    }

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        modules.emplace_back(module, std::move(spec));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        modules[perm_idx] = {module, std::move(spec)};
    }
};

using ShaderInvocationSet = std::array<Shader::ShaderInvocationData, MaxShaderStages>;

struct PipelineBuildContext {
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    ShaderInvocationSet invocations{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
};


struct GraphicsPipelineLookup {
    const GraphicsPipeline* pipeline{};
    ShaderInvocationSet invocations{};

    explicit operator bool() const noexcept {
        return pipeline != nullptr;
    }
};

struct ComputePipelineLookup {
    const ComputePipeline* pipeline{};
    Shader::ShaderInvocationData invocation{};

    explicit operator bool() const noexcept {
        return pipeline != nullptr;
    }
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler);
    ~PipelineCache();

    void WarmUp();
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage,
                           PipelineBuildContext& context);

    GraphicsPipelineLookup GetGraphicsPipeline(const VideoCore::CapturedGraphicsState& state);

    ComputePipelineLookup GetComputePipeline(const VideoCore::CapturedComputeState& state);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::Stage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    bool RefreshGraphicsKey(PipelineBuildContext& context,
                            const VideoCore::CapturedGraphicsState& state);
    bool RefreshGraphicsStages(PipelineBuildContext& context,
                               const VideoCore::CapturedGraphicsState& state);
    bool RefreshComputeKey(PipelineBuildContext& context,
                           const VideoCore::CapturedComputeState& state);

    void DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::Stage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(
        PipelineBuildContext& context, const VideoCore::CapturedGraphicsState* graphics_state,
        const VideoCore::CapturedComputeState* compute_state, Shader::Stage stage,
        Shader::LogicalStage l_stage);

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule,
                              std::optional<Shader::Gcn::FetchShaderData>, u64>;
    Result GetProgram(PipelineBuildContext& context, Shader::Stage stage,
                      Shader::LogicalStage l_stage, const Shader::ShaderParams& params,
                      Shader::Backend::Bindings& binding,
                      const VideoCore::CapturedGraphicsState* graphics_state,
                      const VideoCore::CapturedComputeState* compute_state);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
};

} // namespace Vulkan
