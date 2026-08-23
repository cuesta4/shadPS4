// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <variant>
#include <vector>
#include <tsl/robin_map.h>
#include "common/polyfill_thread.h"
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
struct ShaderCompileResult;

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    struct PendingCompilation {
        std::shared_ptr<ShaderCompileResult> result;
        std::future<void> completion;
    };

    Shader::Info info;
    ModuleList modules{};
    std::optional<PendingCompilation> pending_compilation;

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

    const ComputePipeline* GetComputePipeline();

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule,
                              std::optional<Shader::Gcn::FetchShaderData>, u64>;
    std::optional<Result> GetProgram(Shader::Stage stage, Shader::LogicalStage l_stage,
                                     const Shader::ShaderParams& params,
                                     Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::Stage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    const GraphicsPipeline* CreateGraphicsPipeline();
    const GraphicsPipeline* PublishGraphicsPipeline(
        tsl::robin_map<GraphicsPipelineKey,
                       std::future<std::unique_ptr<GraphicsPipeline>>>::iterator pending_it);
    void StartGraphicsPipelineCompiler();
    void StopGraphicsPipelineCompiler();
    void WaitForGraphicsPipelineCompiler();
    void GraphicsPipelineCompilerThread(u32 worker_index);
    void QueueGraphicsPipelineTask(std::packaged_task<void()>&& task);
    void QueueShaderModuleTask(std::packaged_task<void()>&& task);
    [[nodiscard]] std::vector<u8> LoadNativePipelineCache() const;
    void SaveNativePipelineCache();
    bool PublishProgramCompilation(Program& program);
    void PublishPendingProgramCompilations();
    void QueueProgramCompilation(Program& program, Shader::Stage stage,
                                 Shader::LogicalStage l_stage,
                                 const Shader::ShaderParams& params,
                                 const Shader::RuntimeInfo& runtime_info,
                                 Shader::Backend::Bindings binding,
                                 std::optional<Shader::StageSpecialization> specialization,
                                 size_t permutation_index, u64 permutation_hash,
                                 bool initial_program);

    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    void DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::Stage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding,
                                   ShaderCompileResult* async_result = nullptr);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::Stage stage, Shader::LogicalStage l_stage);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::future<std::unique_ptr<GraphicsPipeline>>>
        pending_graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start
    bool async_shader_recompiling{};

    static constexpr u32 NumGraphicsPipelineWorkers = 6;
    static constexpr u32 NumShaderModulePreferredWorkers = 3;
    static constexpr u32 NativePipelineCacheSaveBatch = 8;
    static_assert(NumShaderModulePreferredWorkers < NumGraphicsPipelineWorkers);
    std::array<std::jthread, NumGraphicsPipelineWorkers> graphics_pipeline_workers;
    std::deque<std::packaged_task<void()>> shader_module_tasks;
    std::deque<std::packaged_task<void()>> graphics_pipeline_tasks;
    std::mutex graphics_pipeline_tasks_mutex;
    std::condition_variable graphics_pipeline_tasks_cv;
    size_t graphics_pipeline_tasks_in_flight{};
    bool graphics_pipeline_compiler_stopping{};
    std::mutex native_pipeline_cache_mutex;
    std::atomic_bool native_pipeline_cache_dirty{};
    std::atomic_bool native_pipeline_cache_save_requested{};
    std::atomic<u32> native_pipeline_cache_updates{};

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
};

} // namespace Vulkan
