// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <boost/container/static_vector.hpp>

#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

namespace {

struct PushConstantCache {
    bool initialized{};
    vk::CommandBuffer command_buffer{};
    vk::PipelineLayout pipeline_layout{};
    u64 scheduler_tick{};
    Shader::PushData push_data{};
};

PushConstantCache& GetPushConstantCache(const bool is_compute) {
    static thread_local PushConstantCache caches[2]{};
    return caches[is_compute ? 1U : 0U];
}

bool ShouldPushConstants(PushConstantCache& cache, const vk::CommandBuffer command_buffer,
                         const vk::PipelineLayout pipeline_layout, const u64 scheduler_tick,
                         const Shader::PushData& push_data) {
    if (!cache.initialized || cache.command_buffer != command_buffer ||
        cache.pipeline_layout != pipeline_layout || cache.scheduler_tick != scheduler_tick ||
        std::memcmp(&cache.push_data, &push_data, sizeof(push_data)) != 0) {
        cache.initialized = true;
        cache.command_buffer = command_buffer;
        cache.pipeline_layout = pipeline_layout;
        cache.scheduler_tick = scheduler_tick;
        std::memcpy(&cache.push_data, &push_data, sizeof(push_data));
        return true;
    }
    return false;
}

} // namespace

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      is_compute{is_compute_} {}

Pipeline::~Pipeline() = default;

void Pipeline::BindResources(DescriptorWrites& set_writes, const BufferBarriers& buffer_barriers,
                             const Shader::PushData& push_data) const {
    const auto cmdbuf = scheduler.CommandBuffer();
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;

    if (!buffer_barriers.empty()) {
        const auto dependencies = vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = u32(buffer_barriers.size()),
            .pBufferMemoryBarriers = buffer_barriers.data(),
        };
        scheduler.EndRendering();
        cmdbuf.pipelineBarrier2(dependencies);
    }

    const auto stage_flags = IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    if (!scheduler.IsHighDrawCallOptimization() ||
        ShouldPushConstants(GetPushConstantCache(IsCompute()), cmdbuf, *pipeline_layout,
                            scheduler.CurrentTick(), push_data)) {
        cmdbuf.pushConstants(*pipeline_layout, stage_flags, 0u, sizeof(push_data), &push_data);
    }

    // Bind descriptor set.
    if (set_writes.empty()) {
        return;
    }

    if (uses_push_descriptors) {
        cmdbuf.pushDescriptorSetKHR(bind_point, *pipeline_layout, 0, set_writes);
        if (bind_point == vk::PipelineBindPoint::eGraphics) {
            scheduler.NotifyGraphicsPushDescriptorSet();
        }
        return;
    }

    const auto desc_set = desc_heap.Commit(*desc_layout);
    for (auto& set_write : set_writes) {
        set_write.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(set_writes, {});
    cmdbuf.bindDescriptorSets(bind_point, *pipeline_layout, 0, desc_set, {});
}

std::string Pipeline::GetDebugString() const {
    std::string stage_desc;
    for (const auto& stage : stages) {
        if (stage) {
            const auto shader_name = PipelineCache::GetShaderName(stage->stage, stage->pgm_hash);
            if (stage_desc.empty()) {
                stage_desc = shader_name;
            } else {
                stage_desc = fmt::format("{},{}", stage_desc, shader_name);
            }
        }
    }
    return stage_desc;
}

} // namespace Vulkan
