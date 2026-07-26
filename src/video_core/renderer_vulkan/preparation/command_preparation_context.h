// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <limits>
#include <utility>
#include <vector>

#include "common/assert.h"
#include "shader_recompiler/resource.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/renderer_vulkan/preparation/command_requests.h"
#include "video_core/renderer_vulkan/preparation/resource_access_plan.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/resources/resolved_resource_refs.h"
#include "video_core/texture_cache/image_use_tracker.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

struct CommandResolutionScratch {
    using RenderTargetInfo =
        std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    using ImageBindingInfo =
        std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;

    static constexpr u8 InvalidTargetIndex = std::numeric_limits<u8>::max();

    std::array<u8, AmdGpu::NUM_COLOR_BUFFERS> color_target_indices{};
    u8 depth_target_index{InvalidTargetIndex};
    std::vector<RenderTargetInfo> render_targets;
    std::vector<VideoCore::ImageId> bound_images;
    std::vector<ImageBindingInfo> image_bindings;
    ShaderResourceRequestSet resource_requests;
    VideoCore::ImageUseTracker image_uses;

    [[nodiscard]] RenderTargetInfo* ColorTarget(u32 slot) noexcept {
        const u8 index = color_target_indices[slot];
        return index == InvalidTargetIndex ? nullptr : &render_targets[index];
    }

    [[nodiscard]] RenderTargetInfo* DepthTarget() noexcept {
        return depth_target_index == InvalidTargetIndex
                   ? nullptr
                   : &render_targets[depth_target_index];
    }

    RenderTargetInfo& AddColorTarget(u32 slot, VideoCore::TextureCache::ImageDesc desc) {
        ASSERT(slot < color_target_indices.size());
        ASSERT(render_targets.size() < InvalidTargetIndex);
        color_target_indices[slot] = static_cast<u8>(render_targets.size());
        return render_targets.emplace_back(VideoCore::ImageId{}, std::move(desc));
    }

    RenderTargetInfo& AddDepthTarget(VideoCore::TextureCache::ImageDesc desc) {
        ASSERT(render_targets.size() < InvalidTargetIndex);
        depth_target_index = static_cast<u8>(render_targets.size());
        return render_targets.emplace_back(VideoCore::ImageId{}, std::move(desc));
    }

    void Reset() {
        color_target_indices.fill(InvalidTargetIndex);
        depth_target_index = InvalidTargetIndex;
        render_targets.clear();
        bound_images.clear();
        image_bindings.clear();
        resource_requests.Reset();
        image_uses.Clear();
    }
};

struct PreparedResourcePayload {
    std::vector<vk::DescriptorImageInfo> image_infos;
    std::vector<vk::DescriptorBufferInfo> buffer_infos;
    std::vector<VideoCore::ResolvedBufferRef> resolved_buffers;
    std::vector<VideoCore::ResolvedImageRef> resolved_images;

    Pipeline::DescriptorWrites descriptor_writes;
    Pipeline::BufferBarriers buffer_barriers;
    BufferAccessPlan buffer_accesses;
    ImageTransitionPlan image_transitions;
    ImageBarrierPlan image_barriers;
    Shader::PushData push_data{};
    u64 buffer_registry_epoch{};
    u64 image_registry_epoch{};
    bool attachment_feedback_loop{};

    void Reset() {
        image_infos.clear();
        buffer_infos.clear();
        resolved_buffers.clear();
        resolved_images.clear();
        descriptor_writes.clear();
        buffer_barriers.clear();
        buffer_accesses.clear();
        image_transitions.clear();
        image_barriers.clear();
        push_data = {};
        buffer_registry_epoch = 0;
        image_registry_epoch = 0;
        attachment_feedback_loop = false;
    }
};

struct CommandPreparationContext : PreparedResourcePayload {
    CommandResolutionScratch* resolution{};

    explicit CommandPreparationContext(CommandResolutionScratch& scratch)
        : resolution{&scratch} {
        scratch.Reset();
    }

    CommandPreparationContext() = default;

    void Reset(CommandResolutionScratch& scratch) {
        PreparedResourcePayload::Reset();
        resolution = &scratch;
        scratch.Reset();
    }

    void Seal() noexcept {
        resolution = nullptr;
    }

    [[nodiscard]] PreparedResourcePayload Finish() && {
        Seal();
        return std::move(static_cast<PreparedResourcePayload&>(*this));
    }
};

} // namespace Vulkan
