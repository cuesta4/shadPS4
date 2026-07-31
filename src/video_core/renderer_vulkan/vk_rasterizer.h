// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstring>
#include <vector>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class Scheduler;
class RenderState;
class GraphicsPipeline;

enum class GuestSyncDomain : u8 {
    ComputeShader,
    PixelShader,
    EndOfPipe,
};

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size);
    bool ReadMemory(VAddr addr, u64 size);
    bool ProcessDownloadImages();
    void InsertGuestSyncBarrier(GuestSyncDomain domain);
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    void CpSync();
    u64 Flush();
    void Finish();
    u64 FlushGuestCompletionPoint();
    void DeferGuestCompletion(u64 tick, Common::UniqueFunction<void>&& callback);
    void OnSubmit();

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);

    void ResetBindings() {
        for (auto& image_id : bound_images) {
            texture_cache.GetImage(image_id).binding = {};
        }
        bound_images.clear();
    }

    struct DescriptorWritePlan {
        bool valid{};
        const Pipeline* pipeline{};
    };

    void BeginDescriptorWritePlan(const Pipeline* pipeline);
    void EnsureDescriptorWriteCapacity(u32 required_size);
    void SetBufferDescriptorWrite(u32 binding, vk::DescriptorType type,
                                  const vk::DescriptorBufferInfo* info);
    void SetImageDescriptorWrite(u32 binding, u32 count, vk::DescriptorType type,
                                 const vk::DescriptorImageInfo* info);
    void ReserveOrSetImageDescriptorWrite(u32 binding, u32 count, vk::DescriptorType type,
                                          const vk::DescriptorImageInfo* info);
    void MaterializeDeferredImageDescriptorWrites();
    /// Returns true when the recorded partial-push state is still valid for this pipeline on the
    /// current command buffer (same texture generation and push-descriptor epoch).
    bool PartialPushStateMatches(const Pipeline* pipeline) const;
    void FinalizeDeferredImageWrites(const Pipeline* pipeline);
    void FinalizeDescriptorWritePlan(const Pipeline* pipeline);
    void BindPipelineResources(const Pipeline* pipeline);
    void ResetDescriptorPartialPushState();
    void BindGraphicsPipelineIfNeeded(vk::CommandBuffer cmdbuf, vk::Pipeline pipeline);
    void ResetCachedCommandBufferState();

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    const bool high_draw_call_optimization;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;

    using RenderTargetInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;
    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Pipeline::DescriptorWrites descriptor_partial_writes;
    Pipeline::BufferBarriers buffer_barriers;
    Shader::PushData push_data;

    struct DescriptorPartialPushState {
        bool valid{};
        const Pipeline* pipeline{};
        vk::CommandBuffer cmdbuf{};
        u64 texture_generation{};
        u64 graphics_push_descriptor_epoch{};
        u32 write_count{};
        boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos{};
    };
    DescriptorPartialPushState descriptor_partial_push_state{};

    struct DeferredImageDescriptorWrite {
        u32 write_index{};
        u32 image_info_index{};
    };
    boost::container::static_vector<DeferredImageDescriptorWrite,
                                    Shader::NUM_IMAGES + Shader::NUM_SAMPLERS>
        deferred_image_writes;
    bool descriptor_partial_materialization_candidate{};
    bool descriptor_partial_image_writes_omitted{};

    using BufferBindingInfo = std::tuple<VideoCore::BufferId, AmdGpu::Buffer, u64>;
    boost::container::static_vector<BufferBindingInfo, Shader::NUM_BUFFERS> buffer_bindings;
    using ImageBindingInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;

    struct CachedTextureView {
        VideoCore::ImageId image_id{};
        vk::Image backing_image{};
        vk::ImageView view{};
        bool valid{};
    };
    struct TextureDescriptorWritePlan {
        vk::DescriptorType type{};
        u32 count{};
    };
    struct TextureBindingPlan {
        bool valid{};
        const Shader::Info* stage{};
        u64 texture_generation{};
        VAddr border_color_address{};
        boost::container::static_vector<AmdGpu::Image, Shader::NUM_IMAGES> image_sharps{};
        boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings{};
        boost::container::small_vector<u32, 8> image_descriptor_array_sizes{};
        boost::container::small_vector<TextureDescriptorWritePlan, 8> descriptor_writes{};
        boost::container::static_vector<CachedTextureView, Shader::NUM_IMAGES> texture_views{};
        boost::container::static_vector<AmdGpu::Sampler, Shader::NUM_SAMPLERS> sampler_sharps{};
        boost::container::static_vector<vk::Sampler, Shader::NUM_SAMPLERS> samplers{};
    };
    std::array<TextureBindingPlan, Shader::MaxStageTypes> texture_binding_plans{};

    struct RenderTargetPlanKey {
        const GraphicsPipeline* pipeline{};
        std::array<AmdGpu::ColorBuffer, AmdGpu::NUM_COLOR_BUFFERS> color_buffers{};
        std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS> color_extents{};
        AmdGpu::ColorControl color_control{};
        AmdGpu::ColorBufferMask color_target_mask{};
        AmdGpu::DepthBuffer depth_buffer{};
        AmdGpu::DepthView depth_view{};
        AmdGpu::DepthControl depth_control{};
        AmdGpu::Address depth_htile_data_base{};
        AmdGpu::CbDbExtent depth_extent{};

        bool operator==(const RenderTargetPlanKey& other) const noexcept {
            return std::memcmp(this, &other, sizeof(*this)) == 0;
        }
    };

    struct CachedRenderTargetView {
        VideoCore::ImageId image_id{};
        vk::Image backing_image{};
        vk::ImageView view{};
        VideoCore::SubresourceRange range{};
        bool valid{};
    };

    struct RenderTargetStatePlan {
        bool valid{};
        u64 texture_generation{};
        RenderTargetPlanKey key{};
        std::array<CachedRenderTargetView, AmdGpu::NUM_COLOR_BUFFERS> color_views{};
        CachedRenderTargetView depth_view{};
    };

    /// Returns the render-target view for the image, reusing the cached handle when the plan is
    /// still valid; refreshes the cache entry otherwise.
    std::pair<vk::ImageView, VideoCore::SubresourceRange> ResolveRenderTargetView(
        CachedRenderTargetView& cached_view, VideoCore::ImageId image_id, VideoCore::Image& image,
        const VideoCore::TextureCache::ImageDesc& desc);

    RenderTargetStatePlan render_target_plan{};
    bool render_target_plan_hit{};

    DescriptorWritePlan descriptor_write_plan{};
    bool descriptor_write_plan_candidate{};
    bool descriptor_write_plan_hit{};

    bool fault_process_pending{};
    bool attachment_feedback_loop{};
    // Vulkan graphics pipeline state is persistent within a command buffer. Keep a tiny L1 cache
    // so repeated draws using the same pipeline do not emit redundant vkCmdBindPipeline calls.
    // The cache is invalidated whenever the scheduler submits or resets command buffer work.
    vk::CommandBuffer last_graphics_cmdbuf{nullptr};
    vk::Pipeline last_bound_graphics_pipeline{nullptr};
};

} // namespace Vulkan
