// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>

#include "common/performance_telemetry.h"
#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/pm4_cmds.h"
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
    bool ReadMemory(VAddr addr, u64 size, void* context = nullptr);
    bool HandleWriteFaultOnReadWatchedPage(VAddr addr, u64 size, void* context);
    void ArmSemanticReadWatch(VAddr addr, u64 size);
    void DisarmSemanticReadWatch(VAddr addr, u64 size);
    VideoCore::MemoryWriteNotifyResult NotifyMemoryWrite(VAddr addr, u64 size,
                                                         VideoCore::MemoryWriteSource source);
    [[nodiscard]] VideoCore::MemoryWriteWatch ArmMemoryWriteWatch(
        VAddr addr, VideoCore::MemoryWriteCallback callback, void* user_data) {
        return page_manager.ArmWriteWatch(addr, callback, user_data);
    }
    bool CancelMemoryWriteWatch(VideoCore::MemoryWriteWatch watch) {
        return page_manager.CancelWriteWatch(watch);
    }
    bool ProcessDownloadImages(const VideoCore::TextureCache::DownloadContext& context,
                               bool* gpu_resident = nullptr);
    bool ProcessDownloadImages(Common::PerformanceTelemetry::WritebackTrigger trigger,
                               u32 trigger_control = 0, u32 trigger_data_control = 0,
                               bool* gpu_resident = nullptr);
    void WaitTick(u64 tick, Common::PerformanceTelemetry::HostWaitReason reason =
                                Common::PerformanceTelemetry::HostWaitReason::Unknown);
    void DeferGpuCompletion(Common::UniqueFunction<void>&& callback,
                            const Common::PerformanceTelemetry::PendingOpTraceToken& trace = {});
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    void AcquireMemory(u32 cp_coher_cntl, VAddr base_address, u64 size);
    void FlushCaches(AmdGpu::EventType event_type);

    void CpSync();
    void GpuFenceWait();
    void FullGpuBarrier();
    [[nodiscard]] u64 CurrentTick() const noexcept;
    [[nodiscard]] u64 KnownGpuTick() const noexcept;
    u64 Flush(Common::PerformanceTelemetry::SubmitReason reason =
                  Common::PerformanceTelemetry::SubmitReason::Generic);
    void Finish();
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

    void PrepareBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    void FinalizeBuffers(Shader::PushData& push_data, bool stream_only);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);
    void SynchronizeDmaBuffers();
    void BindPipelineResources(const Pipeline* pipeline);
    void CaptureDescriptorState(const Pipeline* pipeline);
    void MarkImageWrites(Common::PerformanceTelemetry::ImageWriter writer,
                         bool include_render_targets);

    void ResetBindings() {
        for (auto& image_id : bound_images) {
            texture_cache.GetImage(image_id).binding = {};
        }
        bound_images.clear();
    }

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
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
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES>
        potential_write_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Pipeline::DescriptorWrites partial_set_writes;
    Pipeline::BufferBarriers buffer_barriers;
    Shader::PushData push_data;

    struct PendingBufferBinding {
        const Shader::BufferResource* desc{};
        VideoCore::BufferId buffer_id{};
        AmdGpu::Buffer sharp{};
        u64 size{};
        u64 alignment{};
        u32 unified_binding{};
        u32 buffer_binding{};
        u32 set_write_index{};
        u16 stream_index{std::numeric_limits<u16>::max()};
        VideoCore::BufferCache::StreamCopySource stream_source{};
        bool is_storage{};
        bool finalized{};
    };
    boost::container::static_vector<PendingBufferBinding, Shader::NUM_BUFFERS>
        pending_buffer_bindings;
    using ImageBindingInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;

    struct CachedBufferBinding {
        const Shader::Info* owner{};
        AmdGpu::Buffer sharp{};
        VideoCore::BufferId buffer_id{};
        u64 buffer_uid{};
        u64 topology_epoch{};
        u64 size{};
        bool valid{};
    };
    std::array<std::array<CachedBufferBinding, Shader::NUM_BUFFERS>, MaxShaderStages>
        cached_buffer_bindings{};

    struct CachedImageBinding {
        const Shader::Info* owner{};
        AmdGpu::Image sharp{};
        VideoCore::ImageId image_id{};
        u64 image_uid{};
        u64 topology_epoch{};
        VideoCore::TextureCache::ImageDesc resolved_desc{};
        bool valid{};
    };
    std::array<std::array<CachedImageBinding, Shader::NUM_IMAGES>, MaxShaderStages>
        cached_image_bindings{};

    struct CachedImageView {
        VideoCore::ImageId image_id{};
        u64 image_uid{};
        u64 topology_epoch{};
        vk::Image backing_image{};
        vk::ImageView image_view{};
        VideoCore::ImageViewInfo info{};
        bool valid{};
    };
    std::array<std::array<CachedImageView, Shader::NUM_IMAGES>, MaxShaderStages>
        cached_texture_views{};
    std::array<CachedImageView, AmdGpu::NUM_COLOR_BUFFERS> cached_color_target_views{};
    CachedImageView cached_depth_target_view{};

    struct DescriptorWriteState {
        u64 key0{};
        u64 key1{};
        u32 first_info{};
        bool is_buffer{};
    };

    struct DescriptorState {
        const Pipeline* pipeline{};
        vk::CommandBuffer command_buffer{};
        u64 push_descriptor_epoch{};
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
        u64 layout_signature{};
#endif
        boost::container::static_vector<DescriptorWriteState,
                                        Shader::NUM_BUFFERS + Shader::NUM_IMAGES +
                                            Shader::NUM_SAMPLERS>
            writes;
        boost::container::static_vector<vk::DescriptorImageInfo,
                                        Shader::NUM_IMAGES + Shader::NUM_SAMPLERS>
            image_infos;
        boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
        bool valid{};
    } descriptor_state;

    mutable u64 dynamic_state_generation{};
    mutable const GraphicsPipeline* dynamic_state_pipeline{};
    mutable bool dynamic_state_indexed{};
    mutable bool dynamic_state_feedback_loop{};
    Common::PerformanceTelemetry::Gate telemetry_enabled{};
    bool fault_process_pending{};
    bool attachment_feedback_loop{};
};

} // namespace Vulkan
