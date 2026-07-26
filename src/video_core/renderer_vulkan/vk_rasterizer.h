// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/gpu_commands/command_sink.h"
#include "video_core/memory/gpu_memory_coordinator.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/preparation/command_preparation_context.h"
#include "video_core/renderer_vulkan/preparation/command_requests.h"
#include "video_core/renderer_vulkan/preparation/graphics_request_builder.h"
#include "video_core/renderer_vulkan/execution/ordered_resource_committer.h"
#include "video_core/resources/resource_alias_coordinator.h"
#include "video_core/resources/transfer_buffer_pool.h"
#include "video_core/texture_cache/tile_manager.h"
#include "video_core/renderer_vulkan/vk_command_recorder.h"
#include "video_core/renderer_vulkan/vk_descriptor_binder.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
class GpuThreadDispatcher;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class Instance;
class Scheduler;
class RenderState;
class GraphicsPipeline;

class Rasterizer final : public VideoCore::GpuCommandSink {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler,
                        AmdGpu::GpuThreadDispatcher* gpu_thread_dispatcher);
    ~Rasterizer();

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Execute(const VideoCore::DrawCommand& command) override;
    void Execute(const VideoCore::DrawIndirectCommand& command) override;
    void Execute(const VideoCore::DispatchDirectCommand& command) override;
    void Execute(const VideoCore::DispatchIndirectCommand& command) override;
    void Execute(const VideoCore::FillBufferCommand& command) override;
    void Execute(const VideoCore::CopyBufferCommand& command) override;

    void ScopeMarkerBegin(std::string_view str, bool from_guest = false) override;
    void ScopeMarkerEnd(bool from_guest = false) override;
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(std::string_view str, u32 color,
                                 bool from_guest = false) override;

    u32 ReadDataFromGds(u32 gsd_offset) override;
    void ProcessDownloadImages() override;

    void CpSync() override;
    u64 Flush() override;
    void Finish() override;
    void OnSubmit() override;

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

private:
    void Draw(const VideoCore::DrawCommand& command);
    [[nodiscard]] bool PrepareDraw(const VideoCore::DrawCommand& command, u32 retry_count,
                                   CommandResolutionScratch& scratch);
    void DrawIndirect(const VideoCore::DrawIndirectCommand& command);
    [[nodiscard]] bool PrepareDrawIndirect(const VideoCore::DrawIndirectCommand& command,
                                           u32 retry_count, CommandResolutionScratch& scratch);
    void DispatchDirect(const VideoCore::DispatchDirectCommand& command);
    [[nodiscard]] bool PrepareDispatchDirect(const VideoCore::DispatchDirectCommand& command,
                                             u32 retry_count, CommandResolutionScratch& scratch);
    void DispatchIndirect(const VideoCore::DispatchIndirectCommand& command);
    [[nodiscard]] bool PrepareDispatchIndirect(
        const VideoCore::DispatchIndirectCommand& command, u32 retry_count,
        CommandResolutionScratch& scratch);
    void FillBuffer(VAddr address, u32 size, u32 value, bool is_gds);
    void CopyBuffer(VAddr destination, VAddr source, u32 size, bool destination_is_gds,
                    bool source_is_gds);

    void PrepareRenderState(CommandPreparationContext& context,
                            const VideoCore::CapturedGraphicsState& state, u32 mrt_mask);
    RenderState BeginRendering(CommandPreparationContext& context,
                               const GraphicsPipeline* pipeline,
                               const VideoCore::CapturedGraphicsState& state);
    void Resolve(const VideoCore::CapturedGraphicsState& state);
    void DepthStencilCopy(const VideoCore::CapturedGraphicsState& state, bool is_depth,
                          bool is_stencil);
    void EliminateFastClear(const VideoCore::CapturedGraphicsState& state);

    bool FilterDraw(const VideoCore::CapturedGraphicsState& state);

    void BindBuffers(CommandPreparationContext& context,
                     std::span<const BufferResourceRequest> requests,
                     const Shader::ShaderInvocationData& invocation,
                     Shader::Backend::Bindings& binding, Shader::PushData& push_data,
                     const VideoCore::CapturedComputeState* compute_state);
    void BindTextures(CommandPreparationContext& context,
                      std::span<const ImageResourceRequest> image_requests,
                      std::span<const SamplerResourceRequest> sampler_requests,
                      Shader::Backend::Bindings& binding,
                      const VideoCore::GraphicsResourceState& resource_state);
    bool BindResources(CommandPreparationContext& context, const Pipeline* pipeline,
                       const ShaderResourceRequestSet& requests,
                       const VideoCore::GraphicsResourceState& resource_state,
                       const VideoCore::GraphicsDynamicState* graphics_state,
                       const VideoCore::CapturedComputeState* compute_state);

    bool IsComputeMetaClear(const Pipeline* pipeline,
                            const Shader::ShaderInvocationData& invocation);
    bool IsComputeImageCopy(const Pipeline* pipeline,
                            const Shader::ShaderInvocationData& invocation,
                            const VideoCore::CapturedComputeState& state);
    bool IsComputeImageClear(const Pipeline* pipeline,
                             const Shader::ShaderInvocationData& invocation,
                             const VideoCore::CapturedComputeState& state);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::MappedRangeRegistry mapped_ranges;
    VideoCore::GpuMemoryCoordinator memory_coordinator;
    VideoCore::PageManager page_manager;
    VideoCore::TransferBufferPool transfer_buffers;
    VideoCore::TileManager tile_manager;
    VideoCore::BufferRegistry buffer_registry;
    VideoCore::ImageRegistry image_registry;
    VideoCore::BufferAccessStateStore buffer_access_states;
    VideoCore::ImageAccessStateStore image_access_states;
    VideoCore::ResourceAliasCoordinator alias_coordinator;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    Core::MemoryManager* memory;
    PipelineCache pipeline_cache;
    DescriptorBinder descriptor_binder;
    OrderedResourceCommitter resource_committer;
    VulkanCommandRecorder command_recorder;
    bool fault_process_pending{};
};

} // namespace Vulkan
