// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "common/bit_array.h"
#include "video_core/amdgpu/gpu_queue.h"
#include "video_core/amdgpu/register_file.h"
#include "video_core/buffer_cache/buffer_registry.h"
#include "video_core/renderer_vulkan/execution/graphics_operation_classifier.h"
#include "video_core/texture_cache/image_use_tracker.h"
#include "video_core/texture_cache/image_registry.h"
#include "video_core/texture_cache/surface_metadata_tracker.h"

namespace {

TEST(BitArray, TestsOnlyTheRequestedWordRange) {
    Common::BitArray<256> bits;
    bits.Set(3);
    bits.Set(65);
    bits.Set(191);
    bits.Set(255);

    EXPECT_FALSE(bits.Any(4, 65));
    EXPECT_TRUE(bits.Any(64, 66));
    EXPECT_FALSE(bits.Any(66, 191));
    EXPECT_TRUE(bits.Any(191, 192));
    EXPECT_FALSE(bits.Any(192, 255));
    EXPECT_TRUE(bits.Any(255, 256));
    EXPECT_FALSE(bits.Any(32, 32));
}

TEST(GpuRegisterFile, WritesRegisterDomainsAtExpectedOffsets) {
    AmdGpu::GpuRegisterFile registers;
    const std::array<u32, 3> values{0x11223344, 0x55667788, 0xAABBCCDD};

    registers.WriteContext(7, values);

    const auto& raw = registers.Raw().reg_array;
    EXPECT_EQ(raw[AmdGpu::Regs::ContextRegWordOffset + 7], values[0]);
    EXPECT_EQ(raw[AmdGpu::Regs::ContextRegWordOffset + 8], values[1]);
    EXPECT_EQ(raw[AmdGpu::Regs::ContextRegWordOffset + 9], values[2]);
}

TEST(GpuQueueManager, KeepsQueueStateIndependent) {
    AmdGpu::GpuQueueManager<2> queues;
    queues[0].compute_state.dim_x = 3;
    queues[1].compute_state.dim_x = 9;

    EXPECT_EQ(queues[0].compute_state.dim_x, 3u);
    EXPECT_EQ(queues[1].compute_state.dim_x, 9u);
    EXPECT_EQ(decltype(queues)::Size(), 2u);
}

TEST(ResourceRegistries, RejectStaleGenerationsAfterLifetimeChanges) {
    VideoCore::BufferRegistry buffers;
    const VideoCore::BufferId buffer_id{3};
    const u32 first_buffer_generation = buffers.RegisterBuffer(buffer_id);
    EXPECT_TRUE(buffers.Validate(buffer_id, first_buffer_generation));
    buffers.UnregisterBuffer(buffer_id);
    const u32 second_buffer_generation = buffers.RegisterBuffer(buffer_id);
    EXPECT_NE(first_buffer_generation, second_buffer_generation);
    EXPECT_FALSE(buffers.Validate(buffer_id, first_buffer_generation));

    VideoCore::ImageRegistry images;
    const VideoCore::ImageId image_id{7};
    const u32 first_image_generation = images.RegisterImage(image_id);
    const u32 second_image_generation = images.BumpBackingGeneration(image_id);
    EXPECT_NE(first_image_generation, second_image_generation);
    EXPECT_FALSE(images.Validate(image_id, first_image_generation));
    EXPECT_TRUE(images.Validate(image_id, second_image_generation));
}

TEST(GraphicsOperationClassifier, PreservesPrimitiveNoneDecision) {
    VideoCore::CapturedGraphicsState state{};
    state.pipeline.color_control.mode = AmdGpu::ColorControl::OperationMode::Normal;
    state.pipeline.primitive_type = AmdGpu::PrimitiveType::None;

    const auto result = Vulkan::ClassifyGraphicsOperation(state);
    EXPECT_EQ(result.operation, Vulkan::GraphicsOperation::SkipPrimitiveNone);
}

TEST(GraphicsOperationClassifier, PreservesFastClearDecision) {
    VideoCore::CapturedGraphicsState state{};
    state.pipeline.color_control.mode =
        AmdGpu::ColorControl::OperationMode::EliminateFastClear;

    const auto result = Vulkan::ClassifyGraphicsOperation(state);
    EXPECT_EQ(result.operation, Vulkan::GraphicsOperation::EliminateFastClear);
}

TEST(ImageUseTracker, IsLocalAndResettable) {
    VideoCore::ImageUseTracker tracker;
    const VideoCore::ImageId image_id{1};
    auto& state = tracker.Get(image_id);
    state.is_bound = true;
    state.needs_rebind = true;

    EXPECT_TRUE(tracker.IsBound(image_id));
    EXPECT_TRUE(tracker.NeedsRebind(image_id));
    tracker.Clear();
    EXPECT_FALSE(tracker.IsBound(image_id));
}

TEST(SurfaceMetadataTracker, TracksClearStatePerSlice) {
    VideoCore::SurfaceMetadataTracker tracker;
    constexpr VAddr address = 0x10000;
    tracker.Register(address, VideoCore::SurfaceMetadataTracker::Type::HTile, 0);

    EXPECT_FALSE(tracker.IsCleared(address, 2));
    EXPECT_TRUE(tracker.Touch(address, 2, true));
    EXPECT_TRUE(tracker.IsCleared(address, 2));
    EXPECT_TRUE(tracker.Clear(address));
    EXPECT_TRUE(tracker.IsCleared(address, 7));
    tracker.Unregister(address);
    EXPECT_FALSE(tracker.Contains(address));
}

} // namespace
