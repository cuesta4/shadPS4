// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/buffer_access.h"

namespace {

TEST(BufferAccessTrackerTest, MergesReadOnlyConsumersWithoutBarrier) {
    VideoCore::BufferAccessTracker tracker{vk::AccessFlagBits2::eShaderRead,
                                           vk::PipelineStageFlagBits2::eVertexShader};

    EXPECT_FALSE(tracker.TransitionTo(vk::AccessFlagBits2::eShaderRead,
                                      vk::PipelineStageFlagBits2::eFragmentShader));
    EXPECT_EQ(tracker.Access(), vk::AccessFlagBits2::eShaderRead);
    EXPECT_EQ(tracker.Stages(), vk::PipelineStageFlagBits2::eVertexShader |
                                    vk::PipelineStageFlagBits2::eFragmentShader);
}

TEST(BufferAccessTrackerTest, LaterWriteWaitsForEveryAccumulatedReader) {
    VideoCore::BufferAccessTracker tracker{vk::AccessFlagBits2::eVertexAttributeRead,
                                           vk::PipelineStageFlagBits2::eVertexAttributeInput};
    ASSERT_FALSE(tracker.TransitionTo(vk::AccessFlagBits2::eShaderRead,
                                      vk::PipelineStageFlagBits2::eFragmentShader));

    const auto transition = tracker.TransitionTo(vk::AccessFlagBits2::eShaderWrite,
                                                 vk::PipelineStageFlagBits2::eComputeShader);
    ASSERT_TRUE(transition);
    EXPECT_EQ(transition->source_access,
              vk::AccessFlagBits2::eVertexAttributeRead | vk::AccessFlagBits2::eShaderRead);
    EXPECT_EQ(transition->source_stages, vk::PipelineStageFlagBits2::eVertexAttributeInput |
                                             vk::PipelineStageFlagBits2::eFragmentShader);
    EXPECT_EQ(transition->destination_access, vk::AccessFlagBits2::eShaderWrite);
    EXPECT_EQ(transition->destination_stages, vk::PipelineStageFlagBits2::eComputeShader);
}

TEST(BufferAccessTrackerTest, PreservesWriteHazards) {
    VideoCore::BufferAccessTracker tracker{vk::AccessFlagBits2::eTransferWrite,
                                           vk::PipelineStageFlagBits2::eTransfer};

    const auto transition = tracker.TransitionTo(vk::AccessFlagBits2::eShaderRead,
                                                 vk::PipelineStageFlagBits2::eFragmentShader);
    ASSERT_TRUE(transition);
    EXPECT_EQ(transition->source_access, vk::AccessFlagBits2::eTransferWrite);
    EXPECT_EQ(transition->destination_access, vk::AccessFlagBits2::eShaderRead);
}

} // namespace
