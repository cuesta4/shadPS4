// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <gtest/gtest.h>

#include "common/ring_buffer_queue.h"

TEST(RingBufferQueueTest, PopLatestOnEmptyQueueReturnsNothing) {
    RingBufferQueue<int> queue{4};

    EXPECT_FALSE(queue.PopLatest().has_value());
    EXPECT_EQ(queue.Size(), 0u);
}

TEST(RingBufferQueueTest, PopLatestReturnsNewestAndClearsQueue) {
    RingBufferQueue<int> queue{4};
    queue.Push(10);
    queue.Push(20);
    queue.Push(30);

    EXPECT_EQ(queue.PopLatest(), 30);
    EXPECT_EQ(queue.Size(), 0u);
    EXPECT_FALSE(queue.Pop().has_value());
}

TEST(RingBufferQueueTest, PopLatestHandlesWrappedFullQueue) {
    RingBufferQueue<int> queue{3};
    queue.Push(10);
    queue.Push(20);
    queue.Push(30);
    queue.Push(40);

    EXPECT_EQ(queue.PopLatest(), 40);
    EXPECT_EQ(queue.Size(), 0u);
}

TEST(RingBufferQueueTest, PopNewestDropsStaleEntriesAndPreservesOrder) {
    RingBufferQueue<int> queue{5};
    queue.Push(10);
    queue.Push(20);
    queue.Push(30);
    queue.Push(40);

    std::array<int, 2> output{};
    EXPECT_EQ(queue.PopNewest(output), 2u);
    EXPECT_EQ(output, (std::array{30, 40}));
    EXPECT_EQ(queue.Size(), 0u);
}

TEST(RingBufferQueueTest, EmptyOutputDoesNotConsumeHistory) {
    RingBufferQueue<int> queue{3};
    queue.Push(10);
    queue.Push(20);

    EXPECT_EQ(queue.PopNewest(std::span<int>{}), 0u);
    EXPECT_EQ(queue.Size(), 2u);
    EXPECT_EQ(queue.PopLatest(), 20);
}
