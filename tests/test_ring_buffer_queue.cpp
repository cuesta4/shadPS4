// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "common/ring_buffer_queue.h"

TEST(RingBufferQueueTest, PopOnEmptyQueueReturnsNothing) {
    RingBufferQueue<int> queue{4};

    EXPECT_FALSE(queue.Pop().has_value());
    EXPECT_EQ(queue.Size(), 0u);
}

TEST(RingBufferQueueTest, PopPreservesFifoOrder) {
    RingBufferQueue<int> queue{4};
    queue.Push(10);
    queue.Push(20);
    queue.Push(30);

    EXPECT_EQ(queue.Pop(), 10);
    EXPECT_EQ(queue.Pop(), 20);
    EXPECT_EQ(queue.Pop(), 30);
    EXPECT_FALSE(queue.Pop().has_value());
}

TEST(RingBufferQueueTest, OverflowDropsOldestEntry) {
    RingBufferQueue<int> queue{3};
    queue.Push(10);
    queue.Push(20);
    queue.Push(30);
    queue.Push(40);

    EXPECT_EQ(queue.Pop(), 20);
    EXPECT_EQ(queue.Pop(), 30);
    EXPECT_EQ(queue.Pop(), 40);
    EXPECT_FALSE(queue.Pop().has_value());
}

TEST(RingBufferQueueTest, ClearDiscardsHistory) {
    RingBufferQueue<int> queue{4};
    queue.Push(10);
    queue.Push(20);

    queue.Clear();

    EXPECT_EQ(queue.Size(), 0u);
    EXPECT_FALSE(queue.Pop().has_value());
}
