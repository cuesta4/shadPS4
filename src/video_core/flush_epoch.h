// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "common/types.h"

namespace VideoCore {

/// Counts the points where the guest orders GPU work against earlier GPU work.
///
/// Two accesses of a resource that no such point separates are not ordered by the guest either:
/// on the console they may overlap, so the host needs no barrier between them. A resource
/// records the epochs of its last write, and a barrier that only orders accesses of the same
/// kind is emitted when an epoch passed since then.
class FlushEpoch {
public:
    /// Cache flush and invalidation packets: EVENT_WRITE flushes and ACQUIRE_MEM.
    [[nodiscard]] static u64 Cache() noexcept {
        return cache.load(std::memory_order_relaxed);
    }

    /// Cache packets plus every packet the command processor waits on or signals completion
    /// with: EOP, EOS, RELEASE_MEM, WAIT_REG_MEM, and guest submissions.
    [[nodiscard]] static u64 Sync() noexcept {
        return sync.load(std::memory_order_relaxed);
    }

    static void AdvanceCache() noexcept {
        cache.fetch_add(1, std::memory_order_relaxed);
        sync.fetch_add(1, std::memory_order_relaxed);
    }

    static void AdvanceSync() noexcept {
        sync.fetch_add(1, std::memory_order_relaxed);
    }

private:
    static inline std::atomic<u64> cache{1};
    static inline std::atomic<u64> sync{1};
};

} // namespace VideoCore
