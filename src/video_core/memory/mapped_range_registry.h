// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>

#include <boost/icl/interval_set.hpp>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "common/types.h"

namespace VideoCore {

/**
 * Thread-safe authority for GPU-visible guest virtual-address ranges.
 *
 * Lock order: callers must acquire this registry before entering cache-specific locks. Callbacks
 * passed to ForEach never run while a write lock is held.
 */
class MappedRangeRegistry {
public:
    [[nodiscard]] bool Contains(VAddr address, u64 size) const;
    void Map(VAddr address, u64 size);
    void Unmap(VAddr address, u64 size);

    template <typename Func>
    void ForEach(Func&& func) const {
        Common::RecursiveSharedLock lock{mutex};
        for (const auto& range : ranges) {
            func(range.lower(), range.upper() - range.lower());
        }
    }

    template <typename Func>
    void ForEachInRange(VAddr address, u64 size, Func&& func) const {
        if (size == 0 || address > std::numeric_limits<VAddr>::max() - size) {
            return;
        }
        const auto query = RangeSet::interval_type::right_open(address, address + size);
        Common::RecursiveSharedLock lock{mutex};
        for (const auto& range : (ranges & query)) {
            func(range.lower(), range.upper() - range.lower());
        }
    }

private:
    using RangeSet = boost::icl::interval_set<VAddr>;

    mutable Common::SharedFirstMutex mutex;
    RangeSet ranges;
};

} // namespace VideoCore
