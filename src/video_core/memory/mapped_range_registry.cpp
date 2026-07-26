// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/memory/mapped_range_registry.h"

namespace VideoCore {

bool MappedRangeRegistry::Contains(VAddr address, u64 size) const {
    if (size == 0 || address > std::numeric_limits<VAddr>::max() - size) {
        return false;
    }
    const auto range = RangeSet::interval_type::right_open(address, address + size);
    Common::RecursiveSharedLock lock{mutex};
    return boost::icl::contains(ranges, range);
}

void MappedRangeRegistry::Map(VAddr address, u64 size) {
    if (size == 0 || address > std::numeric_limits<VAddr>::max() - size) {
        return;
    }
    std::scoped_lock lock{mutex};
    ranges += RangeSet::interval_type::right_open(address, address + size);
}

void MappedRangeRegistry::Unmap(VAddr address, u64 size) {
    if (size == 0 || address > std::numeric_limits<VAddr>::max() - size) {
        return;
    }
    std::scoped_lock lock{mutex};
    ranges -= RangeSet::interval_type::right_open(address, address + size);
}

} // namespace VideoCore
