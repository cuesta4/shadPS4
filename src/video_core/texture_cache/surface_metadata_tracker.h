// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <tsl/robin_map.h>

#include "common/types.h"

namespace VideoCore {

class SurfaceMetadataTracker {
public:
    enum class Type {
        CMask,
        FMask,
        HTile,
    };

    void Register(VAddr address, Type type, s32 clear_mask = -1) {
        if (address != 0) {
            surfaces.emplace(address, Entry{.type = type, .clear_mask = clear_mask});
        }
    }

    void Unregister(VAddr address) {
        if (address != 0) {
            surfaces.erase(address);
        }
    }

    [[nodiscard]] bool Contains(VAddr address) const {
        return surfaces.contains(address);
    }

    [[nodiscard]] bool IsCleared(VAddr address, u32 slice) const {
        const auto it = surfaces.find(address);
        return it != surfaces.end() && (it->second.clear_mask & (1u << slice)) != 0;
    }

    bool Clear(VAddr address) {
        auto it = surfaces.find(address);
        if (it == surfaces.end()) {
            return false;
        }
        it.value().clear_mask = u32(-1);
        return true;
    }

    bool Touch(VAddr address, u32 slice, bool is_clear) {
        auto it = surfaces.find(address);
        if (it == surfaces.end()) {
            return false;
        }
        if (is_clear) {
            it.value().clear_mask |= 1u << slice;
        } else {
            it.value().clear_mask &= ~(1u << slice);
        }
        return true;
    }

private:
    struct Entry {
        Type type;
        s32 clear_mask{-1};
    };

    tsl::robin_map<VAddr, Entry> surfaces;
};

} // namespace VideoCore
