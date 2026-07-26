// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <vector>

#include "common/types.h"
#include "video_core/resources/resource_ids.h"

namespace VideoCore {

struct BufferLifetime {
    u32 generation{};
    bool present{};
};

/**
 * Dense buffer lifetime table indexed directly by BufferId::index.
 *
 * BufferCache::page_table is the sole guest-address index. Mutation is serial GPU-owner only.
 */
class BufferRegistry {
public:
    [[nodiscard]] const BufferLifetime* Find(BufferId id) const noexcept {
        if (!id || id.index >= lifetimes.size()) {
            return nullptr;
        }
        const auto& lifetime = lifetimes[id.index];
        return lifetime.present ? &lifetime : nullptr;
    }

    [[nodiscard]] u32 Generation(BufferId id) const noexcept {
        const auto* lifetime = Find(id);
        return lifetime ? lifetime->generation : 0;
    }

    [[nodiscard]] bool Validate(BufferId id, u32 generation) const noexcept {
        return generation != 0 && Generation(id) == generation;
    }

    [[nodiscard]] u64 Epoch() const noexcept {
        return epoch;
    }

    u32 RegisterBuffer(BufferId id) {
        auto& lifetime = GetOrCreate(id);
        if (!lifetime.present) {
            lifetime.generation = NextGeneration(lifetime.generation);
            lifetime.present = true;
            ++epoch;
        }
        return lifetime.generation;
    }

    void UnregisterBuffer(BufferId id) {
        auto* lifetime = FindMutable(id);
        if (!lifetime) {
            return;
        }
        lifetime->present = false;
        ++epoch;
    }

private:
    [[nodiscard]] static u32 NextGeneration(u32 generation) noexcept {
        return generation == std::numeric_limits<u32>::max() ? 1 : generation + 1;
    }

    BufferLifetime& GetOrCreate(BufferId id) {
        if (id.index >= lifetimes.size()) {
            lifetimes.resize(static_cast<size_t>(id.index) + 1);
        }
        return lifetimes[id.index];
    }

    [[nodiscard]] BufferLifetime* FindMutable(BufferId id) noexcept {
        if (!id || id.index >= lifetimes.size() || !lifetimes[id.index].present) {
            return nullptr;
        }
        return &lifetimes[id.index];
    }

    std::vector<BufferLifetime> lifetimes;
    u64 epoch{1};
};

} // namespace VideoCore
