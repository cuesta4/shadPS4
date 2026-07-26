// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <vector>

#include "common/types.h"
#include "video_core/resources/resource_ids.h"

namespace VideoCore {

struct ImageLifetime {
    u32 backing_generation{};
    u32 view_generation{};
    bool present{};
};

/**
 * Dense image lifetime table indexed directly by ImageId::index.
 *
 * Address lookup belongs exclusively to TextureCache::page_table. This table only carries
 * lifetime data needed to validate prepared commands. Mutation is serial GPU-owner only.
 */
class ImageRegistry {
public:
    [[nodiscard]] const ImageLifetime* Find(ImageId id) const noexcept {
        if (!id || id.index >= lifetimes.size()) {
            return nullptr;
        }
        const auto& lifetime = lifetimes[id.index];
        return lifetime.present ? &lifetime : nullptr;
    }

    [[nodiscard]] bool Validate(ImageId id, u32 backing_generation) const noexcept {
        const auto* lifetime = Find(id);
        return lifetime != nullptr && backing_generation != 0 &&
               lifetime->backing_generation == backing_generation;
    }

    [[nodiscard]] bool Validate(ImageId id, u32 backing_generation,
                                u32 view_generation) const noexcept {
        const auto* lifetime = Find(id);
        return lifetime != nullptr && backing_generation != 0 && view_generation != 0 &&
               lifetime->backing_generation == backing_generation &&
               lifetime->view_generation == view_generation;
    }

    [[nodiscard]] u64 Epoch() const noexcept {
        return epoch;
    }

    u32 RegisterImage(ImageId id) {
        auto& lifetime = GetOrCreate(id);
        if (!lifetime.present) {
            lifetime.backing_generation = NextGeneration(lifetime.backing_generation);
            lifetime.view_generation = lifetime.backing_generation;
            lifetime.present = true;
            ++epoch;
        }
        return lifetime.backing_generation;
    }

    u32 BumpBackingGeneration(ImageId id) {
        auto* lifetime = FindMutable(id);
        if (!lifetime) {
            return 0;
        }
        lifetime->backing_generation = NextGeneration(lifetime->backing_generation);
        lifetime->view_generation = NextGeneration(lifetime->view_generation);
        ++epoch;
        return lifetime->backing_generation;
    }

    u32 BumpViewGeneration(ImageId id) {
        auto* lifetime = FindMutable(id);
        if (!lifetime) {
            return 0;
        }
        lifetime->view_generation = NextGeneration(lifetime->view_generation);
        ++epoch;
        return lifetime->view_generation;
    }

    void UnregisterImage(ImageId id) {
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

    ImageLifetime& GetOrCreate(ImageId id) {
        if (id.index >= lifetimes.size()) {
            lifetimes.resize(static_cast<size_t>(id.index) + 1);
        }
        return lifetimes[id.index];
    }

    [[nodiscard]] ImageLifetime* FindMutable(ImageId id) noexcept {
        if (!id || id.index >= lifetimes.size() || !lifetimes[id.index].present) {
            return nullptr;
        }
        return &lifetimes[id.index];
    }

    std::vector<ImageLifetime> lifetimes;
    u64 epoch{1};
};

} // namespace VideoCore
