// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include <boost/container/flat_map.hpp>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/resources/resource_ids.h"

namespace VideoCore {

struct BufferAccessState {
    vk::PipelineStageFlags2 stage_mask{vk::PipelineStageFlagBits2::eAllCommands};
    vk::AccessFlags2 access_mask{vk::AccessFlagBits2::eMemoryRead |
                                 vk::AccessFlagBits2::eMemoryWrite |
                                 vk::AccessFlagBits2::eTransferRead |
                                 vk::AccessFlagBits2::eTransferWrite};
    u32 state_version{1};
    u32 resource_generation{};
};

class BufferAccessStateStore {
public:
    [[nodiscard]] bool Contains(BufferId id) const noexcept {
        return id && id.index < states.size() && states[id.index].present;
    }

    [[nodiscard]] BufferAccessState GetSpecialAccessState(
        u64 resource_uid, vk::PipelineStageFlags2 initial_stage,
        vk::AccessFlags2 initial_access) const noexcept {
        const auto it = special_states.find(resource_uid);
        if (it != special_states.end()) {
            return it->second;
        }
        return {
            .stage_mask = initial_stage,
            .access_mask = initial_access,
        };
    }

    [[nodiscard]] BufferAccessState GetAccessState(BufferId id) const noexcept {
        if (Contains(id)) {
            return states[id.index].state;
        }
        return {};
    }

    [[nodiscard]] bool CanUpdateAccessState(BufferId id, u32 expected_state_version,
                                            u32 resource_generation) const noexcept {
        if (!Contains(id)) {
            return expected_state_version == 1;
        }
        const auto& state = states[id.index].state;
        return state.state_version == expected_state_version &&
               (state.resource_generation == 0 ||
                state.resource_generation == resource_generation);
    }

    void UpdateAccessState(BufferId id, u32 resource_generation,
                           vk::PipelineStageFlags2 stage_mask,
                           vk::AccessFlags2 access_mask) noexcept {
        auto& state = GetOrCreate(id);
        state.stage_mask = stage_mask;
        state.access_mask = access_mask;
        state.resource_generation = resource_generation;
        ++state.state_version;
    }

    void ResetResource(BufferId id, u32 resource_generation) noexcept {
        auto& slot = GetSlot(id);
        slot.state = BufferAccessState{
            .resource_generation = resource_generation,
        };
        slot.present = true;
    }

    void Clear() noexcept {
        states.clear();
        special_states.clear();
    }

    void RemoveResource(BufferId id) noexcept {
        if (id && id.index < states.size()) {
            states[id.index].present = false;
        }
    }

    [[nodiscard]] bool CanUpdateSpecialAccessState(
        u64 resource_uid, u32 expected_state_version) const noexcept {
        const auto it = special_states.find(resource_uid);
        const u32 version = it == special_states.end() ? 1 : it->second.state_version;
        return version == expected_state_version;
    }

    void UpdateSpecialAccessState(u64 resource_uid, vk::PipelineStageFlags2 stage_mask,
                                  vk::AccessFlags2 access_mask) noexcept {
        auto& state = special_states[resource_uid];
        state.stage_mask = stage_mask;
        state.access_mask = access_mask;
        ++state.state_version;
    }

private:
    struct Slot {
        BufferAccessState state{};
        bool present{};
    };

    Slot& GetSlot(BufferId id) {
        if (id.index >= states.size()) {
            states.resize(static_cast<size_t>(id.index) + 1);
        }
        return states[id.index];
    }

    BufferAccessState& GetOrCreate(BufferId id) {
        auto& slot = GetSlot(id);
        slot.present = true;
        return slot.state;
    }

    std::vector<Slot> states;
    // Special utility/stream buffers are few and are looked up from the serial commit path.
    // A flat map keeps their keys and states contiguous instead of allocating hash nodes.
    boost::container::flat_map<u64, BufferAccessState> special_states;
};

} // namespace VideoCore
