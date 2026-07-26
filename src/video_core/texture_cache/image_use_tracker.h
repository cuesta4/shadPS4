// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <utility>

#include <boost/container/small_vector.hpp>

#include "video_core/texture_cache/types.h"

namespace VideoCore {

struct ImageUseState {
    bool is_bound : 1 {};
    bool is_target : 1 {};
    bool needs_rebind : 1 {};
    bool force_general : 1 {};
};

class ImageUseTracker {
public:
    ImageUseState& Get(ImageId image_id) {
        for (auto& [id, state] : states) {
            if (id == image_id) {
                return state;
            }
        }
        return states.emplace_back(image_id, ImageUseState{}).second;
    }

    const ImageUseState* Find(ImageId image_id) const noexcept {
        for (const auto& [id, state] : states) {
            if (id == image_id) {
                return &state;
            }
        }
        return nullptr;
    }

    bool IsBound(ImageId image_id) const noexcept {
        const auto* state = Find(image_id);
        return state && state->is_bound;
    }

    bool IsTarget(ImageId image_id) const noexcept {
        const auto* state = Find(image_id);
        return state && state->is_target;
    }

    bool NeedsRebind(ImageId image_id) const noexcept {
        const auto* state = Find(image_id);
        return state && state->needs_rebind;
    }

    void Erase(ImageId image_id) {
        for (auto it = states.begin(); it != states.end(); ++it) {
            if (it->first == image_id) {
                states.erase(it);
                return;
            }
        }
    }

    void Clear() noexcept {
        states.clear();
    }

    [[nodiscard]] size_t Size() const noexcept {
        return states.size();
    }

private:
    boost::container::small_vector<std::pair<ImageId, ImageUseState>, 8> states;
};

} // namespace VideoCore
