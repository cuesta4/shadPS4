// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace Core::Startup {

enum class Stage { Inactive, Libraries, Shaders, Executable, Modules, Launch, FirstFrame, Complete };

class Progress {
public:
    void Begin() {
        started_ms.store(NowMs(), std::memory_order_relaxed);
        stage.store(Stage::Libraries, std::memory_order_release);
    }

    /// Advances the stage while startup is active; Stage::Complete ends it for good.
    void SetStage(Stage next) {
        auto current = stage.load(std::memory_order_relaxed);
        while (current != Stage::Inactive && current != Stage::Complete) {
            if (stage.compare_exchange_weak(current, next, std::memory_order_release,
                                            std::memory_order_relaxed)) {
                return;
            }
        }
    }

    Stage GetStage() const {
        return stage.load(std::memory_order_acquire);
    }

    /// Startup lasts at least MinimumMs so the splash is never just a flash.
    bool IsActive() const {
        const auto current = GetStage();
        return current != Stage::Inactive &&
               (current != Stage::Complete || ElapsedMs() < MinimumMs);
    }

    /// Ends startup once the guest flipped for two seconds without a pause, even if the frames
    /// never looked visible. Called only by the presentation thread.
    void GuestFlip() {
        const std::int64_t now = NowMs();
        if (now - last_flip_ms > 250) {
            flip_run_start_ms = now;
        } else if (now - flip_run_start_ms >= 2000) {
            SetStage(Stage::Complete);
        }
        last_flip_ms = now;
    }

    std::int64_t ElapsedMs() const {
        return NowMs() - started_ms.load(std::memory_order_relaxed);
    }

private:
    static constexpr std::int64_t MinimumMs = 3000;

    static std::int64_t NowMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    std::atomic<Stage> stage{Stage::Inactive};
    std::atomic<std::int64_t> started_ms{};
    std::int64_t last_flip_ms{};
    std::int64_t flip_run_start_ms{};
};

inline Progress progress;

} // namespace Core::Startup
