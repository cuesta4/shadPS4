// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <coroutine>
#include <exception>

#include "common/assert.h"

namespace AmdGpu {

class GpuTask {
public:
    struct promise_type {
        GpuTask get_return_object() noexcept {
            return GpuTask{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        static constexpr std::suspend_always initial_suspend() noexcept {
            return {};
        }

        static constexpr std::suspend_always final_suspend() noexcept {
            return {};
        }

        void unhandled_exception() {
            try {
                std::rethrow_exception(std::current_exception());
            } catch (const std::exception& e) {
                UNREACHABLE_MSG("Unhandled exception: {}", e.what());
            }
        }

        void return_void() noexcept {}

        struct YieldToken {};

        std::suspend_always yield_value(YieldToken&&) noexcept {
            return {};
        }
    };

    using Handle = std::coroutine_handle<promise_type>;

    GpuTask() = default;
    explicit GpuTask(Handle handle_) : handle{handle_} {}

    Handle handle{};
};

} // namespace AmdGpu
