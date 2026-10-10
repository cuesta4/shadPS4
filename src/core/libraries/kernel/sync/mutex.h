// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>

#include "common/types.h"

#ifdef _WIN64
#include <windows.h>
#else
#include <mutex>
#endif

namespace Libraries::Kernel {

class TimedMutex {
public:
    TimedMutex();
    ~TimedMutex();

    void lock();
    bool try_lock();

    void unlock();

    template <class Rep, class Period>
    bool try_lock_for(const std::chrono::duration<Rep, Period>& rel_time) {
#ifdef _WIN64
        constexpr auto zero = std::chrono::duration<Rep, Period>::zero();
        const auto now = std::chrono::steady_clock::now();

        std::chrono::steady_clock::time_point abs_time = now;
        if (rel_time > zero) {
            constexpr auto max = (std::chrono::steady_clock::time_point::max)();
            if (abs_time < max - rel_time) {
                abs_time += rel_time;
            } else {
                abs_time = max;
            }
        }

        return try_lock_until(abs_time);
#else
        return mtx.try_lock_for(rel_time);
#endif
    }

    template <class Clock, class Duration>
    bool try_lock_until(const std::chrono::time_point<Clock, Duration>& abs_time) {
#ifdef _WIN64
        if (try_lock()) {
            return true;
        }
        waiters.fetch_add(1, std::memory_order_seq_cst);
        bool acquired = false;
        while (!(acquired = TryAcquireWaiting())) {
            const auto now = Clock::now();
            if (abs_time <= now) {
                break;
            }
            const auto rel_ms = std::chrono::ceil<std::chrono::milliseconds>(abs_time - now);
            WaitForSingleObjectEx(
                wake, static_cast<DWORD>((std::min)(rel_ms.count(),
                                                    static_cast<decltype(rel_ms.count())>(
                                                        INFINITE - 1))),
                true);
        }
        waiters.fetch_sub(1, std::memory_order_relaxed);
        return acquired;
#else
        return mtx.try_lock_until(abs_time);
#endif
    }

private:
#ifdef _WIN64
    bool TryAcquireWaiting() {
        return locked.exchange(1, std::memory_order_seq_cst) == 0;
    }

    std::atomic<u32> locked{0};
    std::atomic<u32> waiters{0};
    HANDLE wake;
#else
    std::timed_mutex mtx;
#endif
};

} // namespace Libraries::Kernel
