// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "mutex.h"

#include "common/assert.h"

namespace Libraries::Kernel {

TimedMutex::TimedMutex() {
#ifdef _WIN64
    wake = CreateEvent(nullptr, false, false, nullptr);
    ASSERT(wake);
#endif
}

TimedMutex::~TimedMutex() {
#ifdef _WIN64
    CloseHandle(wake);
#endif
}

void TimedMutex::lock() {
#ifdef _WIN64
    if (try_lock()) {
        return;
    }
    waiters.fetch_add(1, std::memory_order_seq_cst);
    while (!TryAcquireWaiting()) {
        WaitForSingleObjectEx(wake, INFINITE, true);
    }
    waiters.fetch_sub(1, std::memory_order_relaxed);
#else
    mtx.lock();
#endif
}

bool TimedMutex::try_lock() {
#ifdef _WIN64
    u32 expected = 0;
    return locked.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                          std::memory_order_relaxed);
#else
    return mtx.try_lock();
#endif
}

void TimedMutex::unlock() {
#ifdef _WIN64
    locked.store(0, std::memory_order_seq_cst);
    if (waiters.load(std::memory_order_seq_cst) != 0) {
        SetEvent(wake);
    }
#else
    mtx.unlock();
#endif
}

} // namespace Libraries::Kernel