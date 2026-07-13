// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "common/types.h"

template <class T>
class RingBufferQueue {
public:
    RingBufferQueue(u64 size) : m_storage(size) {}

    void Push(T item) {
        const u64 index = (m_begin + m_size) % m_storage.size();
        m_storage[index] = std::move(item);
        if (m_size < m_storage.size()) {
            m_size += 1;
        } else {
            m_begin = (m_begin + 1) % m_storage.size();
        }
    }

    std::optional<T> Pop() {
        if (m_size == 0) {
            return {};
        }
        const u64 index = m_begin;
        m_begin = (m_begin + 1) % m_storage.size();
        m_size -= 1;
        return std::move(m_storage[index]);
    }

    std::optional<T> Peek() {
        if (m_size == 0) {
            return {};
        }
        return m_storage[m_begin];
    }

    std::optional<T> PopLatest() {
        if (m_size == 0) {
            return {};
        }
        const u64 index = (m_begin + m_size - 1) % m_storage.size();
        m_begin = 0;
        m_size = 0;
        return std::move(m_storage[index]);
    }

    /// Removes up to output.size() newest entries while preserving chronological order.
    /// Older entries that do not fit are discarded so consumers never receive stale backlog.
    u64 PopNewest(std::span<T> output) {
        if (m_size == 0 || output.empty()) {
            return 0;
        }
        const u64 count = std::min<u64>(m_size, output.size());
        const u64 discard = m_size - count;
        m_begin = (m_begin + discard) % m_storage.size();
        m_size = count;
        for (u64 i = 0; i < count; ++i) {
            output[i] = std::move(*Pop());
        }
        return count;
    }

    u64 Size() const {
        return m_size;
    }

    void Clear() {
        m_begin = 0;
        m_size = 0;
    }

private:
    u64 m_begin = 0;
    u64 m_size = 0;
    std::vector<T> m_storage;
};
