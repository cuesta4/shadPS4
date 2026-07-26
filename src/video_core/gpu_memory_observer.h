// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

class GpuMemoryObserver {
public:
    virtual ~GpuMemoryObserver() = default;

    virtual bool InvalidateMemory(VAddr address, u64 size) = 0;
    virtual bool ReadMemory(VAddr address, u64 size) = 0;
    virtual bool IsMapped(VAddr address, u64 size) = 0;
    virtual void MapMemory(VAddr address, u64 size) = 0;
    virtual void UnmapMemory(VAddr address, u64 size) = 0;
};

} // namespace VideoCore
