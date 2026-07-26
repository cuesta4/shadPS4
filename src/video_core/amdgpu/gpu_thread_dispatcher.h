// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/unique_function.h"

namespace AmdGpu {

class GpuThreadDispatcher {
public:
    virtual ~GpuThreadDispatcher() = default;
    virtual void ExecuteOnGpuThread(Common::UniqueFunction<void> command) = 0;
};

} // namespace AmdGpu
