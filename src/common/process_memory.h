// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Common {

/// Applies the configured resident-memory limit to the current process when supported.
void ConfigureProcessWorkingSetLimit(u32 limit_mib);

} // namespace Common
