// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::Kernel {

void TestSetSdkVersion(s32 ver);
void TestResetSdkVersion();
void TestSetProcessTime(u64 time);
void TestResetProcessTime();

} // namespace Libraries::Kernel
