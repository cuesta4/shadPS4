// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>
#include "common/types.h"

namespace Shader::Backend::SPIRV {

void ConvertRawAccessChains(std::vector<u32>& code);

} // namespace Shader::Backend::SPIRV
