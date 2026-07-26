// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

enum class DescriptorInfoKind : u8 {
    Buffer,
    Image,
};

struct DescriptorWriteRecord {
    u32 binding{};
    u32 array_element{};
    u32 descriptor_count{};
    u32 info_index{};
    vk::DescriptorType type{};
    DescriptorInfoKind info_kind{};
};

using DescriptorWritePlan = std::vector<DescriptorWriteRecord>;

} // namespace Vulkan
