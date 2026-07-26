// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

namespace Vulkan {

class Instance;
class Scheduler;

class DescriptorBinder {
public:
    DescriptorBinder(const Instance& instance, Scheduler& scheduler);

    void Bind(const Pipeline& pipeline, const Pipeline::DescriptorWrites& writes,
              std::span<const vk::DescriptorBufferInfo> buffer_infos,
              std::span<const vk::DescriptorImageInfo> image_infos,
              const Pipeline::BufferBarriers& barriers,
              const Shader::PushData& push_data, vk::CommandBuffer command_buffer) const;

private:
    const Instance& instance;
    Scheduler& scheduler;
    mutable DescriptorHeap descriptor_heap;
};

} // namespace Vulkan
