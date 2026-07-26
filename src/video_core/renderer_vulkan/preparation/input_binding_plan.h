// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <boost/container/small_vector.hpp>

#include "common/types.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/gpu_commands/captured_state.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Shader {
struct ShaderInvocationData;
}

namespace Vulkan {

class GraphicsPipeline;
class Instance;

struct VertexInputRequest {
    static constexpr size_t InlineBindings = 8;

    boost::container::small_vector<vk::VertexInputAttributeDescription2EXT, InlineBindings>
        attributes;
    boost::container::small_vector<vk::VertexInputBindingDescription2EXT, InlineBindings> bindings;
    boost::container::small_vector<AmdGpu::Buffer, InlineBindings> guest_buffers;
    bool use_dynamic_vertex_input{};
};

struct IndexInputRequest {
    VAddr address{};
    u32 size{};
    vk::IndexType type{vk::IndexType::eUint32};
};

class InputRequestBuilder {
public:
    [[nodiscard]] static VertexInputRequest BuildVertexRequest(
        const GraphicsPipeline& pipeline, const Shader::ShaderInvocationData& invocation,
        const VideoCore::GraphicsDrawState& draw_state, const Instance& instance);
    [[nodiscard]] static IndexInputRequest BuildIndexRequest(
        const VideoCore::GraphicsDrawState& draw_state, u32 index_offset);
};

struct VertexInputPlan {
    static constexpr size_t InlineBindings = 8;

    boost::container::small_vector<vk::VertexInputAttributeDescription2EXT, InlineBindings>
        attributes;
    boost::container::small_vector<vk::VertexInputBindingDescription2EXT, InlineBindings> bindings;
    boost::container::small_vector<vk::Buffer, InlineBindings> buffers;
    boost::container::small_vector<vk::DeviceSize, InlineBindings> offsets;
    boost::container::small_vector<vk::DeviceSize, InlineBindings> sizes;
    boost::container::small_vector<vk::DeviceSize, InlineBindings> strides;
    bool use_dynamic_vertex_input{};

    [[nodiscard]] bool Empty() const noexcept {
        return buffers.empty();
    }
};

struct IndexInputPlan {
    vk::Buffer buffer{};
    vk::DeviceSize offset{};
    vk::IndexType type{vk::IndexType::eUint32};

    [[nodiscard]] bool IsValid() const noexcept {
        return static_cast<bool>(buffer);
    }
};

} // namespace Vulkan
