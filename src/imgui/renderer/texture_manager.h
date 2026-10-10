// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <vector>

#include "common/types.h"
#include "imgui/imgui_texture.h"

namespace vk {
class CommandBuffer;
}

namespace Vulkan {
class Scheduler;
}

namespace ImGui::Core::TextureManager {

struct Inner;

void StartWorker();

void StopWorker();

void DecodePngTexture(std::vector<u8> data, Inner* core);

void DecodePngFile(std::filesystem::path path, Inner* core);

void Submit();

void EndFrame(::Vulkan::Scheduler& scheduler);

}; // namespace ImGui::Core::TextureManager
