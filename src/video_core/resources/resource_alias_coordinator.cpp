// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/resources/resource_alias_coordinator.h"

namespace VideoCore {

bool ResourceAliasCoordinator::ClearMetadata(VAddr address) {
    return clear_metadata(address);
}

bool ResourceAliasCoordinator::HasImage(VAddr address, u32 size) {
    return has_image(address, size);
}

void ResourceAliasCoordinator::InvalidateImagesFromGpu(VAddr address, u32 size) {
    invalidate_images(address, size);
}

std::pair<Buffer*, u32> ResourceAliasCoordinator::ObtainBufferForImage(VAddr address, u32 size) {
    return obtain_buffer_for_image(address, size);
}

bool ResourceAliasCoordinator::SynchronizeBufferFromImage(Buffer& buffer, VAddr address, u32 size) {
    return synchronize_buffer_from_image(buffer, address, size);
}

} // namespace VideoCore
