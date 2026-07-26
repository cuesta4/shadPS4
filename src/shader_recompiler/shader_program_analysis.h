// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/assert.h"
#include "shader_recompiler/info.h"

namespace Shader {

/**
 * Immutable published shader analysis.
 *
 * The recompiler still uses Info as its mutable construction record. Before publication, all
 * invocation-owned fields are removed and the completed record is moved into this const wrapper.
 * Pipelines and commands only receive const references to the published metadata.
 */
class ShaderProgramAnalysis {
public:
    explicit ShaderProgramAnalysis(Info&& completed_analysis)
        : metadata{std::move(completed_analysis)} {
        ASSERT(metadata.user_data.empty());
        ASSERT(metadata.flattened_ud_buf.empty());
        ASSERT(metadata.resolved_buffers.empty());
        ASSERT(metadata.resolved_images.empty());
        ASSERT(metadata.resolved_samplers.empty());
        ASSERT(metadata.resolved_fmasks.empty());
        ASSERT(metadata.resolved_vertex_buffers.empty());
        ASSERT(metadata.pgm_base == 0);
    }

    ShaderProgramAnalysis(const ShaderProgramAnalysis&) = delete;
    ShaderProgramAnalysis& operator=(const ShaderProgramAnalysis&) = delete;
    ShaderProgramAnalysis(ShaderProgramAnalysis&&) = delete;
    ShaderProgramAnalysis& operator=(ShaderProgramAnalysis&&) = delete;

    [[nodiscard]] const Info& Metadata() const noexcept {
        return metadata;
    }

private:
    const Info metadata;
};

} // namespace Shader
