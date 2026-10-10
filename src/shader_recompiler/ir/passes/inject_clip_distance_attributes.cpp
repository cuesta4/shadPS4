// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"

namespace Shader {

void InjectClipDistanceAttributes(IR::Program& program, const RuntimeInfo& runtime_info) {
    if (program.info.l_stage != LogicalStage::Fragment ||
        !runtime_info.fs_info.clip_distance_mask) {
        return;
    }

    auto* first_block = *program.blocks.begin();
    auto it = std::ranges::find_if(first_block->Instructions(), [](const IR::Inst& inst) {
        return inst.GetOpcode() == IR::Opcode::Prologue;
    });
    ASSERT(it != first_block->end());
    ++it;
    ASSERT(it != first_block->end());
    ++it;

    IR::IREmitter ir{*first_block, it};

    auto clipped = ir.Imm1(false);
    for (u32 comp = 0; comp < MaxEmulatedClipDistances; ++comp) {
        if ((runtime_info.fs_info.clip_distance_mask & (1u << comp)) == 0) {
            continue;
        }
        const auto attr_read = ir.GetAttribute(IR::Attribute::ClipDistance, comp);
        clipped = ir.LogicalOr(clipped, ir.FPLessThan(attr_read, ir.Imm32(0.0f)));
    }
    ir.Discard(clipped);
}

} // namespace Shader
