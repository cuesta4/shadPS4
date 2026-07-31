// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "video_core/amdgpu/pm4_cmds.h"

namespace AmdGpu {

/// Returns true when the next packet is a complete memory WAIT_REG_MEM that the supplied
/// immediate signal would satisfy. This deliberately recognizes only adjacent packets so the
/// signal cannot be observed by an intervening command.
[[nodiscard]] inline bool WaitMatchesImmediateSignal(std::span<const u32> next_packet,
                                                     VAddr address, u32 value) {
    constexpr size_t WaitPacketWords = sizeof(PM4CmdWaitRegMem) / sizeof(u32);
    static_assert(sizeof(PM4CmdWaitRegMem) % sizeof(u32) == 0);

    if (next_packet.size() < WaitPacketWords) {
        return false;
    }

    const auto* header = reinterpret_cast<const PM4Header*>(next_packet.data());
    if (header->type.Value() != PM4Type3Header::TYPE ||
        header->type3.opcode.Value() != PM4ItOpcode::WaitRegMem ||
        header->type3.NumWords() + 1 != WaitPacketWords) {
        return false;
    }

    const auto* wait = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
    if (wait->mem_space.Value() != PM4CmdWaitRegMem::MemSpace::Memory ||
        reinterpret_cast<VAddr>(wait->Address()) != address) {
        return false;
    }

    const u32 masked_value = value & wait->mask;
    switch (wait->function.Value()) {
    case PM4CmdWaitRegMem::Function::Always:
        return true;
    case PM4CmdWaitRegMem::Function::LessThan:
        return masked_value < wait->ref;
    case PM4CmdWaitRegMem::Function::LessThanEqual:
        return masked_value <= wait->ref;
    case PM4CmdWaitRegMem::Function::Equal:
        return masked_value == wait->ref;
    case PM4CmdWaitRegMem::Function::NotEqual:
        return masked_value != wait->ref;
    case PM4CmdWaitRegMem::Function::GreaterThanEqual:
        return masked_value >= wait->ref;
    case PM4CmdWaitRegMem::Function::GreaterThan:
        return masked_value > wait->ref;
    case PM4CmdWaitRegMem::Function::Reserved:
        return false;
    }
    return false;
}

} // namespace AmdGpu
