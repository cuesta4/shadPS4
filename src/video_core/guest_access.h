// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

/// Memory access of the instruction that raised a page fault.
struct GuestAccess {
    enum class Kind : u8 {
        /// The accessed bytes are known but the instruction is not emulated.
        Other,
        /// MOV reg, [mem].
        GprLoad,
        /// MOVZX reg, [mem].
        GprLoadZeroExtend,
        /// MOVSX / MOVSXD reg, [mem].
        GprLoadSignExtend,
        /// MOV [mem], reg.
        GprStore,
        /// MOV [mem], imm.
        ImmStore,
        /// SSE load that writes the low bytes of an XMM register and clears the rest of it.
        XmmLoad,
        /// SSE or VEX.128 store of the low bytes of an XMM register.
        XmmStore,
    };

    VAddr address{};
    u32 size{};
    Kind kind{Kind::Other};
    bool is_write{};
    /// Register operand: GPR in encoding order (RAX = 0 ... R15 = 15) or XMM index.
    u8 reg{};
    /// Width in bytes of the destination register of a load.
    u8 reg_width{};
    /// AH, CH, DH or BH.
    bool high_byte{};
    u8 length{};
    u64 imm{};

    [[nodiscard]] bool Emulatable() const noexcept {
        return kind != Kind::Other;
    }
};

/// Decodes the instruction of a faulting context. Returns false when the bytes it accesses
/// cannot be told, for instance with several memory operands.
bool DecodeGuestAccess(void* context, VAddr fault_addr, GuestAccess& access);

/// Performs the access through the backing view of guest memory, which ignores the page
/// protection, and moves the context past the instruction. Returns false, changing nothing, when
/// the access cannot be emulated.
bool EmulateGuestAccess(void* context, const GuestAccess& access);

} // namespace VideoCore
