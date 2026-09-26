// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>

#include "common/arch.h"
#include "common/decoder.h"
#include "core/memory.h"
#include "video_core/guest_access.h"

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__) && defined(ARCH_X86_64)
#include <sys/ucontext.h>
#define SHADPS4_GUEST_ACCESS_UCONTEXT 1
#endif

namespace VideoCore {

namespace {

#if defined(_WIN32) || defined(SHADPS4_GUEST_ACCESS_UCONTEXT)
constexpr bool ContextSupported = true;
#else
constexpr bool ContextSupported = false;
#endif

constexpr u64 PageSize = 4096;

/// Points at a 64-bit general purpose register of the context, by encoding index.
u64* GprPointer(void* context, u32 index) {
#ifdef _WIN32
    auto* ctx = static_cast<EXCEPTION_POINTERS*>(context)->ContextRecord;
    // Rax, Rcx, Rdx, Rbx, Rsp, Rbp, Rsi, Rdi and R8-R15 are laid out in encoding order.
    return reinterpret_cast<u64*>(&ctx->Rax) + index;
#elif defined(SHADPS4_GUEST_ACCESS_UCONTEXT)
    static constexpr std::array<int, 16> GregIndex{
        REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
        REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15,
    };
    auto* uctx = static_cast<ucontext_t*>(context);
    return reinterpret_cast<u64*>(&uctx->uc_mcontext.gregs[GregIndex[index]]);
#else
    static_cast<void>(context);
    static_cast<void>(index);
    return nullptr;
#endif
}

u8* XmmPointer(void* context, u32 index) {
#ifdef _WIN32
    auto* ctx = static_cast<EXCEPTION_POINTERS*>(context)->ContextRecord;
    return reinterpret_cast<u8*>(&ctx->Xmm0) + index * 16;
#elif defined(SHADPS4_GUEST_ACCESS_UCONTEXT)
    auto* uctx = static_cast<ucontext_t*>(context);
    if (uctx->uc_mcontext.fpregs == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<u8*>(&uctx->uc_mcontext.fpregs->_xmm[index]);
#else
    static_cast<void>(context);
    static_cast<void>(index);
    return nullptr;
#endif
}

u64* RipPointer(void* context) {
#ifdef _WIN32
    return reinterpret_cast<u64*>(&static_cast<EXCEPTION_POINTERS*>(context)->ContextRecord->Rip);
#elif defined(SHADPS4_GUEST_ACCESS_UCONTEXT)
    return reinterpret_cast<u64*>(&static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_RIP]);
#else
    static_cast<void>(context);
    return nullptr;
#endif
}

/// Encoding index of a general purpose register, or -1.
int GprIndex(ZydisRegister reg, bool& high_byte) {
    high_byte = reg == ZYDIS_REGISTER_AH || reg == ZYDIS_REGISTER_CH ||
                reg == ZYDIS_REGISTER_DH || reg == ZYDIS_REGISTER_BH;
    const ZydisRegisterClass reg_class = ZydisRegisterGetClass(reg);
    if (reg_class != ZYDIS_REGCLASS_GPR8 && reg_class != ZYDIS_REGCLASS_GPR16 &&
        reg_class != ZYDIS_REGCLASS_GPR32 && reg_class != ZYDIS_REGCLASS_GPR64) {
        return -1;
    }
    const ZydisRegister full = ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, reg);
    if (full < ZYDIS_REGISTER_RAX || full > ZYDIS_REGISTER_R15) {
        return -1;
    }
    return static_cast<int>(full - ZYDIS_REGISTER_RAX);
}

int XmmIndex(ZydisRegister reg) {
    if (ZydisRegisterGetClass(reg) != ZYDIS_REGCLASS_XMM || reg < ZYDIS_REGISTER_XMM0 ||
        reg > ZYDIS_REGISTER_XMM15) {
        return -1;
    }
    return static_cast<int>(reg - ZYDIS_REGISTER_XMM0);
}

bool RegisterValue(void* context, ZydisRegister reg, u64& value) {
    if (reg == ZYDIS_REGISTER_NONE) {
        value = 0;
        return true;
    }
    bool high_byte{};
    const int index = GprIndex(reg, high_byte);
    if (index < 0 || high_byte ||
        ZydisRegisterGetWidth(ZYDIS_MACHINE_MODE_LONG_64, reg) != 64) {
        return false;
    }
    value = *GprPointer(context, static_cast<u32>(index));
    return true;
}

enum class LoadKind : u8 {
    None,
    Gpr,
    GprZeroExtend,
    GprSignExtend,
    Xmm,
};

enum class StoreKind : u8 {
    None,
    Gpr,
    Xmm,
};

/// Classifies the moves the emulation handles. VEX loads are left out: they clear the upper
/// half of the YMM register, which the signal context does not carry.
void Classify(const ZydisDecodedInstruction& instruction, LoadKind& load, StoreKind& store) {
    load = LoadKind::None;
    store = StoreKind::None;
    const bool vex = instruction.encoding == ZYDIS_INSTRUCTION_ENCODING_VEX;
    const bool legacy = instruction.encoding == ZYDIS_INSTRUCTION_ENCODING_LEGACY;
    switch (instruction.mnemonic) {
    case ZYDIS_MNEMONIC_MOV:
        if (legacy) {
            load = LoadKind::Gpr;
            store = StoreKind::Gpr;
        }
        break;
    case ZYDIS_MNEMONIC_MOVZX:
        if (legacy) {
            load = LoadKind::GprZeroExtend;
        }
        break;
    case ZYDIS_MNEMONIC_MOVSX:
    case ZYDIS_MNEMONIC_MOVSXD:
        if (legacy) {
            load = LoadKind::GprSignExtend;
        }
        break;
    case ZYDIS_MNEMONIC_MOVSS:
    case ZYDIS_MNEMONIC_MOVSD:
    case ZYDIS_MNEMONIC_MOVD:
    case ZYDIS_MNEMONIC_MOVQ:
    case ZYDIS_MNEMONIC_MOVUPS:
    case ZYDIS_MNEMONIC_MOVAPS:
    case ZYDIS_MNEMONIC_MOVUPD:
    case ZYDIS_MNEMONIC_MOVAPD:
    case ZYDIS_MNEMONIC_MOVDQU:
    case ZYDIS_MNEMONIC_MOVDQA:
        if (legacy) {
            load = LoadKind::Xmm;
            store = StoreKind::Xmm;
        }
        break;
    case ZYDIS_MNEMONIC_LDDQU:
        if (legacy) {
            load = LoadKind::Xmm;
        }
        break;
    case ZYDIS_MNEMONIC_MOVNTDQ:
    case ZYDIS_MNEMONIC_MOVNTPS:
    case ZYDIS_MNEMONIC_MOVNTPD:
        if (legacy) {
            store = StoreKind::Xmm;
        }
        break;
    case ZYDIS_MNEMONIC_VMOVSS:
    case ZYDIS_MNEMONIC_VMOVSD:
    case ZYDIS_MNEMONIC_VMOVD:
    case ZYDIS_MNEMONIC_VMOVQ:
    case ZYDIS_MNEMONIC_VMOVUPS:
    case ZYDIS_MNEMONIC_VMOVAPS:
    case ZYDIS_MNEMONIC_VMOVUPD:
    case ZYDIS_MNEMONIC_VMOVAPD:
    case ZYDIS_MNEMONIC_VMOVDQU:
    case ZYDIS_MNEMONIC_VMOVDQA:
    case ZYDIS_MNEMONIC_VMOVNTDQ:
    case ZYDIS_MNEMONIC_VMOVNTPS:
    case ZYDIS_MNEMONIC_VMOVNTPD:
        if (vex) {
            store = StoreKind::Xmm;
        }
        break;
    default:
        break;
    }
}

} // namespace

bool DecodeGuestAccess(void* context, VAddr fault_addr, GuestAccess& access) {
    access = {};
    if constexpr (!ContextSupported) {
        return false;
    }
    if (context == nullptr) {
        return false;
    }
    u64* rip_ptr = RipPointer(context);
    if (rip_ptr == nullptr) {
        return false;
    }
    const u64 rip = *rip_ptr;
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    // Stay on the page of the instruction unless the instruction continues on the next one,
    // which is then mapped as well.
    const u64 to_page_end = PageSize - (rip & (PageSize - 1));
    auto* decoder = Common::Decoder::Instance();
    ZyanStatus status = decoder->decodeInstruction(instruction, operands,
                                                   reinterpret_cast<void*>(rip),
                                                   std::min<u64>(15, to_page_end));
    if (status == ZYDIS_STATUS_NO_MORE_DATA && to_page_end < 15) {
        status = decoder->decodeInstruction(instruction, operands, reinterpret_cast<void*>(rip), 15);
    }
    if (!ZYAN_SUCCESS(status) || instruction.address_width != 64) {
        return false;
    }

    // Exactly one explicit memory operand.
    int mem_index = -1;
    for (u32 i = 0; i < instruction.operand_count; ++i) {
        const auto& operand = operands[i];
        if (operand.type != ZYDIS_OPERAND_TYPE_MEMORY) {
            continue;
        }
        if (mem_index >= 0 || operand.visibility != ZYDIS_OPERAND_VISIBILITY_EXPLICIT) {
            return false;
        }
        mem_index = static_cast<int>(i);
    }
    if (mem_index < 0) {
        return false;
    }
    const auto& mem = operands[mem_index];
    if (mem.mem.type != ZYDIS_MEMOP_TYPE_MEM || mem.size == 0 || (mem.size % 8) != 0) {
        return false;
    }
    if (mem.mem.segment != ZYDIS_REGISTER_NONE && mem.mem.segment != ZYDIS_REGISTER_DS &&
        mem.mem.segment != ZYDIS_REGISTER_SS && mem.mem.segment != ZYDIS_REGISTER_ES &&
        mem.mem.segment != ZYDIS_REGISTER_CS) {
        return false;
    }
    u64 base{};
    u64 index{};
    if (mem.mem.base == ZYDIS_REGISTER_RIP) {
        base = rip + instruction.length;
    } else if (!RegisterValue(context, mem.mem.base, base)) {
        return false;
    }
    if (!RegisterValue(context, mem.mem.index, index)) {
        return false;
    }
    const u64 scale = mem.mem.scale == 0 ? 1 : mem.mem.scale;
    const VAddr address = base + index * scale + static_cast<u64>(mem.mem.disp.value);
    const u32 size = mem.size / 8;
    if (fault_addr < address || fault_addr - address >= size) {
        return false;
    }
    access.address = address;
    access.size = size;
    access.length = instruction.length;
    access.is_write = (mem.actions & ZYDIS_OPERAND_ACTION_MASK_WRITE) != 0;
    const bool is_read = (mem.actions & ZYDIS_OPERAND_ACTION_MASK_READ) != 0;
    access.kind = GuestAccess::Kind::Other;

    LoadKind load{};
    StoreKind store{};
    Classify(instruction, load, store);
    if (instruction.operand_count_visible != 2 || (access.is_write && is_read) ||
        (instruction.attributes & ZYDIS_ATTRIB_HAS_LOCK) != 0) {
        return true;
    }
    const auto& other = operands[mem_index == 0 ? 1 : 0];
    if (mem_index == 1 && !access.is_write && other.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        // Load into a register.
        bool high_byte{};
        const int gpr = GprIndex(other.reg.value, high_byte);
        const int xmm = XmmIndex(other.reg.value);
        const u32 width = ZydisRegisterGetWidth(ZYDIS_MACHINE_MODE_LONG_64, other.reg.value) / 8;
        if (gpr >= 0 && (load == LoadKind::Gpr || load == LoadKind::GprZeroExtend ||
                         load == LoadKind::GprSignExtend)) {
            if ((load == LoadKind::Gpr && width != size) || width < size || size > 8) {
                return true;
            }
            access.reg = static_cast<u8>(gpr);
            access.reg_width = static_cast<u8>(width);
            access.high_byte = high_byte;
            access.kind = load == LoadKind::Gpr             ? GuestAccess::Kind::GprLoad
                          : load == LoadKind::GprZeroExtend ? GuestAccess::Kind::GprLoadZeroExtend
                                                            : GuestAccess::Kind::GprLoadSignExtend;
        } else if (xmm >= 0 && load == LoadKind::Xmm && size <= 16) {
            access.reg = static_cast<u8>(xmm);
            access.reg_width = 16;
            access.kind = GuestAccess::Kind::XmmLoad;
        }
        return true;
    }
    if (mem_index == 0 && access.is_write && !is_read) {
        if (other.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            bool high_byte{};
            const int gpr = GprIndex(other.reg.value, high_byte);
            const int xmm = XmmIndex(other.reg.value);
            const u32 width =
                ZydisRegisterGetWidth(ZYDIS_MACHINE_MODE_LONG_64, other.reg.value) / 8;
            if (gpr >= 0 && store == StoreKind::Gpr && width == size) {
                access.reg = static_cast<u8>(gpr);
                access.high_byte = high_byte;
                access.kind = GuestAccess::Kind::GprStore;
            } else if (xmm >= 0 && store == StoreKind::Xmm && size <= 16) {
                access.reg = static_cast<u8>(xmm);
                access.kind = GuestAccess::Kind::XmmStore;
            }
        } else if (other.type == ZYDIS_OPERAND_TYPE_IMMEDIATE && store == StoreKind::Gpr &&
                   size <= 8) {
            access.imm = other.imm.value.u;
            access.kind = GuestAccess::Kind::ImmStore;
        }
    }
    return true;
}

bool EmulateGuestAccess(void* context, const GuestAccess& access) {
    if constexpr (!ContextSupported) {
        return false;
    }
    if (!access.Emulatable() || context == nullptr || access.size == 0 || access.size > 16) {
        return false;
    }
    u64* rip = RipPointer(context);
    if (rip == nullptr) {
        return false;
    }
    auto* memory = Core::Memory::Instance();
    std::array<u8, 16> bytes{};
    switch (access.kind) {
    case GuestAccess::Kind::GprLoad:
    case GuestAccess::Kind::GprLoadZeroExtend:
    case GuestAccess::Kind::GprLoadSignExtend: {
        if (!memory->ReadBacking(access.address, bytes.data(), access.size)) {
            return false;
        }
        u64 value{};
        std::memcpy(&value, bytes.data(), access.size);
        if (access.kind == GuestAccess::Kind::GprLoadSignExtend && access.size < 8) {
            const u32 shift = 64 - access.size * 8;
            value = static_cast<u64>(static_cast<s64>(value << shift) >> shift);
        }
        u64& reg = *GprPointer(context, access.reg);
        switch (access.reg_width) {
        case 8:
            reg = value;
            break;
        case 4:
            // 32-bit destinations clear the upper half.
            reg = value & 0xFFFFFFFFULL;
            break;
        case 2:
            reg = (reg & ~0xFFFFULL) | (value & 0xFFFFULL);
            break;
        case 1:
            if (access.high_byte) {
                reg = (reg & ~0xFF00ULL) | ((value & 0xFFULL) << 8);
            } else {
                reg = (reg & ~0xFFULL) | (value & 0xFFULL);
            }
            break;
        default:
            return false;
        }
        break;
    }
    case GuestAccess::Kind::XmmLoad: {
        u8* xmm = XmmPointer(context, access.reg);
        if (xmm == nullptr || !memory->ReadBacking(access.address, bytes.data(), access.size)) {
            return false;
        }
        // Loads from memory clear the rest of the register.
        std::memcpy(xmm, bytes.data(), 16);
        break;
    }
    case GuestAccess::Kind::GprStore: {
        const u64 reg = *GprPointer(context, access.reg);
        const u64 value = access.high_byte ? (reg >> 8) & 0xFFULL : reg;
        std::memcpy(bytes.data(), &value, sizeof(value));
        if (!memory->TryWriteBacking(reinterpret_cast<void*>(access.address), bytes.data(),
                                     access.size)) {
            return false;
        }
        break;
    }
    case GuestAccess::Kind::ImmStore: {
        std::memcpy(bytes.data(), &access.imm, sizeof(access.imm));
        if (!memory->TryWriteBacking(reinterpret_cast<void*>(access.address), bytes.data(),
                                     access.size)) {
            return false;
        }
        break;
    }
    case GuestAccess::Kind::XmmStore: {
        const u8* xmm = XmmPointer(context, access.reg);
        if (xmm == nullptr) {
            return false;
        }
        std::memcpy(bytes.data(), xmm, 16);
        if (!memory->TryWriteBacking(reinterpret_cast<void*>(access.address), bytes.data(),
                                     access.size)) {
            return false;
        }
        break;
    }
    default:
        return false;
    }
    *rip += access.length;
    return true;
}

} // namespace VideoCore
