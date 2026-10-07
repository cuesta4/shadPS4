// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <stdexcept>

#include "common/arch.h"
#include "common/assert.h"
#include "common/logging/log.h"

void assert_log_message_impl(const char* file, unsigned int line, const char* function,
                             fmt::string_view format, fmt::format_args args) noexcept {
    Common::Log::WriteImpl("Debug", Common::Log::Level::Critical, {file, line, function}, format,
                           args);
}

#if defined(ARCH_X86_64)
#define Crash() __asm__ __volatile__("int $3")
#elif defined(ARCH_ARM64)
#define Crash() __asm__ __volatile__("brk 0")
#else
#error "Missing Crash() implementation for target CPU architecture."
#endif

void assert_fail_impl() {
    Common::Log::Flush();
    Crash();
}

[[noreturn]] void unreachable_impl() {
    Common::Log::Flush();
    Crash();
    throw std::runtime_error("Unreachable code");
}

void assert_fail_debug_msg(const char* msg) {
    LOG_CRITICAL(Debug, "Assertion Failed!\n{}", msg);
    assert_fail_impl();
}
