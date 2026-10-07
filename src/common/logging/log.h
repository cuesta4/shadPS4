// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <fmt/format.h>

namespace Common::Log {

namespace Detail {

// Category tokens use underscores where the backend's canonical names use dots.
template <std::size_t N>
struct ClassToken {
    consteval ClassToken(const char (&token)[N]) {
        for (std::size_t i = 0; i < N; ++i) {
            value[i] = token[i] == '_' ? '.' : token[i];
        }
    }

    char value[N]{};
};

// One constant per category, shared across call sites and translation units.
template <ClassToken Token>
inline constexpr auto class_name = Token;

template <ClassToken Token>
consteval std::string_view ClassName() {
    return {class_name<Token>.value, sizeof(Token.value) - 1};
}

} // namespace Detail

enum class Level : std::uint8_t {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Critical,
    Off,
};

struct SourceLocation {
    const char* file;
    std::uint_least32_t line;
    const char* function;
};

extern bool g_should_append;

void Setup(std::string_view log_filename);
void Shutdown();
void Flush();
void Terminate();

[[nodiscard]] bool IsEnabled(std::string_view log_class, Level level);

// Arguments are consumed synchronously; only the formatted message reaches the sinks.
void WriteImpl(std::string_view log_class, Level level, SourceLocation source,
               fmt::string_view format, fmt::format_args args) noexcept;

template <typename... Args>
void Write(std::string_view log_class, Level level, SourceLocation source,
           fmt::format_string<Args...> format, Args&&... args) noexcept {
    WriteImpl(log_class, level, source, format, fmt::make_format_args(args...));
}

} // namespace Common::Log

#define LOG_CLASS_NAME(log_class) Common::Log::Detail::ClassName<#log_class>()

#define LOG_GENERIC(log_class, log_level, ...)                                                     \
    do {                                                                                           \
        const auto shad_log_class = (log_class);                                                   \
        const auto shad_log_level = (log_level);                                                   \
        if (Common::Log::IsEnabled(shad_log_class, shad_log_level)) {                              \
            Common::Log::Write(shad_log_class, shad_log_level, {__FILE__, __LINE__, __func__},     \
                               __VA_ARGS__);                                                       \
        }                                                                                          \
    } while (false)

#ifdef _DEBUG
#define LOG_TRACE(log_class, ...)                                                                  \
    LOG_GENERIC(LOG_CLASS_NAME(log_class), Common::Log::Level::Trace, __VA_ARGS__)
#else
#define LOG_TRACE(log_class, ...) (void(0))
#endif

#define LOG_DEBUG(log_class, ...)                                                                  \
    LOG_GENERIC(LOG_CLASS_NAME(log_class), Common::Log::Level::Debug, __VA_ARGS__)
#define LOG_INFO(log_class, ...)                                                                   \
    LOG_GENERIC(LOG_CLASS_NAME(log_class), Common::Log::Level::Info, __VA_ARGS__)
#define LOG_WARNING(log_class, ...)                                                                \
    LOG_GENERIC(LOG_CLASS_NAME(log_class), Common::Log::Level::Warning, __VA_ARGS__)
#define LOG_ERROR(log_class, ...)                                                                  \
    LOG_GENERIC(LOG_CLASS_NAME(log_class), Common::Log::Level::Error, __VA_ARGS__)
#define LOG_CRITICAL(log_class, ...)                                                               \
    LOG_GENERIC(LOG_CLASS_NAME(log_class), Common::Log::Level::Critical, __VA_ARGS__)
