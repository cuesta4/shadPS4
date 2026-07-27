// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <memory>
#include <string_view>
#include <unordered_map>

#include <gtest/gtest.h>
#include <spdlog/logger.h>

#include "common/logging/log.h"

namespace Common::Log {
std::atomic_bool g_is_enabled{true};
std::unordered_map<std::string_view, std::shared_ptr<spdlog::logger>> ALL_LOGGERS;
} // namespace Common::Log

namespace {

class LoggingMacroTest : public testing::Test {
protected:
    void SetUp() override {
        Common::Log::ALL_LOGGERS.clear();
        Common::Log::g_is_enabled.store(true, std::memory_order_relaxed);
    }

    void TearDown() override {
        Common::Log::ALL_LOGGERS.clear();
        Common::Log::g_is_enabled.store(true, std::memory_order_relaxed);
    }

    static std::shared_ptr<spdlog::logger> AddCommonLogger(spdlog::level level) {
        auto logger = std::make_shared<spdlog::logger>("CommonTest");
        logger->set_level(level);
        Common::Log::ALL_LOGGERS.emplace(Common::Log::Class::Common, logger);
        return logger;
    }
};

TEST_F(LoggingMacroTest, DisabledLoggingDoesNotEvaluateArguments) {
    AddCommonLogger(spdlog::level::trace);
    Common::Log::g_is_enabled.store(false, std::memory_order_relaxed);
    int evaluations = 0;

    LOG_INFO(Common, "value={}", ++evaluations);

    EXPECT_EQ(evaluations, 0);
}

TEST_F(LoggingMacroTest, FilteredLevelDoesNotEvaluateArguments) {
    AddCommonLogger(spdlog::level::off);
    int evaluations = 0;

    LOG_WARNING(Common, "value={}", ++evaluations);

    EXPECT_EQ(evaluations, 0);
}

TEST_F(LoggingMacroTest, EnabledLevelEvaluatesArgumentsOnce) {
    AddCommonLogger(spdlog::level::info);
    int evaluations = 0;

    LOG_INFO(Common, "value={}", ++evaluations);

    EXPECT_EQ(evaluations, 1);
}

TEST_F(LoggingMacroTest, MissingLoggerDoesNotInsertOrEvaluate) {
    int evaluations = 0;

    LOG_INFO(Common, "value={}", ++evaluations);

    EXPECT_EQ(evaluations, 0);
    EXPECT_TRUE(Common::Log::ALL_LOGGERS.empty());
}

} // namespace
