// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include "common/types.h"
#include "core/emulator_settings.h"

TEST(ArtificialRamLimit, JsonRoundTripPreservesLimit) {
    DebugSettings settings;
    settings.artificial_ram_limit_mib.set(12 * 1024);

    const nlohmann::json json = settings;
    EXPECT_EQ(json.at("artificial_ram_limit_mib").get<u32>(), 12 * 1024u);

    DebugSettings loaded;
    json.get_to(loaded);
    EXPECT_EQ(loaded.artificial_ram_limit_mib.get(), 12 * 1024u);
}

TEST(ArtificialRamLimit, GameOverrideDoesNotReplaceGlobalValue) {
    DebugSettings settings;
    settings.artificial_ram_limit_mib.set(16 * 1024);
    std::vector<std::string> changed;

    const auto fields = settings.GetOverrideableFields();
    const auto field =
        std::ranges::find(fields, std::string_view{"artificial_ram_limit_mib"},
                          [](const OverrideItem& item) { return std::string_view{item.key}; });
    ASSERT_NE(field, fields.end());
    field->apply(&settings, nlohmann::json(10 * 1024), changed);

    EXPECT_EQ(settings.artificial_ram_limit_mib.get(ConfigMode::Default), 10 * 1024u);
    EXPECT_EQ(settings.artificial_ram_limit_mib.get(ConfigMode::Global), 16 * 1024u);
}
