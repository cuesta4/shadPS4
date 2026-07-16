# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

[CmdletBinding()]
param(
    [ValidateRange(1, 256)]
    [int]$Jobs = [Environment]::ProcessorCount
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = $PSScriptRoot
$EnvironmentFile = 'D:\CODING\SDKs\EWDK\EWDK-LLVM.env'
$ReleasePreset = 'x64-Clang-Release'
$ReleaseBuild = Join-Path $Root 'Build\x64-Clang-Release'

function Assert-LastExitCode([string]$Description) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Import-CompilerEnvironment {
    if (-not (Test-Path -LiteralPath $EnvironmentFile)) {
        throw "Compiler environment file was not found at '$EnvironmentFile'."
    }

    foreach ($line in Get-Content -LiteralPath $EnvironmentFile) {
        if (-not $line -or $line.StartsWith('#') -or -not $line.Contains('=')) {
            continue
        }
        $name, $value = $line -split '=', 2
        Set-Item -LiteralPath "Env:$name" -Value $value
    }

    $sdkIncludes = @('ucrt', 'shared', 'um', 'winrt', 'cppwinrt') |
        ForEach-Object { Join-Path $env:WindowsSdkDir "Include\$($env:Version_Number)\$_" }
    $sdkLibraries = @('ucrt', 'um') |
        ForEach-Object { Join-Path $env:WindowsSdkDir "Lib\$($env:Version_Number)\$_\x64" }
    $env:INCLUDE = (@($sdkIncludes) + $env:INCLUDE) -join ';'
    $env:LIB = (@($sdkLibraries) + $env:LIB) -join ';'

    $env:VSCMD_SKIP_SENDTELEMETRY = '1'
    $env:SCCACHE_DIR = 'D:\CODING\SDKs\sccache-cache'
    $env:SCCACHE_CACHE_SIZE = '50G'
    # Strip each checkout root so identical relative paths hash the same across worktrees.
    $env:SCCACHE_BASEDIRS = (Get-ChildItem -LiteralPath 'D:\CODING\PROJETOS' -Directory |
        ForEach-Object FullName) -join ';'

    foreach ($command in @('clang-cl', 'cmake', 'ninja', 'sccache')) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "Required command '$command' is not available after importing '$EnvironmentFile'."
        }
    }
}

Push-Location $Root
try {
    Import-CompilerEnvironment
    $clangCl = Join-Path $env:LLVM_ROOT 'bin\clang-cl.exe'
    $lldLink = Join-Path $env:LLVM_ROOT 'bin\lld-link.exe'
    $llvmLib = Join-Path $env:LLVM_ROOT 'bin\llvm-lib.exe'
    $llvmMt = Join-Path $env:LLVM_ROOT 'bin\llvm-mt.exe'
    $cmakeCache = Join-Path $ReleaseBuild 'CMakeCache.txt'
    $fresh = (Test-Path -LiteralPath $cmakeCache) -and
        (Select-String -LiteralPath $cmakeCache -SimpleMatch 'C:/Program Files/LLVM' -Quiet)
    $configureArgs = @()
    if ($fresh) {
        Write-Host 'Migrating stale CMake toolchain paths with a one-time fresh configure.' -ForegroundColor Yellow
        $configureArgs += '--fresh'
    }

    # Reconfiguring is incremental and ensures changed CMake source lists are picked up.
    $configureArgs += @(
        '--preset', $ReleasePreset,
        "-DCMAKE_C_COMPILER=$clangCl",
        "-DCMAKE_CXX_COMPILER=$clangCl",
        "-DCMAKE_ASM_COMPILER=$clangCl",
        "-DCMAKE_LINKER=$lldLink",
        "-DCMAKE_AR=$llvmLib",
        "-DCMAKE_MT=$llvmMt",
        '-DCMAKE_C_COMPILER_LAUNCHER=sccache',
        '-DCMAKE_CXX_COMPILER_LAUNCHER=sccache'
    )
    & cmake @configureArgs
    Assert-LastExitCode 'Release configuration'

    & cmake --build $ReleaseBuild --config Release --parallel $Jobs
    Assert-LastExitCode 'Incremental release build'

    Write-Host 'Incremental release build completed successfully.' -ForegroundColor Green
    & sccache --show-stats
} finally {
    Pop-Location
}
