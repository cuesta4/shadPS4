# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

[CmdletBinding()]
param(
    [ValidateSet('Instrument', 'Smoke', 'EnableProfile', 'Merge', 'Optimize', 'Reports')]
    [string]$Stage = 'Instrument',
    [ValidateRange(1, 256)]
    [int]$Jobs = [Environment]::ProcessorCount
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = $PSScriptRoot
$EnvironmentFile = 'D:\CODING\SDKs\EWDK\EWDK-LLVM.env'
$QtPrefix = 'D:\CODING\SDKs\Qt\6.10.0\msvc2022_64'
$PgoRoot = Join-Path $Root 'Build\x64-Clang-PGO'
$InstrumentedBuild = Join-Path $PgoRoot 'Instrumented'
$OptimizedBuild = Join-Path $PgoRoot 'Optimized'
$ProfilesRoot = Join-Path $PgoRoot 'profiles'
$BuildProfilesDir = Join-Path $ProfilesRoot 'build'
$RawDir = Join-Path $ProfilesRoot 'raw'
$SmokeRoot = Join-Path $ProfilesRoot 'smoke'
$ReportsDir = Join-Path $ProfilesRoot 'reports'
$ProfileData = Join-Path $ProfilesRoot 'shadps4.profdata'
$ProfileScript = Join-Path $PgoRoot 'enable-profile.ps1'

function Assert-LastExitCode([string]$Description) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Assert-Path([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Description was not found at '$Path'."
    }
}

function Ensure-Directory([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Import-CompilerEnvironment {
    Assert-Path $EnvironmentFile 'EWDK compiler environment file'

    foreach ($line in Get-Content -LiteralPath $EnvironmentFile) {
        if (-not $line -or $line.StartsWith('#') -or -not $line.Contains('=')) {
            continue
        }
        $name, $value = $line -split '=', 2
        Set-Item -LiteralPath "Env:$name" -Value $value
    }

    if (-not $env:WindowsSdkDir -or -not $env:Version_Number) {
        throw 'The EWDK environment did not define WindowsSdkDir and Version_Number.'
    }

    $sdkIncludes = @('ucrt', 'shared', 'um', 'winrt', 'cppwinrt') |
        ForEach-Object { Join-Path $env:WindowsSdkDir "Include\$($env:Version_Number)\$_" }
    $sdkLibraries = @('ucrt', 'um') |
        ForEach-Object { Join-Path $env:WindowsSdkDir "Lib\$($env:Version_Number)\$_\x64" }
    $existingInclude = [Environment]::GetEnvironmentVariable('INCLUDE', 'Process')
    $existingLib = [Environment]::GetEnvironmentVariable('LIB', 'Process')
    $env:INCLUDE = (@($sdkIncludes) + @($existingInclude)) -join ';'
    $env:LIB = (@($sdkLibraries) + @($existingLib)) -join ';'
    $env:VSCMD_SKIP_SENDTELEMETRY = '1'
    $env:SCCACHE_DISABLE = '1'
}

function Resolve-Tool([string]$Name, [string]$Fallback) {
    if (Test-Path -LiteralPath $Fallback) {
        return $Fallback
    }
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "$Name was not found at '$Fallback' or on PATH."
    }
    return $command.Source
}

function Get-Tools {
    $llvmBin = Join-Path $env:LLVM_ROOT 'bin'
    $cmake = Resolve-Tool 'cmake.exe' 'D:\CODING\SDKs\mingw64\bin\cmake.exe'
    $ninja = Resolve-Tool 'ninja.exe' 'D:\CODING\SDKs\mingw64\bin\ninja.exe'
    $tools = [pscustomobject]@{
        CMake = $cmake
        Ninja = $ninja
        ClangCl = Join-Path $llvmBin 'clang-cl.exe'
        LldLink = Join-Path $llvmBin 'lld-link.exe'
        LlvmLib = Join-Path $llvmBin 'llvm-lib.exe'
        LlvmMt = Join-Path $llvmBin 'llvm-mt.exe'
        ProfData = Join-Path $llvmBin 'llvm-profdata.exe'
        ReadObj = Join-Path $llvmBin 'llvm-readobj.exe'
    }
    foreach ($property in @('ClangCl', 'LldLink', 'LlvmLib', 'LlvmMt', 'ProfData', 'ReadObj')) {
        Assert-Path $tools.$property "LLVM tool $property"
    }
    Assert-Path $QtPrefix 'Qt prefix'
    $env:PATH = "$(Split-Path -Parent $tools.CMake);$(Split-Path -Parent $tools.Ninja);$env:PATH"
    return $tools
}

function Get-ProfileEnvironmentScript {
    Ensure-Directory $PgoRoot
    Ensure-Directory $RawDir
    $profilePattern = Join-Path $RawDir 'shadps4-%p-%m.profraw'
    $contents = @(
        '# Dot-source this file in the shell that launches the instrumented emulator.'
        "`$env:LLVM_PROFILE_FILE = '$profilePattern'"
        'Write-Host "LLVM_PROFILE_FILE=$env:LLVM_PROFILE_FILE"'
    ) -join [Environment]::NewLine
    Set-Content -LiteralPath $ProfileScript -Value $contents -Encoding utf8
    return $profilePattern
}

function Get-ReleaseFlags([string]$ProfileFlags) {
    return "/O2 /Ob2 /DNDEBUG /Zi /clang:-fdebug-info-for-profiling /clang:-funique-internal-linkage-names /clang:-fsave-optimization-record $ProfileFlags"
}

function Get-ConfigureArguments([string]$BuildDirectory, [string]$ReleaseFlags) {
    @(
        '-S', $Root,
        '-B', $BuildDirectory,
        '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release',
        "-DCMAKE_C_COMPILER=$($Tools.ClangCl)",
        "-DCMAKE_CXX_COMPILER=$($Tools.ClangCl)",
        "-DCMAKE_ASM_COMPILER=$($Tools.ClangCl)",
        "-DCMAKE_LINKER=$($Tools.LldLink)",
        "-DCMAKE_AR=$($Tools.LlvmLib)",
        "-DCMAKE_MT=$($Tools.LlvmMt)",
        "-DCMAKE_PREFIX_PATH=$QtPrefix",
        "-DCMAKE_C_FLAGS_RELEASE=$ReleaseFlags",
        "-DCMAKE_CXX_FLAGS_RELEASE=$ReleaseFlags",
        '-DCMAKE_C_COMPILER_LAUNCHER=',
        '-DCMAKE_CXX_COMPILER_LAUNCHER=',
        '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
        '-DENABLE_DETAILED_TELEMETRY=OFF'
    )
}

function Invoke-ConfigureBuild([string]$BuildDirectory, [string]$ReleaseFlags) {
    Ensure-Directory $BuildDirectory
    $configureArguments = @(Get-ConfigureArguments $BuildDirectory $ReleaseFlags)
    Write-Host "Configuring $BuildDirectory" -ForegroundColor Cyan
    & $Tools.CMake @configureArguments
    Assert-LastExitCode 'CMake configuration'
    Write-Host "Building $BuildDirectory with $Jobs job(s)" -ForegroundColor Cyan
    & $Tools.CMake '--build' $BuildDirectory '--config' 'Release' '--parallel' $Jobs
    Assert-LastExitCode 'CMake build'
}

function Assert-InstrumentedBinary {
    $executable = Join-Path $InstrumentedBuild 'shadps4.exe'
    Assert-Path $executable 'Instrumented executable'
    $sections = @(& $Tools.ReadObj '--sections' $executable 2>&1)
    Assert-LastExitCode 'LLVM section inspection'
    $profileSections = @($sections | Select-String -Pattern '\.lprf[cn]')
    if ($profileSections.Count -eq 0) {
        throw "Instrumented executable '$executable' has no LLVM profile sections."
    }
    Write-Host "LLVM profile sections found in $executable" -ForegroundColor Green
    return $executable
}

function Invoke-Smoke {
    $executable = Join-Path $InstrumentedBuild 'shadps4.exe'
    Assert-Path $executable 'Instrumented executable'
    $runId = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmssfff')
    $smokeDirectory = Join-Path $SmokeRoot $runId
    Ensure-Directory $smokeDirectory
    $smokePattern = Join-Path $smokeDirectory 'shadps4-%p-%m.profraw'
    $previousProfile = [Environment]::GetEnvironmentVariable('LLVM_PROFILE_FILE', 'Process')
    $previousQuickExitSmoke =
        [Environment]::GetEnvironmentVariable('SHADPS4_PGO_QUICK_EXIT_SMOKE', 'Process')
    $exitCode = 0
    try {
        $env:LLVM_PROFILE_FILE = $smokePattern
        $env:SHADPS4_PGO_QUICK_EXIT_SMOKE = '1'
        & $executable '--help'
        $exitCode = $LASTEXITCODE
    } finally {
        if ($null -eq $previousProfile) {
            Remove-Item -LiteralPath 'Env:LLVM_PROFILE_FILE' -ErrorAction SilentlyContinue
        } else {
            $env:LLVM_PROFILE_FILE = $previousProfile
        }
        if ($null -eq $previousQuickExitSmoke) {
            Remove-Item -LiteralPath 'Env:SHADPS4_PGO_QUICK_EXIT_SMOKE' -ErrorAction SilentlyContinue
        } else {
            $env:SHADPS4_PGO_QUICK_EXIT_SMOKE = $previousQuickExitSmoke
        }
    }
    if ($exitCode -ne 0) {
        throw "Instrumented --help smoke failed with exit code $exitCode."
    }
    $rawFiles = @(Get-ChildItem -LiteralPath $smokeDirectory -Filter '*.profraw' -File |
        Where-Object { $_.Length -gt 0 })
    if ($rawFiles.Count -eq 0) {
        throw "Instrumented --help smoke produced no non-empty .profraw under '$smokeDirectory'."
    }
    Write-Host "Smoke profile: $($rawFiles[0].FullName) ($($rawFiles[0].Length) bytes)" -ForegroundColor Green
}

function Invoke-Merge {
    Ensure-Directory $ReportsDir
    Assert-Path $RawDir 'Scene profile directory'
    $rawFiles = @(Get-ChildItem -LiteralPath $RawDir -Filter '*.profraw' -File |
        Where-Object { $_.Length -gt 0 })
    if ($rawFiles.Count -eq 0) {
        throw "No non-empty scene .profraw files found under '$RawDir'. Run the scene with enable-profile.ps1 first."
    }
    $mergeArguments = @('merge', '-sparse', '-o', $ProfileData) + @($rawFiles.FullName)
    & $Tools.ProfData @mergeArguments
    Assert-LastExitCode 'llvm-profdata merge'
    Assert-Path $ProfileData 'Indexed profile'
    $summaryPath = Join-Path $ReportsDir 'profile-summary.txt'
    $summary = @(& $Tools.ProfData 'show' $ProfileData 2>&1)
    Assert-LastExitCode 'llvm-profdata show'
    $summary | Set-Content -LiteralPath $summaryPath -Encoding utf8
    Write-Host "Indexed profile: $ProfileData" -ForegroundColor Green
    Write-Host "Profile summary: $summaryPath" -ForegroundColor Green
}

function Invoke-Optimize {
    Assert-Path $ProfileData 'Indexed scene profile'
    $profileFlags = "/clang:-fprofile-instr-use=$ProfileData /clang:-Wprofile-instr-unprofiled /clang:-Wprofile-instr-out-of-date /clang:-Wno-error=profile-instr-unprofiled /clang:-Wno-error=profile-instr-out-of-date"
    Invoke-ConfigureBuild $OptimizedBuild (Get-ReleaseFlags $profileFlags)
    Write-Host "Profile-use executable: $(Join-Path $OptimizedBuild 'shadps4.exe')" -ForegroundColor Green
}

function Resolve-Python {
    $candidates = [System.Collections.Generic.List[string]]::new()
    $command = Get-Command python.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($command) {
        $candidates.Add($command.Source)
    }
    $bundled = 'D:\CODING\SDKs\python\python.exe'
    if (Test-Path -LiteralPath $bundled) {
        $candidates.Add($bundled)
    }
    foreach ($candidate in $candidates | Select-Object -Unique) {
        & $candidate '-c' 'import yaml' 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $candidate
        }
    }
    throw 'Python with PyYAML is required for optimization-record reports but was not found.'
}

function Invoke-Reports {
    Assert-Path $ProfileData 'Indexed scene profile'
    Assert-Path (Join-Path $OptimizedBuild 'shadps4.exe') 'Profile-use executable'
    Ensure-Directory $ReportsDir
    $recordNames = @(
        'buffer_cache.opt.yaml'
        'liverpool.opt.yaml'
        'texture_cache.opt.yaml'
        'vk_pipeline_cache.opt.yaml'
        'vk_rasterizer.opt.yaml'
        'vk_scheduler.opt.yaml'
    )
    $yamlFiles = @(
        foreach ($recordName in $recordNames) {
            $recordMatches = @(Get-ChildItem -LiteralPath $OptimizedBuild -Filter $recordName -File -Recurse)
            if ($recordMatches.Count -ne 1) {
                throw "Expected exactly one '$recordName' under '$OptimizedBuild', found $($recordMatches.Count)."
            }
            $recordMatches[0]
        }
    )
    if ($yamlFiles.Count -ne $recordNames.Count) {
        throw "No complete hot-path optimization-record set was found under '$OptimizedBuild'."
    }
    $selectedManifest = Join-Path $ReportsDir 'optimization-record-selected.txt'
    $yamlManifest = Join-Path $ReportsDir 'optimization-record-files.txt'
    $invalidManifest = Join-Path $ReportsDir 'optimization-record-invalid.txt'
    $validationLog = Join-Path $ReportsDir 'optimization-record-validation.txt'
    $yamlFiles.FullName | Set-Content -LiteralPath $selectedManifest -Encoding utf8
    Write-Host 'Stage Reports is limited to the selected hot-path records; global records are skipped because -fsave-optimization-record basename collisions can corrupt YAML.' -ForegroundColor Yellow
    $statsScript = Join-Path $env:LLVM_ROOT 'share\opt-viewer\opt-stats.py'
    Assert-Path $statsScript 'LLVM optimization statistics script'
    $python = Resolve-Python

    $validationScript = @'
import os
import sys

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def validate(path):
    try:
        if os.path.getsize(path) == 0:
            return path, "empty file"
        with open(path, encoding="utf-8") as stream:
            for _ in yaml.parse(stream, Loader=Loader):
                pass
        return path, None
    except Exception as error:
        detail = f"{type(error).__name__}: {error}".replace("\r", " ").replace("\n", " ")
        return path, detail


def main():
    selected_manifest, valid_manifest, invalid_manifest = sys.argv[1:]
    with open(selected_manifest, encoding="utf-8-sig") as stream:
        paths = [line.strip() for line in stream if line.strip()]
    results = [validate(path) for path in paths]
    valid = [path for path, error in results if error is None]
    invalid = [(path, error) for path, error in results if error is not None]
    with open(valid_manifest, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("".join(f"{path}\n" for path in valid))
    with open(invalid_manifest, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("".join(f"{path}\t{error}\n" for path, error in invalid))
    print(f"Validated optimization records: {len(valid)}")
    print(f"Excluded invalid optimization records: {len(invalid)}")


if __name__ == "__main__":
    main()
'@
    $validationOutput = @(& $python '-c' $validationScript $selectedManifest $yamlManifest $invalidManifest 2>&1)
    $validationExitCode = $LASTEXITCODE
    $validationOutput | Set-Content -LiteralPath $validationLog -Encoding utf8
    if ($validationExitCode -ne 0) {
        throw "Optimization-record validation failed with exit code $validationExitCode. See '$validationLog'."
    }
    $validPaths = @(Get-Content -LiteralPath $yamlManifest | Where-Object { $_.Trim() })
    $invalidRecords = @(Get-Content -LiteralPath $invalidManifest | Where-Object { $_.Trim() })
    if ($validPaths.Count -eq 0) {
        throw "No valid optimization records remain after validation. See '$invalidManifest'."
    }
    if ($invalidRecords.Count -gt 0) {
        Write-Warning "Excluded $($invalidRecords.Count) invalid optimization record(s). See '$invalidManifest'."
        $invalidRecords | ForEach-Object { Write-Host "Excluded: $_" -ForegroundColor Yellow }
    }
    $statsPath = Join-Path $ReportsDir 'optimization-stats.txt'
    $statsArguments = @($statsScript, '-n', '-j', $Jobs) + $validPaths
    $stats = @(& $python @statsArguments 2>&1)
    $statsExitCode = $LASTEXITCODE
    $stats | Set-Content -LiteralPath $statsPath -Encoding utf8
    if ($statsExitCode -ne 0) {
        throw "Optimization statistics failed with exit code $statsExitCode. See '$statsPath'."
    }
    Write-Host "Optimization records: $($validPaths.Count) valid, $($invalidRecords.Count) excluded" -ForegroundColor Green
    Write-Host "Selected record manifest: $selectedManifest" -ForegroundColor Green
    Write-Host "Optimization report: $statsPath" -ForegroundColor Green
}

Push-Location $Root
try {
    Import-CompilerEnvironment
    $Tools = Get-Tools
    Ensure-Directory $ProfilesRoot
    Ensure-Directory $BuildProfilesDir

    switch ($Stage) {
        'Instrument' {
            $profilePattern = Get-ProfileEnvironmentScript
            $buildProfilePattern = Join-Path $BuildProfilesDir 'shadps4-%p-%m.profraw'
            $profileFlags = "/clang:-fprofile-update=atomic /clang:-fprofile-instr-generate=$buildProfilePattern"
            Invoke-ConfigureBuild $InstrumentedBuild (Get-ReleaseFlags $profileFlags)
            [void](Assert-InstrumentedBinary)
            Invoke-Smoke
            Write-Host "Scene profile environment: $ProfileScript" -ForegroundColor Green
            Write-Host "Scene raw directory: $RawDir" -ForegroundColor Green
        }
        'Smoke' {
            [void](Assert-InstrumentedBinary)
            Invoke-Smoke
        }
        'EnableProfile' {
            $profilePattern = Get-ProfileEnvironmentScript
            Write-Host "Dot-source $ProfileScript before launching the instrumented emulator." -ForegroundColor Green
            Write-Host "LLVM_PROFILE_FILE=$profilePattern"
        }
        'Merge' { Invoke-Merge }
        'Optimize' { Invoke-Optimize }
        'Reports' { Invoke-Reports }
    }
} finally {
    Pop-Location
}
