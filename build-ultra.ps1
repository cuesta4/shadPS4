# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

[CmdletBinding()]
param(
    [ValidateSet('Instrument', 'Merge', 'Optimize')]
    [string]$Stage = 'Instrument',
    [ValidateSet('Source', 'IR')]
    [string]$Instrumentation = 'Source',
    [ValidateRange(1, 256)]
    [int]$Jobs = [Environment]::ProcessorCount,
    [string]$ProfileRoot,
    [switch]$Fresh
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = $PSScriptRoot
$EnvironmentFile = 'D:\CODING\SDKs\EWDK\EWDK-LLVM.env'
$QtPrefix = 'D:\CODING\SDKs\Qt\6.10.0\msvc2022_64'
$ReleasePreset = 'x64-Clang-Release'

if (-not $ProfileRoot) {
    $ProfileRoot = Join-Path $Root 'Build\x64-Clang-Ultra'
}

$InstrumentBuild = Join-Path $ProfileRoot 'Instrumented'
$OptimizedBuild = Join-Path $ProfileRoot 'Optimized'
$BuildProfileDirectory = Join-Path $ProfileRoot 'profiles\build'
$RawProfileDirectory = Join-Path $ProfileRoot 'profiles\raw'
$ProfileData = Join-Path $ProfileRoot 'profiles\shadps4.profdata'
$ReportDirectory = Join-Path $ProfileRoot 'reports'
$CoverageDirectory = Join-Path $ReportDirectory 'coverage-html'
$ProfilePattern = Join-Path $RawProfileDirectory 'shadps4-%p-%m.profraw'

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
    $env:SCCACHE_DISABLE = '1'
}

function Get-RequiredTool([string]$Name) {
    $path = Join-Path $env:LLVM_ROOT "bin\$Name.exe"
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required LLVM tool was not found at '$path'."
    }
    return $path
}

function Ensure-Directories {
    foreach ($path in @($ProfileRoot, $BuildProfileDirectory, $RawProfileDirectory, $ReportDirectory)) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
    }
}

function Get-ProfileGenerationFlags {
    $common = @(
        '/O2',
        '/Ob2',
        '/DNDEBUG',
        '/DSHADPS4_ULTRA_PROFILE',
        '/Zi',
        '/clang:-fdebug-info-for-profiling',
        '/clang:-funique-internal-linkage-names',
        '/clang:-fsave-optimization-record',
        '/clang:-fprofile-update=atomic'
    )

    if ($Instrumentation -eq 'Source') {
        $common += @(
            "/clang:-fprofile-instr-generate=$ProfilePattern",
            '/clang:-fcoverage-mapping'
        )
    } else {
        $common += @(
            "/clang:-fprofile-generate=$RawProfileDirectory",
            '/clang:-ftemporal-profile'
        )
    }

    return $common -join ' '
}

function Get-ProfileUseFlags {
    return @(
        '/O2',
        '/Ob2',
        '/DNDEBUG',
        '/Zi',
        "/clang:-fprofile-instr-use=$ProfileData",
        '/clang:-fdebug-info-for-profiling',
        '/clang:-funique-internal-linkage-names',
        '/clang:-fsave-optimization-record',
        '/clang:-Wprofile-instr-unprofiled',
        '/clang:-Wprofile-instr-out-of-date',
        '/clang:-Wno-error=profile-instr-unprofiled',
        '/clang:-Wno-error=profile-instr-out-of-date'
    ) -join ' '
}

function Configure-Build([string]$BuildDirectory, [string]$CompileFlags, [string]$LinkFlags) {
    $clangCl = Get-RequiredTool 'clang-cl'
    $lldLink = Get-RequiredTool 'lld-link'
    $llvmLib = Get-RequiredTool 'llvm-lib'
    $llvmMt = Get-RequiredTool 'llvm-mt'

    $configureArgs = @()
    if ($Fresh -or -not (Test-Path -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt'))) {
        $configureArgs += '--fresh'
    }
    $configureArgs += @(
        '--preset', $ReleasePreset,
        '-B', $BuildDirectory,
        "-DCMAKE_C_COMPILER=$clangCl",
        "-DCMAKE_CXX_COMPILER=$clangCl",
        "-DCMAKE_ASM_COMPILER=$clangCl",
        "-DCMAKE_LINKER=$lldLink",
        "-DCMAKE_AR=$llvmLib",
        "-DCMAKE_MT=$llvmMt",
        "-DCMAKE_PREFIX_PATH=$QtPrefix",
        '-DCMAKE_C_COMPILER_LAUNCHER=',
        '-DCMAKE_CXX_COMPILER_LAUNCHER=',
        '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
        "-DCMAKE_C_FLAGS_RELEASE=$CompileFlags",
        "-DCMAKE_CXX_FLAGS_RELEASE=$CompileFlags",
        "-DCMAKE_EXE_LINKER_FLAGS_RELEASE=$LinkFlags"
    )

    & cmake @configureArgs
    Assert-LastExitCode 'Ultra configuration'

    & cmake --build $BuildDirectory --config Release --parallel $Jobs
    Assert-LastExitCode 'Ultra build'
}

function Find-EmulatorBinary([string]$BuildDirectory) {
    $binary = Join-Path $BuildDirectory 'shadps4.exe'
    if (Test-Path -LiteralPath $binary) {
        return $binary
    }

    $binary = Get-ChildItem -LiteralPath $BuildDirectory -Filter 'shadps4.exe' -Recurse -File |
        Select-Object -First 1 -ExpandProperty FullName
    if (-not $binary) {
        throw "The shadps4 executable was not found below '$BuildDirectory'."
    }
    return $binary
}

function Write-ProfileEnvironment {
    $environmentScript = Join-Path $ProfileRoot 'enable-profile.ps1'
    @"
# Dot-source this file before launching the instrumented emulator or Qt GUI.
`$env:LLVM_PROFILE_FILE = '$ProfilePattern'
Write-Host "LLVM_PROFILE_FILE=`$env:LLVM_PROFILE_FILE"
"@ | Set-Content -LiteralPath $environmentScript -Encoding utf8

    $readme = Join-Path $ProfileRoot 'README.txt'
    @"
Ultra PGO experiment
====================

Instrumentation mode: $Instrumentation
Instrumented binary: $(Join-Path $InstrumentBuild 'shadps4.exe')
Raw profiles: $RawProfileDirectory
Indexed profile: $ProfileData
Reports: $ReportDirectory
Build-time profiles (ignored): $BuildProfileDirectory

The profile path is embedded in the instrumented binary. If the emulator is
launched through another process, dot-source enable-profile.ps1 in that
process's parent shell to override the path:

    . .\enable-profile.ps1

Run representative workloads (especially the GOW3 scene) with the instrumented
binary. Multiple processes/runs are kept separate by %p and %m. Then run:

    .\build-ultra.ps1 -Stage Merge

After reviewing the reports, build the profile-optimized binary with:

    .\build-ultra.ps1 -Stage Optimize

This PGO records execution counts and branch outcomes. It does not measure
hardware branch-misprediction events or cycles; those require a PMU profiler
(for example AMD uProf/ETW sampling) and are intentionally kept separate from
the compiler instrumentation.
"@ | Set-Content -LiteralPath $readme -Encoding utf8
}

function Invoke-CapturedTool([string]$Tool, [string[]]$Arguments, [string]$OutputFile) {
    $output = & $Tool @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { $_.ToString() } | Set-Content -LiteralPath $OutputFile -Encoding utf8
    if ($exitCode -ne 0) {
        throw "$Tool failed with exit code $exitCode. See '$OutputFile'."
    }
}

function Get-ProfileFiles {
    $files = @(Get-ChildItem -LiteralPath $RawProfileDirectory -Filter '*.profraw' -File -ErrorAction SilentlyContinue)
    if ($files.Count -eq 0) {
        throw "No .profraw files were found below '$RawProfileDirectory'. Run the instrumented binary first."
    }
    return $files
}

function Write-ProfileReports([string]$InstrumentedBinary) {
    $llvmProfdata = Get-RequiredTool 'llvm-profdata'
    $llvmCov = Get-RequiredTool 'llvm-cov'

    Invoke-CapturedTool $llvmProfdata @(
        'show', '--all-functions', '--counts', '--detailed-summary', '--show-format=text',
        $ProfileData
    ) (Join-Path $ReportDirectory 'profile-functions.txt')

    Invoke-CapturedTool $llvmProfdata @(
        'show', '--topn=200', '--counts', '--detailed-summary', $ProfileData
    ) (Join-Path $ReportDirectory 'profile-top-functions.txt')

    Invoke-CapturedTool $llvmProfdata @(
        'show', '--all-functions', '--counts', '--list-below-cutoff', '--value-cutoff=1',
        $ProfileData
    ) (Join-Path $ReportDirectory 'profile-zero-functions.txt')

    $jsonReport = Join-Path $ReportDirectory 'profile-functions.json'
    $jsonOutput = & $llvmProfdata @(
        'show', '--all-functions', '--counts', '--showcs', '--show-format=json',
        $ProfileData
    ) 2>&1
    $jsonExitCode = $LASTEXITCODE
    $jsonOutput | ForEach-Object { $_.ToString() } | Set-Content -LiteralPath $jsonReport -Encoding utf8
    if ($jsonExitCode -ne 0) {
        Write-Warning "llvm-profdata does not support JSON for this profile format; see '$jsonReport'."
    }

    $coverageReport = Join-Path $ReportDirectory 'coverage-summary.txt'
    $coverageArgs = @(
        'report', $InstrumentedBinary, "--instr-profile=$ProfileData", '--show-branch-summary'
    )
    $coverageOutput = & $llvmCov @coverageArgs 2>&1
    $coverageExitCode = $LASTEXITCODE
    $coverageOutput | ForEach-Object { $_.ToString() } | Set-Content -LiteralPath $coverageReport -Encoding utf8
    if ($coverageExitCode -ne 0) {
        Write-Warning "llvm-cov could not produce '$coverageReport'. The profile report is still valid."
    } else {
        New-Item -ItemType Directory -Force -Path $CoverageDirectory | Out-Null
        $htmlArgs = @(
            'show', $InstrumentedBinary, "--instr-profile=$ProfileData", '--format=html',
            '--output-dir', $CoverageDirectory, '--show-branches=count', '--show-line-counts'
        )
        $htmlOutput = & $llvmCov @htmlArgs 2>&1
        $htmlExitCode = $LASTEXITCODE
        if ($htmlExitCode -ne 0) {
            $htmlOutput | ForEach-Object { $_.ToString() } |
                Set-Content -LiteralPath (Join-Path $ReportDirectory 'coverage-html-error.txt') -Encoding utf8
            Write-Warning "llvm-cov HTML output failed; see coverage-html-error.txt."
        }

        Invoke-CapturedTool $llvmCov @(
            'export', $InstrumentedBinary, "--instr-profile=$ProfileData", '--format=lcov'
        ) (Join-Path $ReportDirectory 'coverage.lcov')
    }
}

function Write-OptimizationSummary([string]$BuildDirectory) {
    $records = @(Get-ChildItem -LiteralPath $BuildDirectory -Filter '*.opt.yaml' -Recurse -File -ErrorAction SilentlyContinue)
    $summaryPath = Join-Path $ReportDirectory 'optimization-record-summary.txt'
    $summary = [System.Collections.Generic.List[string]]::new()
    $summary.Add("Optimization record files: $($records.Count)")

    if ($records.Count -eq 0) {
        $summary.Add('No .opt.yaml files were generated.')
        $summary | Set-Content -LiteralPath $summaryPath -Encoding utf8
        return
    }

    $remarks = [System.Collections.Generic.List[object]]::new()
    foreach ($record in $records) {
        $lastPass = ''
        $lastName = ''
        foreach ($line in Get-Content -LiteralPath $record.FullName) {
            if ($line -match '^\s*Pass:\s*(.+)$') {
                $lastPass = $matches[1].Trim()
            } elseif ($line -match '^\s*Name:\s*(.+)$') {
                $lastName = $matches[1].Trim()
            } elseif ($line -match '^\s*Hotness:\s*(-?\d+)\s*$') {
                $remarks.Add([pscustomobject]@{
                    Hotness = [int64]$matches[1]
                    Pass = $lastPass
                    Name = $lastName
                    File = $record.Name
                })
            }
        }
    }

    $summary.Add("Remarks with hotness: $($remarks.Count)")
    $summary.Add('Top hot optimization remarks:')
    $summary.Add('Hotness`tPass`tName`tRecord')
    foreach ($remark in ($remarks | Sort-Object Hotness -Descending | Select-Object -First 200)) {
        $summary.Add("$($remark.Hotness)`t$($remark.Pass)`t$($remark.Name)`t$($remark.File)")
    }
    $summary.Add('')
    $summary.Add('The complete YAML records remain in the build directory.')
    $summary | Set-Content -LiteralPath $summaryPath -Encoding utf8
}

Push-Location $Root
try {
    Import-CompilerEnvironment
    Ensure-Directories

    $instrumentCompileFlags = Get-ProfileGenerationFlags
    $profileUseFlags = Get-ProfileUseFlags
    $instrumentLinkFlags = '/INCREMENTAL:NO /DEBUG'
    $profileUseLinkFlags = '/INCREMENTAL:NO /DEBUG'

    switch ($Stage) {
    'Instrument' {
        $previousProfileFile = [Environment]::GetEnvironmentVariable('LLVM_PROFILE_FILE', 'Process')
        $env:LLVM_PROFILE_FILE = Join-Path $BuildProfileDirectory 'build-%p-%m.profraw'
        try {
            Configure-Build $InstrumentBuild $instrumentCompileFlags $instrumentLinkFlags
        } finally {
            if ($null -eq $previousProfileFile) {
                Remove-Item Env:LLVM_PROFILE_FILE -ErrorAction SilentlyContinue
            } else {
                $env:LLVM_PROFILE_FILE = $previousProfileFile
            }
        }
        $instrumentedBinary = Find-EmulatorBinary $InstrumentBuild
        Write-ProfileEnvironment
        Write-Output "Instrumented binary: $instrumentedBinary"
        Write-Output "Profile pattern: $ProfilePattern"
        Write-Output "Run representative workloads, then execute:"
        Write-Output "  .\build-ultra.ps1 -Stage Merge -Instrumentation $Instrumentation"
        break
    }
    'Merge' {
        $profileFiles = Get-ProfileFiles
        $llvmProfdata = Get-RequiredTool 'llvm-profdata'
        $mergeArguments = @(
            'merge', '--sparse', '--failure-mode=warn', "--num-threads=$Jobs", '-o', $ProfileData
        )
        $mergeArguments += @($profileFiles | ForEach-Object { $_.FullName })
        & $llvmProfdata @mergeArguments
        Assert-LastExitCode 'Profile merge'

        $instrumentedBinary = Find-EmulatorBinary $InstrumentBuild
        Write-ProfileReports $instrumentedBinary
        Write-Output "Indexed profile: $ProfileData"
        Write-Output "Reports: $ReportDirectory"
        break
    }
    'Optimize' {
        if (-not (Test-Path -LiteralPath $ProfileData)) {
            throw "Indexed profile '$ProfileData' was not found. Run -Stage Merge first."
        }
        Configure-Build $OptimizedBuild $profileUseFlags $profileUseLinkFlags
        Write-OptimizationSummary $OptimizedBuild
        Write-Output "Profile-optimized binary: $(Find-EmulatorBinary $OptimizedBuild)"
        Write-Output "Optimization report: $(Join-Path $ReportDirectory 'optimization-record-summary.txt')"
        break
    }
    }
} finally {
    Pop-Location
}
