# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

[CmdletBinding()]
param(
    [string]$TelemetryFile = "",
    [string]$OutputDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
$PythonExe = "python"

if ($TelemetryFile -ne "") {
    if ($OutputDir -ne "") {
        & $PythonExe "$ScriptDir\analisador.py" $TelemetryFile $OutputDir
    } else {
        & $PythonExe "$ScriptDir\analisador.py" $TelemetryFile
    }
} else {
    & $PythonExe "$ScriptDir\analisador.py"
}
