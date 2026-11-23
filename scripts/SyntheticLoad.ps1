# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

<#
.SYNOPSIS
Genera y controla carga sintética de ancho de banda de memoria para medir RTF.

.DESCRIPTION
Cada hilo de carga es un proceso PowerShell independiente que copia un bloque
de memoria en bucle con [System.Buffer]::BlockCopy. La herramienta solo crea,
inspecciona y detiene esa carga: no conoce el motor, no mide RTF y no escribe
resultados de medición. El estado se guarda en un archivo JSON pequeño en la
carpeta temporal del usuario, clave por Tag, para poder invocar Start y Stop
en procesos distintos.

No usar dot-sourcing ('.\scripts\SyntheticLoad.ps1' sin '&'): se ejecutaría con
los valores por defecto. Invocar siempre como comando:
    & .\scripts\SyntheticLoad.ps1 -Action Start -Tag rtf

.PARAMETER Action
Start crea la carga, Status la inspecciona, Stop la detiene y SelfTest verifica
la herramienta sin dejar procesos.

.PARAMETER Threads
Número de procesos de carga. 0 no inicia ninguno y devuelve un handle vacío.

.PARAMETER ChunkMB
Tamaño del bloque copiado por cada proceso. Cada proceso reserva 2 x ChunkMB.

.PARAMETER DurationSec
Duración automática en segundos. 0 mantiene la carga hasta Stop.

.PARAMETER Tag
Identificador del conjunto de procesos. Solo caracteres A-Z a-z 0-9 _ y -.

.PARAMETER Handle
Handle devuelto por Start. Tiene prioridad sobre Tag en Status y Stop.

.EXAMPLE
$load = & .\scripts\SyntheticLoad.ps1 -Action Start -Threads 1 -Tag rtf
try { & .\target\release\qwen_tts.exe -d models\qwen3-tts-0.6b-base ... }
finally { & .\scripts\SyntheticLoad.ps1 -Action Stop -Handle $load }

.EXAMPLE
& .\scripts\SyntheticLoad.ps1 -Action SelfTest
#>

[CmdletBinding()]
param(
    [ValidateSet('Start','Status','Stop','SelfTest')]
    [string] $Action = 'Start',

    [ValidateRange(0,8)]
    [int] $Threads = 1,

    [ValidateRange(16,1024)]
    [int] $ChunkMB = 256,

    [ValidateRange(0,3600)]
    [int] $DurationSec = 0,

    [ValidatePattern('^[A-Za-z0-9_-]{1,32}$')]
    [string] $Tag = 'load',

    [Parameter(ValueFromPipeline = $true)]
    [object] $Handle
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Evitar que un dot-sourcing arranque carga sin querer: con '.' se omiten todas
# las acciones y solo quedan disponibles las funciones auxiliares.
if ($MyInvocation.InvocationName -eq '.') {
    Write-Warning 'SyntheticLoad.ps1: use & o .\script.ps1; el dot-sourcing no ejecuta acciones.'
    return
}

function Get-SyntheticLoadPwshPath {
    $current = (Get-Process -Id $PID -ErrorAction SilentlyContinue).Path
    if ($current) { return $current }
    foreach ($name in @('pwsh','powershell.exe','powershell')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    throw 'No se encontró el ejecutable de PowerShell.'
}

function Get-SyntheticLoadStatePath {
    param([string] $LoadTag)
    $dir = [System.IO.Path]::GetTempPath()
    return [System.IO.Path]::Combine($dir, "SyntheticLoad_$LoadTag.json")
}

function ConvertTo-SyntheticLoadProcess {
    param([System.Diagnostics.Process] $Process, [string] $LoadTag, [int] $Bytes, [int] $Seconds)
    $started = $null
    try { $started = $Process.StartTime.ToUniversalTime().ToString('o') } catch { }
    return [pscustomobject]@{
        Id        = $Process.Id
        StartTime = $started
        Bytes     = $Bytes
        Duration  = $Seconds
        Tag       = $LoadTag
    }
}

function Get-SyntheticLoadState {
    param([object] $InputHandle, [string] $LoadTag)

    if ($null -ne $InputHandle) {
        if (-not $InputHandle.PSObject.Properties['Processes']) {
            throw 'Handle de carga no válido: falta la propiedad Processes.'
        }
        return $InputHandle
    }

    $path = Get-SyntheticLoadStatePath -LoadTag $LoadTag
    if (-not (Test-Path -LiteralPath $path)) {
        throw "No hay estado de carga para Tag '$LoadTag'. Pase -Handle o inicie primero la carga."
    }
    return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
}

function Save-SyntheticLoadState {
    param([object] $State, [string] $LoadTag)
    $path = Get-SyntheticLoadStatePath -LoadTag $LoadTag
    $State | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $path -Encoding UTF8
}

function Remove-SyntheticLoadState {
    param([string] $LoadTag)
    $path = Get-SyntheticLoadStatePath -LoadTag $LoadTag
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
}

function Test-SyntheticLoadIdentity {
    param([int] $ProcessId, [string] $StartTimeIso)
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $proc) { return $null }
    if (-not $StartTimeIso) { return $proc }
    try {
        $expected = [datetime]::Parse($StartTimeIso, [cultureinfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::RoundtripKind)
        if ([math]::Abs(($proc.StartTime.ToUniversalTime() - $expected).TotalSeconds) -gt 2) { return $null }
    } catch { return $null }
    return $proc
}

function Assert-PhysicalMemory {
    param([int] $Count, [int] $Megabytes)
    $needed = [int64] $Count * 2 * [int64] $Megabytes
    $total = [int64] 0
    try { $total = [int64] (Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory } catch { $total = [int64] 8 * 1GB }
    if ($total -gt 0 -and $needed -gt ($total / 2)) {
        throw "La carga pediría $([math]::Round($needed/1GB,2)) GiB y supera la mitad de la RAM del equipo."
    }
}

function New-SyntheticLoadChildCommand {
    param([int] $Bytes, [int] $Seconds)

    $body = @'
$n = __BYTES__
$src = [byte[]]::new($n)
$dst = [byte[]]::new($n)
__DEADLINE__
while ($true) {
    __CHECK__
    [System.Buffer]::BlockCopy($src, 0, $dst, 0, $n)
}
'@
    if ($Seconds -gt 0) {
        $deadline = '$deadline = (Get-Date).AddSeconds(__SECONDS__)'.Replace('__SECONDS__', "$Seconds")
        $check = 'if ((Get-Date) -ge $deadline) { break }'
    } else {
        $deadline = ''
        $check = ''
    }
    $body = $body.Replace('__BYTES__', "$Bytes").Replace('__DEADLINE__', $deadline).Replace('__CHECK__', $check)
    return [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
}

function New-SyntheticLoadProcess {
    param([string] $Exe, [string] $Encoded, [bool] $Hidden)

    $args = @('-NoProfile','-NonInteractive','-EncodedCommand',$Encoded)
    if ($Hidden) {
        return Start-Process -FilePath $Exe -ArgumentList $args -PassThru -WindowStyle Hidden
    }
    return Start-Process -FilePath $Exe -ArgumentList $args -PassThru
}

function Start-SyntheticLoadAction {
    param(
        [string] $LoadTag,
        [int] $Count,
        [int] $BlockMB,
        [int] $Seconds
    )
    $statePath = Get-SyntheticLoadStatePath -LoadTag $LoadTag
    if (Test-Path -LiteralPath $statePath) {
        $previous = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        foreach ($p in @($previous.Processes)) {
            if ($null -ne (Test-SyntheticLoadIdentity -ProcessId $p.Id -StartTimeIso $p.StartTime)) {
                throw "Ya existe carga activa con Tag '$LoadTag'. Detenla antes de iniciar otra."
            }
        }
    }

    Assert-PhysicalMemory -Count $Count -Megabytes $BlockMB
    $bytes = $BlockMB * 1MB
    $encoded = New-SyntheticLoadChildCommand -Bytes $bytes -Seconds $Seconds
    $exe = Get-SyntheticLoadPwshPath
    $hidden = ($env:OS -eq 'Windows_NT')
    $processes = [System.Collections.Generic.List[object]]::new()

    try {
        for ($i = 0; $i -lt $Count; $i++) {
            $proc = New-SyntheticLoadProcess -Exe $exe -Encoded $encoded -Hidden $hidden
            $processes.Add((ConvertTo-SyntheticLoadProcess -Process $proc -LoadTag $LoadTag -Bytes $bytes -Seconds $Seconds))
        }
        Start-Sleep -Milliseconds 300
        foreach ($p in @($processes)) {
            if ($null -eq (Test-SyntheticLoadIdentity -ProcessId $p.Id -StartTimeIso $p.StartTime)) {
                throw "El proceso de carga $($p.Id) terminó durante el arranque."
            }
        }
    } catch {
        foreach ($p in @($processes)) {
            if ($null -ne (Test-SyntheticLoadIdentity -ProcessId $p.Id -StartTimeIso $p.StartTime)) {
                Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            }
        }
        throw
    }

    $state = [pscustomobject]@{
        Tag         = $LoadTag
        Threads     = $Count
        ChunkMB     = $BlockMB
        DurationSec = $Seconds
        StartedAt   = (Get-Date).ToUniversalTime().ToString('o')
        Processes   = @($processes.ToArray())
    }
    Save-SyntheticLoadState -State $state -LoadTag $LoadTag
    $pidsText = if (@($state.Processes).Count -gt 0) { (@($state.Processes) | ForEach-Object { $_.Id }) -join ',' } else { '(ninguno)' }
    Write-Information "Carga iniciada: Tag='$LoadTag', procesos=$Count, bloque=$BlockMB MiB, PIDs=$pidsText. Parar con: -Action Stop -Tag $LoadTag" -InformationAction Continue
    return $state
}

function Get-SyntheticLoadStatus {
    param([object] $InputHandle, [string] $LoadTag)
    $state = Get-SyntheticLoadState -InputHandle $InputHandle -LoadTag $LoadTag
    $rows = foreach ($p in @($state.Processes)) {
        $proc = Test-SyntheticLoadIdentity -ProcessId $p.Id -StartTimeIso $p.StartTime
        [pscustomobject]@{
            Tag           = $state.Tag
            Id            = $p.Id
            Alive         = ($null -ne $proc)
            WorkingSetMiB = if ($proc) { [math]::Round($proc.WorkingSet64 / 1MB, 1) } else { 0 }
            StartedAt     = $p.StartTime
        }
    }
    if (-not @($rows).Count) {
        Write-Information "Tag='$($state.Tag)': sin procesos de carga (Threads=0 o duración agotada)." -InformationAction Continue
    }
    return @($rows)
}

function Stop-SyntheticLoadAction {
    param([object] $InputHandle, [string] $LoadTag)
    $state = Get-SyntheticLoadState -InputHandle $InputHandle -LoadTag $LoadTag
    $stopped = 0
    $failed = 0
    foreach ($p in @($state.Processes)) {
        $proc = Test-SyntheticLoadIdentity -ProcessId $p.Id -StartTimeIso $p.StartTime
        if ($null -eq $proc) { continue }
        try {
            Stop-Process -Id $p.Id -Force -ErrorAction Stop
            $stopped++
        } catch {
            $failed++
        }
    }
    Start-Sleep -Milliseconds 200
    $remaining = 0
    foreach ($p in @($state.Processes)) {
        if ($null -ne (Test-SyntheticLoadIdentity -ProcessId $p.Id -StartTimeIso $p.StartTime)) { $remaining++ }
    }
    Remove-SyntheticLoadState -LoadTag $state.Tag
    $result = [pscustomobject]@{
        Tag       = $state.Tag
        Stopped   = $stopped
        Failed    = $failed
        Remaining = $remaining
    }
    Write-Information "Carga detenida: Tag='$($state.Tag)', detenidos=$stopped, fallidos=$failed, restantes=$remaining." -InformationAction Continue
    return $result
}

function Test-SyntheticLoadAction {
    $testTag = "selftest_$([guid]::NewGuid().ToString('N').Substring(0,8))"
    $details = $null
    try {
        $state = Start-SyntheticLoadAction -LoadTag $testTag -Count 1 -BlockMB 64 -Seconds 0
        $pid0 = @($state.Processes)[0].Id
        Start-Sleep -Milliseconds 800
        $rows = @(Get-SyntheticLoadStatus -InputHandle $state)
        $alive = ($rows.Count -eq 1 -and $rows[0].Alive)
        $memoryOk = ($rows.Count -eq 1 -and $rows[0].WorkingSetMiB -ge 16)
        $stop = Stop-SyntheticLoadAction -InputHandle $state
        $clean = ($stop.Remaining -eq 0 -and -not (Test-Path -LiteralPath (Get-SyntheticLoadStatePath -LoadTag $testTag)))
        $passed = ($alive -and $memoryOk -and $clean)
        return [pscustomobject]@{
            Result        = if ($passed) { 'OK' } else { 'FALLO' }
            Tag           = $testTag
            Pid           = $pid0
            WorkingSetMiB = if ($rows.Count -gt 0) { $rows[0].WorkingSetMiB } else { 0 }
            Detail        = "vivo=$alive; memoria=$memoryOk; limpieza=$clean"
        }
    } catch {
        Remove-SyntheticLoadState -LoadTag $testTag
        return [pscustomobject]@{
            Result = 'FALLO'; Tag = $testTag; Pid = $null; WorkingSetMiB = 0; Detail = $_.Exception.Message
        }
    }
}

switch ($Action) {
    'Start'    { Start-SyntheticLoadAction -LoadTag $Tag -Count $Threads -BlockMB $ChunkMB -Seconds $DurationSec }
    'Status'   { Get-SyntheticLoadStatus -InputHandle $Handle -LoadTag $Tag }
    'Stop'     { Stop-SyntheticLoadAction -InputHandle $Handle -LoadTag $Tag }
    'SelfTest' { Test-SyntheticLoadAction }
}
