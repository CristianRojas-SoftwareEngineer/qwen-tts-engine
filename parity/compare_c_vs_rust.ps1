param(
    [string]$RustBin = "qwen3-tts-rs/target/x86_64-pc-windows-gnu/release/qwen-app.exe",
    [string]$GoldenDir = "parity/golden",
    [string]$CorpusPath = "parity/corpus.json",
    [string]$OutDir = "parity/runs/latest"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Get-Location).Path
$RustAbs = Join-Path $RepoRoot $RustBin
$GoldenAbs = Join-Path $RepoRoot $GoldenDir
$OutAbs = Join-Path $RepoRoot $OutDir
New-Item -ItemType Directory -Force -Path $OutAbs | Out-Null

if (-not (Test-Path $RustAbs)) { throw "Rust binary not found: $RustAbs" }
$MetricsC = Get-Content -Raw -Encoding UTF8 (Join-Path $GoldenAbs "metrics_c.json") | ConvertFrom-Json
$Corpus = Get-Content -Raw -Encoding UTF8 $CorpusPath | ConvertFrom-Json

$env:OMP_NUM_THREADS = "$($Corpus.threads)"
Remove-Item Env:OPENBLAS_NUM_THREADS -ErrorAction SilentlyContinue

Write-Host "=== Comparing C Golden vs Rust ($RustBin) ==="

# 1. Decode-dump SNR check
$codesFile = Join-Path $RepoRoot $MetricsC.decoder_tool.codes_file
$cDecWav = Join-Path $RepoRoot $MetricsC.decoder_tool.wav_file
$rDecWav = Join-Path $OutAbs "rust_decoder_c2_en_mixed.wav"
& $RustAbs decode-dump $codesFile $Corpus.model_06b $rDecWav
if ($LASTEXITCODE -ne 0) { throw "Rust decode-dump failed" }
$decCmp = (python (Join-Path $RepoRoot "parity/analyze_audio.py") compare $cDecWav $rDecWav --with-mel) | ConvertFrom-Json
Write-Host ("decode-dump vs C: SNR={0} dB, mel_corr={1}, max_abs={2}" -f $decCmp.snr_db, $decCmp.mel_corr, $decCmp.max_abs_diff)
if ($decCmp.snr_db -lt 100.0) { throw "Gate G3 FAIL: decode-dump SNR $($decCmp.snr_db) < 100 dB" }

# 2. Corpus cases C1..C4, C6
$comparisons = @()
foreach ($cCase in $MetricsC.cases) {
    $spec = $Corpus.cases | Where-Object { $_.id -eq $cCase.id }
    if (-not $spec) { continue }
    $rWav = Join-Path $OutAbs "$($cCase.id).wav"
    $args = @("-d", $Corpus.model_06b, "-j", "$($Corpus.threads)", "--seed", "$($Corpus.seed)", "-s", $spec.speaker, "-l", $spec.language, "--text", $spec.text, "-o", $rWav) + @($spec.extra_args)

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $out = & $RustAbs @args 2>&1
    $sw.Stop()
    if ($LASTEXITCODE -ne 0) { throw "Rust failed on $($cCase.id): $out" }

    $rRtf = $null; $rTtfa = $null
    $errStr = ($out | Out-String)
    if ($errStr -match 'RTF\s+([0-9.]+)') { $rRtf = [double]$Matches[1] }
    if ($errStr -match 'TTFA:\s+([0-9.]+)\s+ms') { $rTtfa = [double]$Matches[1] }

    $cWav = Join-Path $RepoRoot $cCase.wav_rel
    $cmp = (python (Join-Path $RepoRoot "parity/analyze_audio.py") compare $cWav $rWav --with-mel) | ConvertFrom-Json
    Write-Host ("{0,-28} C_RTF={1} R_RTF={2} mel_corr={3} SNR={4}dB" -f $cCase.id, $cCase.rtf, $rRtf, $cmp.mel_corr, $cmp.snr_db)
    $comparisons += [ordered]@{
        id = $cCase.id
        c_rtf = $cCase.rtf
        r_rtf = $rRtf
        c_ttfa_ms = $cCase.ttfa_ms
        r_ttfa_ms = $rTtfa
        mel_corr = $cmp.mel_corr
        snr_db = $cmp.snr_db
    }
}

$comparisons | ConvertTo-Json -Depth 4 | Out-File -Encoding UTF8 (Join-Path $OutAbs "comparison.json")
Write-Host "=== Comparison saved to $OutDir/comparison.json ==="
