param(
    [string]$Bin = "qwen3-tts/qwen_tts.exe",
    [string]$DecoderTool = "qwen3-tts/qwen_tts_decoder_tool.exe",
    [string]$CorpusPath = "parity/corpus.json",
    [string]$GoldenDir = "parity/golden"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Get-Location).Path
$BinAbs = Join-Path $RepoRoot $Bin
$DecAbs = Join-Path $RepoRoot $DecoderTool
$GoldenAbs = Join-Path $RepoRoot $GoldenDir
$WavDir = Join-Path $GoldenAbs "wavs"
New-Item -ItemType Directory -Force -Path $GoldenAbs, $WavDir | Out-Null

$Corpus = Get-Content -Raw -Encoding UTF8 $CorpusPath | ConvertFrom-Json
$Threads = [int]$Corpus.threads
$Seed = [int]$Corpus.seed

# Contract: OMP_NUM_THREADS = -j, NEVER set OPENBLAS_NUM_THREADS
$env:OMP_NUM_THREADS = "$Threads"
Remove-Item Env:OPENBLAS_NUM_THREADS -ErrorAction SilentlyContinue

function Invoke-MonitoredProcess {
    param(
        [string]$Exe,
        [string[]]$ArgList,
        [hashtable]$ExtraEnv = @{}
    )
    $prevEnv = @{}
    foreach ($k in $ExtraEnv.Keys) {
        $prevEnv[$k] = [Environment]::GetEnvironmentVariable($k, "Process")
        [Environment]::SetEnvironmentVariable($k, $ExtraEnv[$k], "Process")
    }
    try {
        $tmpOut = [System.IO.Path]::GetTempFileName()
        $tmpErr = [System.IO.Path]::GetTempFileName()
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Exe
        foreach ($a in $ArgList) { $psi.ArgumentList.Add($a) }
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
        $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $proc = [System.Diagnostics.Process]::Start($psi)
        $outTask = $proc.StandardOutput.ReadToEndAsync()
        $errTask = $proc.StandardError.ReadToEndAsync()
        [long]$peakWs = 0
        while (-not $proc.HasExited) {
            try {
                $proc.Refresh()
                if ($proc.PeakWorkingSet64 -gt $peakWs) { $peakWs = $proc.PeakWorkingSet64 }
            } catch {}
            Start-Sleep -Milliseconds 30
        }
        $proc.WaitForExit()
        try {
            $proc.Refresh()
            if ($proc.PeakWorkingSet64 -gt $peakWs) { $peakWs = $proc.PeakWorkingSet64 }
        } catch {}
        $sw.Stop()
        $stdout = $outTask.GetAwaiter().GetResult()
        $stderr = $errTask.GetAwaiter().GetResult()
        Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue
        return @{
            exit_code = $proc.ExitCode
            wall_ms   = [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
            peak_rss_mb = [math]::Round($peakWs / 1MB, 2)
            stdout    = $stdout
            stderr    = $stderr
        }
    } finally {
        foreach ($k in $ExtraEnv.Keys) {
            [Environment]::SetEnvironmentVariable($k, $prevEnv[$k], "Process")
        }
    }
}

function Parse-InferenceMetrics {
    param([string]$Stderr, [string]$WavPath)
    $frames = $null; $audioS = $null; $rtf = $null; $ttfaMs = $null
    if ($Stderr -match 'Generated\s+(\d+)\s+frames\s+\(([0-9.]+)s\s+audio\)') {
        $frames = [int]$Matches[1]
        $audioS = [double]$Matches[2]
    }
    if ($Stderr -match 'Audio:\s+([0-9.]+)s\s+generated\s+in\s+[0-9.]+s\s+\(RTF\s+([0-9.]+)\)') {
        $audioS = [double]$Matches[1]
        $rtf = [double]$Matches[2]
    }
    if ($Stderr -match 'TTFA:\s+([0-9.]+)\s+ms') {
        $ttfaMs = [double]$Matches[1]
    }
    $wavBytes = 0; $sha = $null
    if (Test-Path $WavPath) {
        $wavBytes = (Get-Item $WavPath).Length
        $sha = (Get-FileHash -Algorithm SHA256 $WavPath).Hash.ToLower()
    }
    return @{
        frames = $frames
        audio_s = $audioS
        rtf = $rtf
        ttfa_ms = $ttfaMs
        wav_bytes = $wavBytes
        wav_sha256 = $sha
    }
}

Write-Host "=== F0: Capturing C baseline into $GoldenDir ==="

# 1. Non-model introspections
$capsRes = Invoke-MonitoredProcess -Exe $BinAbs -ArgList @("--caps")
$capsRes.stdout | Out-File -Encoding UTF8 (Join-Path $GoldenAbs "caps.txt")
if ($capsRes.exit_code -ne 0) { throw "--caps failed" }

$stRes = Invoke-MonitoredProcess -Exe $BinAbs -ArgList @("--self-test")
$stRes.stdout | Out-File -Encoding UTF8 (Join-Path $GoldenAbs "selftest.txt")
if ($stRes.exit_code -ne 0) { throw "--self-test failed" }

$mm4 = Invoke-MonitoredProcess -Exe $BinAbs -ArgList @("--matmat-bench", "-j", "$Threads")
$mm4.stdout | Out-File -Encoding UTF8 (Join-Path $GoldenAbs "matmat_bench_j4.txt")
$mm1 = Invoke-MonitoredProcess -Exe $BinAbs -ArgList @("--matmat-bench", "-j", "1") -ExtraEnv @{ OMP_NUM_THREADS = "1" }
$mm1.stdout | Out-File -Encoding UTF8 (Join-Path $GoldenAbs "matmat_bench_j1.txt")

# Error handling capture
$err1 = Invoke-MonitoredProcess -Exe $BinAbs -ArgList @("-d", $Corpus.model_06b)
$err2 = Invoke-MonitoredProcess -Exe $BinAbs -ArgList @("-d", "nonexistent_dir_xyz", "--text", "x", "-o", "nul")
$err3 = Invoke-MonitoredProcess -Exe $BinAbs -ArgList @("--load-voice", "nonexistent.qvoice", "-d", $Corpus.model_06b, "--text", "x", "-o", "nul")
if ($err1.exit_code -eq 0 -or $err2.exit_code -eq 0 -or $err3.exit_code -eq 0) {
    throw "Expected non-zero exit codes on invalid CLI invocations"
}
@(
    "err_notext exit=$($err1.exit_code): $($err1.stderr.Trim())"
    "err_nomodel exit=$($err2.exit_code): $($err2.stderr.Trim())"
    "err_novoice exit=$($err3.exit_code): $($err3.stderr.Trim())"
) | Out-File -Encoding UTF8 (Join-Path $GoldenAbs "errors_test.txt")

# 2. Corpus C1..C4, C6
$Results = @()
foreach ($c in $Corpus.cases) {
    Write-Host "Running $($c.id)..."
    $modelDir = if ($c.model -eq "0.6b-base") { $Corpus.model_06b_base } else { $Corpus.model_06b }
    $wavOut = Join-Path $WavDir "$($c.id).wav"
    $args = @("-d", $modelDir, "-j", "$Threads", "--seed", "$Seed", "-s", $c.speaker, "-l", $c.language, "--text", $c.text, "-o", $wavOut)
    $args += @($c.extra_args)

    $extraEnv = @{}
    if ($c.dump_codes -eq $true) {
        $codesPath = Join-Path $GoldenAbs "codes_c2_en_mixed.txt"
        if (Test-Path $codesPath) { Remove-Item $codesPath }
        $extraEnv["QWEN_DUMP_CODES"] = $codesPath
    }

    $mlProc = $null
    $stopFlag = $null
    if ($c.memload_mb) {
        $stopFlag = [System.IO.Path]::GetTempFileName()
        Remove-Item $stopFlag -ErrorAction SilentlyContinue
        $mlProc = Start-Process -FilePath "python" -ArgumentList @((Join-Path $RepoRoot "parity/analyze_audio.py"), "memload", "--mb", "$($c.memload_mb)", "--stop-file", $stopFlag) -PassThru -WindowStyle Hidden
        Start-Sleep -Milliseconds 300
    }
    try {
        $res = Invoke-MonitoredProcess -Exe $BinAbs -ArgList $args -ExtraEnv $extraEnv
    } finally {
        if ($mlProc) {
            New-Item -ItemType File -Path $stopFlag -Force | Out-Null
            $mlProc.WaitForExit(3000) | Out-Null
            if (-not $mlProc.HasExited) { $mlProc.Kill() }
            Remove-Item $stopFlag -ErrorAction SilentlyContinue
        }
    }
    if ($res.exit_code -ne 0) {
        throw "Case $($c.id) failed with exit $($res.exit_code): $($res.stderr)"
    }
    $m = Parse-InferenceMetrics -Stderr $res.stderr -WavPath $wavOut
    Write-Host ("  RTF={0} TTFA={1}ms Audio={2}s Wall={3}ms PeakRSS={4}MB" -f $m.rtf, $m.ttfa_ms, $m.audio_s, $res.wall_ms, $res.peak_rss_mb)
    $Results += [ordered]@{
        id = $c.id
        group = $c.group
        quant = $c.quant
        stream = [bool]$c.stream
        wall_ms = $res.wall_ms
        peak_rss_mb = $res.peak_rss_mb
        frames = $m.frames
        audio_s = $m.audio_s
        rtf = $m.rtf
        ttfa_ms = $m.ttfa_ms
        wav_bytes = $m.wav_bytes
        wav_sha256 = $m.wav_sha256
        wav_rel = "parity/golden/wavs/$($c.id).wav"
    }
}

# 3. Standalone SpeechDecoder baseline on QWEN_DUMP_CODES
$codesFile = Join-Path $GoldenAbs "codes_c2_en_mixed.txt"
$decWav = Join-Path $WavDir "decoder_c2_en_mixed.wav"
$decRes = Invoke-MonitoredProcess -Exe $DecAbs -ArgList @($codesFile, $Corpus.model_06b, $decWav)
if ($decRes.exit_code -ne 0) { throw "qwen_tts_decoder_tool failed: $($decRes.stderr)" }
$decHash = (Get-FileHash -Algorithm SHA256 $decWav).Hash.ToLower()
Write-Host "Decoder tool golden: $decWav ($((Get-Item $decWav).Length) B, sha256=$decHash)"

# 4. C5_clone (0.6B -> ref.wav -> 0.6B-Base save .qvoice & .bin -> 0.6B --icl-only & --graft)
Write-Host "Running C5_clone..."
$c5Ref = Join-Path $WavDir "C5_ref.wav"
$rRef = Invoke-MonitoredProcess -Exe $BinAbs -ArgList @("-d", $Corpus.model_06b, "-j", "$Threads", "--seed", "$Seed", "-s", "ryan", "-l", "English", "--text", $Corpus.clone_case.ref_text, "-o", $c5Ref)
if ($rRef.exit_code -ne 0) { throw "C5 ref failed" }

$c5Qvoice = Join-Path $GoldenAbs "c5_ryan.qvoice"
$c5Bin = Join-Path $GoldenAbs "c5_ryan.bin"
$rSaveQ = Invoke-MonitoredProcess -Exe $BinAbs -ArgList @("-d", $Corpus.model_06b_base, "-j", "$Threads", "--ref-audio", $c5Ref, "--ref-text", $Corpus.clone_case.ref_text, "--voice-name", "c5_ryan", "--save-voice", $c5Qvoice)
if ($rSaveQ.exit_code -ne 0) { throw "C5 save .qvoice failed: $($rSaveQ.stderr)" }
$rSaveB = Invoke-MonitoredProcess -Exe $BinAbs -ArgList @("-d", $Corpus.model_06b_base, "-j", "$Threads", "--ref-audio", $c5Ref, "--xvector-only", "--save-voice", $c5Bin)
if ($rSaveB.exit_code -ne 0) { throw "C5 save .bin failed: $($rSaveB.stderr)" }

$c5IclWav = Join-Path $WavDir "C5_clone_icl.wav"
$rIcl = Invoke-MonitoredProcess -Exe $BinAbs -ArgList @("-d", $Corpus.model_06b, "-j", "$Threads", "--seed", "$Seed", "-l", $Corpus.clone_case.language, "--quant-mixed", "--load-voice", $c5Qvoice, "--icl-only", "--text", $Corpus.clone_case.target_text, "-o", $c5IclWav)
if ($rIcl.exit_code -ne 0) { throw "C5 --icl-only failed: $($rIcl.stderr)" }
$mIcl = Parse-InferenceMetrics -Stderr $rIcl.stderr -WavPath $c5IclWav
$Results += [ordered]@{
    id = "C5_clone_icl"; group = "C5_clone"; quant = "mixed"; stream = $false
    wall_ms = $rIcl.wall_ms; peak_rss_mb = $rIcl.peak_rss_mb; frames = $mIcl.frames
    audio_s = $mIcl.audio_s; rtf = $mIcl.rtf; ttfa_ms = $mIcl.ttfa_ms
    wav_bytes = $mIcl.wav_bytes; wav_sha256 = $mIcl.wav_sha256; wav_rel = "parity/golden/wavs/C5_clone_icl.wav"
}

$c5GraftWav = Join-Path $WavDir "C5_clone_graft.wav"
$rGraft = Invoke-MonitoredProcess -Exe $BinAbs -ArgList @("-d", $Corpus.model_06b, "-j", "$Threads", "--seed", "$Seed", "-l", $Corpus.clone_case.language, "--quant-mixed", "--load-voice", $c5Qvoice, "--graft", "--text", $Corpus.clone_case.target_text, "-o", $c5GraftWav)
if ($rGraft.exit_code -ne 0) { throw "C5 --graft failed: $($rGraft.stderr)" }
$mGraft = Parse-InferenceMetrics -Stderr $rGraft.stderr -WavPath $c5GraftWav
$Results += [ordered]@{
    id = "C5_clone_graft"; group = "C5_clone"; quant = "mixed"; stream = $false
    wall_ms = $rGraft.wall_ms; peak_rss_mb = $rGraft.peak_rss_mb; frames = $mGraft.frames
    audio_s = $mGraft.audio_s; rtf = $mGraft.rtf; ttfa_ms = $mGraft.ttfa_ms
    wav_bytes = $mGraft.wav_bytes; wav_sha256 = $mGraft.wav_sha256; wav_rel = "parity/golden/wavs/C5_clone_graft.wav"
}

# 5. C7_server
Write-Host "Running C7_server..."
$srvPort = [int]$Corpus.server_case.port
$srvUrl = "http://127.0.0.1:$srvPort"
$srvErr = Join-Path $GoldenAbs "c7_server_stderr.log"
$srvOut = Join-Path $GoldenAbs "c7_server_stdout.log"
$srvProc = Start-Process -FilePath $BinAbs -ArgumentList @("-d", $Corpus.model_06b, "-j", "$Threads", "--quant-mixed", "--serve", "$srvPort") -RedirectStandardError $srvErr -RedirectStandardOutput $srvOut -PassThru -WindowStyle Hidden
$c7Summary = [ordered]@{}
try {
    $up = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $h = Invoke-RestMethod -Uri "$srvUrl/v1/health" -TimeoutSec 2
            if ($h.status -eq "ok") { $up = $true; break }
        } catch {}
        Start-Sleep -Milliseconds 300
    }
    if (-not $up) { throw "C7 server failed to start" }

    $spk = Invoke-RestMethod -Uri "$srvUrl/v1/speakers" -TimeoutSec 5
    $c7Summary.speakers_count = $spk.speakers.Count

    $bodyA = @{ text = $Corpus.server_case.text_a; speaker = "ryan"; language = "English"; seed = $Seed } | ConvertTo-Json -Compress
    $bodyB = @{ text = $Corpus.server_case.text_b; speaker = "ryan"; language = "English"; seed = $Seed } | ConvertTo-Json -Compress
    $bodyOai = @{ input = $Corpus.server_case.text_a; voice = "ryan"; language = "English"; seed = $Seed } | ConvertTo-Json -Compress

    $c7TtsWav = Join-Path $WavDir "C7_tts.wav"
    $c7OaiWav = Join-Path $WavDir "C7_openai.wav"
    $c7Pcm = Join-Path $WavDir "C7_stream.pcm"

    & curl.exe -s -X POST "$srvUrl/v1/tts" -H "Content-Type: application/json" -d $bodyA -o $c7TtsWav
    & curl.exe -s -X POST "$srvUrl/v1/audio/speech" -H "Content-Type: application/json" -d $bodyOai -o $c7OaiWav
    & curl.exe -sN -X POST "$srvUrl/v1/tts/stream" -H "Content-Type: application/json" -d $bodyA -o $c7Pcm

    $wavLen = (Get-Item $c7TtsWav).Length
    $pcmLen = (Get-Item $c7Pcm).Length
    if (($wavLen - 44) -ne $pcmLen) { throw "C7 WAV($wavLen)-44 != PCM($pcmLen)" }
    $hTts = (Get-FileHash -Algorithm SHA256 $c7TtsWav).Hash.ToLower()
    $hOai = (Get-FileHash -Algorithm SHA256 $c7OaiWav).Hash.ToLower()
    if ($hTts -ne $hOai) { throw "C7 /v1/tts != /v1/audio/speech" }

    # Validations 400
    $codeWs = & curl.exe -s -o NUL -w "%{http_code}" -X POST "$srvUrl/v1/tts" -H "Content-Type: application/json" -d '{"text":"   \n\t "}'
    $codeLang = & curl.exe -s -o NUL -w "%{http_code}" -X POST "$srvUrl/v1/tts" -H "Content-Type: application/json" -d '{"text":"Hi.","language":"Klingon"}'
    $codeSpk = & curl.exe -s -o NUL -w "%{http_code}" -X POST "$srvUrl/v1/tts" -H "Content-Type: application/json" -d '{"text":"Hi.","speaker":"nadie"}'
    $codeRateStream = & curl.exe -s -o NUL -w "%{http_code}" -X POST "$srvUrl/v1/tts/stream" -H "Content-Type: application/json" -d '{"text":"Hi.","speaker":"ryan","rate":1.2}'
    if ($codeWs -ne "400" -or $codeLang -ne "400" -or $codeSpk -ne "400" -or $codeRateStream -ne "400") {
        throw "C7 400 checks failed: ws=$codeWs lang=$codeLang spk=$codeSpk rateStream=$codeRateStream"
    }

    # Cancellation
    $bodyLong = @{ text = $Corpus.server_case.text_long; speaker = "ryan"; language = "Spanish"; seed = $Seed } | ConvertTo-Json -Compress
    & curl.exe -sN --max-time 1 -X POST "$srvUrl/v1/tts/stream" -H "Content-Type: application/json" -d $bodyLong -o NUL | Out-Null
    $hAfter = Invoke-RestMethod -Uri "$srvUrl/v1/health" -TimeoutSec 5
    if ($hAfter.status -ne "ok") { throw "C7 health failed after cancel" }

    $srvProc.Refresh()
    $c7Summary.peak_rss_mb = [math]::Round($srvProc.PeakWorkingSet64 / 1MB, 2)
    $c7Summary.tts_wav_bytes = $wavLen
    $c7Summary.tts_wav_sha256 = $hTts
    $c7Summary.stream_pcm_bytes = $pcmLen
    $c7Summary.http_400_verified = $true
} finally {
    if ($srvProc -and -not $srvProc.HasExited) {
        $srvProc.Kill()
        $srvProc.WaitForExit(3000) | Out-Null
    }
}
$desglose = Select-String -Path $srvErr -Pattern '\[HTTP\] Desglose:' | ForEach-Object { $_.Line.Trim() }
$c7Summary.http_desglose = @($desglose)

$manifest = [ordered]@{
    generated_at = (Get-Date).ToString("o")
    c_git_ref = (git rev-parse --short HEAD)
    threads = $Threads
    seed = $Seed
    decoder_tool = [ordered]@{
        codes_file = "parity/golden/codes_c2_en_mixed.txt"
        wav_file = "parity/golden/wavs/decoder_c2_en_mixed.wav"
        wav_bytes = (Get-Item $decWav).Length
        wav_sha256 = $decHash
    }
    cases = $Results
    server = $c7Summary
}
$manifest | ConvertTo-Json -Depth 6 | Out-File -Encoding UTF8 (Join-Path $GoldenAbs "metrics_c.json")
Write-Host "=== F0 Golden Baseline Complete: parity/golden/metrics_c.json ==="
