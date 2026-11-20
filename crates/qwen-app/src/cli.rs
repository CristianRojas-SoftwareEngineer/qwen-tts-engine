// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::net::Ipv4Addr;

use qwen_core::config::CODEC_VOCAB_SIZE;
use qwen_core::{SamplingParams, SeedChoice};

/// Argumentos CLI parseados para el binario `qwen_tts`.
#[derive(Debug, Clone)]
pub struct CliArgs {
    pub model_dir: Option<String>,
    pub text: Option<String>,
    pub output: String,
    pub language: Option<String>,
    /// Temperatura pedida con `-T`; `None` deja el valor por defecto (o el de la paralingüística).
    pub temperature: Option<f32>,
    pub top_k: i32,
    pub top_p: f32,
    pub rep_penalty: f32,
    pub max_tokens: usize,
    pub max_duration: Option<f32>,
    pub threads: Option<usize>,
    /// Semilla pedida con `--seed` (un valor negativo activa semilla temporal).
    pub seed: SeedChoice,
    pub greedy_warmup: usize,

    pub use_int8: bool,
    pub use_int4: bool,
    pub quant_mixed: bool,

    pub stream: bool,
    pub stdout_pcm: bool,
    pub stream_chunk: usize,
    pub volume: f32,
    pub rate: f32,

    pub ref_audio: Option<String>,
    pub ref_text: Option<String>,
    pub xvector_only: bool,
    pub save_voice: Option<String>,
    pub load_voice: Option<String>,
    pub icl_frames: usize,
    pub voice_name: Option<String>,
    pub max_ref_duration: f32,

    pub roughness: f32,

    pub compose_spec: Option<String>,
    pub compose_pause: f32,
    pub no_compose: bool,

    pub serve_port: Option<u16>,
    pub serve_host: Ipv4Addr,
    pub watch_stdin: bool,

    pub silent: bool,
    pub debug: bool,

    pub onset_fade_ms: u32,
    pub tail_trim: bool,
    pub seed_audition: usize,
    pub audition_keep: bool,

    pub show_caps: bool,
    pub self_test: bool,
    pub matmat_bench: bool,
    pub list_voices_dir: Option<String>,
    pub delete_voice_path: Option<String>,
}

impl Default for CliArgs {
    fn default() -> Self {
        let sampling = SamplingParams::default();
        Self {
            model_dir: None,
            text: None,
            output: "output.wav".to_string(),
            language: None,
            temperature: None,
            top_k: sampling.top_k as i32,
            top_p: sampling.top_p,
            rep_penalty: sampling.rep_penalty,
            max_tokens: 8192,
            max_duration: None,
            threads: None,
            seed: SeedChoice::Default,
            greedy_warmup: sampling.greedy_warmup,
            use_int8: false,
            use_int4: false,
            quant_mixed: false,
            stream: false,
            stdout_pcm: false,
            stream_chunk: 10,
            volume: 1.0,
            rate: 1.0,
            ref_audio: None,
            ref_text: None,
            xvector_only: false,
            save_voice: None,
            load_voice: None,
            icl_frames: 0,
            voice_name: None,
            max_ref_duration: 30.0,
            roughness: 0.0,
            compose_spec: None,
            compose_pause: 0.12,
            no_compose: false,
            serve_port: None,
            serve_host: Ipv4Addr::new(127, 0, 0, 1),
            watch_stdin: false,
            silent: false,
            debug: false,
            onset_fade_ms: 0,
            tail_trim: false,
            seed_audition: 0,
            audition_keep: false,
            show_caps: false,
            self_test: false,
            matmat_bench: false,
            list_voices_dir: None,
            delete_voice_path: None,
        }
    }
}

/// Resultado del parseo de argumentos CLI.
pub enum ParseOutcome {
    Run(Box<CliArgs>),
    Help,
}

/// Imprime la ayuda de uso de `qwen_tts` por `stderr`.
pub fn print_usage(prog: &str) {
    eprintln!("Usage: {prog} -d <model_dir> -t <text> [options]");
    eprintln!("Options (Qwen3-TTS 0.6B):");
    eprintln!("  -d, --model-dir <path>     Model directory (Qwen3-TTS 0.6B Base)");
    eprintln!("  -t, --text <string>        Text to synthesize");
    eprintln!("  -o, --output <path>        Output WAV file (default: output.wav)");
    eprintln!("  -l, --language <name>      Language (default: the voice language, else English)");
    eprintln!("  -T, --temperature <float>  Sampling temperature (default: 0.325)");
    eprintln!("  -k, --top-k <int>          Top-k sampling (default: 50)");
    eprintln!("  -p, --top-p <float>        Top-p sampling (default: 1.0)");
    eprintln!("  -r, --rep-penalty <float>  Repetition penalty (default: 1.05)");
    eprintln!("  -m, --max-tokens <int>     Max tokens (default: 8192)");
    eprintln!("  -j, --threads <int>        Number of threads (0=auto)");
    eprintln!("  --int8                     INT8 quantized Talker + Code Predictor");
    eprintln!("  --int4                     Q4_0 quantized Talker + Code Predictor");
    eprintln!("  --quant-mixed              Q4_0 Talker + CP with INT8 attention and Q4_0 FFN");
    eprintln!("  --stream                   Stream audio progressively to WAV");
    eprintln!("  --stdout                   Output raw s16le PCM to stdout (implies --stream)");
    eprintln!("  --stream-chunk <n>         Max frames per stream chunk (2..250, default: 10)");
    eprintln!("  --volume <f>               Output gain (default: 1.0)");
    eprintln!("  --rate <f>                 Speaking rate, pitch-preserving (default: 1.0)");
    eprintln!("  --seed <n>                 Random seed (default: 4; < 0 for time-based)");
    eprintln!("  --greedy-warmup <n>        First n frames greedy (T=0, k=1)");
    eprintln!("  --max-duration <secs>      Max audio duration in seconds");
    eprintln!("  --ref-audio <path>         Reference 24 kHz mono WAV for voice cloning");
    eprintln!("  --ref-text <string>        Transcript of --ref-audio for ICL cloning");
    eprintln!("  --xvector-only             Use speaker embedding only (no ICL)");
    eprintln!("  --save-voice <path>        Save voice (.qvoice)");
    eprintln!("  --load-voice <path>        Load voice (.qvoice; default: the built-in default voice)");
    eprintln!("  --icl-frames <n>           Cap ICL ref frames (0=all)");
    eprintln!("  --voice-name <name>        Voice metadata name for .qvoice");
    eprintln!("  --max-ref-duration <secs>  Max ref audio seconds (default: 30, 0=all)");
    eprintln!("  --list-voices <dir>        List .qvoice files in directory");
    eprintln!("  --delete-voice <path>      Delete a .qvoice file");
    eprintln!("  --roughness <0..1>         Code Predictor Q2_0 down_proj blend");
    eprintln!("  --compose <spec>           Inline markup multi-span synthesis");
    eprintln!("  --compose-pause <s>        Default pause between spoken spans (default: 0.12)");
    eprintln!("  --no-compose               Pass [tags] literally to the model");
    eprintln!("  --serve <port>             Start HTTP server on port");
    eprintln!("  --host <ipv4>              Bind IPv4 address (default: 127.0.0.1)");
    eprintln!("  --watch-stdin              Exit immediately when stdin reaches EOF");
    eprintln!("  --onset-fade <ms>          Onset fade-in duration (ms)");
    eprintln!("  --tail-trim                Trim degenerate metallic tail");
    eprintln!("  --seed-audition <N>        Sequential best-of-N seed audition (up to 64)");
    eprintln!("  --audition-keep            Keep all audition takes as <out>.seed<S>.wav");
    eprintln!("  -S, --silent               Suppress stderr logs");
    eprintln!("  -D, --debug                Verbose internal diagnostics");
    eprintln!("  --caps                     Print CPU/SIMD/BLAS capabilities and exit");
    eprintln!("  --self-test                Run kernel numeric self-test and exit");
    eprintln!("  --matmat-bench             Run matmat vs B*matvec microbenchmark and exit");
    eprintln!("  -h, --help                 Show this help and exit");
}

fn take_val(
    args: &[String],
    idx: &mut usize,
    inline: Option<String>,
    flag: &str,
) -> Result<String, String> {
    if let Some(v) = inline {
        return Ok(v);
    }
    *idx += 1;
    if *idx >= args.len() {
        return Err(format!("Error: {flag} requires an argument"));
    }
    Ok(args[*idx].clone())
}

fn parse_finite_f32(s: &str, flag_name: &str) -> Result<f32, String> {
    s.parse::<f32>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("Error: invalid {flag_name} '{s}'"))
}

fn take_bounded_f32(
    args: &[String],
    idx: &mut usize,
    inline: Option<String>,
    flag: &str,
    name: &str,
    min: f32,
    max: f32,
) -> Result<f32, String> {
    let s = take_val(args, idx, inline, flag)?;
    let v = parse_finite_f32(&s, name)?;
    if !(min..=max).contains(&v) {
        return Err(format!("Error: {name} {v} out of range [{min:.1}, {max:.1}]"));
    }
    Ok(v)
}

fn take_usize(
    args: &[String],
    idx: &mut usize,
    inline: Option<String>,
    flag: &str,
    name: &str,
) -> Result<usize, String> {
    let s = take_val(args, idx, inline, flag)?;
    s.parse::<usize>()
        .map_err(|_| format!("Error: invalid {name} '{s}'"))
}

/// Parsea y valida los argumentos de línea de comandos en [`ParseOutcome`].
pub fn parse_cli(args: &[String]) -> Result<ParseOutcome, String> {
    let mut out = CliArgs::default();
    let mut i = 1usize;

    while i < args.len() {
        let raw = &args[i];
        if raw == "-h" || raw == "--help" {
            return Ok(ParseOutcome::Help);
        }

        let (flag, inline_val) = if raw.starts_with("--") {
            if let Some((k, v)) = raw.split_once('=') {
                (k, Some(v.to_string()))
            } else {
                (raw.as_str(), None)
            }
        } else {
            (raw.as_str(), None)
        };

        match flag {
            "-d" | "--model-dir" => {
                out.model_dir = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "-t" | "--text" => {
                out.text = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "-o" | "--output" => {
                out.output = take_val(args, &mut i, inline_val, flag)?;
            }
            "-l" | "--language" => {
                out.language = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "-T" | "--temperature" => {
                out.temperature = Some(take_bounded_f32(
                    args,
                    &mut i,
                    inline_val,
                    flag,
                    "temperature",
                    0.0,
                    2.0,
                )?);
            }
            "-k" | "--top-k" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                let v = s
                    .parse::<i32>()
                    .map_err(|_| format!("Error: invalid top-k '{s}'"))?;
                if v < 0 || v > CODEC_VOCAB_SIZE as i32 {
                    return Err(format!("Error: top-k {v} out of range [0, {CODEC_VOCAB_SIZE}]"));
                }
                out.top_k = v;
            }
            "-p" | "--top-p" => {
                out.top_p = take_bounded_f32(args, &mut i, inline_val, flag, "top-p", 0.0, 1.0)?;
            }
            "-r" | "--rep-penalty" => {
                out.rep_penalty =
                    take_bounded_f32(args, &mut i, inline_val, flag, "rep-penalty", 0.5, 2.0)?;
            }
            "-m" | "--max-tokens" => {
                let m = take_usize(args, &mut i, inline_val, flag, "max-tokens")?;
                if m == 0 {
                    return Err("Error: max-tokens must be >= 1".to_string());
                }
                out.max_tokens = m;
            }
            "-j" | "--threads" => {
                let n = take_usize(args, &mut i, inline_val, flag, "threads")?;
                if n > 0 {
                    out.threads = Some(n);
                }
            }
            "--seed" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                let val = s
                    .parse::<i64>()
                    .map_err(|_| format!("Error: invalid seed '{s}'"))?;
                if val < 0 {
                    out.seed = SeedChoice::Clock;
                } else if val <= u32::MAX as i64 {
                    out.seed = SeedChoice::Fixed(val as u32);
                } else {
                    return Err(format!("Error: seed {val} exceeds u32::MAX"));
                }
            }
            "--greedy-warmup" => {
                out.greedy_warmup = take_usize(args, &mut i, inline_val, flag, "greedy-warmup")?;
            }
            "--max-duration" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.max_duration = Some(parse_finite_f32(&s, "max-duration")?);
            }
            "--int8" => out.use_int8 = true,
            "--int4" => out.use_int4 = true,
            "--quant-mixed" => {
                out.use_int4 = true;
                out.quant_mixed = true;
            }
            "--stream" => out.stream = true,
            "--stdout" => {
                out.stdout_pcm = true;
                out.stream = true;
            }
            "--stream-chunk" => {
                out.stream_chunk = take_usize(args, &mut i, inline_val, flag, "stream-chunk")?;
            }
            "--volume" => {
                out.volume = take_bounded_f32(args, &mut i, inline_val, flag, "volume", 0.0, 10.0)?;
            }
            "--rate" => {
                out.rate = take_bounded_f32(args, &mut i, inline_val, flag, "rate", 0.25, 4.0)?;
            }
            "--ref-audio" => {
                out.ref_audio = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--ref-text" => {
                out.ref_text = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--xvector-only" => out.xvector_only = true,
            "--save-voice" => {
                out.save_voice = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--load-voice" => {
                out.load_voice = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--icl-frames" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.icl_frames = s
                    .parse::<usize>()
                    .map_err(|_| format!("Error: invalid icl-frames '{s}'"))?;
            }
            "--voice-name" => {
                out.voice_name = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--max-ref-duration" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.max_ref_duration = parse_finite_f32(&s, "max-ref-duration")?;
            }
            "--roughness" => {
                out.roughness =
                    take_bounded_f32(args, &mut i, inline_val, flag, "roughness", 0.0, 1.0)?;
            }
            "--compose" => {
                out.compose_spec = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--compose-pause" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.compose_pause = parse_finite_f32(&s, "compose-pause")?;
            }
            "--no-compose" => out.no_compose = true,
            "--serve" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                let p = s
                    .parse::<u16>()
                    .map_err(|_| format!("Error: invalid port '{s}'"))?;
                if p == 0 {
                    return Err("Error: --serve port must be > 0".to_string());
                }
                out.serve_port = Some(p);
            }
            "--host" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                let ip = s.parse::<Ipv4Addr>().map_err(|_| {
                    format!("Error: --host requires an IPv4 address (got '{s}')")
                })?;
                out.serve_host = ip;
            }
            "--watch-stdin" => out.watch_stdin = true,
            "-S" | "--silent" => out.silent = true,
            "-D" | "--debug" => out.debug = true,
            "--onset-fade" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.onset_fade_ms = s
                    .parse::<i32>()
                    .map_err(|_| format!("Error: invalid onset-fade '{s}'"))?
                    .max(0) as u32;
            }
            "--tail-trim" => out.tail_trim = true,
            "--seed-audition" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.seed_audition = s
                    .parse::<usize>()
                    .map_err(|_| format!("Error: invalid seed-audition '{s}'"))?
                    .clamp(1, 64);
            }
            "--audition-keep" => out.audition_keep = true,
            "--caps" => out.show_caps = true,
            "--self-test" => out.self_test = true,
            "--matmat-bench" => out.matmat_bench = true,
            "--list-voices" => {
                out.list_voices_dir = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--delete-voice" => {
                out.delete_voice_path = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            other => {
                return Err(format!("Error: unknown option '{other}'"));
            }
        }

        i += 1;
    }

    Ok(ParseOutcome::Run(Box::new(out)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reject_17b_flags_and_invalid_host() {
        let a = vec![
            "qwen_tts".to_string(),
            "--voice-design".to_string(),
        ];
        assert!(parse_cli(&a).is_err());

        let b = vec![
            "qwen_tts".to_string(),
            "--instruct".to_string(),
            "angry".to_string(),
        ];
        assert!(parse_cli(&b).is_err());

        let c = vec![
            "qwen_tts".to_string(),
            "--host".to_string(),
            "not-an-ip".to_string(),
        ];
        assert!(parse_cli(&c).is_err());

        let d = vec![
            "qwen_tts".to_string(),
            "--rate".to_string(),
            "NaN".to_string(),
        ];
        assert!(parse_cli(&d).is_err());

        for removed in [
            vec!["qwen_tts", "-s", "ryan"],
            vec!["qwen_tts", "--speaker", "ryan"],
            vec!["qwen_tts", "--graft"],
            vec!["qwen_tts", "--icl-only"],
            vec!["qwen_tts", "--target-cv", "x"],
            vec!["qwen_tts", "--voice-strength", "0.5"],
            vec!["qwen_tts", "--vs-layers", "1-2"],
            vec!["qwen_tts", "--emotion", "sad"],
            vec!["qwen_tts", "--emotion-strength", "0.5"],
            vec!["qwen_tts", "--emo-ref", "x.wav"],
            vec!["qwen_tts", "--emo-ref-text", "x"],
            vec!["qwen_tts", "--batch-size", "2"],
            vec!["qwen_tts", "--workers", "2"],
            vec!["qwen_tts", "--batch"],
            vec!["qwen_tts", "--batch-words", "16"],
            vec!["qwen_tts", "--batch-dry"],
            vec!["qwen_tts", "--batch-multi-test", "2"],
            vec!["qwen_tts", "--batch-test"],
            vec!["qwen_tts", "--batch-bench"],
        ] {
            let args: Vec<String> = removed.into_iter().map(str::to_string).collect();
            assert!(parse_cli(&args).is_err(), "expected error for {args:?}");
        }

        for bad in [
            vec!["qwen_tts", "--temperature", "3.0"],
            vec!["qwen_tts", "--top-k", "4000"],
            vec!["qwen_tts", "--top-p", "1.5"],
            vec!["qwen_tts", "--rep-penalty", "0.2"],
            vec!["qwen_tts", "--volume", "-0.5"],
            vec!["qwen_tts", "--rate", "0.1"],
            vec!["qwen_tts", "--roughness", "1.2"],
            vec!["qwen_tts", "--seed", "99999999999"],
            vec!["qwen_tts", "--max-tokens", "0"],
        ] {
            let args: Vec<String> = bad.into_iter().map(str::to_string).collect();
            assert!(parse_cli(&args).is_err(), "expected error for {args:?}");
        }
    }

    #[test]
    fn test_stdout_implies_stream() {
        let a = vec![
            "qwen_tts".to_string(),
            "--stdout".to_string(),
        ];
        let ParseOutcome::Run(cfg) = parse_cli(&a).unwrap() else {
            panic!("expected Run");
        };
        assert!(cfg.stdout_pcm);
        assert!(cfg.stream);
        assert_eq!(cfg.temperature, None);
        assert_eq!(cfg.seed, SeedChoice::Default);

        let neg_seed = vec!["qwen_tts".to_string(), "--seed".to_string(), "-1".to_string()];
        let ParseOutcome::Run(neg_cfg) = parse_cli(&neg_seed).unwrap() else {
            panic!("expected Run");
        };
        assert_eq!(neg_cfg.seed, SeedChoice::Clock);
    }
}
