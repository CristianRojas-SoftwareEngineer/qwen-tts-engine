// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::net::Ipv4Addr;

#[derive(Debug, Clone)]
pub struct CliArgs {
    pub model_dir: Option<String>,
    pub text: Option<String>,
    pub output: String,
    pub speaker_name: Option<String>,
    pub language: Option<String>,
    pub temperature: f32,
    pub temp_set: bool,
    pub top_k: i32,
    pub top_p: f32,
    pub rep_penalty: f32,
    pub max_tokens: usize,
    pub max_duration: Option<f32>,
    pub threads: Option<usize>,
    pub seed: Option<u32>,
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
    pub icl_only: bool,
    pub graft: bool,
    pub icl_frames: usize,
    pub voice_name: Option<String>,
    pub max_ref_duration: f32,
    pub target_cv: Option<String>,
    pub voice_strength: f32,
    pub vs_l0: i32,
    pub vs_l1: i32,

    pub emotion: Option<String>,
    pub emotion_strength: f32,
    pub emo_ref: Option<String>,
    pub emo_ref_text: Option<String>,
    pub roughness: f32,

    pub compose_spec: Option<String>,
    pub compose_pause: f32,
    pub no_compose: bool,

    pub serve_port: Option<u16>,
    pub serve_host: Ipv4Addr,
    pub workers: usize,
    pub batch_size: usize,
    pub watch_stdin: bool,

    pub silent: bool,
    pub debug: bool,

    pub batch_mode: bool,
    pub batch_words: usize,
    pub batch_dry: bool,
    pub batch_test: bool,
    pub batch_bench: bool,
    pub batch_multi_test: Option<usize>,

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
        Self {
            model_dir: None,
            text: None,
            output: "output.wav".to_string(),
            speaker_name: None,
            language: None,
            temperature: 0.9,
            temp_set: false,
            top_k: 50,
            top_p: 1.0,
            rep_penalty: 1.05,
            max_tokens: 8192,
            max_duration: None,
            threads: None,
            seed: None,
            greedy_warmup: 0,
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
            icl_only: false,
            graft: false,
            icl_frames: 0,
            voice_name: None,
            max_ref_duration: 30.0,
            target_cv: None,
            voice_strength: 1.0,
            vs_l0: 14,
            vs_l1: 27,
            emotion: None,
            emotion_strength: 0.25,
            emo_ref: None,
            emo_ref_text: None,
            roughness: 0.0,
            compose_spec: None,
            compose_pause: 0.12,
            no_compose: false,
            serve_port: None,
            serve_host: Ipv4Addr::new(127, 0, 0, 1),
            workers: 1,
            batch_size: 1,
            watch_stdin: false,
            silent: false,
            debug: false,
            batch_mode: false,
            batch_words: 16,
            batch_dry: false,
            batch_test: false,
            batch_bench: false,
            batch_multi_test: None,
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

pub enum ParseOutcome {
    Run(Box<CliArgs>),
    Help,
}

pub fn print_usage(prog: &str) {
    eprintln!("Usage: {prog} -d <model_dir> -t <text> [options]");
    eprintln!("Options (Qwen3-TTS 0.6B):");
    eprintln!("  -d, --model-dir <path>     Model directory (0.6B CustomVoice or Base)");
    eprintln!("  -t, --text <string>        Text to synthesize");
    eprintln!("  -o, --output <path>        Output WAV file (default: output.wav)");
    eprintln!("  -s, --speaker <name>       Speaker preset (ryan, vivian, serena, aiden, eric, dylan, uncle_fu, ono_anna, sohee)");
    eprintln!("  -l, --language <name>      Language (default: English)");
    eprintln!("  -T, --temperature <float>  Sampling temperature (default: 0.9 CLI / 0.5 server)");
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
    eprintln!("  --seed <n>                 Random seed (default: time-based)");
    eprintln!("  --greedy-warmup <n>        First n frames greedy (T=0, k=1)");
    eprintln!("  --max-duration <secs>      Max audio duration in seconds");
    eprintln!("  --ref-audio <path>         Reference 24 kHz mono WAV for voice cloning (0.6B-Base)");
    eprintln!("  --ref-text <string>        Transcript of --ref-audio for ICL cloning");
    eprintln!("  --xvector-only             Use speaker embedding only (no ICL)");
    eprintln!("  --save-voice <path>        Save voice (.qvoice or .bin)");
    eprintln!("  --load-voice <path>        Load voice (.qvoice or .bin)");
    eprintln!("  --icl-only                 Skip WDELTA when loading .qvoice");
    eprintln!("  --graft                    Ignore ref_codes, clone via x-vector (implies --icl-only)");
    eprintln!("  --icl-frames <n>           Cap ICL ref frames (0=all)");
    eprintln!("  --voice-name <name>        Voice metadata name for .qvoice");
    eprintln!("  --max-ref-duration <secs>  Max ref audio seconds (default: 30, 0=all)");
    eprintln!("  --target-cv <dir>          CustomVoice dir for WDELTA compression in --save-voice");
    eprintln!("  --voice-strength <a>       .qvoice WDELTA scale (default: 1.0)");
    eprintln!("  --vs-layers <a-b>          Layer range for --voice-strength (default: 14-27)");
    eprintln!("  --list-voices <dir>        List .qvoice files in directory");
    eprintln!("  --delete-voice <path>      Delete a .qvoice file");
    eprintln!("  --emotion <name>           Emotion asset/direction (0.6B emovoice)");
    eprintln!("  --emotion-strength <s>     Dose of dir_<tok>.bin on clone (default: 0.25)");
    eprintln!("  --emo-ref <path>           Emotion by example (24 kHz mono WAV)");
    eprintln!("  --emo-ref-text <text>      Transcript of --emo-ref");
    eprintln!("  --roughness <0..1>         Code Predictor Q2_0 down_proj blend");
    eprintln!("  --compose <spec>           Inline markup multi-span synthesis");
    eprintln!("  --compose-pause <s>        Default pause between spoken spans (default: 0.12)");
    eprintln!("  --no-compose               Pass [tags] literally to the model");
    eprintln!("  --serve <port>             Start HTTP server on port");
    eprintln!("  --host <ipv4>              Bind IPv4 address (default: 127.0.0.1)");
    eprintln!("  --workers <n>              Concurrent synthesis workers (default: 1)");
    eprintln!("  --batch-size <n>           Continuous batching width (n>=2)");
    eprintln!("  --watch-stdin              Exit immediately when stdin reaches EOF");
    eprintln!("  --batch                    Long-form sentence-packed batch synthesis");
    eprintln!("  --batch-words <n>          Target words per batch chunk (min 4, default: 16)");
    eprintln!("  --batch-dry                Print batch chunking and exit");
    eprintln!("  --batch-test               Verify batched Talker step vs single-stream");
    eprintln!("  --batch-bench              Measure batched throughput vs single-stream");
    eprintln!("  --batch-multi-test <N>     Synthesize N batch copies to bm_<i>.wav");
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
            "-s" | "--speaker" => {
                out.speaker_name = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "-l" | "--language" => {
                out.language = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "-T" | "--temperature" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.temperature = s
                    .parse::<f32>()
                    .map_err(|_| format!("Error: invalid temperature '{s}'"))?;
                out.temp_set = true;
            }
            "-k" | "--top-k" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.top_k = s
                    .parse::<i32>()
                    .map_err(|_| format!("Error: invalid top-k '{s}'"))?;
            }
            "-p" | "--top-p" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.top_p = s
                    .parse::<f32>()
                    .map_err(|_| format!("Error: invalid top-p '{s}'"))?;
            }
            "-r" | "--rep-penalty" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.rep_penalty = s
                    .parse::<f32>()
                    .map_err(|_| format!("Error: invalid rep-penalty '{s}'"))?;
            }
            "-m" | "--max-tokens" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.max_tokens = s
                    .parse::<usize>()
                    .map_err(|_| format!("Error: invalid max-tokens '{s}'"))?;
            }
            "-j" | "--threads" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                let n = s
                    .parse::<usize>()
                    .map_err(|_| format!("Error: invalid threads '{s}'"))?;
                if n > 0 {
                    out.threads = Some(n);
                }
            }
            "--seed" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                let val = s
                    .parse::<i64>()
                    .map_err(|_| format!("Error: invalid seed '{s}'"))?;
                if val >= 0 {
                    out.seed = Some(val as u32);
                } else {
                    out.seed = None;
                }
            }
            "--greedy-warmup" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.greedy_warmup = s
                    .parse::<usize>()
                    .map_err(|_| format!("Error: invalid greedy-warmup '{s}'"))?;
            }
            "--max-duration" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                let d = s
                    .parse::<f32>()
                    .map_err(|_| format!("Error: invalid max-duration '{s}'"))?;
                out.max_duration = Some(d);
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
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.stream_chunk = s
                    .parse::<usize>()
                    .map_err(|_| format!("Error: invalid stream-chunk '{s}'"))?;
            }
            "--volume" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.volume = s
                    .parse::<f32>()
                    .map_err(|_| format!("Error: invalid volume '{s}'"))?;
            }
            "--rate" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.rate = s
                    .parse::<f32>()
                    .map_err(|_| format!("Error: invalid rate '{s}'"))?;
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
            "--icl-only" => out.icl_only = true,
            "--graft" => {
                out.graft = true;
                out.icl_only = true;
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
                out.max_ref_duration = s
                    .parse::<f32>()
                    .map_err(|_| format!("Error: invalid max-ref-duration '{s}'"))?;
            }
            "--target-cv" => {
                out.target_cv = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--voice-strength" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.voice_strength = s
                    .parse::<f32>()
                    .map_err(|_| format!("Error: invalid voice-strength '{s}'"))?;
            }
            "--vs-layers" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                if let Some((a, b)) = s.split_once('-') {
                    if let (Ok(l0), Ok(l1)) = (a.parse::<i32>(), b.parse::<i32>()) {
                        out.vs_l0 = l0;
                        out.vs_l1 = l1;
                    } else {
                        return Err(format!("Error: invalid --vs-layers '{s}' (expected a-b)"));
                    }
                } else {
                    return Err(format!("Error: invalid --vs-layers '{s}' (expected a-b)"));
                }
            }
            "--emotion" => {
                out.emotion = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--emotion-strength" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.emotion_strength = s
                    .parse::<f32>()
                    .map_err(|_| format!("Error: invalid emotion-strength '{s}'"))?;
            }
            "--emo-ref" => {
                out.emo_ref = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--emo-ref-text" => {
                out.emo_ref_text = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--roughness" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.roughness = s
                    .parse::<f32>()
                    .map_err(|_| format!("Error: invalid roughness '{s}'"))?;
            }
            "--compose" => {
                out.compose_spec = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--compose-pause" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.compose_pause = s
                    .parse::<f32>()
                    .map_err(|_| format!("Error: invalid compose-pause '{s}'"))?;
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
            "--workers" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.workers = s
                    .parse::<usize>()
                    .map_err(|_| format!("Error: invalid workers '{s}'"))?
                    .max(1);
            }
            "--batch-size" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.batch_size = s
                    .parse::<usize>()
                    .map_err(|_| format!("Error: invalid batch-size '{s}'"))?
                    .max(1);
            }
            "--watch-stdin" => out.watch_stdin = true,
            "-S" | "--silent" => out.silent = true,
            "-D" | "--debug" => out.debug = true,
            "--batch" => out.batch_mode = true,
            "--batch-words" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                let w = s
                    .parse::<usize>()
                    .map_err(|_| format!("Error: invalid batch-words '{s}'"))?;
                out.batch_words = w.max(4);
            }
            "--batch-dry" => {
                out.batch_mode = true;
                out.batch_dry = true;
            }
            "--batch-test" => out.batch_test = true,
            "--batch-bench" => out.batch_bench = true,
            "--batch-multi-test" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                let n = s
                    .parse::<usize>()
                    .map_err(|_| format!("Error: invalid batch-multi-test '{s}'"))?;
                out.batch_multi_test = Some(n.max(1));
            }
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
    }

    #[test]
    fn test_graft_implies_icl_only_and_stdout_implies_stream() {
        let a = vec![
            "qwen_tts".to_string(),
            "--graft".to_string(),
            "--stdout".to_string(),
            "--batch-words".to_string(),
            "2".to_string(),
        ];
        let ParseOutcome::Run(cfg) = parse_cli(&a).unwrap() else {
            panic!("expected Run");
        };
        assert!(cfg.graft);
        assert!(cfg.icl_only);
        assert!(cfg.stdout_pcm);
        assert!(cfg.stream);
        assert_eq!(cfg.batch_words, 4);
    }
}
