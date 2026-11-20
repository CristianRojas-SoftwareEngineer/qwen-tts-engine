// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::net::Ipv4Addr;

use qwen_core::config::CODEC_VOCAB_SIZE;
use qwen_core::{SamplingParams, SeedChoice};
use qwen_engine::{DEFAULT_CHUNK_FRAMES, DEFAULT_COMPOSE_PAUSE_S, DEFAULT_MAX_FRAMES};

/// Qué hace el proceso, resuelto una sola vez al parsear (ver [`CliMode::resolve`]).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CliMode {
    /// `--caps`.
    Caps,
    /// `--self-test`.
    SelfTest,
    /// `--matmat-bench`.
    MatmatBench,
    /// `--list-voices <carpeta|archivo>`.
    ListVoices(String),
    /// `--delete-voice <archivo>`.
    DeleteVoice(String),
    /// `--save-voice` sin `--text` ni `--serve`: solo crea el perfil de voz.
    CreateVoice,
    /// `--serve <puerto>`.
    Serve(u16),
    /// `--text` o `--compose`: una síntesis.
    Synthesize,
    /// Sin ningún trabajo que hacer (se reporta como error tras exigir `--model-dir`).
    #[default]
    NoJob,
}

impl CliMode {
    /// Precedencia: diagnóstico (`--caps`, `--self-test`, `--matmat-bench`), `--list-voices`,
    /// `--delete-voice`, crear voz, servidor y síntesis.
    pub fn resolve(cfg: &CliArgs) -> Self {
        if cfg.show_caps {
            return Self::Caps;
        }
        if cfg.self_test {
            return Self::SelfTest;
        }
        if cfg.matmat_bench {
            return Self::MatmatBench;
        }
        if let Some(dir) = &cfg.list_voices_dir {
            return Self::ListVoices(dir.clone());
        }
        if let Some(path) = &cfg.delete_voice_path {
            return Self::DeleteVoice(path.clone());
        }
        if cfg.save_voice.is_some() && cfg.text.is_none() && cfg.serve_port.is_none() {
            return Self::CreateVoice;
        }
        if let Some(port) = cfg.serve_port {
            return Self::Serve(port);
        }
        if cfg.text.is_some() || cfg.compose_spec.is_some() {
            return Self::Synthesize;
        }
        Self::NoJob
    }
}

/// Argumentos CLI parseados para el binario `qwen_tts`.
#[derive(Debug, Clone)]
pub struct CliArgs {
    /// Modo resuelto por [`CliMode::resolve`] al terminar el parseo.
    pub mode: CliMode,
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
            mode: CliMode::default(),
            model_dir: None,
            text: None,
            output: "output.wav".to_string(),
            language: None,
            temperature: None,
            top_k: sampling.top_k as i32,
            top_p: sampling.top_p,
            rep_penalty: sampling.rep_penalty,
            max_tokens: DEFAULT_MAX_FRAMES,
            max_duration: None,
            threads: None,
            seed: SeedChoice::Default,
            greedy_warmup: sampling.greedy_warmup,
            use_int8: false,
            use_int4: false,
            quant_mixed: false,
            stream: false,
            stdout_pcm: false,
            stream_chunk: DEFAULT_CHUNK_FRAMES,
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
            compose_pause: DEFAULT_COMPOSE_PAUSE_S,
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
    eprintln!("Uso: {prog} -d <model_dir> -t <text> [opciones]");
    eprintln!("Opciones (Qwen3-TTS 0.6B):");
    eprintln!("  -d, --model-dir <path>     Directorio del modelo (Qwen3-TTS 0.6B Base)");
    eprintln!("  -t, --text <string>        Texto a sintetizar");
    eprintln!("  -o, --output <path>        Archivo WAV de salida (por defecto: output.wav)");
    eprintln!("  -l, --language <name>      Idioma (por defecto: el de la voz; si no, Spanish)");
    eprintln!("  -T, --temperature <float>  Temperatura de muestreo (por defecto: 0.325)");
    eprintln!("  -k, --top-k <int>          Muestreo top-k (por defecto: 50)");
    eprintln!("  -p, --top-p <float>        Muestreo top-p (por defecto: 1.0)");
    eprintln!("  -r, --rep-penalty <float>  Penalización por repetición (por defecto: 1.05)");
    eprintln!("  -m, --max-tokens <int>     Máximo de tokens (por defecto: {DEFAULT_MAX_FRAMES})");
    eprintln!("  -j, --threads <int>        Número de hilos (0=automático)");
    eprintln!("  --int8                     Talker + Code Predictor cuantizados a INT8");
    eprintln!("  --int4                     Talker + Code Predictor cuantizados a Q4_0");
    eprintln!("  --quant-mixed              Talker + CP en Q4_0 con atención INT8 y FFN Q4_0");
    eprintln!("  --stream                   Escribe el audio al WAV de forma progresiva");
    eprintln!(
        "  --stdout                   Escribe PCM s16le sin cabecera por stdout (implica --stream)"
    );
    eprintln!("  --stream-chunk <n>         Máximo de frames por chunk de streaming (2..250, por defecto: {DEFAULT_CHUNK_FRAMES})");
    eprintln!("  --volume <f>               Ganancia de salida (por defecto: 1.0)");
    eprintln!(
        "  --rate <f>                 Velocidad del habla, conservando el tono (por defecto: 1.0)"
    );
    eprintln!("  --seed <n>                 Semilla aleatoria (por defecto: 4; < 0 para semilla temporal)");
    eprintln!("  --greedy-warmup <n>        Primeros n frames en modo voraz (T=0, k=1)");
    eprintln!("  --max-duration <secs>      Duración máxima del audio en segundos");
    eprintln!("  --ref-audio <path>         WAV de referencia mono de 24 kHz para clonar la voz");
    eprintln!("  --ref-text <string>        Transcripción de --ref-audio para la clonación ICL");
    eprintln!("  --xvector-only             Usa solo el embedding del locutor (sin ICL)");
    eprintln!("  --save-voice <path>        Guarda la voz (.qvoice)");
    eprintln!("  --load-voice <path>        Carga una voz (.qvoice; por defecto: la voz default integrada)");
    eprintln!("  --icl-frames <n>           Limita los frames de referencia ICL (0=todos)");
    eprintln!("  --voice-name <name>        Nombre de voz en los metadatos del .qvoice");
    eprintln!("  --max-ref-duration <secs>  Segundos máximos del audio de referencia (por defecto: 30, 0=todo)");
    eprintln!("  --list-voices <dir>        Lista los archivos .qvoice de un directorio");
    eprintln!("  --delete-voice <path>      Borra un archivo .qvoice");
    eprintln!("  --roughness <0..1>         Mezcla Q2_0 de down_proj del Code Predictor");
    eprintln!("  --compose <spec>           Síntesis de varios tramos con marcado en línea");
    eprintln!("  --compose-pause <s>        Pausa por defecto entre tramos hablados (por defecto: {DEFAULT_COMPOSE_PAUSE_S})");
    eprintln!("  --no-compose               Pasa los [tags] literalmente al modelo");
    eprintln!("  --serve <port>             Inicia el servidor HTTP en el puerto indicado");
    eprintln!("  --host <ipv4>              Dirección IPv4 de enlace (por defecto: 127.0.0.1)");
    eprintln!("  --watch-stdin              Termina de inmediato cuando stdin llega a EOF");
    eprintln!("  --onset-fade <ms>          Duración de la rampa de entrada inicial (ms)");
    eprintln!("  --tail-trim                Recorta la cola metálica degenerada");
    eprintln!(
        "  --seed-audition <N>        Audición secuencial de las N mejores semillas (hasta 64)"
    );
    eprintln!("  --audition-keep            Conserva todas las tomas de la audición como <out>.seed<S>.wav");
    eprintln!("  -S, --silent               Suprime los logs de stderr");
    eprintln!("  -D, --debug                Diagnósticos internos detallados");
    eprintln!("  --caps                     Imprime las capacidades de CPU/SIMD/BLAS y termina");
    eprintln!("  --self-test                Ejecuta la autoprueba numérica de los kernels y termina");
    eprintln!(
        "  --matmat-bench             Ejecuta el microbenchmark matmat frente a B*matvec y termina"
    );
    eprintln!("  -h, --help                 Muestra esta ayuda y termina");
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
        return Err(format!("Error: {flag} requiere un argumento"));
    }
    Ok(args[*idx].clone())
}

fn parse_finite_f32(s: &str, flag_name: &str) -> Result<f32, String> {
    s.parse::<f32>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("Error: {flag_name} inválido '{s}'"))
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
        return Err(format!(
            "Error: {name} {v} fuera de rango [{min:.1}, {max:.1}]"
        ));
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
        .map_err(|_| format!("Error: {name} inválido '{s}'"))
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
                    .map_err(|_| format!("Error: top-k inválido '{s}'"))?;
                if v < 0 || v > CODEC_VOCAB_SIZE as i32 {
                    return Err(format!(
                        "Error: top-k {v} fuera de rango [0, {CODEC_VOCAB_SIZE}]"
                    ));
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
                    return Err("Error: max-tokens debe ser >= 1".to_string());
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
                    .map_err(|_| format!("Error: seed inválido '{s}'"))?;
                if val < 0 {
                    out.seed = SeedChoice::Clock;
                } else if val <= u32::MAX as i64 {
                    out.seed = SeedChoice::Fixed(val as u32);
                } else {
                    return Err(format!("Error: seed {val} excede u32::MAX"));
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
                    .map_err(|_| format!("Error: icl-frames inválido '{s}'"))?;
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
                    .map_err(|_| format!("Error: puerto inválido '{s}'"))?;
                if p == 0 {
                    return Err("Error: el puerto de --serve debe ser > 0".to_string());
                }
                out.serve_port = Some(p);
            }
            "--host" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                let ip = s.parse::<Ipv4Addr>().map_err(|_| {
                    format!("Error: --host requiere una dirección IPv4 (recibido '{s}')")
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
                    .map_err(|_| format!("Error: onset-fade inválido '{s}'"))?
                    .max(0) as u32;
            }
            "--tail-trim" => out.tail_trim = true,
            "--seed-audition" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                out.seed_audition = s
                    .parse::<usize>()
                    .map_err(|_| format!("Error: seed-audition inválido '{s}'"))?
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
                return Err(format!("Error: opción desconocida '{other}'"));
            }
        }

        i += 1;
    }

    out.mode = CliMode::resolve(&out);
    Ok(ParseOutcome::Run(Box::new(out)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reject_17b_flags_and_invalid_host() {
        let a = vec!["qwen_tts".to_string(), "--voice-design".to_string()];
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
            assert!(
                parse_cli(&args).is_err(),
                "se esperaba un error para {args:?}"
            );
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
            assert!(
                parse_cli(&args).is_err(),
                "se esperaba un error para {args:?}"
            );
        }
    }

    fn mode_of(args: &[&str]) -> CliMode {
        let v: Vec<String> = std::iter::once("qwen_tts")
            .chain(args.iter().copied())
            .map(str::to_string)
            .collect();
        let ParseOutcome::Run(cfg) = parse_cli(&v).unwrap() else {
            panic!("se esperaba Run");
        };
        cfg.mode
    }

    #[test]
    fn test_cli_mode_precedence() {
        assert_eq!(mode_of(&[]), CliMode::NoJob);
        assert_eq!(mode_of(&["-t", "hola"]), CliMode::Synthesize);
        assert_eq!(mode_of(&["--compose", "hola"]), CliMode::Synthesize);
        assert_eq!(
            mode_of(&["--serve", "8080", "-t", "x"]),
            CliMode::Serve(8080)
        );
        assert_eq!(mode_of(&["--save-voice", "v.qvoice"]), CliMode::CreateVoice);
        assert_eq!(
            mode_of(&["--save-voice", "v.qvoice", "-t", "x"]),
            CliMode::Synthesize
        );
        assert_eq!(
            mode_of(&["--save-voice", "v.qvoice", "--serve", "8080"]),
            CliMode::Serve(8080)
        );
        assert_eq!(
            mode_of(&["--delete-voice", "a.qvoice", "-t", "x"]),
            CliMode::DeleteVoice("a.qvoice".to_string())
        );
        assert_eq!(
            mode_of(&["--list-voices", "d", "--delete-voice", "a.qvoice"]),
            CliMode::ListVoices("d".to_string())
        );
        assert_eq!(
            mode_of(&["--matmat-bench", "--list-voices", "d"]),
            CliMode::MatmatBench
        );
        assert_eq!(
            mode_of(&["--self-test", "--matmat-bench"]),
            CliMode::SelfTest
        );
        assert_eq!(
            mode_of(&["--caps", "--self-test", "-t", "x"]),
            CliMode::Caps
        );
    }

    #[test]
    fn test_stdout_implies_stream() {
        let a = vec!["qwen_tts".to_string(), "--stdout".to_string()];
        let ParseOutcome::Run(cfg) = parse_cli(&a).unwrap() else {
            panic!("se esperaba Run");
        };
        assert!(cfg.stdout_pcm);
        assert!(cfg.stream);
        assert_eq!(cfg.temperature, None);
        assert_eq!(cfg.seed, SeedChoice::Default);

        let neg_seed = vec![
            "qwen_tts".to_string(),
            "--seed".to_string(),
            "-1".to_string(),
        ];
        let ParseOutcome::Run(neg_cfg) = parse_cli(&neg_seed).unwrap() else {
            panic!("se esperaba Run");
        };
        assert_eq!(neg_cfg.seed, SeedChoice::Clock);
    }
}
