// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::net::Ipv4Addr;
use std::ops::RangeInclusive;

use qwen_core::config::CODEC_VOCAB_SIZE;
use qwen_core::{SamplingParams, SeedChoice};
use qwen_engine::{
    DEFAULT_CHUNK_FRAMES, DEFAULT_COMPOSE_PAUSE_S, MAX_CHUNK_FRAMES, MIN_CHUNK_FRAMES,
};

use crate::request::{
    RATE_RANGE, REP_PENALTY_RANGE, TEMPERATURE_RANGE, TOP_P_RANGE, VOLUME_RANGE,
};

/// Qué hace el proceso, resuelto una sola vez al parsear (ver [`CliMode::resolve`]).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CliMode {
    /// `--caps`.
    Caps,
    /// `--self-test`.
    SelfTest,
    /// `--list-voices <carpeta|archivo>`.
    ListVoices(String),
    /// `--ref-audio` y `--save-voice` sin `--text` ni `--serve`: solo crea el perfil de voz.
    CreateVoice,
    /// `--serve <puerto>`.
    Serve(u16),
    /// `--text`: una síntesis.
    Synthesize,
    /// Sin ningún trabajo que hacer (se reporta como error tras exigir `--model-dir`).
    #[default]
    NoJob,
}

impl CliMode {
    /// Precedencia: diagnóstico (`--caps`, `--self-test`), `--list-voices`, crear voz, servidor y síntesis.
    pub fn resolve(cfg: &CliArgs) -> Self {
        if cfg.show_caps {
            return Self::Caps;
        }
        if cfg.self_test {
            return Self::SelfTest;
        }
        if let Some(dir) = &cfg.list_voices_dir {
            return Self::ListVoices(dir.clone());
        }
        if cfg.save_voice.is_some() && cfg.text.is_none() && cfg.serve_port.is_none() {
            return Self::CreateVoice;
        }
        if let Some(port) = cfg.serve_port {
            return Self::Serve(port);
        }
        if cfg.text.is_some() {
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
    /// Temperatura pedida con `-T`; `None` deja el valor por defecto.
    pub temperature: Option<f32>,
    pub top_k: i32,
    pub top_p: f32,
    pub rep_penalty: f32,
    pub max_duration: Option<f32>,
    pub threads: Option<usize>,
    /// Semilla pedida con `--seed` (un valor negativo activa semilla temporal).
    pub seed: SeedChoice,

    pub stdout_pcm: bool,
    pub stream_chunk: usize,
    pub volume: f32,
    pub rate: f32,

    pub ref_audio: Option<String>,
    pub save_voice: Option<String>,
    pub load_voice: Option<String>,

    pub compose_pause: f32,
    pub no_compose: bool,

    pub serve_port: Option<u16>,
    pub serve_host: Ipv4Addr,
    pub watch_stdin: bool,

    pub silent: bool,

    pub show_caps: bool,
    pub self_test: bool,
    pub list_voices_dir: Option<String>,
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
            max_duration: None,
            threads: None,
            seed: SeedChoice::Default,
            stdout_pcm: false,
            stream_chunk: DEFAULT_CHUNK_FRAMES,
            volume: 1.0,
            rate: 1.0,
            ref_audio: None,
            save_voice: None,
            load_voice: None,
            compose_pause: DEFAULT_COMPOSE_PAUSE_S,
            no_compose: false,
            serve_port: None,
            serve_host: Ipv4Addr::new(127, 0, 0, 1),
            watch_stdin: false,
            silent: false,
            show_caps: false,
            self_test: false,
            list_voices_dir: None,
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
    eprintln!("  -k, --top-k <int>          Candidatos más probables entre los que se muestrea (por defecto: 50)");
    eprintln!("  -p, --top-p <float>        Probabilidad acumulada máxima de esos candidatos (por defecto: 1.0)");
    eprintln!("  -r, --rep-penalty <float>  Penalización por repetición (por defecto: 1.05)");
    eprintln!("  -j, --threads <int>        Número de hilos (0=automático)");
    eprintln!(
        "  --stdout                   Escribe PCM s16le sin cabecera por stdout (implica --silent)"
    );
    eprintln!("  --stream-chunk <n>         Máximo de frames por chunk del decodificador ({MIN_CHUNK_FRAMES}..{MAX_CHUNK_FRAMES}, por defecto: {DEFAULT_CHUNK_FRAMES})");
    eprintln!("  --volume <f>               Ganancia de salida (por defecto: 1.0)");
    eprintln!(
        "  --rate <f>                 Velocidad del habla, conservando el tono (por defecto: 1.0)"
    );
    eprintln!("  --seed <n>                 Semilla aleatoria (por defecto: 4; < 0 para semilla temporal)");
    eprintln!("  --max-duration <secs>      Duración máxima del audio en segundos");
    eprintln!("  --ref-audio <path>         WAV PCM mono de 16 bits a 24 kHz para clonar la voz");
    eprintln!("  --save-voice <path>        Guarda la voz (.qvoice)");
    eprintln!("  --load-voice <path>        Carga una voz (.qvoice; por defecto: la voz default integrada)");
    eprintln!("  --list-voices <dir>        Lista las voces .qvoice de un directorio (o la de un archivo .qvoice)");
    eprintln!("  --compose-pause <s>        Pausa por defecto entre tramos hablados (por defecto: {DEFAULT_COMPOSE_PAUSE_S})");
    eprintln!("  --no-compose               Pasa el texto tal cual: los [tags] se leen como texto");
    eprintln!("  --serve <port>             Inicia el servidor HTTP en el puerto indicado");
    eprintln!("  --host <ipv4>              Dirección IPv4 de enlace (por defecto: 127.0.0.1)");
    eprintln!("  --watch-stdin              Termina de inmediato cuando stdin llega a EOF");
    eprintln!("  -S, --silent               Suprime los logs de stderr");
    eprintln!("  --caps                     Imprime las capacidades de CPU/SIMD/BLAS y termina");
    eprintln!("  --self-test                Ejecuta la autoprueba numérica de los kernels y termina");
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
    range: RangeInclusive<f32>,
) -> Result<f32, String> {
    let s = take_val(args, idx, inline, flag)?;
    let v = parse_finite_f32(&s, name)?;
    if !range.contains(&v) {
        return Err(format!(
            "Error: {name} {v} fuera de rango [{}, {}]",
            range.start(),
            range.end()
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

/// Reglas de las opciones de voz, válidas en todos los modos. Se evalúan en
/// orden y se devuelve el error de la primera que falle.
fn validate_voice_flags(cfg: &CliArgs) -> Result<(), String> {
    if cfg.ref_audio.is_some() && cfg.load_voice.is_some() {
        return Err("Error: --ref-audio y --load-voice no se pueden combinar: clona con --ref-audio o carga con --load-voice".to_string());
    }
    if cfg.save_voice.is_some() && cfg.ref_audio.is_none() {
        return Err("Error: --save-voice requiere --ref-audio".to_string());
    }
    Ok(())
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
                    TEMPERATURE_RANGE,
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
                out.top_p = take_bounded_f32(args, &mut i, inline_val, flag, "top-p", TOP_P_RANGE)?;
            }
            "-r" | "--rep-penalty" => {
                out.rep_penalty =
                    take_bounded_f32(args, &mut i, inline_val, flag, "rep-penalty", REP_PENALTY_RANGE)?;
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
            "--max-duration" => {
                let s = take_val(args, &mut i, inline_val, flag)?;
                let v = parse_finite_f32(&s, "max-duration")?;
                if v <= 0.0 {
                    return Err("Error: --max-duration debe ser mayor que 0".to_string());
                }
                out.max_duration = Some(v);
            }
            "--stdout" => {
                out.stdout_pcm = true;
            }
            "--stream-chunk" => {
                out.stream_chunk = take_usize(args, &mut i, inline_val, flag, "stream-chunk")?;
            }
            "--volume" => {
                out.volume = take_bounded_f32(args, &mut i, inline_val, flag, "volume", VOLUME_RANGE)?;
            }
            "--rate" => {
                out.rate = take_bounded_f32(args, &mut i, inline_val, flag, "rate", RATE_RANGE)?;
            }
            "--ref-audio" => {
                out.ref_audio = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--save-voice" => {
                out.save_voice = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            "--load-voice" => {
                out.load_voice = Some(take_val(args, &mut i, inline_val, flag)?);
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
            "--caps" => out.show_caps = true,
            "--self-test" => out.self_test = true,
            "--list-voices" => {
                out.list_voices_dir = Some(take_val(args, &mut i, inline_val, flag)?);
            }
            other => {
                return Err(format!("Error: opción desconocida '{other}'"));
            }
        }

        i += 1;
    }

    validate_voice_flags(&out)?;
    out.mode = CliMode::resolve(&out);
    Ok(ParseOutcome::Run(Box::new(out)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reject_invalid_host_and_values() {
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

        for bad in [
            vec!["qwen_tts", "--temperature", "3.0"],
            vec!["qwen_tts", "--top-k", "4000"],
            vec!["qwen_tts", "--top-p", "1.5"],
            vec!["qwen_tts", "--rep-penalty", "0.2"],
            vec!["qwen_tts", "--volume", "-0.5"],
            vec!["qwen_tts", "--rate", "0.1"],
            vec!["qwen_tts", "--seed", "99999999999"],
            vec!["qwen_tts", "--max-duration", "0"],
            vec!["qwen_tts", "--max-duration", "-1"],
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
        assert_eq!(
            mode_of(&["--serve", "8080", "-t", "x"]),
            CliMode::Serve(8080)
        );
        assert_eq!(
            mode_of(&["--ref-audio", "a.wav", "--save-voice", "v.qvoice"]),
            CliMode::CreateVoice
        );
        assert_eq!(
            mode_of(&["--ref-audio", "a.wav", "--save-voice", "v.qvoice", "-t", "x"]),
            CliMode::Synthesize
        );
        assert_eq!(
            mode_of(&[
                "--ref-audio",
                "a.wav",
                "--save-voice",
                "v.qvoice",
                "--serve",
                "8080"
            ]),
            CliMode::Serve(8080)
        );
        assert_eq!(
            mode_of(&["--list-voices", "d", "-t", "x"]),
            CliMode::ListVoices("d".to_string())
        );
        assert_eq!(
            mode_of(&["--caps", "--self-test", "-t", "x"]),
            CliMode::Caps
        );
    }

    #[test]
    fn test_unknown_flag_is_rejected() {
        let args: Vec<String> = ["qwen_tts", "--opcion-inexistente"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let Err(e) = parse_cli(&args) else {
            panic!("se esperaba un error");
        };
        assert_eq!(e, "Error: opción desconocida '--opcion-inexistente'");
    }

    fn voice_flags_result(args: &[&str]) -> Result<(), String> {
        let v: Vec<String> = std::iter::once("qwen_tts")
            .chain(args.iter().copied())
            .map(str::to_string)
            .collect();
        parse_cli(&v).map(|_| ())
    }

    #[test]
    fn test_voice_flag_rules() {
        let r1 = "Error: --ref-audio y --load-voice no se pueden combinar: clona con --ref-audio o carga con --load-voice";
        let r2 = "Error: --save-voice requiere --ref-audio";
        assert_eq!(
            voice_flags_result(&["--ref-audio", "a.wav", "--load-voice", "v.qvoice"]),
            Err(r1.to_string())
        );
        assert_eq!(
            voice_flags_result(&["--save-voice", "v.qvoice"]),
            Err(r2.to_string())
        );
        // Orden: gana la primera regla que falle.
        assert_eq!(
            voice_flags_result(&[
                "--ref-audio",
                "a",
                "--load-voice",
                "b",
                "--save-voice",
                "c.qvoice"
            ]),
            Err(r1.to_string())
        );
        // Valen también con los modos de diagnóstico.
        assert_eq!(
            voice_flags_result(&["--caps", "--save-voice", "v.qvoice"]),
            Err(r2.to_string())
        );
        // Combinaciones válidas.
        assert_eq!(voice_flags_result(&["--ref-audio", "a.wav"]), Ok(()));
        assert_eq!(
            voice_flags_result(&[
                "--ref-audio",
                "a.wav",
                "--save-voice",
                "v.qvoice"
            ]),
            Ok(())
        );
        assert_eq!(
            voice_flags_result(&["--load-voice", "v.qvoice", "-t", "x"]),
            Ok(())
        );
    }

    #[test]
    fn test_stdout_and_clock_seed() {
        let a = vec!["qwen_tts".to_string(), "--stdout".to_string()];
        let ParseOutcome::Run(cfg) = parse_cli(&a).unwrap() else {
            panic!("se esperaba Run");
        };
        assert!(cfg.stdout_pcm);
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
