// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::fs;
use std::path::PathBuf;

use qwen_core::config::{DEFAULT_SEED, ENC_DIM, SAMPLE_RATE};
use qwen_core::{
    plan_script, resolve_sampling, MarkupMode, QVoiceFile, SamplingParams, Script, SeedChoice,
    SpanSpec,
};
use qwen_engine::{
    clone_voice, encode_wav_bytes, mel_spectrogram, AbortHandle,
    read_wav_24khz, trim_trailing_silence, validate_wav_path_or_header, write_wav, CloneOptions,
    Delivery, LogLevel, MemorySink, Model06B, SessionScratch,
    SynthesisRequest, Synthesizer, Voice, VoiceLoadOptions,
};
use qwen_kernels::ComputeRuntime;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("qwen_engine_voice_clone_tests")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_wav_validation_and_ffmpeg_hints() {
    let dir = temp_dir("wav_validation");

    // 1. Rechazo por extensión (.mp3, .mp4, .flac, .ogg)
    for ext in ["mp3", "mp4", "ogg", "flac"] {
        let p = dir.join(format!("fake.{ext}"));
        fs::write(&p, b"dummy").unwrap();
        let err = validate_wav_path_or_header(&p).unwrap_err().to_string();
        assert!(
            err.contains("ffmpeg -i"),
            "Se esperaba la pista de ffmpeg para .{ext}: {err}"
        );
        assert!(
            err.contains("24000"),
            "Se esperaba la pista de 24000 Hz para .{ext}: {err}"
        );
    }

    // 2. Rechazo por magic bytes incluso con extensión .wav
    let bad_mp3 = dir.join("disguised_mp3.wav");
    fs::write(&bad_mp3, b"ID3\x03\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
    let err = read_wav_24khz(&bad_mp3).unwrap_err().to_string();
    assert!(
        err.contains("MP3") && err.contains("ffmpeg -i"),
        "se obtuvo: {err}"
    );

    let bad_ogg = dir.join("disguised_ogg.wav");
    fs::write(&bad_ogg, b"OggS\x00\x02\x00\x00\x00\x00\x00\x00").unwrap();
    let err = read_wav_24khz(&bad_ogg).unwrap_err().to_string();
    assert!(
        err.contains("OGG") && err.contains("ffmpeg -i"),
        "se obtuvo: {err}"
    );

    // 3. Un WAV válido con frecuencia distinta (16000 Hz) debe fallar explícitamente
    let wav_16k = dir.join("ref_16k.wav");
    let samples = vec![0.1f32; 16000];
    write_wav(&wav_16k, &samples, 16000).unwrap();
    let err = read_wav_24khz(&wav_16k).unwrap_err().to_string();
    assert!(
        err.contains("24000 Hz") && err.contains("16000 Hz") && err.contains("ffmpeg -i"),
        "Se esperaba un error explícito de 24 kHz, se obtuvo: {err}"
    );

    // 4. Un WAV válido a 24 kHz tiene éxito
    let wav_24k = dir.join("ref_24k.wav");
    let samples24: Vec<f32> = (0..24000)
        .map(|i| ((i as f32) * 0.05).sin() * 0.2)
        .collect();
    write_wav(&wav_24k, &samples24, 24000).unwrap();
    let read_back = read_wav_24khz(&wav_24k).unwrap();
    assert_eq!(read_back.len(), 24000);

    // 5. Un WAV con sample_rate = 0 falla en read_wav_pcm
    let wav_sr0 = dir.join("bad_sr0.wav");
    let bad_hdr = encode_wav_bytes(&samples24[..128], 0);
    fs::write(&wav_sr0, bad_hdr).unwrap();
    assert!(qwen_engine::read_wav_pcm(&wav_sr0).is_err());
}

#[test]
fn test_trim_trailing_silence_and_mel_shape() {
    let mut clip = vec![0.0f32; 24000 * 5];
    for (i, v) in clip[..24000 * 3].iter_mut().enumerate() {
        *v = ((i as f32) * 0.05).sin() * 0.3;
    }
    trim_trailing_silence(&mut clip, 24000, true);
    assert!(clip.len() < 24000 * 5 && clip.len() >= 24000 * 3);

    let (mel, n_frames) = mel_spectrogram(&clip, 24000).unwrap();
    assert!(n_frames > 250);
    assert_eq!(mel.len(), n_frames * 128);
    assert!(mel.iter().all(|v| v.is_finite()));
}

/// Bits exactos (`f32::to_bits`) de los valores 0, 511 y 1023 de `assets/default_voice.xvec`.
const DEFAULT_VOICE_BITS: [(usize, u32); 3] =
    [(0, 0x3d00fcba), (511, 0xbe193748), (1023, 0x3d73bff4)];

#[test]
fn test_default_voice_is_embedded() {
    let voice = Voice::builtin_default();
    assert_eq!(voice.name, "default");
    assert!(voice.icl.is_none(), "la voz por defecto debe ser un x-vector simple");
    assert_eq!(voice.language_id, Some(2054));
    let emb = &voice.xvector;
    assert_eq!(emb.len(), ENC_DIM);
    assert!(emb.iter().all(|v| v.is_finite()));
    let norm = emb
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();
    assert!((norm - 10.8499).abs() < 1e-3, "norma {norm}");
    for (idx, bits) in DEFAULT_VOICE_BITS {
        assert_eq!(emb[idx].to_bits(), bits, "valor {idx}");
    }
    // El archivo embebido es el mismo que el del repositorio
    let raw = fs::read(repo_root().join("crates/qwen-engine/assets/default_voice.xvec")).unwrap();
    assert_eq!(raw.len(), 4096);
    for (idx, bits) in DEFAULT_VOICE_BITS {
        let b = &raw[idx * 4..idx * 4 + 4];
        assert_eq!(u32::from_le_bytes([b[0], b[1], b[2], b[3]]), bits);
    }
}

#[test]
fn test_voice_file_extension_and_version_errors() {
    let dir = temp_dir("voice_errors");
    let opts = VoiceLoadOptions {
        use_icl: true,
        log: LogLevel::Silent,
        ..Default::default()
    };

    // Una extensión distinta de .qvoice se rechaza antes de leer (ni siquiera existe el archivo)
    let bin = dir.join("voice.bin");
    let err = Voice::load(&bin, &opts)
        .err()
        .unwrap()
        .to_string();
    assert_eq!(
        err,
        format!(
            "los archivos de voz deben ser .qvoice (se obtuvo {})",
            bin.display()
        )
    );

    // Un .qvoice v3 se rechaza nombrando la ruta
    let v3 = dir.join("old.qvoice");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"QVCE");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1024u32.to_le_bytes());
    fs::write(&v3, bytes).unwrap();
    let err = Voice::load(&v3, &opts)
        .err()
        .unwrap()
        .to_string();
    assert!(
        err.contains("version 3") && err.contains("old.qvoice"),
        "se obtuvo: {err}"
    );
}

#[test]
fn test_markup_with_default_voice_texts() {
    // `[sad]` ya no es marcado: queda como texto y no activa el modo compose
    let plan = plan_script("Devo andare [sad] Ciao.", MarkupMode::Auto);
    assert!(plan.paralinguistic.is_none());
    let Script::Plain(text) = &plan.script else {
        panic!("se esperaba Plain");
    };
    assert!(text.contains("[sad]"));

    // Pausas y `[laugh]` siguen funcionando
    let plan = plan_script("Oh! [laugh] [pause:0.5] Great!", MarkupMode::Auto);
    assert!(plan.paralinguistic.is_some());
    let Script::Spans(spans) = &plan.script else {
        panic!("se esperaba Spans");
    };
    assert!(spans.iter().any(
        |s| matches!(s, qwen_core::SpanSpec::Pause { seconds } if (*seconds - 0.5).abs() < 1e-6)
    ));
}

#[test]
fn test_clone_from_default_voice_e2e() {
    let root = repo_root();
    let base_dir = root.join("models/qwen3-tts-0.6b-base");
    if !base_dir.exists() {
        eprintln!("Se omite test_clone_from_default_voice_e2e (falta el modelo Base)");
        return;
    }

    let dir = temp_dir("e2e_clone");
    let runtime = ComputeRuntime::new(Some(4));
    let model = Model06B::load(
        &base_dir,
        &runtime,
        true,
    )
    .unwrap();
    let default_voice = Voice::builtin_default();
    let mut scratch = SessionScratch::new();

    // 1. Referencia: la voz `default` dice un texto conocido (WAV de 24 kHz)
    let ref_wav = dir.join("ref.wav");
    let ref_text = "The weather is beautiful today, perfect for a walk in the park.";
    let req = |lang: i32| SynthesisRequest {
        language_id: lang,
        sampling: SamplingParams {
            seed: DEFAULT_SEED,
            ..SamplingParams::default()
        },
        log: LogLevel::Silent,
        ..Default::default()
    };
    let abort = AbortHandle::new();
    let out_ref = Synthesizer {
        runtime: &runtime,
        model: &model,
        voice: &default_voice,
    }
    .synthesize(
        &mut scratch,
        &req(2050),
        &abort,
        &Script::Plain(ref_text.to_string()),
        Delivery::Buffer,
    )
    .unwrap();
    let (ref_samples, t_ref) = (out_ref.samples.unwrap(), out_ref.timing);
    assert!(t_ref.frames >= 10 && !ref_samples.is_empty());
    write_wav(&ref_wav, &ref_samples, SAMPLE_RATE as u32).unwrap();

    // 2. Clonado con ICL (con ref_text) y solo con x-vector
    let icl_path = dir.join("clone_icl.qvoice");
    let (icl_qv, icl_built) = clone_voice(
        &runtime,
        &model,
        &ref_wav,
        &CloneOptions {
            ref_text: Some(ref_text.to_string()),
            max_ref_seconds: 30.0,
            name: Some("clone_icl".to_string()),
            language_id: Some(2050),
            log: LogLevel::Silent,
            ..Default::default()
        },
    )
    .unwrap();
    icl_qv.write_to_file(&icl_path).unwrap();
    assert!(icl_built.icl.is_some());
    assert_eq!(icl_built.name, "clone_icl");

    let xvec_path = dir.join("xvec.qvoice");
    let (xvec_qv, xvec_built) = clone_voice(
        &runtime,
        &model,
        &ref_wav,
        &CloneOptions {
            ref_text: None,
            log: LogLevel::Silent,
            ..Default::default()
        },
    )
    .unwrap();
    xvec_qv.write_to_file(&xvec_path).unwrap();
    assert!(xvec_built.icl.is_none());
    // Sin `--voice-name` el nombre en memoria es el del WAV; el archivo guarda el nombre vacío
    assert_eq!(xvec_built.name, "ref");

    // 3. Tamaño exacto de cada .qvoice
    let icl_file = QVoiceFile::read_from_file(&icl_path).unwrap();
    assert!(icl_file.has_icl() && icl_file.n_ref_frames() >= 1);
    assert_eq!(icl_file.ref_text, ref_text);
    assert_eq!(icl_file.lang_id, 2050);
    assert_eq!(
        fs::metadata(&icl_path).unwrap().len() as usize,
        4124 + "clone_icl".len() + ref_text.len() + 64 * icl_file.n_ref_frames()
    );
    assert_eq!(fs::metadata(&xvec_path).unwrap().len() as usize, 4124);
    let xvec_file = QVoiceFile::read_from_file(&xvec_path).unwrap();
    assert!(!xvec_file.has_icl());
    assert_eq!(xvec_file.lang_id, 0, "clonar sin -l no guarda idioma");
    assert_eq!(xvec_file.voice_name, "");

    // 4. Carga de cada voz y síntesis de un texto corto
    let load_opts = VoiceLoadOptions {
        use_icl: true,
        log: LogLevel::Silent,
        ..Default::default()
    };
    let icl_voice = Voice::load(&icl_path, &load_opts).unwrap();
    assert!(icl_voice.icl.is_some());
    assert_eq!(icl_voice.name, "clone_icl");
    assert_eq!(icl_voice.language_id, Some(2050));
    let xvec_voice = Voice::load(&xvec_path, &load_opts).unwrap();
    assert!(xvec_voice.icl.is_none());
    assert_eq!(
        xvec_voice.name, "xvec",
        "un nombre guardado vacío recurre al nombre del archivo"
    );
    assert_eq!(xvec_voice.language_id, None);

    // `use_icl = false` al cargar descarta el ICL del archivo
    let icl_as_xvec = Voice::load(
        &icl_path,
        &VoiceLoadOptions {
            use_icl: false,
            log: LogLevel::Silent,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(icl_as_xvec.icl.is_none());

    // `ref_text_override` al cargar sustituye el texto ICL del archivo
    let icl_override = Voice::load(
        &icl_path,
        &VoiceLoadOptions {
            ref_text_override: Some("Another transcript.".to_string()),
            ..load_opts.clone()
        },
    )
    .unwrap();
    assert_eq!(
        icl_override.icl.as_ref().map(|i| i.ref_text.as_str()),
        Some("Another transcript.")
    );

    // El tope de frames ICL se aplica al construir la voz
    let n_file_frames = icl_file.n_ref_frames();
    assert!(n_file_frames > 3);
    let capped = Voice::load(
        &icl_path,
        &VoiceLoadOptions {
            icl_frames_cap: 3,
            ..load_opts.clone()
        },
    )
    .unwrap();
    let capped_icl = capped.icl.as_ref().unwrap();
    assert_eq!(capped_icl.ref_codes.len(), 3 * 16);
    assert_eq!(capped_icl.ref_codes[..], icl_file.ref_codes[..3 * 16]);

    for voice in [&icl_voice, &xvec_voice] {
        let out = Synthesizer {
            runtime: &runtime,
            model: &model,
            voice,
        }
        .synthesize(
            &mut scratch,
            &req(2050),
            &abort,
            &Script::Plain(
                "Hello, this is a test of the cloned voice running with the new engine.".to_string(),
            ),
            Delivery::Buffer,
        )
        .unwrap();
        let (samples, t) = (out.samples.unwrap(), out.timing);
        assert!(t.frames >= 10 && !samples.is_empty());
    }

    // 5. Guardar con una extensión que no sea .qvoice es un error
    let err = qwen_engine::ensure_qvoice_path(&dir.join("clone.bin"))
        .err()
        .unwrap()
        .to_string();
    assert!(
        err.contains("los archivos de voz deben ser .qvoice"),
        "se obtuvo: {err}"
    );

    // 6. Marcado multitramo y streaming de tramos, con la voz `default`
    let plan = plan_script(
        "Che bella notizia! [pause:400ms] Devo andare... [sigh] Ciao.",
        MarkupMode::Auto,
    );
    assert!(plan.paralinguistic.is_some());
    let Script::Spans(spans) = &plan.script else {
        panic!("se esperaba Spans");
    };
    let sampling = resolve_sampling(
        SamplingParams::default(),
        SeedChoice::Fixed(DEFAULT_SEED),
        None,
        plan.paralinguistic,
    );
    assert!(spans.len() >= 3);
    let compose_req = SynthesisRequest {
        language_id: 2070,
        sampling,
        log: LogLevel::Silent,
        ..Default::default()
    };
    let n_spoken = spans
        .iter()
        .filter(|sp| matches!(sp, SpanSpec::Utterance { .. }))
        .count();
    let default_synth = Synthesizer {
        runtime: &runtime,
        model: &model,
        voice: &default_voice,
    };
    let composed = default_synth
        .synthesize(&mut scratch, &compose_req, &abort, &plan.script, Delivery::Buffer)
        .unwrap()
        .samples
        .unwrap();
    assert!(n_spoken >= 2 && composed.len() > SAMPLE_RATE);
    let _ = encode_wav_bytes(&composed, SAMPLE_RATE as u32);

    // La entrega en streaming de los tramos no aplica doble ganancia base
    let mut stream_sink = MemorySink::new(composed.len());
    let streamed = default_synth
        .synthesize(
            &mut scratch,
            &SynthesisRequest {
                volume: 2.0,
                ..compose_req
            },
            &abort,
            &plan.script,
            Delivery::Stream(&mut stream_sink),
        )
        .unwrap();
    assert!(streamed.samples.is_none());
    assert!(!stream_sink.samples.is_empty());
}
