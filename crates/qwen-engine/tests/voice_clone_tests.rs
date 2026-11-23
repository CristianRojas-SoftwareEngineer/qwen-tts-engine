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
    read_wav_24khz, trim_trailing_silence, write_wav, CloneOptions,
    Delivery, MemorySink, Model06B, SessionScratch,
    SynthesisRequest, Synthesizer, Voice,
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
    let samples24: Vec<f32> = (0..24000).map(|i| ((i as f32) * 0.05).sin() * 0.2).collect();

    // 1. Un archivo que no es WAV da la pista de ffmpeg
    let fake = dir.join("fake.mp3");
    fs::write(&fake, b"dummy").unwrap();
    let err = read_wav_24khz(&fake).unwrap_err().to_string();
    assert!(
        err.contains("ffmpeg -i") && err.contains("24000"),
        "se obtuvo: {err}"
    );

    // 2. Una firma que no es RIFF/WAVE se rechaza aunque la extensión sea .wav
    let bad_mp3 = dir.join("disguised_mp3.wav");
    fs::write(&bad_mp3, b"ID3\x03\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
    let err = read_wav_24khz(&bad_mp3).unwrap_err().to_string();
    assert!(
        err.contains("no es un archivo WAV") && err.contains("ffmpeg -i"),
        "se obtuvo: {err}"
    );

    // 3. Un WAV de otra frecuencia (16000 Hz) falla de forma explícita
    let wav_16k = dir.join("ref_16k.wav");
    let mut bytes = encode_wav_bytes(&vec![0.1f32; 16000]);
    bytes[24..28].copy_from_slice(&16000u32.to_le_bytes());
    fs::write(&wav_16k, bytes).unwrap();
    let err = read_wav_24khz(&wav_16k).unwrap_err().to_string();
    assert!(
        err.contains("16000 Hz") && err.contains("ffmpeg -i"),
        "se obtuvo: {err}"
    );

    // 4. Un WAV estéreo se rechaza
    let wav_stereo = dir.join("stereo.wav");
    let mut bytes = encode_wav_bytes(&samples24[..2400]);
    bytes[22..24].copy_from_slice(&2u16.to_le_bytes());
    fs::write(&wav_stereo, bytes).unwrap();
    let err = read_wav_24khz(&wav_stereo).unwrap_err().to_string();
    assert!(err.contains("2 canales"), "se obtuvo: {err}");

    // 5. Un WAV válido a 24 kHz se lee entero
    let wav_24k = dir.join("ref_24k.wav");
    write_wav(&wav_24k, &samples24).unwrap();
    assert_eq!(read_wav_24khz(&wav_24k).unwrap().len(), 24000);

    // 6. Un WAV con sample_rate = 0 falla
    let wav_sr0 = dir.join("bad_sr0.wav");
    let mut bytes = encode_wav_bytes(&samples24[..128]);
    bytes[24..28].copy_from_slice(&0u32.to_le_bytes());
    fs::write(&wav_sr0, bytes).unwrap();
    assert!(read_wav_24khz(&wav_sr0).is_err());
}

#[test]
fn test_trim_trailing_silence_and_mel_shape() {
    let mut clip = vec![0.0f32; 24000 * 5];
    for (i, v) in clip[..24000 * 3].iter_mut().enumerate() {
        *v = ((i as f32) * 0.05).sin() * 0.3;
    }
    trim_trailing_silence(&mut clip, true);
    assert!(clip.len() < 24000 * 5 && clip.len() >= 24000 * 3);

    let (mel, n_frames) = mel_spectrogram(&clip).unwrap();
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

    // Una extensión distinta de .qvoice se rechaza antes de leer (ni siquiera existe el archivo)
    let bin = dir.join("voice.bin");
    let err = Voice::load(&bin, true)
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

    // Un .qvoice de otra versión se rechaza nombrando la ruta
    let other = dir.join("other_version.qvoice");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"QVCE");
    let old_version = 5u32;
    bytes.extend_from_slice(&old_version.to_le_bytes());
    bytes.extend_from_slice(&1024u32.to_le_bytes());
    fs::write(&other, bytes).unwrap();
    let err = Voice::load(&other, true)
        .err()
        .unwrap()
        .to_string();
    assert!(
        err.contains(&format!("version {old_version}"))
            && err.contains("version 6")
            && err.contains("other_version.qvoice"),
        "se obtuvo: {err}"
    );
}

#[test]
fn test_markup_with_default_voice_texts() {
    // Un tag desconocido no es marcado: queda como texto y no activa el modo compose
    let plan = plan_script("Devo andare [xyz] Ciao.", MarkupMode::Auto);
    let Script::Plain(text) = &plan else {
        panic!("se esperaba Plain");
    };
    assert!(text.contains("[xyz]"));

    // Pausas y muletillas siguen funcionando
    let plan = plan_script("Oh! [hmm] [pause:0.5] Great!", MarkupMode::Auto);
    let Script::Spans(spans) = &plan else {
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
    let ref_phrase = "The weather is beautiful today, perfect for a walk in the park.";
    let req = |lang: i32| SynthesisRequest {
        language_id: lang,
        sampling: SamplingParams {
            seed: DEFAULT_SEED,
            ..SamplingParams::default()
        },
        silent: true,
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
        &Script::Plain(ref_phrase.to_string()),
        Delivery::Buffer,
    )
    .unwrap();
    let (ref_samples, t_ref) = (out_ref.samples.unwrap(), out_ref.timing);
    assert!(t_ref.frames >= 10 && !ref_samples.is_empty());
    write_wav(&ref_wav, &ref_samples).unwrap();

    // 2. Clonado solo con x-vector
    let xvec_path = dir.join("xvec.qvoice");
    let xvec_qv = clone_voice(
        &runtime,
        &model,
        &ref_wav,
        &CloneOptions {
            silent: true,
            ..Default::default()
        },
    )
    .unwrap();
    xvec_qv.write_to_file(&xvec_path).unwrap();
    let xvec_built = Voice::from_qvoice(xvec_qv.clone(), "ref".to_string());
    assert_eq!(xvec_built.name, "ref");

    // 3. Tamaño exacto del .qvoice
    assert_eq!(fs::metadata(&xvec_path).unwrap().len() as usize, 4112);
    let xvec_file = QVoiceFile::read_from_file(&xvec_path).unwrap();
    assert_eq!(xvec_file.lang_id, 0, "clonar sin -l no guarda idioma");

    // 4. Carga de la voz y síntesis de un texto corto
    let xvec_voice = Voice::load(&xvec_path, true).unwrap();
    assert_eq!(xvec_voice.name, "xvec", "la voz se llama como su archivo");
    assert_eq!(xvec_voice.language_id, None);

    let out = Synthesizer {
        runtime: &runtime,
        model: &model,
        voice: &xvec_voice,
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
        "Che bella notizia! [pause:400ms] Devo andare... Ciao.",
        MarkupMode::Auto,
    );
    let Script::Spans(spans) = &plan else {
        panic!("se esperaba Spans");
    };
    let sampling = resolve_sampling(
        SamplingParams::default(),
        SeedChoice::Fixed(DEFAULT_SEED),
        None,
    );
    assert!(spans.len() >= 3);
    let compose_req = SynthesisRequest {
        language_id: 2070,
        sampling,
        silent: true,
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
        .synthesize(&mut scratch, &compose_req, &abort, &plan, Delivery::Buffer)
        .unwrap()
        .samples
        .unwrap();
    assert!(n_spoken >= 2 && composed.len() > SAMPLE_RATE);
    let _ = encode_wav_bytes(&composed);

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
            &plan,
            Delivery::Stream(&mut stream_sink),
        )
        .unwrap();
    assert!(streamed.samples.is_none());
    assert!(!stream_sink.samples.is_empty());
}
