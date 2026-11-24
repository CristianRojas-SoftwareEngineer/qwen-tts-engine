// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::path::PathBuf;
use std::sync::Mutex;

use qwen_core::config::{
    lookup_language, DEFAULT_SEED, DEFAULT_TEMPERATURE, NUM_CODEBOOKS,
};
use qwen_engine::model::{DecoderWorkspace, SdStreamState};
use qwen_engine::{
    encode_wav_bytes, speech_decoder_decode_streaming,
    AbortHandle, Delivery, EngineError, MemorySink, Model06B,
    SessionScratch, SynthesisRequest, Synthesizer, Voice,
};
use qwen_core::Script;
use qwen_kernels::ComputeRuntime;

/// Las pruebas con el modelo comparten el número de hilos de BLAS, que es global al proceso: si
/// dos síntesis corren a la vez, el orden de las sumas cambia y los bits dejan de coincidir. Se
/// ejecutan una a una.
static MODEL_LOCK: Mutex<()> = Mutex::new(());

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn test_e2e_determinism_and_abort() {
    let _serial = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = repo_root();
    let model_dir = root.join("models/qwen3-tts-0.6b-base");
    if !model_dir.exists() {
        eprintln!("Se omite test_e2e_determinism_and_abort (falta el modelo)");
        return;
    }

    let runtime = ComputeRuntime::new(Some(4));
    let model = Model06B::load(
        &model_dir,
        &runtime,
        true,
    )
    .unwrap();

    let lang = lookup_language("Spanish").unwrap();
    let voice = Voice::builtin_default();

    let req = SynthesisRequest {
        language_id: lang,
        silent: true,
        ..Default::default()
    };
    assert_eq!(req.sampling.temperature, DEFAULT_TEMPERATURE);
    assert_eq!(req.sampling.seed, DEFAULT_SEED);

    let abort = AbortHandle::new();
    let mut scratch1 = SessionScratch::new();
    let script = Script::Plain("Hola, ¿cómo estás?".to_string());
    let synth = Synthesizer {
        runtime: &runtime,
        model: &model,
        voice: &voice,
    };
    let out1 = synth
        .synthesize(&mut scratch1, &req, &abort, &script, Delivery::Buffer)
        .unwrap();
    let (s1, t1) = (out1.samples.unwrap(), out1.timing);
    assert!(t1.frames >= 10 && !s1.is_empty());

    let mut scratch2 = SessionScratch::new();
    let out2 = synth
        .synthesize(&mut scratch2, &req, &abort, &script, Delivery::Buffer)
        .unwrap();
    let (s2, t2) = (out2.samples.unwrap(), out2.timing);
    assert_eq!(t1.frames, t2.frames);
    assert_eq!(
        encode_wav_bytes(&s1),
        encode_wav_bytes(&s2),
        "Dos ejecuciones idénticas de Rust deben producir bytes WAV idénticos bit a bit"
    );

    // Comprobación de cancelación mediante AbortHandle
    let abort_now = AbortHandle::new();
    abort_now.cancel();
    let mut sink = MemorySink::new(0);
    let err = synth
        .synthesize(
            &mut scratch2,
            &req,
            &abort_now,
            &script,
            Delivery::Stream(&mut sink),
        )
        .err()
        .unwrap();
    assert!(matches!(err, EngineError::Aborted));

    // Rechazo explícito de buffers de códigos truncados en SpeechDecoder
    let bad_codes = vec![0i32; NUM_CODEBOOKS - 1];
    let mut bad_st = SdStreamState::new();
    let mut bad_ws = DecoderWorkspace::new();
    assert!(matches!(
        speech_decoder_decode_streaming(&runtime, &model.decoder, &mut bad_st, &mut bad_ws, &bad_codes, 1),
        Err(EngineError::InvalidArg(_))
    ));
}

/// Frames que el motor retiene tras el primer EOS (el mismo valor que `EOS_HOLD_FRAMES`).
const EOS_HOLD_FRAMES: usize = 2;
/// Muestras por frame: 80 ms a 24 kHz.
const SAMPLES_PER_FRAME: usize = 1920;

#[test]
fn test_eos_hold_frames() {
    let _serial = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = repo_root();
    let model_dir = root.join("models/qwen3-tts-0.6b-base");
    if !model_dir.exists() {
        eprintln!("Se omite test_eos_hold_frames (falta el modelo)");
        return;
    }

    let runtime = ComputeRuntime::new(Some(4));
    let model = Model06B::load(&model_dir, &runtime, true).unwrap();
    let voice = Voice::builtin_default();
    let synth = Synthesizer {
        runtime: &runtime,
        model: &model,
        voice: &voice,
    };
    let abort = AbortHandle::new();
    let lang = lookup_language("Spanish").unwrap();
    let make_req = |max_frames: usize| SynthesisRequest {
        language_id: lang,
        silent: true,
        max_frames,
        ..Default::default()
    };

    // Tras el primer EOS se generan exactamente `EOS_HOLD_FRAMES` frames más, y cada frame
    // entregado aporta sus 1 920 muestras (los de la cadena de silencio inicial no se entregan).
    let full_req = make_req(SynthesisRequest::default().max_frames);
    for text in ["Hola mundo", "Hmmm..."] {
        let mut scratch = SessionScratch::new();
        let script = Script::Plain(text.to_string());
        let t = synth
            .synthesize(&mut scratch, &full_req, &abort, &script, Delivery::Buffer)
            .unwrap()
            .timing;
        let eos = t
            .first_eos_frame
            .unwrap_or_else(|| panic!("«{text}» debe terminar con un EOS"));
        assert!(eos + EOS_HOLD_FRAMES <= full_req.max_frames);
        assert_eq!(t.frames, eos + EOS_HOLD_FRAMES, "frames de «{text}»");
        assert_eq!(
            t.samples_produced,
            (t.frames - t.leading_silence_frames) * SAMPLES_PER_FRAME,
            "muestras de «{text}»"
        );
        if text == "Hola mundo" {
            // La retención no pasa del tope: con `max_frames` igual al frame del primer EOS
            // más 1, solo cabe el frame retenido que sustituye al EOS.
            let capped_req = make_req(eos + 1);
            let mut scratch = SessionScratch::new();
            let capped = synth
                .synthesize(&mut scratch, &capped_req, &abort, &script, Delivery::Buffer)
                .unwrap()
                .timing;
            assert_eq!(capped.first_eos_frame, Some(eos));
            assert_eq!(capped.frames, capped_req.max_frames);
        }
    }
}

/// `code0` de la cadena de silencio inicial (la misma lista que `LEADING_SILENCE_CODES`).
const LEADING_SILENCE_CODES: [i32; 5] = [1995, 215, 212, 1181, 462];

#[test]
fn test_leading_silence_discard() {
    let _serial = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = repo_root();
    let model_dir = root.join("models/qwen3-tts-0.6b-base");
    if !model_dir.exists() {
        eprintln!("Se omite test_leading_silence_discard (falta el modelo)");
        return;
    }

    let runtime = ComputeRuntime::new(Some(4));
    let model = Model06B::load(&model_dir, &runtime, true).unwrap();
    let voice = Voice::builtin_default();
    let synth = Synthesizer {
        runtime: &runtime,
        model: &model,
        voice: &voice,
    };
    let abort = AbortHandle::new();
    let lang = lookup_language("Spanish").unwrap();
    let script = Script::Plain("Hola mundo.".to_string());
    let make_req = |max_frames: usize| SynthesisRequest {
        language_id: lang,
        silent: true,
        max_frames,
        ..Default::default()
    };
    // Frames iniciales seguidos de `codec_codes` cuyo `code0` pertenece a la cadena.
    let chain_len = |scratch: &SessionScratch| {
        scratch
            .codec_codes
            .chunks(NUM_CODEBOOKS)
            .take_while(|frame| LEADING_SILENCE_CODES.contains(&frame[0]))
            .count()
    };

    // Con el tope por defecto se descarta exactamente la cadena inicial.
    let mut scratch = SessionScratch::new();
    let t = synth
        .synthesize(
            &mut scratch,
            &make_req(SynthesisRequest::default().max_frames),
            &abort,
            &script,
            Delivery::Buffer,
        )
        .unwrap()
        .timing;
    assert!(
        t.leading_silence_frames > 0,
        "la cadena de silencio inicial de este checkpoint ha dejado de aparecer al principio de la voz default en español: revisar los ids de LEADING_SILENCE_CODES"
    );
    assert_eq!(t.leading_silence_frames, chain_len(&scratch));
    assert_eq!(
        t.samples_produced,
        (t.frames - t.leading_silence_frames) * SAMPLES_PER_FRAME
    );

    // Si la síntesis termina sin salir de la cadena, se entrega todo su audio.
    let mut scratch = SessionScratch::new();
    let t = synth
        .synthesize(
            &mut scratch,
            &make_req(3),
            &abort,
            &script,
            Delivery::Buffer,
        )
        .unwrap()
        .timing;
    assert_eq!(t.frames, 3);
    assert_eq!(
        chain_len(&scratch),
        3,
        "los tres frames deben ser de la cadena"
    );
    assert_eq!(t.leading_silence_frames, 0);
    assert_eq!(t.samples_produced, 3 * SAMPLES_PER_FRAME);
}
