use std::path::PathBuf;

use qwen_core::config::{lookup_language, lookup_speaker, NUM_CODEBOOKS, SAMPLE_RATE};
use qwen_engine::model::{DecoderWorkspace, SdStreamState};
use qwen_engine::{
    encode_wav_bytes, speech_decoder_decode, speech_decoder_decode_streaming,
    synthesize_to_samples, synthesize_to_sink, EngineError, MemorySink, Model06B, ModelLoadOptions,
    RequestConfig, RequestExecution, SessionScratch, VoiceBindings,
};
use qwen_kernels::ComputeRuntime;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn snr_db_f32(ref_s: &[f32], test_s: &[f32]) -> f64 {
    let n = ref_s.len().min(test_s.len());
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for i in 0..n {
        let r = ref_s[i] as f64;
        let t = test_s[i] as f64;
        sig += r * r;
        let d = r - t;
        err += d * d;
    }
    if err == 0.0 {
        return 999.0;
    }
    10.0 * (sig / err).log10()
}

#[test]
fn test_decoder_streaming_parity() {
    let root = repo_root();
    let model_dir = root.join("models/qwen3-tts-0.6b");
    if !model_dir.exists() {
        eprintln!("Skipping test_decoder_streaming_parity (model absent)");
        return;
    }

    let n_frames = 32usize;
    let mut codes = Vec::with_capacity(n_frames * NUM_CODEBOOKS);
    for f in 0..n_frames {
        for g in 0..NUM_CODEBOOKS {
            codes.push(((f * 37 + g * 131) % 2048) as i32);
        }
    }

    let runtime = ComputeRuntime::new(Some(4));
    let model = Model06B::load(
        &model_dir,
        &runtime,
        ModelLoadOptions {
            silent: true,
            ..Default::default()
        },
    )
    .unwrap();

    // 1. Full decode
    let full_samples = speech_decoder_decode(&runtime, &model.decoder, &codes, n_frames).unwrap();
    assert_eq!(full_samples.len(), n_frames * 1920);

    // 2. Streaming decode (ramp 2, 4x8, 6x8, ..., tiled) vs full decode inside Rust
    let mut st = SdStreamState::new();
    st.cs_tile = true;
    let mut ws = DecoderWorkspace::new();
    let mut stream_samples = Vec::with_capacity(full_samples.len());
    let mut pos = 0usize;
    let mut chunk_idx = 0usize;
    while pos < n_frames {
        let target = if chunk_idx == 0 {
            2
        } else {
            let r = 4 + 2 * ((chunk_idx - 1) / 8);
            r.min(10)
        };
        let take = target.min(n_frames - pos);
        let chunk = speech_decoder_decode_streaming(
            &runtime,
            &model.decoder,
            &mut st,
            &mut ws,
            &codes[pos * NUM_CODEBOOKS..(pos + take) * NUM_CODEBOOKS],
            take,
        )
        .unwrap();
        stream_samples.extend_from_slice(&chunk);
        pos += take;
        chunk_idx += 1;
    }
    assert_eq!(stream_samples.len(), full_samples.len());
    let snr_stream = snr_db_f32(&full_samples, &stream_samples);
    assert!(
        snr_stream >= 100.0,
        "Expected streaming vs non-streaming SNR >= 100 dB, got {snr_stream:.2} dB"
    );
}

#[test]
fn test_e2e_determinism_and_abort() {
    let root = repo_root();
    let model_dir = root.join("models/qwen3-tts-0.6b");
    if !model_dir.exists() {
        eprintln!("Skipping test_e2e_determinism_and_abort (model absent)");
        return;
    }

    let runtime = ComputeRuntime::new(Some(4));
    let model = Model06B::load(
        &model_dir,
        &runtime,
        ModelLoadOptions {
            quant_mixed: true,
            silent: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(model.cp.prec_label(), "INT8+Q4_0-FFN");

    let spk = lookup_speaker("ryan").unwrap();
    let lang = lookup_language("Spanish").unwrap();
    let voice = VoiceBindings::preset(spk.id);

    let req = RequestConfig {
        speaker_id: spk.id,
        language_id: Some(lang),
        seed: 42,
        silent: true,
        ..Default::default()
    };

    let mut scratch1 = SessionScratch::new();
    let (s1, t1) = synthesize_to_samples(
        &runtime,
        &model,
        &voice,
        &mut scratch1,
        &RequestExecution::new(req.clone()),
        "Hola, ¿cómo estás?",
    )
    .unwrap();
    assert!(t1.frames >= 10 && !s1.is_empty());

    let mut scratch2 = SessionScratch::new();
    let (s2, t2) = synthesize_to_samples(
        &runtime,
        &model,
        &voice,
        &mut scratch2,
        &RequestExecution::new(req.clone()),
        "Hola, ¿cómo estás?",
    )
    .unwrap();
    assert_eq!(t1.frames, t2.frames);
    assert_eq!(
        encode_wav_bytes(&s1, SAMPLE_RATE as u32),
        encode_wav_bytes(&s2, SAMPLE_RATE as u32),
        "Two identical Rust runs must produce bit-identical WAV bytes"
    );

    // Abort handle check
    let exec_abort = RequestExecution::new(req);
    exec_abort.abort.cancel();
    let mut sink = MemorySink::new(0);
    let err = synthesize_to_sink(
        &runtime,
        &model,
        &voice,
        &mut scratch2,
        &exec_abort,
        "Hola, ¿cómo estás?",
        &mut sink,
    )
    .unwrap_err();
    assert!(matches!(err, EngineError::Aborted));
}
