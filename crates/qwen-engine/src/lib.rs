// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![forbid(unsafe_code)]

//! Motor de inferencia en CPU para **Qwen3-TTS 0.6B** (`Base`)
//! con soporte de streaming progresivo, clonación de voz.

pub mod audio;
pub mod code_predictor;
pub mod compose;
pub mod error;
pub mod model;
pub mod pipeline;
mod prompt;
pub mod request;
pub mod speech_decoder;
pub mod speech_encoder;
pub mod talker;
pub mod voice;
pub mod voice_clone;

pub use audio::{
    apply_gain, encode_wav_bytes, f32_to_s16le_clamped, onset_fade, read_wav_24khz, read_wav_pcm,
    tail_glitch_score, tail_trim, time_stretch, trim_trailing_silence,
    validate_wav_path_or_header, write_wav, AudioSink, EdgeRampSink, MemorySink, PcmWrite, StdoutWriter, WavFileWriter,
};
pub use compose::{render_spans_buffer, render_spans_stream};
pub use error::EngineError;
pub use model::{
    AbortHandle, Bf16Weight, CodePredictorWeights, F32Weight, Model06B, ModelLoadOptions,
    SessionScratch, SpeechDecoderWeights, SynthesisTiming,
    TalkerWeights,
};
pub use pipeline::{finish_full_delivery, synthesize_to_samples, synthesize_to_sink};
pub use request::{
    LogLevel, SynthesisRequest, DEFAULT_CHUNK_FRAMES, DEFAULT_COMPOSE_PAUSE_S, DEFAULT_MAX_FRAMES,
};
pub use speech_decoder::{speech_decoder_decode, speech_decoder_decode_streaming};
pub use speech_encoder::{speech_encoder_encode, SpeechEncoderWeights};
pub use voice::{
    clone_voice, ensure_qvoice_path, CloneOptions, IclPrompt, Voice, VoiceLoadOptions,
    DEFAULT_VOICE_NAME,
};
pub use voice_clone::{mel_spectrogram, speaker_encoder_forward, SpeakerEncoderWeights};
