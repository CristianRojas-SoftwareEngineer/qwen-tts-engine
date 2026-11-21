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
pub mod synth;
pub mod talker;
pub mod voice;
pub mod voice_clone;

pub use audio::{
    encode_wav_bytes, read_wav_24khz, read_wav_pcm, trim_trailing_silence,
    validate_wav_path_or_header, write_wav, AudioSink, EdgeRampSink, MemorySink, PcmWrite,
    StdoutWriter, WavFileWriter,
};
pub use error::EngineError;
pub use model::{AbortHandle, Model06B, SessionScratch, SynthesisTiming};
pub use request::{
    LogLevel, SynthesisRequest, DEFAULT_CHUNK_FRAMES, DEFAULT_COMPOSE_PAUSE_S, DEFAULT_MAX_FRAMES,
};
pub use speech_decoder::speech_decoder_decode_streaming;
pub use synth::{Delivery, SynthesisOutput, Synthesizer};
pub use voice::{clone_voice, ensure_qvoice_path, CloneOptions, IclPrompt, Voice, VoiceLoadOptions};
pub use voice_clone::mel_spectrogram;
