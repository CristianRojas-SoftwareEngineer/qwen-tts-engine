#![forbid(unsafe_code)]

pub mod audio;
pub mod code_predictor;
pub mod error;
pub mod model;
pub mod pipeline;
pub mod speech_decoder;
pub mod talker;

pub use audio::{
    apply_gain, encode_wav_bytes, f32_to_s16le_clamped, onset_fade, tail_glitch_score, tail_trim,
    time_stretch, write_wav, AudioSink, MemorySink, StdoutRawSink, WavStreamSink,
};
pub use error::EngineError;
pub use model::{
    AbortHandle, CodePredictorWeights, CpPrecision, Model06B, ModelLoadOptions, RequestConfig,
    RequestExecution, SessionScratch, SpeechDecoderWeights, SynthesisTiming, TalkerPrecision,
    TalkerWeights, VoiceBindings, VoiceIdentity,
};
pub use pipeline::{synthesize_to_samples, synthesize_to_sink};
pub use speech_decoder::{speech_decoder_decode, speech_decoder_decode_streaming};
