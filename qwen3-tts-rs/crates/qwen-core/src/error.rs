use std::fmt;

#[derive(Debug)]
pub enum CoreError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidConfig(String),
    Safetensors(String),
    Tokenizer(String),
    QVoice(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::Io(e) => write!(f, "I/O error: {e}"),
            CoreError::Json(e) => write!(f, "JSON error: {e}"),
            CoreError::InvalidConfig(s) => write!(f, "{s}"),
            CoreError::Safetensors(s) => write!(f, "Safetensors error: {s}"),
            CoreError::Tokenizer(s) => write!(f, "Tokenizer error: {s}"),
            CoreError::QVoice(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for CoreError {}

impl From<std::io::Error> for CoreError {
    fn from(e: std::io::Error) -> Self {
        CoreError::Io(e)
    }
}

impl From<serde_json::Error> for CoreError {
    fn from(e: serde_json::Error) -> Self {
        CoreError::Json(e)
    }
}
