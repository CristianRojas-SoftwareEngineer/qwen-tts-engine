// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::fmt;

use qwen_core::CoreError;

#[derive(Debug)]
pub enum EngineError {
    Core(CoreError),
    Io(std::io::Error),
    InvalidArg(String),
    Model(String),
    Runtime(String),
    Aborted,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Core(e) => write!(f, "{e}"),
            EngineError::Io(e) => write!(f, "I/O error: {e}"),
            EngineError::InvalidArg(s) => write!(f, "{s}"),
            EngineError::Model(s) => write!(f, "{s}"),
            EngineError::Runtime(s) => write!(f, "{s}"),
            EngineError::Aborted => write!(f, "Generation aborted"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<CoreError> for EngineError {
    fn from(e: CoreError) -> Self {
        EngineError::Core(e)
    }
}

impl From<std::io::Error> for EngineError {
    fn from(e: std::io::Error) -> Self {
        EngineError::Io(e)
    }
}
