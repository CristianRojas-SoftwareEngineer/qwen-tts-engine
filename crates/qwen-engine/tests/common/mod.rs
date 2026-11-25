// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::path::{Path, PathBuf};

/// Carpeta temporal de una prueba. Se borra al soltar la guarda, también si la prueba falla,
/// porque `Drop` corre al desenrollar el pánico.
pub struct TestDir(PathBuf);

impl TestDir {
    /// Crea la carpeta `name` en el temporal del sistema, descartando un resto previo.
    pub fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
