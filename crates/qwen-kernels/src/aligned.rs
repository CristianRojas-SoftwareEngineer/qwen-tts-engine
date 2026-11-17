// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// Buffer contiguo alineado a línea de caché (alineación por defecto de 64 bytes)
/// para SIMD, BLAS, KV cache y workspaces del decodificador.
pub struct AlignedBuf<T, const ALIGN: usize = 64> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send, const ALIGN: usize> Send for AlignedBuf<T, ALIGN> {}
unsafe impl<T: Sync, const ALIGN: usize> Sync for AlignedBuf<T, ALIGN> {}

impl<T: Copy + Default, const ALIGN: usize> AlignedBuf<T, ALIGN> {
    pub fn new_zeroed(len: usize) -> Self {
        const {
            assert!(ALIGN >= std::mem::align_of::<T>());
            assert!(ALIGN.is_power_of_two());
        }
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                cap: 0,
                _marker: PhantomData,
            };
        }
        let layout = Self::layout_for(len);
        let raw = unsafe { alloc_zeroed(layout) };
        let Some(nn) = NonNull::new(raw as *mut T) else {
            handle_alloc_error(layout);
        };
        Self {
            ptr: nn,
            len,
            cap: len,
            _marker: PhantomData,
        }
    }

    pub fn from_slice(src: &[T]) -> Self {
        let mut buf = Self::new_zeroed(src.len());
        buf.as_mut_slice().copy_from_slice(src);
        buf
    }

    /// Garantiza que la capacidad sea al menos `new_len` y ajusta la longitud a `new_len`,
    /// inicializando a cero los nuevos elementos (ranura monótona de workspace).
    pub fn resize_zeroed(&mut self, new_len: usize) {
        const {
            assert!(ALIGN >= std::mem::align_of::<T>());
            assert!(ALIGN.is_power_of_two());
        }
        if new_len <= self.cap {
            if new_len > self.len {
                unsafe {
                    std::ptr::write_bytes(self.ptr.as_ptr().add(self.len), 0, new_len - self.len);
                }
            }
            self.len = new_len;
            return;
        }
        let new_cap = new_len.max(self.cap.saturating_mul(2));
        let new_layout = Self::layout_for(new_cap);
        let raw = unsafe { alloc_zeroed(new_layout) };
        let Some(nn) = NonNull::new(raw as *mut T) else {
            handle_alloc_error(new_layout);
        };
        if self.len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), nn.as_ptr(), self.len);
            }
        }
        self.dealloc_current();
        self.ptr = nn;
        self.cap = new_cap;
        self.len = new_len;
    }

    /// Rellena con ceros todos los elementos activos `0..len`.
    pub fn fill_zero(&mut self) {
        if self.len > 0 {
            unsafe {
                std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.len);
            }
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    fn layout_for(cap: usize) -> Layout {
        let bytes = cap
            .checked_mul(std::mem::size_of::<T>())
            .expect("AlignedBuf byte size overflow");
        Layout::from_size_align(bytes.max(1), ALIGN).expect("invalid AlignedBuf layout")
    }

    fn dealloc_current(&mut self) {
        if self.cap > 0 {
            let layout = Self::layout_for(self.cap);
            unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) };
            self.cap = 0;
        }
    }
}

impl<T: Copy + Default, const ALIGN: usize> Clone for AlignedBuf<T, ALIGN> {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

impl<T, const ALIGN: usize> Drop for AlignedBuf<T, ALIGN> {
    fn drop(&mut self) {
        if self.cap > 0 {
            let bytes = self
                .cap
                .checked_mul(std::mem::size_of::<T>())
                .expect("AlignedBuf byte size overflow");
            let layout =
                Layout::from_size_align(bytes.max(1), ALIGN).expect("invalid AlignedBuf layout");
            unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) };
        }
    }
}

impl<T: Copy + Default, const ALIGN: usize> Deref for AlignedBuf<T, ALIGN> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy + Default, const ALIGN: usize> DerefMut for AlignedBuf<T, ALIGN> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}
