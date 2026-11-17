// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use crate::ftz::ftz_on;

pub const QWEN_POOL_SPIN_DEFAULT: usize = 4096;

pub fn default_num_threads() -> usize {
    thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(1)
        .max(1)
}

fn env_spin_budget() -> usize {
    match std::env::var("QWEN_POOL_SPIN") {
        Ok(val) => val
            .trim()
            .parse::<isize>()
            .map(|v| v.max(0) as usize)
            .unwrap_or(QWEN_POOL_SPIN_DEFAULT),
        Err(_) => QWEN_POOL_SPIN_DEFAULT,
    }
}

struct JobSlot {
    fn_ptr: Option<unsafe fn(usize, usize, *const ())>,
    ctx_ptr: *const (),
    nt: usize,
    next: AtomicUsize,
}

impl Default for JobSlot {
    fn default() -> Self {
        Self {
            fn_ptr: None,
            ctx_ptr: std::ptr::null(),
            nt: 0,
            next: AtomicUsize::new(0),
        }
    }
}

struct SleepState {
    sleeping: usize,
    main_sleeping: bool,
}

struct PoolInner {
    n_threads: AtomicUsize,
    n_workers: AtomicUsize,
    spin_budget: usize,
    submit_mtx: Mutex<()>,
    mtx: Mutex<SleepState>,
    wake: Condvar,
    complete: Condvar,
    sleeping_atomic: AtomicUsize,
    generation: AtomicU64,
    completed: AtomicUsize,
    stop: AtomicBool,
    // Protegido por `submit_mtx` durante `parallel()`; los workers solo leen entre
    // el incremento de `generation` (Release/SeqCst) y el de `completed` (AcqRel).
    job_slot: UnsafeCell<JobSlot>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

unsafe impl Send for PoolInner {}
unsafe impl Sync for PoolInner {}

struct PoolGuard(Arc<PoolInner>);

impl Drop for PoolGuard {
    fn drop(&mut self) {
        self.0.stop.store(true, Ordering::SeqCst);
        {
            let _guard = self.0.mtx.lock().unwrap_or_else(|e| e.into_inner());
            self.0.generation.fetch_add(1, Ordering::SeqCst);
            self.0.wake.notify_all();
        }
        let mut handles = self.0.workers.lock().unwrap_or_else(|e| e.into_inner());
        for h in handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn wait_workers_complete(inner: &PoolInner, actual_workers: usize) {
    let mut budget = inner.spin_budget;
    while budget > 0 && inner.completed.load(Ordering::Acquire) != actual_workers {
        budget -= 1;
        std::hint::spin_loop();
    }
    if inner.completed.load(Ordering::Acquire) != actual_workers {
        let mut guard = inner.mtx.lock().unwrap_or_else(|e| e.into_inner());
        guard.main_sleeping = true;
        while inner.completed.load(Ordering::Acquire) != actual_workers {
            guard = inner.complete.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
        guard.main_sleeping = false;
    }
}

struct ActiveJobGuard<'a> {
    inner: &'a PoolInner,
    actual_workers: usize,
    done: bool,
}

impl Drop for ActiveJobGuard<'_> {
    fn drop(&mut self) {
        if !self.done {
            wait_workers_complete(self.inner, self.actual_workers);
        }
    }
}

struct WorkerCompletionGuard<'a> {
    inner: &'a PoolInner,
}

impl Drop for WorkerCompletionGuard<'_> {
    fn drop(&mut self) {
        if self.inner.completed.fetch_add(1, Ordering::AcqRel) + 1
            == self.inner.n_workers.load(Ordering::Acquire)
        {
            let guard = self.inner.mtx.lock().unwrap_or_else(|e| e.into_inner());
            if guard.main_sleeping {
                self.inner.complete.notify_one();
            }
        }
    }
}

/// Pool de hilos persistente e instanciable (sin estado global) con espera activa
/// (`QWEN_POOL_SPIN=4096` + `spin_loop()`), `Condvar` y `ftz_on()` por trabajador.
#[derive(Clone)]
pub struct ComputePool {
    guard: Arc<PoolGuard>,
}

impl ComputePool {
    pub fn new(n_threads: usize) -> Self {
        Self::with_spin_budget(n_threads, env_spin_budget())
    }

    pub fn with_spin_budget(n_threads: usize, spin_budget: usize) -> Self {
        ftz_on();
        let requested_threads = n_threads.max(1);
        let requested_workers = requested_threads.saturating_sub(1);

        let inner = Arc::new(PoolInner {
            n_threads: AtomicUsize::new(requested_threads),
            n_workers: AtomicUsize::new(requested_workers),
            spin_budget,
            submit_mtx: Mutex::new(()),
            mtx: Mutex::new(SleepState {
                sleeping: 0,
                main_sleeping: false,
            }),
            wake: Condvar::new(),
            complete: Condvar::new(),
            sleeping_atomic: AtomicUsize::new(0),
            generation: AtomicU64::new(0),
            completed: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            job_slot: UnsafeCell::new(JobSlot::default()),
            workers: Mutex::new(Vec::with_capacity(requested_workers)),
        });

        if requested_workers > 0 {
            let gen0 = inner.generation.load(Ordering::Acquire);
            let mut spawned = 0usize;
            {
                let mut handles = inner.workers.lock().unwrap_or_else(|e| e.into_inner());
                for _ in 0..requested_workers {
                    let p = Arc::clone(&inner);
                    match thread::Builder::new()
                        .name("qwen-worker".to_string())
                        .spawn(move || worker_main(p, gen0))
                    {
                        Ok(h) => {
                            handles.push(h);
                            spawned += 1;
                        }
                        Err(_) => break,
                    }
                }
            }
            if spawned != requested_workers {
                inner.n_workers.store(spawned, Ordering::Release);
                inner.n_threads.store(spawned + 1, Ordering::Release);
            }
        }

        Self {
            guard: Arc::new(PoolGuard(inner)),
        }
    }

    #[inline]
    pub fn n_threads(&self) -> usize {
        self.guard.0.n_threads.load(Ordering::Relaxed)
    }

    /// Ejecuta `f(tid, nt)` para `tid` en `0..nt` sobre el pool (el hilo llamador participa)
    /// y bloquea hasta completar todos los fragmentos.
    pub fn parallel<F>(&self, nt: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if nt == 0 {
            return;
        }
        let inner = &self.guard.0;
        let actual_workers = inner.n_workers.load(Ordering::Acquire);
        if actual_workers == 0 || nt == 1 {
            for tid in 0..nt {
                f(tid, nt);
            }
            return;
        }

        unsafe fn trampoline<F: Fn(usize, usize) + Sync>(tid: usize, nt: usize, ctx: *const ()) {
            let func = &*(ctx as *const F);
            func(tid, nt);
        }

        let _submit_guard = inner.submit_mtx.lock().unwrap_or_else(|e| e.into_inner());
        let job = unsafe { &mut *inner.job_slot.get() };
        job.fn_ptr = Some(trampoline::<F>);
        job.ctx_ptr = (&f as *const F).cast();
        job.nt = nt;
        job.next.store(0, Ordering::Relaxed);
        inner.completed.store(0, Ordering::Relaxed);

        inner.generation.fetch_add(1, Ordering::SeqCst);
        if inner.sleeping_atomic.load(Ordering::SeqCst) > 0 {
            let _guard = inner.mtx.lock().unwrap_or_else(|e| e.into_inner());
            inner.wake.notify_all();
        }

        // Garantiza la espera de los workers antes de liberar `f` incluso ante pánico en el hilo principal.
        let mut active_guard = ActiveJobGuard {
            inner,
            actual_workers,
            done: false,
        };
        run_chunks(job);
        wait_workers_complete(inner, actual_workers);
        active_guard.done = true;
    }
}

#[inline]
fn run_chunks(job: &JobSlot) {
    let Some(fn_ptr) = job.fn_ptr else {
        return;
    };
    loop {
        let tid = job.next.fetch_add(1, Ordering::Relaxed);
        if tid >= job.nt {
            break;
        }
        unsafe { fn_ptr(tid, job.nt, job.ctx_ptr) };
    }
}

fn worker_main(inner: Arc<PoolInner>, mut seen: u64) {
    ftz_on();
    loop {
        let mut budget = inner.spin_budget;
        while budget > 0
            && inner.generation.load(Ordering::Acquire) == seen
            && !inner.stop.load(Ordering::Acquire)
        {
            budget -= 1;
            std::hint::spin_loop();
        }

        if inner.generation.load(Ordering::Acquire) == seen && !inner.stop.load(Ordering::Acquire) {
            let mut guard = inner.mtx.lock().unwrap_or_else(|e| e.into_inner());
            guard.sleeping += 1;
            inner.sleeping_atomic.store(guard.sleeping, Ordering::SeqCst);
            while !inner.stop.load(Ordering::SeqCst)
                && inner.generation.load(Ordering::SeqCst) == seen
            {
                guard = inner.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
            }
            guard.sleeping -= 1;
            inner.sleeping_atomic.store(guard.sleeping, Ordering::SeqCst);
        }

        if inner.stop.load(Ordering::Acquire) {
            break;
        }

        seen = inner.generation.load(Ordering::Acquire);
        let _completion = WorkerCompletionGuard { inner: &inner };
        let job = unsafe { &*inner.job_slot.get() };
        run_chunks(job);
    }
}
