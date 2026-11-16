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

#[derive(Default)]
struct JobSlot {
    fn_ptr: usize,
    ctx_ptr: usize,
    nt: usize,
    next: AtomicUsize,
}

struct SleepState {
    sleeping: usize,
    main_sleeping: bool,
}

struct PoolInner {
    n_threads: usize,
    n_workers: usize,
    spin_budget: usize,
    submit_mtx: Mutex<()>,
    mtx: Mutex<SleepState>,
    wake: Condvar,
    complete: Condvar,
    sleeping_atomic: AtomicUsize,
    generation: AtomicU64,
    completed: AtomicUsize,
    stop: AtomicBool,
    // Protected by `submit_mtx` during `parallel()`; workers only read between
    // `generation` increment (Release/SeqCst) and `completed` increment (AcqRel).
    job_slot: UnsafeCell<JobSlot>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

unsafe impl Sync for PoolInner {}

/// Instanciable persistent thread pool (no global singleton state), matching C's
/// spin-wait (`QWEN_POOL_SPIN=4096` + `spin_loop()`) + `Condvar` + per-worker `ftz_on()`.
#[derive(Clone)]
pub struct ComputePool {
    inner: Arc<PoolInner>,
}

impl ComputePool {
    pub fn new(n_threads: usize) -> Self {
        Self::with_spin_budget(n_threads, env_spin_budget())
    }

    pub fn with_spin_budget(n_threads: usize, spin_budget: usize) -> Self {
        ftz_on();
        let n_threads = n_threads.max(1);
        let n_workers = n_threads.saturating_sub(1);

        let inner = Arc::new(PoolInner {
            n_threads,
            n_workers,
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
            workers: Mutex::new(Vec::with_capacity(n_workers)),
        });

        if n_workers > 0 {
            let gen0 = inner.generation.load(Ordering::Acquire);
            let mut handles = inner.workers.lock().unwrap();
            for _ in 0..n_workers {
                let p = Arc::clone(&inner);
                match thread::Builder::new()
                    .name("qwen-worker".to_string())
                    .spawn(move || worker_main(p, gen0))
                {
                    Ok(h) => handles.push(h),
                    Err(_) => break,
                }
            }
        }

        Self { inner }
    }

    #[inline]
    pub fn n_threads(&self) -> usize {
        self.inner.n_threads
    }

    #[inline]
    pub fn spin_budget(&self) -> usize {
        self.inner.spin_budget
    }

    /// Runs `f(tid, nt)` for `tid` in `0..nt` across the pool (caller participates),
    /// blocking until all chunks finish.
    pub fn parallel<F>(&self, nt: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if nt == 0 {
            return;
        }
        let actual_workers = self.inner.n_workers;
        if actual_workers == 0 || nt == 1 {
            for tid in 0..nt {
                f(tid, nt);
            }
            return;
        }

        unsafe fn trampoline<F: Fn(usize, usize) + Sync>(tid: usize, nt: usize, ctx: usize) {
            let func = &*(ctx as *const F);
            func(tid, nt);
        }

        let _submit_guard = self.inner.submit_mtx.lock().unwrap();
        let job = unsafe { &mut *self.inner.job_slot.get() };
        job.fn_ptr = trampoline::<F> as *const () as usize;
        job.ctx_ptr = (&f as *const F) as usize;
        job.nt = nt;
        job.next.store(0, Ordering::Relaxed);
        self.inner.completed.store(0, Ordering::Relaxed);

        self.inner.generation.fetch_add(1, Ordering::SeqCst);
        if self.inner.sleeping_atomic.load(Ordering::SeqCst) > 0 {
            let _guard = self.inner.mtx.lock().unwrap();
            self.inner.wake.notify_all();
        }

        run_chunks(job);

        let mut budget = self.inner.spin_budget;
        while budget > 0
            && self.inner.completed.load(Ordering::Acquire) != self.inner.n_workers
        {
            budget -= 1;
            std::hint::spin_loop();
        }
        if self.inner.completed.load(Ordering::Acquire) != self.inner.n_workers {
            let mut guard = self.inner.mtx.lock().unwrap();
            guard.main_sleeping = true;
            while self.inner.completed.load(Ordering::Acquire) != self.inner.n_workers {
                guard = self.inner.complete.wait(guard).unwrap();
            }
            guard.main_sleeping = false;
        }
    }
}

impl Drop for PoolInner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        {
            let _guard = self.mtx.lock().unwrap();
            self.generation.fetch_add(1, Ordering::SeqCst);
            self.wake.notify_all();
        }
        let mut handles = self.workers.lock().unwrap();
        for h in handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[inline]
fn run_chunks(job: &JobSlot) {
    let fn_ptr: unsafe fn(usize, usize, usize) = unsafe { std::mem::transmute(job.fn_ptr) };
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
            let mut guard = inner.mtx.lock().unwrap();
            guard.sleeping += 1;
            inner.sleeping_atomic.store(guard.sleeping, Ordering::SeqCst);
            while !inner.stop.load(Ordering::SeqCst)
                && inner.generation.load(Ordering::SeqCst) == seen
            {
                guard = inner.wake.wait(guard).unwrap();
            }
            guard.sleeping -= 1;
            inner.sleeping_atomic.store(guard.sleeping, Ordering::Release);
        }

        if inner.stop.load(Ordering::Acquire) {
            break;
        }

        seen = inner.generation.load(Ordering::Acquire);
        let job = unsafe { &*inner.job_slot.get() };
        run_chunks(job);

        if inner.completed.fetch_add(1, Ordering::AcqRel) + 1 == inner.n_workers {
            let guard = inner.mtx.lock().unwrap();
            if guard.main_sleeping {
                inner.complete.notify_one();
            }
        }
    }
}
