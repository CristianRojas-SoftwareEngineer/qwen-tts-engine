use qwen_kernels::bench::run_matmat_bench;
use qwen_kernels::selftest::run_kernel_selftest;
use qwen_kernels::ComputeRuntime;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut threads: Option<usize> = None;
    let mut mode = "caps";
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--caps" => mode = "caps",
            "--self-test" => mode = "selftest",
            "--matmat-bench" => mode = "bench",
            "-j" | "--threads" => {
                i += 1;
                if i < args.len() {
                    threads = args[i].parse::<usize>().ok();
                }
            }
            _ => {}
        }
        i += 1;
    }

    let rt = ComputeRuntime::new(threads);
    match mode {
        "caps" => print!("{}", rt.caps_report()),
        "selftest" => {
            let (fails, report) = run_kernel_selftest(&rt);
            print!("{report}");
            if fails > 0 {
                std::process::exit(1);
            }
        }
        "bench" => print!("{}", run_matmat_bench(&rt)),
        _ => {}
    }
}
