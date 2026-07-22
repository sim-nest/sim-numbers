#![forbid(unsafe_code)]
//! Repository automation wrapper for sim-numbers generated documentation and
//! policy checks.

mod file_sizes;
mod index_check;
mod simdoc;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let program = args.first().map(String::as_str).unwrap_or("xtask");
    let result = match args.get(1).map(String::as_str) {
        Some("simdoc") => simdoc::run(args),
        Some("index-check") => index_check::run(args),
        Some("check-file-sizes") => file_sizes::run(),
        _ => Err(format!(
            "usage: {program} simdoc [--check] | index-check [--repo PATH] [--strict SPEC] | check-file-sizes"
        )),
    };

    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
