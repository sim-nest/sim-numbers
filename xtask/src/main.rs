#![forbid(unsafe_code)]
//! Repository automation wrapper for sim-numbers generated documentation.

mod simdoc;

fn main() {
    if let Err(err) = simdoc::run(std::env::args().collect()) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
