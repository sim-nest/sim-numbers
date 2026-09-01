//! Enforces the host-blind boundary for published numbers crates.

use std::{fs, path::Path};

const FORBIDDEN_SOURCE: &[&str] = &[
    "std::time::Instant",
    "std::time::SystemTime",
    "std::env::temp_dir",
    "std::thread::sleep",
    "available_parallelism",
];

pub fn run() -> Result<(), String> {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("xtask manifest must be inside the repository")?;
    let crates = repo.join("crates");
    let mut violations = Vec::new();
    visit(&crates, &mut |path, text| {
        let is_manifest = path.file_name().is_some_and(|name| name == "Cargo.toml");
        if is_manifest {
            for dependency in ["sim-capsule", "sim-platform"] {
                if text.contains(dependency) {
                    violations.push(format!(
                        "{}: forbidden dependency {dependency}",
                        path.display()
                    ));
                }
            }
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            for host_api in FORBIDDEN_SOURCE {
                if text.contains(host_api) {
                    violations.push(format!("{}: forbidden host API {host_api}", path.display()));
                }
            }
        }
    })?;
    if violations.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "numbers must remain host-blind:\n{}",
            violations.join("\n")
        ))
    }
}

fn visit(directory: &Path, inspect: &mut impl FnMut(&Path, &str)) -> Result<(), String> {
    let entries = fs::read_dir(directory)
        .map_err(|error| format!("read {}: {error}", directory.display()))?;
    for entry in entries {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            visit(&path, inspect)?;
        } else if path.file_name().is_some_and(|name| name == "Cargo.toml")
            || path.extension().is_some_and(|extension| extension == "rs")
        {
            let text = fs::read_to_string(&path)
                .map_err(|error| format!("read {}: {error}", path.display()))?;
            inspect(&path, &text);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn published_numbers_crates_are_host_blind() {
        super::run().unwrap();
    }
}
