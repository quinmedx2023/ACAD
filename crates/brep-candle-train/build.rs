use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");

    if let Some(commit) = git_output(["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env=ACAD_GIT_COMMIT={commit}");
    }
    if let Some(status) = git_output(["status", "--porcelain"]) {
        println!(
            "cargo:rustc-env=ACAD_GIT_DIRTY={}",
            if status.trim().is_empty() {
                "false"
            } else {
                "true"
            }
        );
    }
}

fn git_output<const N: usize>(args: [&str; N]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}
