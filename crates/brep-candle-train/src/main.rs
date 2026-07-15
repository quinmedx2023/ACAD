use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use acad_brep_candle_train::{
    train_dataset, train_dataset_and_save, train_face_segmentation,
    train_face_segmentation_and_save, train_synthetic, train_synthetic_and_save,
    FaceSamplingStrategy, FaceSegmentationConfig, FaceSegmentationReport, TrainingConfig,
    TrainingReport,
};
use acad_brep_dataset::{
    generate_synthetic_dataset, manifest_hash, summarize_dataset, write_dataset_harness,
    DatasetConfig, DatasetHarnessReport, DatasetSplit, DatasetSummary, HarnessConfig,
};
use serde::Serialize;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("dataset") => {
            let args = parse_dataset_args(args);
            let summary = generate_synthetic_dataset(
                &args.out,
                DatasetConfig {
                    samples_per_class: args.samples_per_class,
                    val_fraction: args.val_fraction,
                },
            )?;
            println!("BRep dataset written: {}", args.out.display());
            print_summary(&summary);
        }
        Some("inspect") => {
            let data = parse_inspect_args(args);
            let summary = summarize_dataset(&data)?;
            println!("BRep dataset: {}", data.display());
            print_summary(&summary);
        }
        Some("inspect-harness") => {
            let args = parse_harness_args(args);
            let report = run_inspect_harness(args)?;
            print_harness_report(&report);
        }
        Some("clean-fusion") => {
            let args = parse_clean_fusion_args(args);
            run_clean_fusion(args)?;
        }
        Some("train") => {
            let args = parse_train_args(None, args);
            let report = run_train(args)?;
            print_report(&report);
        }
        Some("face-train") => {
            let args = parse_face_train_args(args);
            let report = run_face_train(args)?;
            print_face_report(&report);
        }
        Some("--help") | Some("-h") => print_help(),
        Some(first) => {
            let args = parse_train_args(Some(first.to_string()), args);
            let report = run_train(args)?;
            print_report(&report);
        }
        None => {
            let report = train_synthetic(TrainingConfig::default())?;
            print_report(&report);
        }
    }

    Ok(())
}

struct DatasetArgs {
    out: PathBuf,
    samples_per_class: usize,
    val_fraction: f32,
}

struct TrainArgs {
    config: TrainingConfig,
    data: Option<PathBuf>,
    save: Option<PathBuf>,
}

struct FaceTrainArgs {
    config: FaceSegmentationConfig,
    data: PathBuf,
    save: Option<PathBuf>,
    report: Option<PathBuf>,
}

struct CleanFusionArgs {
    raw: PathBuf,
    out: PathBuf,
    exe: PathBuf,
    split_file: Option<PathBuf>,
    use_default_split_file: bool,
    limit: Option<usize>,
    allow_boundary: bool,
}

struct HarnessArgs {
    data: PathBuf,
    out: Option<PathBuf>,
    config: HarnessConfig,
}

#[derive(Debug, Serialize)]
struct FaceTrainJsonReport {
    report_version: &'static str,
    git_commit: Option<String>,
    git_dirty: Option<bool>,
    dataset_path: String,
    manifest_hash: Option<String>,
    config: FaceTrainConfigReport,
    sampler: FaceTrainSamplerReport,
    final_loss: f32,
    train_metrics: acad_brep_candle_train::FaceEvaluationMetrics,
    eval_metrics: acad_brep_candle_train::FaceEvaluationMetrics,
}

#[derive(Debug, Serialize)]
struct FaceTrainConfigReport {
    epochs: usize,
    learning_rate: f64,
    hidden_dim: usize,
    rounds: usize,
    seed: u64,
    batch_size: usize,
    max_train_samples: Option<usize>,
    max_eval_samples: Option<usize>,
    use_class_weights: bool,
    sampling_strategy: String,
    eval_split: String,
    shuffle_each_epoch: bool,
}

#[derive(Debug, Serialize)]
struct FaceTrainSamplerReport {
    train_record_ids_hash: String,
    eval_record_ids_hash: String,
    train_samples: usize,
    eval_samples: usize,
    train_faces: usize,
    eval_faces: usize,
}

fn parse_dataset_args(args: impl Iterator<Item = String>) -> DatasetArgs {
    let mut out = PathBuf::from("data/synthetic-v1");
    let mut samples_per_class = 32;
    let mut val_fraction = 0.25;
    let mut args = args.peekable();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = PathBuf::from(next_string(&mut args, "--out")),
            "--samples" | "--samples-per-class" => {
                samples_per_class = parse_next(&mut args, "--samples-per-class")
            }
            "--val-fraction" => val_fraction = parse_next(&mut args, "--val-fraction"),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => die_unknown(unknown),
        }
    }

    DatasetArgs {
        out,
        samples_per_class,
        val_fraction,
    }
}

fn parse_inspect_args(args: impl Iterator<Item = String>) -> PathBuf {
    let mut data = PathBuf::from("data/synthetic-v1");
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data" => data = PathBuf::from(next_string(&mut args, "--data")),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => die_unknown(unknown),
        }
    }
    data
}

fn parse_harness_args(args: impl Iterator<Item = String>) -> HarnessArgs {
    let mut data = PathBuf::from("data/fusion-seg-v1");
    let mut out = None;
    let mut config = HarnessConfig::default();
    let mut args = args.peekable();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data" => data = PathBuf::from(next_string(&mut args, "--data")),
            "--out" => out = Some(PathBuf::from(next_string(&mut args, "--out"))),
            "--val-percent" => config.validation_percent = parse_next(&mut args, "--val-percent"),
            "--seed" => config.validation_seed = parse_next(&mut args, "--seed"),
            "--rare-count" => config.rare_count_threshold = parse_next(&mut args, "--rare-count"),
            "--rare-fraction" => {
                config.rare_fraction_threshold = parse_next(&mut args, "--rare-fraction")
            }
            "--split-file" => {
                config.split_file = Some(PathBuf::from(next_string(&mut args, "--split-file")))
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => die_unknown(unknown),
        }
    }

    if config.validation_percent > 100 {
        eprintln!("invalid value for --val-percent: expected 0..100");
        std::process::exit(2);
    }
    if config.rare_fraction_threshold < 0.0 {
        eprintln!("invalid value for --rare-fraction: expected a non-negative number");
        std::process::exit(2);
    }

    HarnessArgs { data, out, config }
}

fn parse_clean_fusion_args(args: impl Iterator<Item = String>) -> CleanFusionArgs {
    let mut raw = None;
    let mut out = PathBuf::from("data/fusion-seg-v1");
    let mut exe = default_occt_cleaner_path();
    let mut split_file = None;
    let mut use_default_split_file = true;
    let mut limit = None;
    let mut allow_boundary = false;
    let mut args = args.peekable();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--raw" => raw = Some(PathBuf::from(next_string(&mut args, "--raw"))),
            "--out" => out = PathBuf::from(next_string(&mut args, "--out")),
            "--exe" => exe = PathBuf::from(next_string(&mut args, "--exe")),
            "--split-file" => {
                split_file = Some(PathBuf::from(next_string(&mut args, "--split-file")))
            }
            "--no-split-file" => use_default_split_file = false,
            "--limit" => limit = Some(parse_next(&mut args, "--limit")),
            "--allow-boundary" => allow_boundary = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => die_unknown(unknown),
        }
    }

    let raw = raw.unwrap_or_else(|| {
        eprintln!("missing required --raw DIR for Fusion 360 Gallery Segmentation");
        std::process::exit(2);
    });

    CleanFusionArgs {
        raw,
        out,
        exe,
        split_file,
        use_default_split_file,
        limit,
        allow_boundary,
    }
}

fn parse_train_args(first: Option<String>, rest: impl Iterator<Item = String>) -> TrainArgs {
    let mut config = TrainingConfig::default();
    let mut data = None;
    let mut save = None;
    let mut args = first.into_iter().chain(rest).peekable();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data" => data = Some(PathBuf::from(next_string(&mut args, "--data"))),
            "--epochs" => config.epochs = parse_next(&mut args, "--epochs"),
            "--lr" | "--learning-rate" => {
                config.learning_rate = parse_next(&mut args, "--learning-rate")
            }
            "--hidden" | "--hidden-dim" => {
                config.hidden_dim = parse_next(&mut args, "--hidden-dim")
            }
            "--samples-per-class" => {
                config.samples_per_class = parse_next(&mut args, "--samples-per-class")
            }
            "--rounds" => config.rounds = parse_next(&mut args, "--rounds"),
            "--seed" => config.seed = parse_next(&mut args, "--seed"),
            "--val-fraction" => config.val_fraction = parse_next(&mut args, "--val-fraction"),
            "--save" => save = Some(PathBuf::from(next_string(&mut args, "--save"))),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => die_unknown(unknown),
        }
    }

    TrainArgs { config, data, save }
}

fn parse_face_train_args(args: impl Iterator<Item = String>) -> FaceTrainArgs {
    let mut config = FaceSegmentationConfig::default();
    let mut data = PathBuf::from("data/fusion-seg-v1");
    let mut save = None;
    let mut report = None;
    let mut args = args.peekable();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data" => data = PathBuf::from(next_string(&mut args, "--data")),
            "--epochs" => config.epochs = parse_next(&mut args, "--epochs"),
            "--lr" | "--learning-rate" => {
                config.learning_rate = parse_next(&mut args, "--learning-rate")
            }
            "--hidden" | "--hidden-dim" => {
                config.hidden_dim = parse_next(&mut args, "--hidden-dim")
            }
            "--rounds" => config.rounds = parse_next(&mut args, "--rounds"),
            "--seed" => config.seed = parse_next(&mut args, "--seed"),
            "--batch-size" => config.batch_size = parse_next(&mut args, "--batch-size"),
            "--max-train-samples" => {
                config.max_train_samples =
                    parse_optional_limit(parse_next(&mut args, "--max-train-samples"));
            }
            "--max-eval-samples" => {
                config.max_eval_samples =
                    parse_optional_limit(parse_next(&mut args, "--max-eval-samples"));
            }
            "--class-weights" => config.use_class_weights = true,
            "--no-class-weights" => config.use_class_weights = false,
            "--sample-strategy" => {
                config.sampling_strategy =
                    parse_face_sampling_strategy(next_string(&mut args, "--sample-strategy"))
            }
            "--eval-split" => {
                config.eval_split = parse_eval_split(next_string(&mut args, "--eval-split"))
            }
            "--no-shuffle" => config.shuffle_each_epoch = false,
            "--save" => save = Some(PathBuf::from(next_string(&mut args, "--save"))),
            "--report" => report = Some(PathBuf::from(next_string(&mut args, "--report"))),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => die_unknown(unknown),
        }
    }

    FaceTrainArgs {
        config,
        data,
        save,
        report,
    }
}

fn run_clean_fusion(args: CleanFusionArgs) -> Result<(), Box<dyn Error>> {
    let mut command = Command::new(&args.exe);
    command
        .arg("--raw")
        .arg(&args.raw)
        .arg("--out")
        .arg(&args.out);
    let split_file = if let Some(split_file) = args.split_file {
        if !split_file.is_file() {
            return Err(format!("split file does not exist: {}", split_file.display()).into());
        }
        Some(split_file)
    } else {
        args.use_default_split_file
            .then(|| args.raw.join("train_test.json"))
            .filter(|path| path.is_file())
    };
    if let Some(split_file) = split_file {
        command.arg("--split-file").arg(split_file);
    }
    if let Some(limit) = args.limit {
        command.arg("--limit").arg(limit.to_string());
    }
    if args.allow_boundary {
        command.arg("--allow-boundary");
    }
    add_occt_runtime_env(&mut command);

    let status = command.status()?;
    if !status.success() {
        return Err(format!(
            "Fusion cleanup failed with status {status}. Build tools/occt_cleaner with CMake and ensure OCCT/OpenCascade plus the raw Fusion dataset are installed."
        )
        .into());
    }

    let summary = summarize_dataset(&args.out)?;
    println!("Fusion BRep dataset written: {}", args.out.display());
    print_summary(&summary);
    Ok(())
}

fn default_occt_cleaner_path() -> PathBuf {
    if cfg!(windows) {
        let openenv = PathBuf::from("tools/occt_cleaner/build-openenv-release/occt_cleaner.exe");
        if openenv.exists() {
            return openenv;
        }
        let multi_config = PathBuf::from("tools/occt_cleaner/build/Release/occt_cleaner.exe");
        if multi_config.exists() {
            multi_config
        } else {
            PathBuf::from("tools/occt_cleaner/build/occt_cleaner.exe")
        }
    } else {
        PathBuf::from("tools/occt_cleaner/build/occt_cleaner")
    }
}

fn add_occt_runtime_env(command: &mut Command) {
    if !cfg!(windows) {
        return;
    }

    let runtime_dirs = occt_runtime_dirs();
    if runtime_dirs.is_empty() {
        return;
    }

    let current_path = env::var_os("PATH").unwrap_or_default();
    let merged = runtime_dirs
        .into_iter()
        .chain(env::split_paths(&current_path))
        .collect::<Vec<_>>();
    if let Ok(path) = env::join_paths(merged) {
        command.env("PATH", path);
    }
    if let Some(root) = first_occt_root() {
        add_occt_resource_env(command, &root);
    }
}

fn occt_runtime_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut seen = HashSet::new();
    for root in occt_root_candidates() {
        if !looks_like_occt_root(&root) {
            continue;
        }

        for bin in occt_bin_dirs(&root) {
            push_existing_dir(&mut dirs, &mut seen, bin);
        }

        for thirdparty_root in thirdparty_candidates(&root) {
            push_thirdparty_runtime_dirs(&mut dirs, &mut seen, thirdparty_root);
        }
    }
    dirs
}

fn first_occt_root() -> Option<PathBuf> {
    occt_root_candidates()
        .into_iter()
        .find(|root| looks_like_occt_root(root))
}

fn occt_root_candidates() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Some(root) = env_path("ACAD_OCCT_ROOT") {
        roots.push(root);
    }
    if let Some(root) = env_path("CASROOT") {
        roots.push(root);
    }
    if let Some(open_cascade_dir) = env_path("OpenCASCADE_DIR") {
        if open_cascade_dir.join("OpenCASCADEConfig.cmake").is_file() {
            if let Some(parent) = open_cascade_dir.parent() {
                roots.push(parent.to_path_buf());
            }
        }
        roots.push(open_cascade_dir);
    }

    roots.push(PathBuf::from(r"C:\tools\OpenCascade"));
    roots
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn looks_like_occt_root(root: &std::path::Path) -> bool {
    root.join("inc").join("Standard.hxx").is_file() && root.join("win64").is_dir()
}

fn occt_bin_dirs(root: &std::path::Path) -> Vec<PathBuf> {
    let mut bins = Vec::new();
    let win64 = root.join("win64");
    if let Ok(entries) = std::fs::read_dir(win64) {
        for entry in entries.flatten() {
            bins.push(entry.path().join("bin"));
        }
    }
    bins.push(root.join("bin"));
    bins
}

fn thirdparty_candidates(root: &std::path::Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env_path("THIRDPARTY_DIR") {
        candidates.push(path);
    }
    candidates.push(root.join("3rdparty"));
    if let Some(parent) = root.parent() {
        candidates.push(parent.join("3rdparty-vc14-64"));
    }
    candidates
}

fn add_occt_resource_env(command: &mut Command, root: &std::path::Path) {
    let resource = root.join("src");
    let std_resource = resource.join("StdResource");
    let xstep_resource = resource.join("XSTEPResource");

    command.env("CASROOT", root);
    if let Some(thirdparty) = thirdparty_candidates(root)
        .into_iter()
        .find(|path| path.is_dir())
    {
        command.env("THIRDPARTY_DIR", thirdparty);
    }
    if let Some(bin) = occt_bin_dirs(root).into_iter().find(|path| path.is_dir()) {
        command.env("CSF_OCCTBinPath", bin);
    }
    let lib = root.join("win64").join("vc14").join("lib");
    if lib.is_dir() {
        command.env("CSF_OCCTLibPath", lib);
    }
    command.env("CSF_OCCTIncludePath", root.join("inc"));
    command.env("CSF_OCCTResourcePath", &resource);
    command.env("CSF_OCCTDataPath", root.join("data"));
    command.env("CSF_LANGUAGE", "us");
    command.env("MMGT_CLEAR", "1");
    command.env("CSF_SHMessage", resource.join("SHMessage"));
    command.env("CSF_MDTVTexturesDirectory", resource.join("Textures"));
    command.env("CSF_ShadersDirectory", resource.join("Shaders"));
    command.env("CSF_XSMessage", resource.join("XSMessage"));
    command.env("CSF_TObjMessage", resource.join("TObj"));
    command.env("CSF_StandardDefaults", &std_resource);
    command.env("CSF_PluginDefaults", &std_resource);
    command.env("CSF_XCAFDefaults", &std_resource);
    command.env("CSF_TObjDefaults", &std_resource);
    command.env("CSF_StandardLiteDefaults", &std_resource);
    command.env("CSF_IGESDefaults", &xstep_resource);
    command.env("CSF_STEPDefaults", &xstep_resource);
    command.env("CSF_XmlOcafResource", resource.join("XmlOcafResource"));
    command.env(
        "CSF_MIGRATION_TYPES",
        std_resource.join("MigrationSheet.txt"),
    );
}

fn push_thirdparty_runtime_dirs(
    dirs: &mut Vec<PathBuf>,
    seen: &mut HashSet<String>,
    thirdparty_root: PathBuf,
) {
    if !thirdparty_root.is_dir() {
        return;
    }
    if contains_dll(&thirdparty_root) {
        push_existing_dir(dirs, seen, thirdparty_root.clone());
    }
    if let Ok(entries) = std::fs::read_dir(thirdparty_root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if contains_dll(&path) {
                push_existing_dir(dirs, seen, path.clone());
            }
            push_existing_dir(dirs, seen, path.join("bin"));
            push_existing_dir(dirs, seen, path.join("bin").join("win64"));
        }
    }
}

fn push_existing_dir(dirs: &mut Vec<PathBuf>, seen: &mut HashSet<String>, path: PathBuf) {
    if !path.is_dir() {
        return;
    }
    let key = path.to_string_lossy().to_ascii_lowercase();
    if seen.insert(key) {
        dirs.push(path);
    }
}

fn contains_dll(path: &std::path::Path) -> bool {
    std::fs::read_dir(path)
        .map(|entries| {
            entries.flatten().any(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("dll"))
            })
        })
        .unwrap_or(false)
}

fn run_train(args: TrainArgs) -> candle_core::Result<TrainingReport> {
    match (args.data, args.save) {
        (Some(data), Some(save)) => train_dataset_and_save(args.config, &data, &save),
        (Some(data), None) => train_dataset(args.config, &data),
        (None, Some(save)) => train_synthetic_and_save(args.config, &save),
        (None, None) => train_synthetic(args.config),
    }
}

fn run_face_train(args: FaceTrainArgs) -> candle_core::Result<FaceSegmentationReport> {
    let report = match args.save {
        Some(save) => train_face_segmentation_and_save(args.config, &args.data, &save),
        None => train_face_segmentation(args.config, &args.data),
    }?;
    if let Some(path) = args.report {
        write_face_train_report(&path, args.config, &args.data, &report)?;
    }
    Ok(report)
}

fn run_inspect_harness(args: HarnessArgs) -> Result<DatasetHarnessReport, Box<dyn Error>> {
    let report = write_dataset_harness(&args.data, args.config, args.out.as_deref())?;
    Ok(report)
}

fn write_face_train_report(
    path: &Path,
    config: FaceSegmentationConfig,
    data: &Path,
    report: &FaceSegmentationReport,
) -> candle_core::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(to_candle_error)?;
    }
    let json_report = FaceTrainJsonReport {
        report_version: "acad-brep-face-train-report-v1",
        git_commit: git_output(["rev-parse", "HEAD"]),
        git_dirty: git_output(["status", "--porcelain"]).map(|output| !output.trim().is_empty()),
        dataset_path: data.display().to_string(),
        manifest_hash: manifest_hash(data).ok(),
        config: FaceTrainConfigReport {
            epochs: config.epochs,
            learning_rate: config.learning_rate,
            hidden_dim: config.hidden_dim,
            rounds: config.rounds,
            seed: config.seed,
            batch_size: config.batch_size,
            max_train_samples: config.max_train_samples,
            max_eval_samples: config.max_eval_samples,
            use_class_weights: config.use_class_weights,
            sampling_strategy: config.sampling_strategy.as_str().to_string(),
            eval_split: config.eval_split.as_str().to_string(),
            shuffle_each_epoch: config.shuffle_each_epoch,
        },
        sampler: FaceTrainSamplerReport {
            train_record_ids_hash: report.train_record_ids_hash.clone(),
            eval_record_ids_hash: report.eval_record_ids_hash.clone(),
            train_samples: report.train_samples,
            eval_samples: report.eval_samples,
            train_faces: report.train_faces,
            eval_faces: report.eval_faces,
        },
        final_loss: report.final_loss,
        train_metrics: report.train_metrics.clone(),
        eval_metrics: report.eval_metrics.clone(),
    };
    let json = serde_json::to_string_pretty(&json_report).map_err(to_candle_error)?;
    fs::write(path, json).map_err(to_candle_error)
}

fn git_output<const N: usize>(args: [&str; N]) -> Option<String> {
    let safe_directory = format!(
        "safe.directory={}",
        env::current_dir()
            .ok()?
            .to_string_lossy()
            .replace('\\', "/")
    );
    let output = Command::new("git")
        .arg("-c")
        .arg(safe_directory)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn print_report(report: &TrainingReport) {
    println!("Hybrid BRep Candle training complete");
    println!("epochs: {}", report.epochs);
    println!("train_samples: {}", report.train_samples);
    println!("val_samples: {}", report.val_samples);
    println!("hidden_dim: {}", report.hidden_dim);
    println!("rounds: {}", report.rounds);
    println!("final_loss: {:.6}", report.final_loss);
    println!("train_accuracy: {:.2}%", report.train_accuracy * 100.0);
    println!("val_accuracy: {:.2}%", report.val_accuracy * 100.0);
    println!("val_macro_f1: {:.4}", report.val_macro_f1);
}

fn print_face_report(report: &FaceSegmentationReport) {
    println!("Fusion face segmentation training complete");
    println!("epochs: {}", report.epochs);
    println!("train_samples: {}", report.train_samples);
    println!("eval_split: {}", report.eval_split.as_str());
    println!("eval_samples: {}", report.eval_samples);
    println!("train_faces: {}", report.train_faces);
    println!("eval_faces: {}", report.eval_faces);
    println!("face_classes: {}", report.face_classes);
    println!("hidden_dim: {}", report.hidden_dim);
    println!("rounds: {}", report.rounds);
    println!("batch_size: {}", report.batch_size);
    println!(
        "train_face_label_counts: {}",
        format_face_label_counts(&report.face_label_names, &report.train_face_label_counts)
    );
    println!(
        "eval_face_label_counts: {}",
        format_face_label_counts(&report.face_label_names, &report.eval_face_label_counts)
    );
    println!("final_loss: {:.6}", report.final_loss);
    println!("train_face_accuracy: {:.2}%", report.train_accuracy * 100.0);
    println!("eval_face_accuracy: {:.2}%", report.eval_accuracy * 100.0);
    println!("eval_face_macro_f1: {:.4}", report.eval_macro_f1);
    println!(
        "eval_face_weighted_f1: {:.4}",
        report.eval_metrics.weighted_f1
    );
    println!(
        "eval_face_macro_iou: {:.4}",
        report.eval_metrics.macro_iou_present
    );
}

fn print_summary(summary: &DatasetSummary) {
    println!("version: {}", summary.metadata.version);
    println!("records: {}", summary.metadata.records);
    println!("classes: {}", summary.metadata.classes.join(", "));
    println!("splits: {:?}", summary.split_counts);
    println!("class_counts: {:?}", summary.class_counts);
    println!("face_labels: {:?}", summary.face_label_counts);
    println!("edge_labels: {:?}", summary.edge_label_counts);
}

fn print_harness_report(report: &DatasetHarnessReport) {
    println!("Dataset harness report written");
    println!("harness_version: {}", report.harness_version);
    println!("dataset_version: {}", report.dataset_version);
    println!("manifest_hash: {}", report.manifest_hash);
    println!(
        "split_policy: validation_source={}, validation_percent={}, test_policy={}",
        report.split_policy.validation_source,
        report.split_policy.validation_percent,
        report.split_policy.test_policy
    );
    println!("record_counts: {:?}", report.record_counts_by_split);
    println!("rare_face_labels: {:?}", report.rare_face_labels);
    println!(
        "train_val_total_variation: {:?}",
        report.train_val_label_drift.total_variation
    );
    println!(
        "missing_from_train_inner: {:?}",
        report.labels_missing_from_train_inner
    );
    println!(
        "missing_from_val_inner: {:?}",
        report.labels_missing_from_val_inner
    );
}

fn parse_next<T>(args: &mut impl Iterator<Item = String>, name: &str) -> T
where
    T: std::str::FromStr,
{
    let value = next_string(args, name);
    value.parse().unwrap_or_else(|_| {
        eprintln!("invalid value for {name}: {value}");
        std::process::exit(2);
    })
}

fn to_candle_error(error: impl std::error::Error) -> candle_core::Error {
    candle_core::Error::Msg(error.to_string())
}

fn parse_optional_limit(value: usize) -> Option<usize> {
    if value == 0 {
        None
    } else {
        Some(value)
    }
}

fn parse_face_sampling_strategy(value: String) -> FaceSamplingStrategy {
    match value.as_str() {
        "uniform" => FaceSamplingStrategy::Uniform,
        "face-balanced" | "balanced" => FaceSamplingStrategy::FaceBalanced,
        _ => {
            eprintln!("invalid value for --sample-strategy: {value}");
            eprintln!("expected one of: uniform, face-balanced");
            std::process::exit(2);
        }
    }
}

fn parse_eval_split(value: String) -> DatasetSplit {
    match value.as_str() {
        "val" | "validation" => DatasetSplit::Val,
        "test" => DatasetSplit::Test,
        other => {
            eprintln!("invalid eval split {other:?}; expected val or test");
            std::process::exit(2);
        }
    }
}

fn format_face_label_counts(names: &[String], counts: &[usize]) -> String {
    names
        .iter()
        .zip(counts.iter())
        .map(|(name, count)| format!("{name}:{count}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn next_string(args: &mut impl Iterator<Item = String>, name: &str) -> String {
    args.next().unwrap_or_else(|| {
        eprintln!("missing value for {name}");
        std::process::exit(2);
    })
}

fn die_unknown(arg: &str) -> ! {
    eprintln!("unknown argument: {arg}");
    print_help();
    std::process::exit(2);
}

fn print_help() {
    println!(
        "Usage:\n  \
         brep-candle-train dataset [--out DIR] [--samples-per-class N] [--val-fraction F]\n  \
         brep-candle-train inspect [--data DIR]\n  \
         brep-candle-train inspect-harness [--data DIR] [--out PATH] [--val-percent N] \
         [--seed N] [--rare-count N] [--rare-fraction F] [--split-file PATH]\n  \
         brep-candle-train clean-fusion --raw DIR [--out DIR] [--exe PATH] [--split-file PATH] \
         [--no-split-file] [--limit N] [--allow-boundary]\n  \
         brep-candle-train train [--data DIR] [--epochs N] [--lr LR] [--hidden N] \
         [--samples-per-class N] [--rounds N] [--seed N] [--val-fraction F] [--save PATH]\n  \
         brep-candle-train face-train [--data DIR] [--epochs N] [--lr LR] [--hidden N] \
         [--rounds N] [--seed N] [--batch-size N] [--max-train-samples N] \
         [--max-eval-samples N] [--sample-strategy uniform|face-balanced] \
         [--class-weights] [--no-class-weights] [--eval-split val|test] \
         [--no-shuffle] [--save PATH] [--report PATH]\n\n  \
         Without a subcommand, training uses the in-memory synthetic dataset for backward compatibility."
    );
}
