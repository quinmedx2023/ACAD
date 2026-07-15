use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

use acad_brep_candle_train::{
    train_dataset, train_dataset_and_save, train_face_segmentation,
    train_face_segmentation_and_save, train_synthetic, train_synthetic_and_save,
    FaceSamplingStrategy, FaceSegmentationConfig, FaceSegmentationReport, TrainingConfig,
    TrainingReport,
};
use acad_brep_dataset::{
    generate_synthetic_dataset, summarize_dataset, DatasetConfig, DatasetSummary,
};

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
}

struct CleanFusionArgs {
    raw: PathBuf,
    out: PathBuf,
    exe: PathBuf,
    limit: Option<usize>,
    allow_boundary: bool,
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

fn parse_clean_fusion_args(args: impl Iterator<Item = String>) -> CleanFusionArgs {
    let mut raw = None;
    let mut out = PathBuf::from("data/fusion-seg-v1");
    let mut exe = default_occt_cleaner_path();
    let mut limit = None;
    let mut allow_boundary = false;
    let mut args = args.peekable();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--raw" => raw = Some(PathBuf::from(next_string(&mut args, "--raw"))),
            "--out" => out = PathBuf::from(next_string(&mut args, "--out")),
            "--exe" => exe = PathBuf::from(next_string(&mut args, "--exe")),
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
            "--max-val-samples" => {
                config.max_val_samples =
                    parse_optional_limit(parse_next(&mut args, "--max-val-samples"));
            }
            "--class-weights" => config.use_class_weights = true,
            "--no-class-weights" => config.use_class_weights = false,
            "--sample-strategy" => {
                config.sampling_strategy =
                    parse_face_sampling_strategy(next_string(&mut args, "--sample-strategy"))
            }
            "--val-sample-strategy" => {
                config.val_sampling_strategy =
                    parse_face_sampling_strategy(next_string(&mut args, "--val-sample-strategy"))
            }
            "--no-shuffle" => config.shuffle_each_epoch = false,
            "--save" => save = Some(PathBuf::from(next_string(&mut args, "--save"))),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => die_unknown(unknown),
        }
    }

    FaceTrainArgs { config, data, save }
}

fn run_clean_fusion(args: CleanFusionArgs) -> Result<(), Box<dyn Error>> {
    let mut command = Command::new(&args.exe);
    command
        .arg("--raw")
        .arg(&args.raw)
        .arg("--out")
        .arg(&args.out);
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
    match args.save {
        Some(save) => train_face_segmentation_and_save(args.config, &args.data, &save),
        None => train_face_segmentation(args.config, &args.data),
    }
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
    println!("val_samples: {}", report.val_samples);
    println!("train_faces: {}", report.train_faces);
    println!("val_faces: {}", report.val_faces);
    println!("face_classes: {}", report.face_classes);
    println!("hidden_dim: {}", report.hidden_dim);
    println!("rounds: {}", report.rounds);
    println!("batch_size: {}", report.batch_size);
    println!("class_weighting: {}", report.class_weighting);
    println!(
        "train_sampling_strategy: {}",
        report.sampling_strategy.as_str()
    );
    println!(
        "val_sampling_strategy: {}",
        report.val_sampling_strategy.as_str()
    );
    println!("shuffle_each_epoch: {}", report.shuffle_each_epoch);
    println!(
        "train_face_label_counts: {}",
        format_face_label_counts(&report.face_label_names, &report.train_face_label_counts)
    );
    println!(
        "val_face_label_counts: {}",
        format_face_label_counts(&report.face_label_names, &report.val_face_label_counts)
    );
    println!("final_loss: {:.6}", report.final_loss);
    println!("train_face_accuracy: {:.2}%", report.train_accuracy * 100.0);
    println!("val_face_accuracy: {:.2}%", report.val_accuracy * 100.0);
    println!("val_face_macro_f1: {:.4}", report.val_macro_f1);
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
         brep-candle-train clean-fusion --raw DIR [--out DIR] [--exe PATH] [--limit N] [--allow-boundary]\n  \
         brep-candle-train train [--data DIR] [--epochs N] [--lr LR] [--hidden N] \
         [--samples-per-class N] [--rounds N] [--seed N] [--val-fraction F] [--save PATH]\n  \
         brep-candle-train face-train [--data DIR] [--epochs N] [--lr LR] [--hidden N] \
         [--rounds N] [--seed N] [--batch-size N] [--max-train-samples N] \
         [--max-val-samples N] [--sample-strategy uniform|face-balanced] \
         [--val-sample-strategy uniform|face-balanced] [--class-weights] \
         [--no-class-weights] [--no-shuffle] [--save PATH]\n\n  \
         Without a subcommand, training uses the in-memory synthetic dataset for backward compatibility."
    );
}
