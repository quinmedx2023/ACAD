use std::path::PathBuf;

use acad_brep_candle_train::{train_synthetic, train_synthetic_and_save, TrainingConfig};

fn main() -> candle_core::Result<()> {
    let (config, save_path) = parse_args();
    let report = match save_path {
        Some(path) => train_synthetic_and_save(config, &path)?,
        None => train_synthetic(config)?,
    };

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

    Ok(())
}

fn parse_args() -> (TrainingConfig, Option<PathBuf>) {
    let mut config = TrainingConfig::default();
    let mut save_path = None;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
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
            "--save" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("missing value for --save");
                    std::process::exit(2);
                });
                save_path = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => {
                eprintln!("unknown argument: {unknown}");
                print_help();
                std::process::exit(2);
            }
        }
    }

    (config, save_path)
}

fn parse_next<T>(args: &mut impl Iterator<Item = String>, name: &str) -> T
where
    T: std::str::FromStr,
{
    let value = args.next().unwrap_or_else(|| {
        eprintln!("missing value for {name}");
        std::process::exit(2);
    });
    value.parse().unwrap_or_else(|_| {
        eprintln!("invalid value for {name}: {value}");
        std::process::exit(2);
    })
}

fn print_help() {
    println!(
        "Usage: brep-candle-train [--epochs N] [--lr LR] [--hidden N] \
         [--samples-per-class N] [--rounds N] [--seed N] [--val-fraction F] [--save PATH]"
    );
}
