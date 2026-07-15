#!/usr/bin/env python3
"""Run reusable Fusion face-segmentation benchmark variants."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Variant:
    run: str
    hidden: int
    rounds: int
    strategy: str
    class_weights: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_exe(root: Path) -> Path:
    name = "brep-candle-train.exe" if os.name == "nt" else "brep-candle-train"
    return root / "target" / "debug" / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fusion face-segmentation benchmark variants."
    )
    parser.add_argument("--data", default="data/fusion-seg-v1")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--exe", default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--large-hidden", type=int, default=64)
    parser.add_argument("--large-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["uniform", "weighted", "face-balanced", "large"],
        help="Variants: uniform, weighted, face-balanced, large.",
    )
    parser.add_argument("--skip-harness", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def variant_spec(name: str, args: argparse.Namespace) -> Variant:
    normalized = name.lower()
    if normalized in {"uniform", "baseline"}:
        return Variant(
            run=f"uniform_h{args.hidden}_r{args.rounds}",
            hidden=args.hidden,
            rounds=args.rounds,
            strategy="uniform",
            class_weights=False,
        )
    if normalized == "weighted":
        return Variant(
            run=f"uniform_weighted_h{args.hidden}_r{args.rounds}",
            hidden=args.hidden,
            rounds=args.rounds,
            strategy="uniform",
            class_weights=True,
        )
    if normalized in {"face-balanced", "balanced"}:
        return Variant(
            run=f"face_balanced_h{args.hidden}_r{args.rounds}",
            hidden=args.hidden,
            rounds=args.rounds,
            strategy="face-balanced",
            class_weights=False,
        )
    if normalized == "large":
        return Variant(
            run=f"uniform_h{args.large_hidden}_r{args.large_rounds}",
            hidden=args.large_hidden,
            rounds=args.large_rounds,
            strategy="uniform",
            class_weights=False,
        )
    raise ValueError(
        f"unknown variant {name!r}; use uniform, weighted, face-balanced, or large"
    )


def command_string(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else " ".join(command)


def invoke_checked(exe: Path, args: list[str], log_path: Path, cwd: Path) -> None:
    command = [str(exe), *args]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"command: {command_string(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        status = process.wait()
    if status != 0:
        raise RuntimeError(f"command failed with exit code {status}: {command_string(command)}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def class_metric(report: dict[str, Any], split: str, label: str) -> dict[str, Any] | None:
    metrics = report.get(f"{split}_metrics", {}).get("class_metrics", [])
    for metric in metrics:
        if metric.get("label") == label:
            return metric
    return None


def parse_duration(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    duration = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("duration_seconds: "):
            duration = float(line.removeprefix("duration_seconds: "))
    return duration


def fmt_decimal(value: Any) -> str:
    if value is None:
        return ""
    text = f"{float(value):.4f}"
    return text.rstrip("0").rstrip(".")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    data: str,
    manifest_hash: str,
    eval_split: str,
    epochs: int,
    train_samples: int,
    eval_samples: int,
    seed: int,
) -> None:
    lines = [
        "# Fusion Face Benchmark",
        "",
        f"- dataset: `{data}`",
        f"- manifest hash: `{manifest_hash}`",
        f"- eval split: `{eval_split}`",
        f"- epochs: `{epochs}`",
        f"- train graph budget: `{train_samples}`",
        f"- eval graph budget: `{eval_samples}`",
        f"- seed: `{seed}`",
        "",
        "| Run | Hidden | Rounds | Weights | Sampling | Seconds | Eval Acc | Macro-F1 | Weighted-F1 | Macro-IoU | Seg7 Support | Seg7 F1 |",
        "|-----|-------:|-------:|---------|----------|--------:|---------:|---------:|------------:|----------:|-------------:|--------:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    str(row["hidden"]),
                    str(row["rounds"]),
                    str(row["class_weights"]),
                    str(row["sampling_strategy"]),
                    fmt_decimal(row["duration_seconds"]),
                    fmt_decimal(row["eval_accuracy"]),
                    fmt_decimal(row["eval_macro_f1"]),
                    fmt_decimal(row["eval_weighted_f1"]),
                    fmt_decimal(row["eval_macro_iou"]),
                    str(row["segment_7_eval_support"]),
                    fmt_decimal(row["segment_7_eval_f1"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "Generated by `scripts/train_fusion_face_benchmark.py`."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run",
        "duration_seconds",
        "eval_accuracy",
        "eval_macro_f1",
        "eval_weighted_f1",
        "eval_macro_iou",
    ]
    widths = {
        column: max(len(column), *(len(fmt_decimal(row[column]) if column != "run" else str(row[column])) for row in rows))
        for column in columns
    }
    print("")
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        cells = []
        for column in columns:
            value = str(row[column]) if column == "run" else fmt_decimal(row[column])
            cells.append(value.ljust(widths[column]))
        print("  ".join(cells))


def main() -> int:
    args = parse_args()
    root = repo_root()
    exe = resolve_path(root, args.exe) if args.exe else default_exe(root)
    data = resolve_path(root, args.data)
    out_dir = (
        resolve_path(root, args.out_dir)
        if args.out_dir
        else root
        / "target"
        / "benchmarks"
        / f"fusion-face-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    if not exe.exists():
        raise RuntimeError(
            f"training executable not found: {exe}. Build it first with: cargo build -p acad-brep-candle-train"
        )
    if not data.exists():
        raise RuntimeError(f"dataset not found: {data}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_harness:
        harness_path = out_dir / "harness.json"
        harness_log = out_dir / "harness.log"
        if harness_path.exists() and not args.force:
            print(f"Keeping existing harness report: {harness_path}")
        else:
            invoke_checked(
                exe,
                [
                    "inspect-harness",
                    "--data",
                    str(data),
                    "--out",
                    str(harness_path),
                    "--rare-count",
                    "20",
                ],
                harness_log,
                root,
            )

    rows: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    for variant_name in args.variants:
        variant = variant_spec(variant_name, args)
        if variant.run in seen_runs:
            raise RuntimeError(f"duplicate run name from variants: {variant.run}")
        seen_runs.add(variant.run)

        report_path = out_dir / f"{variant.run}.json"
        model_path = out_dir / f"{variant.run}.safetensors"
        log_path = out_dir / f"{variant.run}.log"

        if report_path.exists() and not args.force:
            print(f"Keeping existing run report: {report_path}")
        else:
            run_args = [
                "face-train",
                "--data",
                str(data),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--hidden",
                str(variant.hidden),
                "--rounds",
                str(variant.rounds),
                "--seed",
                str(args.seed),
                "--batch-size",
                str(args.batch_size),
                "--max-train-samples",
                str(args.train_samples),
                "--max-eval-samples",
                str(args.eval_samples),
                "--sample-strategy",
                variant.strategy,
                "--eval-split",
                args.eval_split,
                "--report",
                str(report_path),
                "--class-weights" if variant.class_weights else "--no-class-weights",
            ]
            if args.eval_split == "test":
                run_args.append("--final-test")
            if not args.no_save_models:
                run_args.extend(["--save", str(model_path)])

            start = time.perf_counter()
            invoke_checked(exe, run_args, log_path, root)
            elapsed = round(time.perf_counter() - start, 3)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"duration_seconds: {elapsed}\n")

        report = load_json(report_path)
        seg7 = class_metric(report, "eval", "segment_7")
        train_seg7 = class_metric(report, "train", "segment_7")
        rows.append(
            {
                "run": variant.run,
                "variant": variant_name,
                "hidden": report["config"]["hidden_dim"],
                "rounds": report["config"]["rounds"],
                "class_weights": report["config"]["use_class_weights"],
                "sampling_strategy": report["config"]["sampling_strategy"],
                "epochs": report["config"]["epochs"],
                "seed": report["config"]["seed"],
                "duration_seconds": parse_duration(log_path),
                "train_samples": report["sampler"]["train_samples"],
                "eval_samples": report["sampler"]["eval_samples"],
                "train_faces": report["sampler"]["train_faces"],
                "eval_faces": report["sampler"]["eval_faces"],
                "final_loss": report["final_loss"],
                "train_accuracy": report["train_metrics"]["accuracy"],
                "eval_accuracy": report["eval_metrics"]["accuracy"],
                "eval_macro_f1": report["eval_metrics"]["macro_f1"],
                "eval_weighted_f1": report["eval_metrics"]["weighted_f1"],
                "eval_macro_iou": report["eval_metrics"]["macro_iou"],
                "eval_macro_iou_present": report["eval_metrics"]["macro_iou_present"],
                "segment_7_train_support": train_seg7["support"] if train_seg7 else None,
                "segment_7_eval_support": seg7["support"] if seg7 else None,
                "segment_7_eval_f1": seg7["f1"] if seg7 else None,
                "segment_7_eval_iou": seg7["iou"] if seg7 else None,
                "train_ids_hash": report["sampler"]["train_record_ids_hash"],
                "eval_ids_hash": report["sampler"]["eval_record_ids_hash"],
            }
        )

    summary_csv = out_dir / "summary.csv"
    summary_md = out_dir / "summary.md"
    write_summary_csv(summary_csv, rows)

    harness_path = out_dir / "harness.json"
    manifest_hash = ""
    if harness_path.exists():
        manifest_hash = str(load_json(harness_path).get("manifest_hash", ""))
    write_summary_markdown(
        summary_md,
        rows,
        data=args.data,
        manifest_hash=manifest_hash,
        eval_split=args.eval_split,
        epochs=args.epochs,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )

    print_table(rows)
    print("")
    print("Benchmark complete")
    print(f"out_dir: {out_dir}")
    print(f"summary_csv: {summary_csv}")
    print(f"summary_md: {summary_md}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
