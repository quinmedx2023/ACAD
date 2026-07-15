# Dataset Harness Plan

This phase should make Fusion face-segmentation experiments comparable before
changing model architecture. The harness is responsible for data provenance,
distribution diagnostics, split control, sampler inputs, and evaluation outputs.

## Goals

- Make every training run traceable to a dataset version, cleaner config, split
  source, and sampler config.
- Detect rare-class and train/eval distribution problems before training.
- Produce per-class metrics so macro-F1 failures are actionable.
- Keep samplers comparable by running them under the same split, budget, and
  report format.

## Dataset Manifest Contract

Each cleaned dataset should have a harness report next to `dataset.json`:

```text
dataset.json
manifest.jsonl
graphs/*.json
labels/*.json
harness.json
harness.md
```

`harness.json` should record:

- dataset format version and harness version
- raw dataset path or source id
- raw file counts and optional content hashes for split/source files
- cleaner executable path/version and relevant flags, especially
  `--allow-boundary` and `--split-file`
- split provenance: official Fusion `train_test.json` vs modulo fallback
- record counts by split
- face/edge/graph count statistics by split
- face-label and edge-label histograms by split
- missing-label, unknown-label, empty-graph, and validation error counts

## Distribution Diagnostics

The harness should compute these before training:

- global label histogram
- per-split label histogram
- rare-label list using configurable thresholds, for example `<1000` faces or
  `<0.5%` of total faces
- train/eval label ratio table
- labels present in eval but absent in train
- labels present in train but absent in eval
- graph face-count percentiles by split: p50, p90, p95, p99, max
- face budget estimate for sampled runs

This report answers whether poor metrics are caused by model failure, sampler
failure, or an invalid evaluation slice.

## Evaluation Artifacts

Every `face-train` run should optionally write a run report:

```text
target/runs/<run-id>/run.json
target/runs/<run-id>/metrics.csv
target/runs/<run-id>/confusion.csv
target/runs/<run-id>/checkpoint.safetensors
target/runs/<run-id>/checkpoint.metadata.json
```

Minimum per-class metrics:

- label name
- support
- predicted count
- true positives, false positives, false negatives
- precision
- recall
- F1

Aggregate metrics:

- accuracy
- macro-F1 over all classes
- macro-F1 over classes present in eval
- weighted-F1
- rare-label macro-F1

## Sampler Experiments

Samplers should be evaluated only after the harness report is available.

Baseline samplers:

- `uniform-graphs`: current deterministic graph-count selection.
- `face-budget-uniform`: deterministic graph order capped by total faces.
- `rare-balanced-graphs`: greedy graph selection using rare-label gain.
- `rare-balanced-face-budget`: rare-label gain with a hard total-face budget.

Each sampler report should include:

- selected graph count
- selected face count
- selected face-label histogram
- coverage ratio versus the full train split
- rare-label coverage ratio
- train/eval drift summary

## Sanity Checks

Before running large experiments:

1. `inspect-harness`: dataset validates and reports official split provenance.
2. `overfit-small`: train on 10-20 graphs and verify training accuracy can climb
   near 100%.
3. `sampler-dry-run`: run sampler without tensorizing graphs and inspect label
   coverage.
4. `eval-only-report`: run metrics on a saved checkpoint and write per-class
   artifacts.

## Implementation Order

1. Add a dataset harness command that reads `dataset.json` + `manifest.jsonl`
   and writes `harness.json` / `harness.md`.
2. Add per-split histograms and distribution drift diagnostics.
3. Add dry-run sampler reports without changing training.
4. Add per-class metrics and confusion matrix to `face-train`.
5. Add face-budget and rare-balanced samplers using the harness diagnostics.
6. Run comparable smoke experiments under the same harness.

## Acceptance Criteria

- A regenerated Fusion dataset can be inspected without tensorizing graphs.
- The harness clearly reports whether official `train_test.json` was used.
- Rare labels and train/eval missing-label cases are visible before training.
- A `face-train` run writes per-class metrics and confusion matrix.
- Sampler changes can be compared by face budget and label coverage, not only
  graph count.
