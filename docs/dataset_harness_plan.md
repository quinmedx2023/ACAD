# Dataset Harness Plan

The harness should make Fusion face-segmentation experiments honest and
repeatable before adding more sampler or model complexity. Keep the first
implementation small: validate the data, freeze the split policy, expose label
distribution, and write enough metrics to explain failures.

## Versioned Scope

Build this in versions. Do not implement the full harness in one patch.

### V0: Policy

- Freeze evaluation discipline.
- Decide the stable validation split rule.
- Define the minimal `harness.json` schema.
- No code required beyond documentation.

### V1: Dataset Inspect

- Add `inspect-harness`.
- Read only `dataset.json` and `manifest.jsonl`.
- Write `harness.json`.
- Compute manifest hash, split counts, label histograms, rare-label set,
  missing-label checks, graph face-count percentiles, and train/val drift.
- No tensorization.
- No model training changes.

### V2: Training Report

- Add `face-train --report`.
- Record resolved config, dataset manifest hash, seed, sampler name, selected ID
  hashes, final aggregate metrics, per-class F1, and per-class IoU.
- Keep output to one JSON file.
- No run directory tree unless it becomes necessary.

### V3: Sampler Dry-Run

- Add sampler dry-runs that operate on manifest label counts only.
- Compare graph count, face count, label coverage, selected ID hash, rare-label
  coverage, and drift against validation distribution.
- Do not train new samplers until dry-runs show better data slices.

### V4: Sampler Implementation

- Add face-budget and rare-balanced samplers only after V3 reports are useful.
- Multi-seed comparison helpers can wait until the single-run harness is stable.

## Current Commands

Implemented:

```powershell
cargo run -p acad-brep-candle-train -- inspect-harness `
  --data data\fusion-seg-v1 `
  --out target\fusion-harness.json

cargo run -p acad-brep-candle-train -- face-train `
  --data data\fusion-seg-v1 `
  --epochs 1 `
  --report target\face-train-report.json
```

`inspect-harness` is V1. `face-train --report` is V2. V3 sampler dry-runs and
V4 sampler implementations are not implemented yet.

## Non-Goals For V1

- No experiment database.
- No mandatory run directory tree.
- No automatic hyperparameter search.
- No new model architecture.
- No sampler changes until sampler dry-runs can be compared without training.

## Step 0: Evaluation Discipline

Fusion Gallery provides an official train/test split. The harness must prevent
test-set leakage:

- Official `test` is reserved for final, infrequent reports.
- All sampler and hyperparameter selection uses a deterministic validation
  split carved from official `train`.
- The validation split policy is stored in the harness report.
- Any run that evaluates on official `test` must be clearly marked as
  `final_test=true`.

Minimal policy for V1:

```text
official train -> train_inner + val_inner
official test  -> test_final
```

The split can be ID-hash based so it is stable across machines:

```text
hash(sample_id + seed) % 100 < val_percent
```

## V1 Dataset Inspect Report

Add one source-of-truth report next to `dataset.json`:

```text
harness.json
```

`harness.md` can be generated later. Avoid maintaining two hand-written report
formats.

`harness.json` should contain:

- harness version
- dataset format version
- manifest SHA-256 hash
- split-file hash when official `train_test.json` is used
- cleaner config that changes data semantics, especially `allow_boundary` and
  split source
- split policy: `train_inner`, `val_inner`, `test_final`
- record counts by split
- face-label histograms by split
- edge-label histograms by split
- graph face-count percentiles by split: p50, p90, p95, p99, max
- frozen rare-label definition and rare-label set
- labels missing from train_inner or val_inner
- train/val distribution drift

Use the manifest SHA-256 hash as the cleaned-dataset fingerprint. Short hashes
are acceptable for split assignment and selected-ID hashes, but dataset
provenance should use a collision-resistant fingerprint.

## Rare Labels

Rare labels must be frozen per dataset version. Otherwise `rare_label_macro_f1`
changes meaning between regenerations.

V1 default:

```text
rare if count < 1000 OR count / total_faces < 0.005
```

Store both the threshold and resulting label list in `harness.json`.

## Distribution Drift

Do not leave "drift summary" as prose. V1 should compute:

- per-label train/eval ratio table
- total variation distance between train_inner and val_inner label
  distributions

Total variation distance:

```text
0.5 * sum(abs(p_train[label] - p_val[label]))
```

This gives one scalar for comparing split and sampler quality.

## V2 Training Run Report

`face-train` should optionally write a single JSON report:

```text
--report target/face-train-report.json
```

V2 does not need a directory tree. The report should contain:

- git commit SHA and dirty flag
- full resolved training config
- dataset path and manifest hash
- harness version, rare-label set, and rare-label macro-F1/IoU
- sampler name, sampler config, seed
- selected train IDs hash
- selected eval IDs hash
- final aggregate metrics
- per-class metrics

Per-class metrics:

- label
- support
- predicted count
- true positives
- false positives
- false negatives
- precision
- recall
- F1
- IoU

Aggregate metrics:

- accuracy
- macro-F1
- weighted-F1
- macro-IoU over all classes
- macro-IoU over classes present in eval
- rare-label macro-F1
- rare-label macro-IoU

IoU is required because Fusion-style segmentation baselines commonly report
per-class IoU and mean IoU.

## V3 Sampler Dry-Runs

Before adding or trusting new samplers, implement sampler dry-runs that do not
load graph JSON or train the model.

Dry-run output:

- sampler name and config
- seed
- selected graph count
- selected face count
- selected ID list hash
- selected face-label histogram
- rare-label coverage ratio
- total variation distance against eval distribution

V3 samplers to compare:

- `uniform-graphs`: current deterministic graph-count selection.
- `face-budget-uniform`: deterministic selection capped by total faces.
- `rare-balanced-dry-run`: greedy rare-label coverage score, dry-run only until
  the report proves it improves coverage without extreme drift.

Sampler rule:

```text
selected_ids = f(manifest_hash, sampler_config, seed)
```

If selected ID hashes differ, two runs did not train on the same subset.

## Seed Policy

Single-seed runs are smoke tests only.

For real sampler comparison:

- run at least 3 seeds
- report mean and standard deviation for accuracy, macro-F1, macro-IoU, and
  rare-label metrics

Do not block V2/V3 implementation on multi-seed automation. Just record the seed
and label one-seed output as `smoke`.

## Sanity Checks

Required before large experiments:

1. `inspect-harness` (V1): reads `dataset.json` and `manifest.jsonl`, writes
   `harness.json`, and fails on unknown dataset format versions.
2. `face-train --report` (V2): writes final per-class F1 and IoU.
3. `sampler-dry-run` (V3): reports selected ID hash and label coverage without
   tensorizing graphs.
4. `overfit-small` (V3 or later): uses pinned graph IDs and must reach a fixed threshold, for
   example `>=99%` train accuracy within a configured epoch budget.

## Implementation Order

1. V0: document validation/test discipline and validation split policy.
2. V1: implement `inspect-harness` with manifest hash and distribution report.
3. V2: implement `face-train --report` with per-class F1, IoU, and run provenance.
4. V3: implement sampler dry-run reports with selected ID hashes.
5. V4: implement face-budget sampler only after dry-run reports show the data
   slice is better than uniform graph sampling.
6. Later: add multi-seed comparison helpers.

## Acceptance Criteria

- The harness can inspect a cleaned Fusion dataset without tensorizing graphs.
- The report records whether official `train_test.json` was used.
- Official test rows are not used for sampler or hyperparameter selection.
- The cleaned dataset is pinned by manifest hash.
- Rare labels are frozen in `harness.json`.
- Train/eval drift has a numeric total-variation score.
- `face-train --report` includes per-class F1 and IoU.
- Sampler dry-runs compare face budget and label coverage, not only graph count.
