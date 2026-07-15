# Fusion Face Benchmark 2026-07-15

## Scope

This is a small engineering benchmark for the Candle face-segmentation pipeline,
not a final model-selection result.

Dataset:

```text
data\fusion-seg-v1
```

Dataset fingerprint:

```text
manifest_hash = d290dac73d051d268a56516231b641d4f8875f5eea71c69c4b0235a36068030c
manifest_hash_algorithm = sha256
```

Harness summary:

| Split | Graphs |
|-------|-------:|
| train_inner | 32,873 |
| val_inner | 3,585 |
| test_final | 6,454 |

These historical runs used the official `test` split as an engineering
benchmark. Current `face-train` defaults to the harness inner validation split;
official test evaluation requires explicit `--eval-split test --final-test`.
Treat the numbers below as pipeline benchmarks, not model-selection results.

## Reusable Script

Use the Python benchmark script for future runs:

```powershell
python scripts/train_fusion_face_benchmark.py `
  --train-samples 1024 `
  --eval-samples 256 `
  --epochs 3
```

Outputs:

```text
target\benchmarks\fusion-face-<timestamp>\
  harness.json
  <run>.json
  <run>.log
  <run>.safetensors
  <run>.metadata.json
  summary.csv
  summary.md
```

For a fast script check:

```powershell
python scripts/train_fusion_face_benchmark.py `
  --out-dir target\benchmarks\fusion-face-script-smoke `
  --variants uniform `
  --train-samples 8 `
  --eval-samples 4 `
  --epochs 1 `
  --batch-size 2 `
  --hidden 16 `
  --rounds 1 `
  --no-save-models `
  --force
```

## Benchmark Setup

Common settings:

| Setting | Value |
|---------|------:|
| epochs | 3 |
| learning rate | 0.003 |
| seed | 42 |
| batch size | 8 graphs |
| train graph budget | 1,024 |
| eval graph budget | 256 |
| eval split | official test with final-test opt-in |

The run artifacts are under:

```text
target\benchmarks\fusion-face-20260715
```

## Results

| Run | Hidden | Rounds | Class Weights | Sampling | Seconds | Train Faces | Eval Acc | Macro-F1 | Weighted-F1 | Macro-IoU |
|-----|-------:|-------:|---------------|----------|--------:|------------:|---------:|---------:|------------:|----------:|
| uniform_h32_r1 | 32 | 1 | no | uniform | 60.5 | 22,468 | 52.47% | 0.2020 | 0.4246 | 0.1387 |
| uniform_weighted_h32_r1 | 32 | 1 | yes | uniform | 54.3 | 22,468 | 36.30% | 0.2372 | 0.3332 | 0.1494 |
| face_balanced_h32_r1 | 32 | 1 | no | face-balanced | 313.3 | 118,689 | 43.26% | 0.3028 | 0.4322 | 0.1926 |
| uniform_h64_r2 | 64 | 2 | no | uniform | 107.5 | 22,468 | 55.53% | 0.2501 | 0.4839 | 0.1712 |

Reproducibility hashes from the original run:

| Run | Train IDs Hash | Eval IDs Hash |
|-----|----------------|---------------|
| uniform_h32_r1 | 65c04726738a1679 | dfab0ece981b3f0e |
| uniform_weighted_h32_r1 | 65c04726738a1679 | dfab0ece981b3f0e |
| face_balanced_h32_r1 | cf3521122ea5cfd7 | dfab0ece981b3f0e |
| uniform_h64_r2 | 65c04726738a1679 | dfab0ece981b3f0e |

Current code emits SHA-256 selected-ID hashes in new reports.

## Interpretation

The larger uniform model is the best current default smoke:

- best eval accuracy: 55.53%
- best weighted-F1: 0.4839
- better macro-F1 than the small uniform baseline
- moderate runtime increase

Class weights help macro metrics slightly but hurt accuracy and weighted-F1:

- macro-F1 improves from 0.2020 to 0.2372
- eval accuracy drops from 52.47% to 36.30%
- weighted-F1 drops from 0.4246 to 0.3332

Face-balanced sampling gives the best macro-F1 and mIoU, but the comparison is
not clean by compute budget:

- macro-F1 improves to 0.3028
- mIoU improves to 0.1926
- selected train faces jump from 22,468 to 118,689 at the same graph budget
- runtime jumps from about 60 seconds to about 313 seconds

This confirms the earlier suspicion: graph-count budgets are misleading for BRep
parts. Sampler comparisons should be done by face budget and label coverage, not
only by graph count.

## Rare Class

`segment_7` remains unresolved in this benchmark:

| Run | Train segment_7 Faces | Eval segment_7 Faces | Eval segment_7 F1 |
|-----|----------------------:|---------------------:|------------------:|
| uniform_h32_r1 | 23 | 1 | 0.0 |
| uniform_weighted_h32_r1 | 23 | 1 | 0.0 |
| face_balanced_h32_r1 | 593 | 1 | 0.0 |
| uniform_h64_r2 | 23 | 1 | 0.0 |

The eval sample has only one `segment_7` face, so this benchmark cannot measure
rare-class performance reliably. The next benchmark should either evaluate a
larger test sample or report rare-label metrics on a fixed validation slice with
enough rare support.

## Next Steps

1. Add sampler dry-runs from Phase L V3 so selected graph IDs, face budget, and
   label coverage are visible before training.
2. Add a face-budgeted sampler. The target should be comparable face counts, not
   comparable graph counts.
3. Repeat the best two candidates on the default `train_inner`/`val_inner`
   policy with at least three seeds.
