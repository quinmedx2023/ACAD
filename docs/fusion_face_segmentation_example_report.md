# Fusion Face Segmentation Example Report

## Summary

This report documents the current real-data face segmentation example for the
ACAD hybrid BRep encoder. The example trains a Candle-native model on cleaned
Fusion 360 Gallery STEP-derived BRep graphs and uses `labels/*.json` face
segment labels as the supervised target.

The run is a smoke experiment, not a converged model. Its value is that the
end-to-end training path is now real:

```text
Fusion STEP + .seg
  -> OCCT cleaner
  -> ACAD BRep graph JSON + face labels
  -> GraphTensorizer
  -> hybrid UV-grid + coedge message-passing encoder
  -> per-face classifier head
  -> face accuracy + macro-F1
```

## Data

Cleaned dataset:

```text
data/fusion-seg-v1
```

Full cleaned dataset size:

| Split | Graphs |
|-------|-------:|
| train | 32,184 |
| val | 10,728 |
| total | 42,912 |

Full dataset face-label distribution:

| Label | Faces |
|-------|------:|
| segment_0 | 454,428 |
| segment_1 | 116,119 |
| segment_2 | 165,023 |
| segment_3 | 24,978 |
| segment_4 | 111,100 |
| segment_5 | 23,494 |
| segment_6 | 46,546 |
| segment_7 | 695 |

`segment_7` is extremely rare, so macro-F1 is a more useful sanity metric than
accuracy alone.

## Model

The example uses `FaceSegmentationModel`:

```text
HybridBrepEncoder
  face UV-grid conv2d front-end
  edge curve conv1d front-end
  coedge <-> face/edge/mate message passing
  per-face embeddings
  linear face classifier head
```

Loss is per-face cross entropy over the concatenated ragged batch faces.

## Default Smoke Command

```powershell
cargo run -p acad-brep-candle-train -- face-train `
  --data data\fusion-seg-v1 `
  --epochs 3 `
  --rounds 1 `
  --hidden 32 `
  --batch-size 8 `
  --max-train-samples 512 `
  --max-eval-samples 128 `
  --eval-split test `
  --save target\fusion-face-seg-default-smoke.safetensors
```

Default settings for this example:

| Setting | Value |
|---------|-------|
| train sampling | uniform |
| eval sampling | uniform |
| class weights | disabled |
| shuffle each epoch | enabled |
| epochs | 3 |
| hidden dim | 32 |
| message-passing rounds | 1 |
| batch size | 8 graphs |

## Sampled Label Coverage

The default smoke loads 512 train graphs and 128 eval graphs from the
manifest. It does not load the full dataset.

| Split | Graphs | Faces |
|-------|-------:|------:|
| train | 512 | 10,375 |
| eval | 128 | 3,222 |

Sampled train face counts:

| Label | Faces |
|-------|------:|
| segment_0 | 4,899 |
| segment_1 | 1,218 |
| segment_2 | 1,690 |
| segment_3 | 242 |
| segment_4 | 1,429 |
| segment_5 | 193 |
| segment_6 | 696 |
| segment_7 | 8 |

Sampled eval face counts:

| Label | Faces |
|-------|------:|
| segment_0 | 1,611 |
| segment_1 | 351 |
| segment_2 | 565 |
| segment_3 | 110 |
| segment_4 | 386 |
| segment_5 | 48 |
| segment_6 | 151 |
| segment_7 | 0 |

The sampled eval split contains no `segment_7` faces, so this smoke cannot
measure performance on that rare class.

## Result

Historical default smoke result from the legacy modulo-split dataset, before
unit-box tensor normalization and official `test` split preservation:

| Metric | Value |
|--------|------:|
| final loss | 1.496397 |
| train face accuracy | 45.21% |
| eval face accuracy | 50.87% |
| eval macro-F1 | 0.1832 |

Checkpoint:

```text
target\fusion-face-seg-default-smoke.safetensors
```

## Comparison Runs

All comparison runs used the same 512 train graph / 128 eval graph budget.

| Run | Class Weights | Train Sampling | Eval Sampling | Eval Accuracy | Eval Macro-F1 |
|-----|---------------|----------------|--------------|-------------:|-------------:|
| default smoke | no | uniform | uniform | 50.87% | 0.1832 |
| uniform weighted | yes | uniform | uniform | 22.16% | 0.1600 |
| uniform unweighted earlier run | no | uniform | uniform | 44.85% | 0.1937 |
| face-balanced train | yes | face-balanced | uniform | 5.40% | 0.0632 |

The face-balanced train run improved rare-label coverage in training:

```text
segment_7 train faces:
  uniform:       8
  face-balanced: 496
```

However, it produced a strong train/eval distribution shift at this small graph
budget and performed worse on the uniform eval sample.

## Interpretation

The example proves the real supervised task is wired correctly:

- targets come from `labels/*.json` face labels, not graph class ids
- ragged graph batches preserve per-face target order
- the encoder produces per-face embeddings
- the classifier trains on real Fusion-derived BRep data
- the CLI reports face accuracy, macro-F1, and sampled label coverage
- checkpoint saving works, with a metadata sidecar that records label names and
  model shape

Recent reliability optimizations are now part of the example pipeline:

- training tensors use per-solid unit-box geometry normalization
- cleaned Fusion manifests preserve the official `train_test.json` split
- manifest rows include label histograms for faster sampling diagnostics
- face checkpoints save label vocabulary and model shape sidecars, and
  `load_face_checkpoint` reconstructs the model from that metadata

The result is not yet a strong segmentation model. The current smoke is short,
CPU-only, and sampled. It is useful as a regression test and baseline for the
training pipeline.

## Current Recommendation

Use the default smoke configuration for quick validation:

```powershell
cargo run -p acad-brep-candle-train -- face-train `
  --data data\fusion-seg-v1 `
  --epochs 3 `
  --rounds 1 `
  --hidden 32 `
  --batch-size 8 `
  --max-train-samples 512 `
  --max-eval-samples 128 `
  --eval-split test
```

Use `--class-weights` or `--sample-strategy face-balanced` only for diagnostics
until a better face-budgeted sampler is implemented.

## Next Work

1. Add a face-budgeted sampler that limits total faces, not just graph count.
2. Regenerate the full Fusion dataset with official `test` preservation and
   rerun comparable smoke metrics.
3. Add per-class precision/recall/F1 reporting.
4. Train longer with a larger sample budget after the sampler is stable.
5. Add trimming-aware UV masks from OCCT face classification.
