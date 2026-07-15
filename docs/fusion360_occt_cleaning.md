# Fusion 360 Gallery Segmentation Cleaning

## Source Dataset

The intended source is Autodesk's Fusion 360 Gallery segmentation data. The
repository documents a segmentation dataset with 35,680 designs and an extended
STEP dataset with 42,912 STEP files plus associated segmentation information.
See:

- https://github.com/AutodeskAILab/Fusion360GalleryDataset
- https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/docs/segmentation.md

## OCCT Path

The default cleanup path uses a C++ Open Cascade sidecar:

```text
Fusion raw STEP + .seg
  -> tools/occt_cleaner (C++ OCCT STEPControl_Reader)
  -> faces, edges, coedges, mate links
  -> UV/curve sampled geometry
  -> graph labels + face segmentation labels + edge labels
  -> ACAD dataset format
```

The generated output matches the existing ACAD dataset layout:

```text
dataset.json
manifest.jsonl
graphs/*.json
labels/*.json
```

## Commands

Build the sidecar:

For the official Windows OCCT package extracted as `C:\tools\OpenCascade`,
prefer direct root mode:

```powershell
cmake -S tools/occt_cleaner -B tools/occt_cleaner/build `
  -DCMAKE_BUILD_TYPE=Release `
  -DACAD_OCCT_ROOT=C:\tools\OpenCascade
cmake --build tools/occt_cleaner/build
```

`ACAD_OCCT_ROOT` should point at the package root containing `inc`, `cmake`,
and `win64\vc14\lib`. This avoids importing optional Visualization/Draw CMake
targets from the official package.

If `OpenCASCADE_DIR` is set to `C:\tools\OpenCascade\cmake`, the cleaner CMake
will infer `ACAD_OCCT_ROOT=C:\tools\OpenCascade` automatically.

Installed CMake package mode is also supported:

```powershell
cmake -S tools/occt_cleaner -B tools/occt_cleaner/build `
  -DOpenCASCADE_DIR=<path-to-occt-cmake-config>
cmake --build tools/occt_cleaner/build --config Release
```

For the local package above, the `OpenCASCADE_DIR` value would be:

```text
C:\tools\OpenCascade\cmake
```

Alternatively, let CMake fetch and build OCCT 8.0.0:

```powershell
cmake -S tools/occt_cleaner -B tools/occt_cleaner/build `
  -DACAD_FETCH_OCCT=ON
cmake --build tools/occt_cleaner/build --config Release
```

This uses `ExternalProject_Add` with:

```text
https://github.com/Open-Cascade-SAS/OCCT/archive/refs/tags/V8_0_0.zip
```

Clean a small subset first:

```powershell
cargo run -p acad-brep-candle-train -- clean-fusion `
  --raw data/s2.0.1_extended_step `
  --out data/fusion-seg-v1 `
  --limit 100 `
  --allow-boundary
```

When `data/s2.0.1_extended_step/train_test.json` is present, the Rust wrapper
passes it to the sidecar automatically. Fusion `train` ids stay `train`; Fusion
`test` ids stay `test`. Override this with `--split-file <path>`, or disable it
with `--no-split-file` to return to deterministic modulo splitting.

Split files are strict: duplicate ids or STEP files missing from the split file
fail the cleanup instead of silently falling back to modulo assignment.

The generated manifest also includes `face_label_counts` and
`edge_label_counts` per record. This keeps face-balanced sampling diagnostics
from reparsing all label JSON files on each training run.

The Rust wrapper defaults to:

```text
tools/occt_cleaner/build/Release/occt_cleaner.exe if present, otherwise
tools/occt_cleaner/build/occt_cleaner.exe
```

Override it with `--exe <path>` if needed.

Before running the cleaner on Windows, initialize the OCCT runtime environment
so dependent DLLs are on `PATH` if you run `occt_cleaner.exe` directly. The
Rust `clean-fusion` wrapper automatically adds these runtime directories when
it detects `ACAD_OCCT_ROOT`, `OpenCASCADE_DIR`, `CASROOT`, or the default
`C:\tools\OpenCascade` layout. The wrapper also sets the key OCCT resource
environment variables used by STEP import, including `CASROOT`,
`CSF_XSMessage`, and `CSF_STEPDefaults`.

For direct sidecar runs, this workspace's third-party DLL root is
`C:\tools\OpenCascade\3rdparty`:

```powershell
cmd /c "set THIRDPARTY_DIR=C:\tools\OpenCascade\3rdparty&& call C:\tools\OpenCascade\env.bat vc143 64&& tools\occt_cleaner\build\occt_cleaner.exe --help"
```

Inspect:

```powershell
cargo run -p acad-brep-candle-train -- inspect --data data/fusion-seg-v1
```

Train a real face-segmentation smoke run:

```powershell
cargo run -p acad-brep-candle-train -- face-train `
  --data data/fusion-seg-v1 `
  --epochs 3 `
  --rounds 1 `
  --hidden 32 `
  --batch-size 8 `
  --max-train-samples 512 `
  --max-eval-samples 128 `
  --eval-split test `
  --save target/fusion-face-seg-smoke.safetensors
```

The graph tensorizer normalizes geometry per solid by default: it centers each
graph on its bounding-box center and scales coordinates and lengths by the
largest bbox extent. The source JSON remains in CAD units; normalization is
applied only to training tensors.

When `--save` is used, `face-train` writes both the `.safetensors` weights and a
`*.metadata.json` sidecar containing the face-label vocabulary and model shape
needed to interpret the checkpoint.

`face-train` uses `labels/*.json` face labels as the supervised target. The
default config uses uniform manifest sampling plus deterministic per-epoch
shuffle, instead of loading all 42,912 cleaned graphs. Pass
`--max-train-samples 0 --max-eval-samples 0` for a full in-memory run.

For regenerated official Fusion datasets, use `--eval-split test` intentionally.
The default `--eval-split val` is retained for synthetic or custom datasets with
a validation split and fails clearly when no validation rows exist.

Optional imbalance controls:

```powershell
--class-weights
--sample-strategy face-balanced
```

The face-balanced selector is useful for train coverage diagnostics, but it
changes the sampled training distribution. Evaluation sampling is always uniform.

Historical local sampled run on the legacy modulo-split `data\fusion-seg-v1`
before unit-box normalization and official test preservation:

```text
train_samples: 512 graphs / 10,375 faces
eval_samples:  128 graphs / 3,222 faces
face_classes:  8
class_weights: disabled
train_sample:  uniform
final_loss:    1.496397
train_acc:     45.21%
eval_acc:      50.87%
eval_macro_f1: 0.1832
checkpoint:    target\fusion-face-seg-default-smoke.safetensors
train_counts:  segment_0:4899, segment_1:1218, segment_2:1690, segment_3:242,
               segment_4:1429, segment_5:193, segment_6:696, segment_7:8
eval_counts:   segment_0:1611, segment_1:351, segment_2:565, segment_3:110,
               segment_4:386, segment_5:48, segment_6:151, segment_7:0
```

Comparison runs at the same 512/128 graph budget:

```text
uniform + class weights:         eval_acc 22.16%, eval_macro_f1 0.1600
face-balanced train + uniform eval: eval_acc 5.40%, eval_macro_f1 0.0632
```

## Current Environment Status

This workspace has a local OCCT package at `C:\tools\OpenCascade`, the raw
Fusion extended STEP dataset at `data\s2.0.1_extended_step`, and the cleaned
ACAD-format dataset at `data\fusion-seg-v1`. The existing full cleaned dataset
was generated before official `test` preservation and manifest label histograms;
regenerate it before using published-style metrics.

## Known Simplifications

- Coedges are derived from edge-to-face incidence. Ordered loop `next`/`prev`
  walks are not emitted yet.
- Face UV masks are currently `1.0` everywhere. Real trimming-aware UV masking
  should be added with OCCT face classification or an occwl-style sampler.
- Edge labels are inferred from curve type; Fusion `.seg` supplies face labels.
