# Progress Log

## Session: 2026-07-15 (afternoon) — Hybrid encoder optimization

Optimized the Candle smoke test into a real hybrid BRep encoder using the mature
architectures (UV-Net geometry + BRepNet coedge message passing), staying
Candle-native per user request ("candle not OK?" → yes, Candle is capable).

- Schema (`acad-brep-graph`): added `SurfaceGeometry` (UV grid) + `CurveGeometry`
  (curve grid) with procedural samplers, coedge `mate` links, multi-hole plate
  structural variation, serde JSON export, and geometry/mate validation.
- Tensorizer (`acad-brep-encoder`): added `GraphTensorizer`/`GraphTensors`
  emitting channel-major UV/curve grids + coedge topology index arrays; kept the
  deterministic pooled encoder for the CLI.
- Encoder (`acad-brep-candle-train`): `BrepBatch` ragged batching, conv2d/conv1d
  geometry front-ends, coedge hetero message passing via `index_add`, residual +
  layer norm, graph/face/edge readouts, train/val split, accuracy + macro-F1,
  checkpoint save (`--save`) and load (`load_encoder`).
- Build env: identified that VS 18 preview lacks C++ headers (`vcruntime.h`);
  used the complete VS 2022 toolchain (`14.44.35207`) for `LIB`/`INCLUDE`/`CC`.
  `Device::set_seed` is unsupported on Candle CPU → made best-effort.
- Verification: `cargo fmt --all -- --check` clean; `cargo test --workspace` =
  16 tests pass (graph 7, encoder 4, candle-train 5); training run (54 train / 18
  held-out val) → final_loss ~0.005, train/val accuracy 100%, val macro-F1 1.0;
  checkpoint `target/hybrid-brep-encoder.safetensors` (259 KB) written.
- Honest caveat recorded in docs: the 3 synthetic classes are still easily
  separable, so 100% val proves the pipeline is correct, not that the encoder
  captures subtle geometry.

### Follow-up verification after user refactor
- Re-read `task_plan.md`, `progress.md`, and key implementation files.
- Confirmed implementation contains `SurfaceGeometry`, `CurveGeometry`, coedge mates, `GraphTensorizer`, `BrepBatch`, Candle conv geometry front-ends, `index_add` message passing, metrics, and checkpoint load.
- Ran `cargo fmt --all -- --check` with explicit VS2022/Windows SDK env: passed.
- Ran `cargo test --workspace`: 16 tests passed.
- Ran short training smoke:
  - command: `cargo run -p acad-brep-candle-train -- --epochs 5 --samples-per-class 6 --save target\hybrid-brep-encoder-smoke.safetensors`
  - train samples: 14
  - val samples: 4
  - final_loss: `0.373883`
  - train_accuracy: `100.00%`
  - val_accuracy: `100.00%`
  - val_macro_f1: `1.0000`
- Updated `task_plan.md` checklist statuses to match the completed hybrid implementation.

## Session: 2026-07-15 — Real BRep dataset

- Added `acad-brep-dataset`, an ML-runtime-free crate for the on-disk dataset format.
- Dataset format:
  - `dataset.json`
  - `manifest.jsonl`
  - `graphs/*.json`
  - `labels/*.json`
- Added graph/face/edge labels for `box`, `cylinder`, and `plate_with_holes`.
- Added CLI subcommands:
  - `brep-candle-train dataset --out data/synthetic-v1 --samples-per-class 12 --val-fraction 0.25`
  - `brep-candle-train inspect --data data/synthetic-v1`
  - `brep-candle-train train --data data/synthetic-v1 ...`
- Generated `data/synthetic-v1`:
  - 36 records
  - 27 train / 9 val
  - 74 files total
  - face labels: `bottom`, `bottom_cap`, `cylinder_side`, `hole_wall`, `outer_side`, `side`, `top`, `top_cap`
  - edge labels: `cap_edge`, `convex_line`, `hole_edge`
- Verification:
  - `cargo test --workspace`: 18 tests passed
  - `cargo run -p acad-brep-candle-train -- inspect --data data\synthetic-v1`: passed
  - `cargo run -p acad-brep-candle-train -- train --data data\synthetic-v1 --epochs 30 --rounds 2 --hidden 48 --save target\dataset-v1-hybrid-brep-encoder.safetensors`: final_loss `0.061632`, train accuracy `100%`, val accuracy `100%`, val macro-F1 `1.0`

## Session: 2026-07-15 — OCCT Fusion cleanup scaffold

- User requested using OCCT to clean Fusion 360 Gallery Segmentation.
- Checked local environment:
  - `DRAWEXE` not found.
  - Python has no `OCC`, `OCP`, or `FreeCAD` module.
  - No OCCT/OpenCascade env vars found.
  - vcpkg exists but OpenCascade is not installed.
- User then requested using C++ sidecar directly.
- Added `tools/occt_cleaner`, a C++ OCCT sidecar:
  - `CMakeLists.txt`
  - `src/main.cpp`
  - `README.md`
  - imports STEP with `STEPControl_Reader`;
  - emits ACAD dataset JSON directly, without Python.
- Updated Rust CLI wrapper:
  - `brep-candle-train clean-fusion --raw DIR --out DIR --limit N` defaults to C++ sidecar.
- User then requested no Python fallback; removed `tools/fusion_occt_clean.py` and Python backend flags.
- `clean-fusion` now has only the C++ OCCT sidecar path. Default executable:
  - `tools/occt_cleaner/build/Release/occt_cleaner.exe` on Windows.
  - `tools/occt_cleaner/build/occt_cleaner` elsewhere.
- Added CMake `ACAD_FETCH_OCCT=ON` superbuild support:
  - downloads `https://github.com/Open-Cascade-SAS/OCCT/archive/refs/tags/V8_0_0.zip`;
  - builds/installs OCCT under the cleaner build tree;
  - links `occt_cleaner` against the fetched install.
- Added docs:
  - `docs/fusion360_occt_cleaning.md`
- Verification:
  - `cargo fmt --all -- --check`: passed.
  - `cargo test --workspace`: 18 tests passed.
  - `cargo run -p acad-brep-candle-train -- clean-fusion --help`: passed with explicit VS2022/SDK env after Python fallback removal.
  - `cmake -S tools\occt_cleaner -B tools\occt_cleaner\build-ninja -G Ninja ...`: C++ compiler works after adding Windows SDK bin path, then configuration stops at missing `OpenCASCADEConfig.cmake`, as expected in this environment.
  - `cmake -S tools\occt_cleaner -B tools\occt_cleaner\build-ninja-fetch -G Ninja -DACAD_FETCH_OCCT=ON ...`: configure/generate passed without downloading; download/build occurs at build time.
- Full cleanup is blocked locally until C++ Open Cascade and the raw Fusion segmentation dataset are installed.
- Raw dataset search found no `Fusion360GalleryDataset`/Fusion raw folder under common local roots.

### Follow-up: local OCCT package at `C:\tools\OpenCascade`
- User reorganized the official Windows OCCT package so the root is
  `C:\tools\OpenCascade` with `cmake`, `inc`, `win64`, and `3rdparty` directly
  under it.
- Added `ACAD_OCCT_ROOT` CMake mode for `tools/occt_cleaner` and automatic root
  inference when `OpenCASCADE_DIR=C:\tools\OpenCascade\cmake`.
- Avoided the official package's full `OpenCASCADEConfig.cmake` target import
  path for this sidecar, because it can pull optional Visualization/Draw targets
  requiring VTK.
- Updated the sidecar link list for OCCT 8.0.0 libraries:
  - replaced old `TKSTEPBase` / `TKSTEPAttr` / `TKSTEP`;
  - linked `TKDE` and `TKDESTEP`.
- Updated `main.cpp` for OCCT 8.0 by replacing the removed
  `TopTools_ListIteratorOfListOfShape.hxx` include with `NCollection_List`
  iteration.
- Verified:
  - CMake configure with `OpenCASCADE_DIR=C:\tools\OpenCascade\cmake` inferred
    `ACAD_OCCT_ROOT=C:/tools/OpenCascade`.
  - Release Ninja build passed in `tools\occt_cleaner\build-openenv-release`.
  - `cmd /c "set THIRDPARTY_DIR=C:\tools\OpenCascade\3rdparty&& call C:\tools\OpenCascade\env.bat vc143 64&& tools\occt_cleaner\build-openenv-release\occt_cleaner.exe --help"` printed usage.
  - `cargo fmt --all -- --check` passed.
  - `cargo test --workspace` passed: 18 tests.
  - `cargo run -p acad-brep-candle-train -- clean-fusion --help` passed.
- Note:
  - `tools\occt_cleaner\build` already had a Visual Studio generator cache; a
    Ninja configure into that same directory failed with generator mismatch.
    Reconfiguring with the cached Visual Studio generator hit sandbox access
    denial under `C:\Users\huang\AppData\Local\Microsoft SDKs`. The verified
    sidecar binary is therefore in `build-openenv-release`; use a fresh build
    directory when switching generators.
- Remaining blocker:
  - raw Fusion 360 Gallery segmentation files are still not present under
    `raw/`.

### Follow-up: automatic OCCT runtime environment
- Confirmed local third-party DLLs are under:
  - `C:\tools\OpenCascade\3rdparty`
- Updated the Rust `clean-fusion` wrapper so the sidecar child process receives
  OCCT runtime setup automatically:
  - detects OCCT root from `ACAD_OCCT_ROOT`, `CASROOT`, `OpenCASCADE_DIR`, or
    the default `C:\tools\OpenCascade`;
  - prepends `win64\vc*\bin` and relevant `3rdparty` DLL/bin directories to the
    child `PATH`;
  - sets key OCCT resource variables such as `CASROOT`, `THIRDPARTY_DIR`,
    `CSF_OCCTResourcePath`, `CSF_XSMessage`, and `CSF_STEPDefaults`.
- Updated `tools/occt_cleaner/README.md` and
  `docs/fusion360_occt_cleaning.md` to document that Cargo/Rust cleanup no
  longer needs a manual `cmd /c call env.bat` wrapper; direct sidecar runs can
  still use `env.bat`.
- Verification:
  - `cargo fmt --all -- --check`: passed.
  - `cargo test --workspace`: passed, 18 tests.
  - `cargo build -p acad-brep-candle-train`: passed.
  - With `OpenCASCADE_DIR=C:\tools\OpenCascade\cmake` and no `THIRDPARTY_DIR`,
    `target\debug\brep-candle-train.exe clean-fusion --raw raw\missing-fusion-segmentation --out data\occt-runtime-test --exe tools\occt_cleaner\build-openenv-release\occt_cleaner.exe`
    successfully launched the sidecar; it failed only because the intentionally
    missing raw input directory does not exist. The previous Windows DLL load
    failure `0xc0000135` is gone.

### Follow-up: real Fusion extended STEP dataset smoke
- User extracted the Fusion 360 Gallery Segmentation Extended STEP dataset to:
  - `C:\repositories\ACAD\data\s2.0.1_extended_step`
- Confirmed extracted layout:
  - `breps\step`: 42,912 `.stp` files
  - `breps\seg`: 42,912 `.seg` files
  - `timeline_info`: 42,916 `.json` files counted overall with metadata files
  - `segment_names.json`, `train_test.json`, `additional_breps*.json`
- Added `/data/s2.0.1_extended_step/` to `.gitignore` to prevent the raw
  dataset from being accidentally tracked.
- First cleanup smoke without `--allow-boundary`:
  - command used `--limit 5`
  - records: 2
  - skipped: 3
  - skip reason: boundary edges with only 1 incident face
- Added `--allow-boundary` to the Rust `clean-fusion` wrapper and help text.
- Cleanup smoke with `--allow-boundary`:
  - `--limit 5`
  - records: 5
  - skipped: 0
- Generated real cleaned dataset:
  - command: `target\debug\brep-candle-train.exe clean-fusion --raw data\s2.0.1_extended_step --out data\fusion-seg-v1-100 --exe tools\occt_cleaner\build-openenv-release\occt_cleaner.exe --limit 100 --allow-boundary`
  - records: 100
  - skipped: 0
  - train: 75
  - val: 25
  - face labels: `segment_0` 766, `segment_1` 283, `segment_2` 511,
    `segment_3` 44, `segment_4` 529, `segment_5` 19, `segment_6` 140
  - edge labels: `circle_edge` 1605, `line_edge` 2885, `other_edge` 905
- Verified trainer-side dataset reading:
  - `cargo run -p acad-brep-candle-train -- inspect --data data\fusion-seg-v1-100`
  - passed with the same summary counts.

### Follow-up: full Fusion cleanup
- Ran full cleanup against all extracted Fusion 360 Gallery Extended STEP data:
  - command: `target\debug\brep-candle-train.exe clean-fusion --raw data\s2.0.1_extended_step --out data\fusion-seg-v1 --exe tools\occt_cleaner\build-openenv-release\occt_cleaner.exe --allow-boundary`
  - duration: about 48 minutes
  - records: 42,912
  - skipped: 0
  - train: 32,184
  - val: 10,728
- Verified output file counts:
  - `data\fusion-seg-v1\graphs`: 42,912 JSON files
  - `data\fusion-seg-v1\labels`: 42,912 JSON files
  - `skipped.jsonl`: 0 lines
- Dataset summary:
  - face labels: `segment_0` 454,428, `segment_1` 116,119,
    `segment_2` 165,023, `segment_3` 24,978, `segment_4` 111,100,
    `segment_5` 23,494, `segment_6` 46,546, `segment_7` 695
  - edge labels: `circle_edge` 558,704, `line_edge` 1,473,964,
    `other_edge` 322,006
- Verified full dataset read:
  - `cargo run -p acad-brep-candle-train -- inspect --data data\fusion-seg-v1`
  - passed with matching counts.
- Ran a real-data training pipeline smoke on `data\fusion-seg-v1-100`:
  - command: `cargo run -p acad-brep-candle-train -- train --data data\fusion-seg-v1-100 --epochs 1 --rounds 1 --hidden 16`
  - train samples: 75
  - val samples: 25
  - final_loss: 1.171474
  - train_accuracy: 45.33%
  - val_accuracy: 52.00%
  - val_macro_f1: 0.3421
- Caveat:
  - This training smoke only validates the current graph-level training
    pipeline on real STEP-derived graph JSON. The meaningful Fusion task is
    face segmentation using `labels/*.json`; Phase G should add a face-label
    head/loss and evaluate face accuracy + macro-F1.

## Session: 2026-07-15

### Phase G: Fusion face segmentation training
- **Status:** complete
- Actions taken:
  - Added `FaceSegmentationConfig`, `FaceSegmentationDataset`, and
    `FaceSegmentationReport` to `acad-brep-candle-train`.
  - Added manifest-sampled dataset loading for `labels/*.json` face labels so
    real Fusion data can be trained without loading all 42,912 graphs by
    default.
  - Added `FaceSegmentationModel` as hybrid encoder + per-face linear head.
  - Added optional inverse-frequency face class weights, clipped at `20.0`, for
    the imbalanced Fusion segment labels.
  - Added mini-batch face segmentation training and per-face accuracy +
    dynamic-class macro-F1 evaluation.
  - Added `face-train` CLI command.
- Verification so far:
  - `cargo fmt --all` passed.
  - `cargo test -p acad-brep-candle-train --lib` passed: 8 tests.
  - `cargo fmt --all -- --check` passed.
  - `cargo test --workspace` passed: 20 unit tests plus doc tests.
  - Real Fusion face-label training run:
    - command: `cargo run -p acad-brep-candle-train -- face-train --data data\fusion-seg-v1 --epochs 3 --rounds 1 --hidden 32 --batch-size 8 --max-train-samples 512 --max-val-samples 128 --save target\fusion-face-seg-smoke.safetensors`
    - train samples: 512 graphs / 10,375 faces
    - val samples: 128 graphs / 3,222 faces
    - face classes: 8
    - class weighting: enabled
    - final_loss: 1.909786
    - train_face_accuracy: 27.33%
    - val_face_accuracy: 22.78%
    - val_face_macro_f1: 0.1730
    - checkpoint: `target\fusion-face-seg-smoke.safetensors` (101,316 bytes)
- Caveat:
  - This is a real per-face supervised Fusion training run, but still a small
    sampled smoke. Metrics are expected to be weak; the value is that the target,
    batching, loss, weighting, metrics, and checkpoint path are now real.

### Phase H: Face segmentation sampling/training quality
- **Status:** complete
- Actions taken:
  - Added `FaceSamplingStrategy::{Uniform, FaceBalanced}`.
  - Added `--sample-strategy` and `--val-sample-strategy` to `face-train`.
  - Added deterministic per-epoch training-order shuffle, enabled by default,
    with `--no-shuffle` as an escape hatch.
  - Added train/val sampled face-label count reporting.
  - Added `--class-weights`; class weights are now opt-in because the short
    real-data smoke performed better without them.
  - Kept face-balanced sampling available for coverage diagnostics, but changed
    defaults to uniform train + uniform val after comparison.
- Verification:
  - `cargo fmt --all -- --check` passed.
  - `cargo test --workspace` passed: 20 unit tests plus doc tests.
  - Default real Fusion smoke:
    - command: `cargo run -p acad-brep-candle-train -- face-train --data data\fusion-seg-v1 --epochs 3 --rounds 1 --hidden 32 --batch-size 8 --max-train-samples 512 --max-val-samples 128 --save target\fusion-face-seg-default-smoke.safetensors`
    - train samples: 512 graphs / 10,375 faces
    - val samples: 128 graphs / 3,222 faces
    - class weighting: disabled
    - train sampling: uniform
    - val sampling: uniform
    - train label counts: `segment_0:4899`, `segment_1:1218`,
      `segment_2:1690`, `segment_3:242`, `segment_4:1429`,
      `segment_5:193`, `segment_6:696`, `segment_7:8`
    - val label counts: `segment_0:1611`, `segment_1:351`,
      `segment_2:565`, `segment_3:110`, `segment_4:386`,
      `segment_5:48`, `segment_6:151`, `segment_7:0`
    - final_loss: 1.496397
    - train_face_accuracy: 45.21%
    - val_face_accuracy: 50.87%
    - val_face_macro_f1: 0.1832
    - checkpoint: `target\fusion-face-seg-default-smoke.safetensors`
- Comparison:
  - uniform + class weights:
    - val_face_accuracy: 22.16%
    - val_face_macro_f1: 0.1600
    - checkpoint: `target\fusion-face-seg-uniform-v2.safetensors`
  - uniform + no class weights:
    - val_face_accuracy: 44.85%
    - val_face_macro_f1: 0.1937
    - checkpoint: `target\fusion-face-seg-uniform-unweighted-smoke.safetensors`
  - face-balanced train + uniform val:
    - train label counts included `segment_7:496`, much better rare-label
      coverage than uniform's `segment_7:8`
    - val_face_accuracy: 5.40%
    - val_face_macro_f1: 0.0632
    - checkpoint: `target\fusion-face-seg-balanced-train-smoke.safetensors`
- Conclusion:
  - Naive face-balanced graph sampling improves rare-label coverage but creates
    a strong train/val distribution shift at this small graph budget. It should
    remain a diagnostic option until a better face-budgeted sampler is added.

### Documentation: Fusion face segmentation example report
- **Status:** complete
- Actions taken:
  - Added `docs/fusion_face_segmentation_example_report.md`.
  - Linked the report from `README.md`.
  - The report documents the real Fusion face segmentation smoke command,
    dataset slice, sampled label coverage, metrics, comparison runs, current
    interpretation, and next work.
- Verification:
  - Documentation-only change; Rust tests were not rerun.

### Phase 1: Requirements & Discovery
- **Status:** complete
- **Started:** 2026-07-15
- Actions taken:
  - Read the planning-with-files skill instructions.
  - Inspected repository root.
  - Confirmed the repository has no source files yet.
  - Attempted `git status --short`; it failed due Git safe.directory ownership protection.
- Files created/modified:
  - `task_plan.md` created.
  - `findings.md` created.
  - `progress.md` created.

### Phase 2: Workspace Structure
- **Status:** complete
- Actions taken:
  - Created root Cargo workspace.
  - Added `acad-brep-graph`.
  - Added `acad-brep-encoder`.
  - Added `acad-cli`.
  - Added `acad-brep-candle-train` after user requested Candle training.
- Files created/modified:
  - `Cargo.toml`
  - `.gitignore`
  - `README.md`
  - `crates/brep-graph/Cargo.toml`
  - `crates/brep-graph/src/lib.rs`
  - `crates/brep-encoder/Cargo.toml`
  - `crates/brep-encoder/src/lib.rs`
  - `crates/acad-cli/Cargo.toml`
  - `crates/acad-cli/src/main.rs`
  - `crates/brep-candle-train/Cargo.toml`
  - `crates/brep-candle-train/src/lib.rs`
  - `crates/brep-candle-train/src/main.rs`

### Phase 3: BRep Graph Encoder Plan
- **Status:** complete
- Actions taken:
  - Created detailed BRep Graph Encoder plan with Candle baseline, message-passing roadmap, data strategy, evaluation metrics, and agent integration notes.
- Files created/modified:
  - `docs/brep_graph_encoder_plan.md`

### Phase 4: Testing & Verification
- **Status:** complete
- Actions taken:
  - Ran `cargo fmt --all`.
  - Ran `cargo test --workspace` after downloading Candle dependencies.
  - Fixed an unused import warning in the Candle training crate.
  - Ran a short Candle training smoke test and saved `target\minimal-brep-classifier.safetensors`.
  - Verified formatting with `cargo fmt --all -- --check`.
  - Verified the checkpoint path exists.
- Files created/modified:
  - `Cargo.lock` generated by Cargo.
  - `target\minimal-brep-classifier.safetensors` generated by training smoke test.

### Phase 5: Delivery
- **Status:** complete
- Actions taken:
  - Added concrete MVP documentation for the minimal dataset and Candle training pipeline.
  - Checked generated file list.
- Files created/modified:
  - `docs/minimal_dataset_training_pipeline.md`
  - `README.md`
  - `docs/brep_graph_encoder_plan.md`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Repository listing | `Get-ChildItem -Force` | Empty source tree with `.git` | Only `.git` present | pass |
| Git status | `git status --short` | Working tree status | Blocked by safe.directory ownership check | noted |
| Format | `cargo fmt --all -- --check` | No diff | Passed | pass |
| Workspace tests | `cargo test --workspace` with explicit VS2022/SDK env | Tests pass | 6 unit tests and doc tests passed | pass |
| Training smoke | `cargo run -p acad-brep-candle-train -- --epochs 20 --samples-per-class 8 --save target\minimal-brep-classifier.safetensors` | Training runs and saves checkpoint | final_loss `0.240214`, train_accuracy `100.00%`, checkpoint exists | pass |
| File list | `rg --files` | Generated workspace files listed | Listed root, docs, and crate files | pass |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-07-15 | `git status --short` failed with dubious ownership | 1 | Did not alter global config; proceed with scaffold. |
| 2026-07-15 | `cargo test` failed to download `candle-core` because crates.io was blocked | 1 | Re-ran with approved network escalation. |
| 2026-07-15 | MSVC linker could not open `msvcrt.lib` | 1 | Set explicit VS2022/Windows SDK `LIB` paths for cargo verification. |
| 2026-07-15 | `onig_sys` could not find `vcruntime.h` | 1 | Set explicit VS2022 `PATH` and `INCLUDE` paths for cargo verification. |
| 2026-07-15 | `git -c safe.directory=... status --short` warned about unreadable user git ignore | 1 | Noted only; command still listed generated files. |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 5 complete |
| Where am I going? | Deliver summary to user. |
| What's the goal? | Create a Rust Cargo workspace for agentic CAD work, include Candle-based training, and document a practical BRep Graph Encoder implementation plan. |
| What have I learned? | Candle training should be isolated from core graph crates; git safe.directory blocks status. |
| What have I done? | Created workspace crates, Candle training code, encoder docs, ran tests, and ran a training smoke test. |
