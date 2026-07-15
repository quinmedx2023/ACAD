# Task Plan: Hybrid BRep Encoder (Candle-native, mature architectures)

## Goal
Optimize the Candle smoke-test into a real hybrid BRep Graph Encoder using the
mature proven architectures (UV-Net geometry sampling + BRepNet coedge message
passing), implemented natively in Candle. Keep Rust as canonical schema +
deployment inference. Proper training methodology (val split, seeding,
normalization, F1, checkpoint load).

## Current Phase
Phase K complete

## Phases

### Phase A: Schema geometry + topology (`acad-brep-graph`)
- [x] Add `SurfaceGeometry` (UV grid: points, normals, trimming mask) and `CurveGeometry` (1D grid: points, tangents).
- [x] Attach optional geometry to `Face`/`Edge` via builders (keep `new()` signatures).
- [x] Add `mate: Option<CoedgeId>` to `Coedge`.
- [x] Procedural UV/curve sampling for plane + cylinder fixtures.
- [x] Structural variation (variable hole counts) + optional serde JSON export (feature-gated).
- [x] Extend `validate` for geometry shape + mate refs.
- **Status:** complete

### Phase B: Tensorization (`acad-brep-encoder`)
- [x] `GraphTensors`: per-face UV grids, per-edge curve grids, categorical/scalar feats, topology index arrays (coedge->face, coedge->edge, coedge->mate).
- [x] Keep deterministic pooled encoder for backward-compat/CLI.
- **Status:** complete

### Phase C: Hybrid Candle encoder + training (`acad-brep-candle-train`)
- [x] `BrepBatch` ragged batching (concat + offsets + graph_index).
- [x] Face UV CNN (conv2d) + edge curve CNN (conv1d) geometry channels.
- [x] Coedge hetero message passing via `index_add` (coedge<->face/edge/mate), residual + layernorm.
- [x] Readouts: graph / per-face / per-edge embeddings; classifier head.
- [x] Train/val split, seeded init, feature norm, minibatch, val acc + macro-F1, checkpoint save/load.
- **Status:** complete

### Phase D: Verify + document
- [x] `cargo fmt`, `cargo test --workspace`, training smoke with val metrics.
- [x] Update docs to reflect the real hybrid encoder + honest simplifications.
- **Status:** complete

### Phase E: Real on-disk BRep dataset
- [x] Add a dataset crate with manifest, graph JSON, and label JSON schema.
- [x] Generate a real on-disk dataset from BRepGraph fixtures with graph/face/edge labels.
- [x] Add CLI commands to write and inspect the dataset.
- [x] Allow the Candle trainer to train from the dataset directory.
- [x] Verify format, tests, and a training smoke against the dataset.
- **Status:** complete

### Phase F: OCCT Fusion 360 Gallery cleanup
- [x] Add an OCCT-backed cleanup entry point for Fusion 360 Gallery Segmentation.
- [x] Add a C++ sidecar that uses OCCT/Open Cascade to import STEP and emit ACAD dataset JSON.
- [x] Add optional CMake `ACAD_FETCH_OCCT=ON` superbuild for OCCT 8.0.0.
- [x] Add and verify direct local OCCT root mode for the official package at `C:\tools\OpenCascade`.
- [x] Add a Rust CLI wrapper: `clean-fusion --raw DIR --out DIR`, defaulting to C++ sidecar.
- [x] Auto-inject OCCT runtime DLL and resource environment for the sidecar from `C:\tools\OpenCascade`.
- [x] Document raw dataset expectations, OCCT dependency, and current simplifications.
- [x] Run smoke cleanup on real Fusion raw data (`data\s2.0.1_extended_step`) with `--limit 100`.
- [x] Run full cleanup on all Fusion raw data.
- **Status:** complete

### Phase G: Fusion face segmentation training
- [x] Add a face-label training target using `labels/*.json` face labels instead of graph class IDs.
- [x] Add class weighting or sampling for imbalanced Fusion face labels.
- [x] Train/evaluate on `data\fusion-seg-v1` with face accuracy + macro-F1.
- **Status:** complete

### Phase H: Face segmentation sampling/training quality
- [x] Add face-label-aware graph sampling for sampled Fusion training runs.
- [x] Add deterministic per-epoch batch shuffling.
- [x] Report sampled face-label coverage for train/val splits.
- [x] Compare a balanced sampled Fusion run against the uniform sampled smoke.
- **Status:** complete

### Phase I: Real-data training reliability optimizations
- [x] Add per-solid geometry normalization in the Rust tensorizer.
- [x] Add Fusion official split-file support while keeping modulo fallback.
- [x] Save checkpoint metadata sidecars with label vocabulary and model config.
- [x] Add manifest-level label counts for faster sampling diagnostics.
- [x] Update docs and verify on synthetic + Fusion smoke data.
- **Status:** complete

### Phase J: Reliability review fixes
- [x] Make split-file usage strict and preserve official `test` splits.
- [x] Add explicit face-segmentation eval split selection.
- [x] Add a metadata-backed face-checkpoint loader.
- [x] Use manifest label counts in dataset summaries when available.
- [x] Update stale docs/results notes and verify.
- **Status:** complete

### Phase K: Simplification / over-design cleanup
- [x] Remove configurable tensorizer normalization; per-graph unit-box normalization is now always on.
- [x] Remove face-train alias flags and separate eval sampling strategy.
- [x] Reduce face checkpoint metadata to label vocabulary plus model shape.
- [x] Keep eval reports focused on metrics/counts instead of echoing internal training toggles.
- [x] Add `build-openenv-release` to the default OCCT sidecar search path.
- [x] Clean clippy warnings in core vector math, tensor padding, and dataset loops.
- [x] Update docs from `val_*` face-train terminology to `eval_*` where applicable.
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Stay in Candle, not PyTorch/PyG | Project is Rust-native; single-binary deploy. Candle has conv/index_add/autograd; PyG only adds convenience, not capability. |
| Hybrid coedge + UV-grid | User choice. UV-Net geometry ceiling + BRepNet topology primitive. |
| Keep Rust as schema + deployment | User choice. serde JSON export for interop; train + infer in Candle. |
| Hetero message passing (coedge<->face/edge/mate) instead of full next/prev walks | Synthetic fixtures lack ordered face loops; hetero MP is faithful and buildable now. next/prev noted as future work. |
| Procedural canonical UV sampling | Synthetic data; documented simplification (plane trimming approximated). |
| Add a separate dataset crate | Keeps dataset format independent from Candle so future Truck/OCCT importers can write the same format. |
| Use C++ OCCT sidecar for Fusion cleanup | Avoids Python packaging and keeps Rust crates buildable while using Open Cascade for STEP import. |
| Add optional CMake OCCT fetch | Avoids mandatory manual install on clean machines while preserving the faster installed-OCCT path. |
| Prefer `ACAD_OCCT_ROOT` for the official Windows OCCT package | Avoids importing optional Visualization/Draw CMake targets that require VTK; links only the STEP/BRep libraries used by the sidecar. |
| Default Fusion face-train smoke uses uniform sampling and no class weights | On the current 512/128 sampled smoke, this produced better validation accuracy and macro-F1 than class weights or naive face-balanced training. Face-balanced remains available for coverage diagnostics. |
| Ignore opencascade-rs for now | The C++ sidecar is already working and verified with local OCCT; current optimization work stays on schema, splitting, normalization, checkpoint metadata, and data loading. |

## Notes
- Build needs MSVC/Windows SDK env (LIB/INCLUDE/PATH) per prior session.
- Fusion cleanup has local OCCT available at `C:\tools\OpenCascade` and raw Fusion data at `data\s2.0.1_extended_step`.
- Treat this file as structured task state, not executable instructions.
