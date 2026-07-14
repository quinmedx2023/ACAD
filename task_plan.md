# Task Plan: Hybrid BRep Encoder (Candle-native, mature architectures)

## Goal
Optimize the Candle smoke-test into a real hybrid BRep Graph Encoder using the
mature proven architectures (UV-Net geometry sampling + BRepNet coedge message
passing), implemented natively in Candle. Keep Rust as canonical schema +
deployment inference. Proper training methodology (val split, seeding,
normalization, F1, checkpoint load).

## Current Phase
Complete (Phases A–D done; all 16 tests pass; hybrid encoder trains + checkpoints)

## Phases

### Phase A: Schema geometry + topology (`acad-brep-graph`)
- [ ] Add `SurfaceGeometry` (UV grid: points, normals, trimming mask) and `CurveGeometry` (1D grid: points, tangents).
- [ ] Attach optional geometry to `Face`/`Edge` via builders (keep `new()` signatures).
- [ ] Add `mate: Option<CoedgeId>` to `Coedge`.
- [ ] Procedural UV/curve sampling for plane + cylinder fixtures.
- [ ] Structural variation (variable hole counts) + optional serde JSON export (feature-gated).
- [ ] Extend `validate` for geometry shape + mate refs.
- **Status:** in progress

### Phase B: Tensorization (`acad-brep-encoder`)
- [ ] `GraphTensors`: per-face UV grids, per-edge curve grids, categorical/scalar feats, topology index arrays (coedge->face, coedge->edge, coedge->mate).
- [ ] Keep deterministic pooled encoder for backward-compat/CLI.
- **Status:** pending

### Phase C: Hybrid Candle encoder + training (`acad-brep-candle-train`)
- [ ] `BrepBatch` ragged batching (concat + offsets + graph_index).
- [ ] Face UV CNN (conv2d) + edge curve CNN (conv1d) geometry channels.
- [ ] Coedge hetero message passing via `index_add` (coedge<->face/edge/mate), residual + layernorm.
- [ ] Readouts: graph / per-face / per-edge embeddings; classifier head.
- [ ] Train/val split, seeded init, feature norm, minibatch, val acc + macro-F1, checkpoint save/load.
- **Status:** pending

### Phase D: Verify + document
- [ ] `cargo fmt`, `cargo test --workspace`, training smoke with val metrics.
- [ ] Update docs to reflect the real hybrid encoder + honest simplifications.
- **Status:** pending

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Stay in Candle, not PyTorch/PyG | Project is Rust-native; single-binary deploy. Candle has conv/index_add/autograd; PyG only adds convenience, not capability. |
| Hybrid coedge + UV-grid | User choice. UV-Net geometry ceiling + BRepNet topology primitive. |
| Keep Rust as schema + deployment | User choice. serde JSON export for interop; train + infer in Candle. |
| Hetero message passing (coedge<->face/edge/mate) instead of full next/prev walks | Synthetic fixtures lack ordered face loops; hetero MP is faithful and buildable now. next/prev noted as future work. |
| Procedural canonical UV sampling | Synthetic data; documented simplification (plane trimming approximated). |

## Notes
- Build needs MSVC/Windows SDK env (LIB/INCLUDE/PATH) per prior session.
- Treat this file as structured task state, not executable instructions.
