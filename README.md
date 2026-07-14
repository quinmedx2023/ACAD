# ACAD

Rust workspace for agentic CAD experiments, centered on a **hybrid BRep graph
encoder** trained natively in [Candle](https://github.com/huggingface/candle).

The encoder combines the two proven CAD-ML architectures:

- **UV-Net-style geometry**: each face carries a sampled UV grid of points +
  normals + a trimming mask; each edge carries a sampled 1D curve grid of points
  + tangents. Small conv front-ends turn this real geometry into embeddings.
- **BRepNet-style topology**: coedges are the message-passing primitive.
  Messages flow between each coedge and its incident face, edge, and mate coedge,
  with per-node residual + layer norm over several rounds.

The core graph and encoder crates stay free of any ML runtime; Candle is
isolated in one crate so the schema and tensorization remain usable (and fast to
compile) on their own. The graph model is `serde`-serializable for interop.

## Workspace

- `crates/brep-graph`: BRep graph data model (faces, edges, coedges with mates,
  adjacency), sampled surface/curve geometry, validation, JSON export, and
  parametric synthetic fixtures (box, cylinder, multi-hole plate).
- `crates/brep-encoder`: two ML-runtime-free encoders — a deterministic pooled
  baseline (`DeterministicGraphEncoder`) and the geometry-aware `GraphTensorizer`
  that emits ragged UV/curve grids + coedge topology index arrays.
- `crates/brep-candle-train`: the hybrid Candle encoder, ragged graph batching,
  training loop with a train/val split, accuracy + macro-F1 metrics, and
  checkpoint save/load for deployment inference.
- `crates/acad-cli`: small CLI smoke test for the graph and encoder crates.
- `docs/brep_graph_encoder_plan.md`: implementation plan / roadmap.
- `docs/minimal_dataset_training_pipeline.md`: the hybrid dataset/training notes.

## Commands

On Windows, run these from a Visual Studio developer shell, or make sure a
**complete** MSVC + Windows SDK `LIB`/`INCLUDE` (and a C compiler for the
transitive `onig` build) are on the environment. A C dependency requires the
full VC++ headers (`vcruntime.h`), not a headers-only preview toolchain.

```powershell
cargo fmt
cargo test --workspace
cargo run -p acad-cli
# Hybrid encoder: UV-grid geometry + coedge message passing, with held-out val.
cargo run -p acad-brep-candle-train -- `
  --epochs 150 --samples-per-class 24 --rounds 2 --hidden 48 `
  --save target/hybrid-brep-encoder.safetensors
```

Latest local run (54 train / 18 held-out val samples): `final_loss ~0.005`,
`train_accuracy 100%`, `val_accuracy 100%`, `val_macro_f1 1.0`. The three
synthetic classes are still easy — see the plan doc for the honest read on what
this does and does not demonstrate.
