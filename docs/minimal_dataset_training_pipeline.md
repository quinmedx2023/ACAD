# Hybrid BRep Dataset and Candle Training Pipeline

## Scope

This is the working training loop for the hybrid BRep Graph Encoder. It replaces
the earlier pooled-MLP smoke test with a real geometry-aware, topology-aware
model implemented natively in Candle. It still runs on small synthetic data so
the whole path (data → tensors → conv front-ends → message passing → metrics →
checkpoint) stays debuggable, but every stage is now the real architecture.

## Dataset

Generated in `acad_brep_candle_train::synthetic_graphs` and tensorized by
`acad_brep_encoder::GraphTensorizer`.

| Label | Class |
|-------|-------|
| `0` | box |
| `1` | cylinder |
| `2` | plate with holes (1–3 holes) |

Variation per class:

- Parameter sweeps over dimensions (size), and
- **structural** variation for the plate class: the number of through-holes
  varies (1, 2, or 3), changing face/edge/coedge counts and topology — so the
  class is not a single fixed shape.

Each graph carries sampled geometry:

```text
face -> SurfaceGeometry { UV grid of points, normals, trimming mask }
edge -> CurveGeometry   { 1D grid of points, tangents }
```

Pipeline:

```text
synthetic BRep fixture (+ sampled geometry)
  -> BrepGraph  (topology + geometry, serde-serializable)
  -> GraphTensorizer
       -> per-face UV grids      [F, 7, uv, uv]
       -> per-edge curve grids   [E, 6, res]
       -> categorical/scalar node features
       -> coedge topology arrays (coedge->face, ->edge, ->mate)
  -> BrepBatch (ragged batching: concat nodes, offset coedge indices)
  -> HybridBrepEncoder (Candle)
```

Default grid resolution: `uv_res = curve_res = 6`.

## Model

`HybridBrepEncoder`:

```text
face UV grid  -> conv2d(7->16->32) -> global avg pool -> face geom (32)
edge curve    -> conv1d(6->16->32) -> global avg pool -> edge geom (32)

face_in = [surface one-hot(7) | scalars(7) | face geom(32)] -> Linear -> h
edge_in = [curve/convexity one-hot(9) | scalars(4) | edge geom(32)] -> Linear -> h
coedge  = [face_h | edge_h] -> Linear -> h

repeat `rounds` times:
  coedge <- LN(coedge + MLP([coedge | face | edge | mate]))   # BRepNet-style
  face   <- LN(face   + MLP([face | mean(incident coedges)]))
  edge   <- LN(edge   + MLP([edge | mean(incident coedges)]))

graph_emb = [mean_pool(face by graph) | mean_pool(edge by graph)]
logits    = MLP(graph_emb) -> 3 classes
```

Aggregation is scatter-mean built from `Tensor::index_add`; ragged graphs are
batched by concatenating nodes and offsetting the coedge index arrays. Per-face
and per-edge embeddings are returned alongside the graph logits for downstream
face/edge selection and repair-hint heads.

## Training Command

```powershell
cargo run -p acad-brep-candle-train -- `
  --epochs 150 --samples-per-class 24 --rounds 2 --hidden 48 `
  --save target/hybrid-brep-encoder.safetensors
```

Flags: `--epochs --lr --hidden --samples-per-class --rounds --seed
--val-fraction --save`.

Output fields: `epochs, train_samples, val_samples, hidden_dim, rounds,
final_loss, train_accuracy, val_accuracy, val_macro_f1`.

Local run (54 train / 18 held-out val):

```text
final_loss:     ~0.005
train_accuracy: 100.00%
val_accuracy:   100.00%
val_macro_f1:   1.0000
```

## Honest read on the result

The three classes remain easy to separate, so 100% val accuracy is **not**
evidence that the encoder captures subtle geometry — it is evidence that the
full hybrid path (conv geometry channels + coedge message passing + ragged
batching + held-out evaluation + checkpointing) is wired correctly and trains.
Real difficulty comes from harder labels (feature/face-level recognition) and
real kernel topology; see the plan's data and evaluation stages.

## Known simplifications

- Synthetic curve geometry uses a canonical parameterization (lines along local
  X, circles in local XY) because the fixtures do not carry real vertices.
- Coedge message passing uses `mate` + face/edge membership; ordered loop walks
  (`next`/`previous`) are future work once fixtures carry ordered loops.
- `set_seed` is a no-op on Candle's CPU backend (0.11), so CPU init is not bit-
  reproducible; the flag still applies on CUDA.

## Next upgrades

1. Add ordered coedge loops (`next`/`previous`) for full BRepNet walks.
2. Add face-level / edge-level supervised heads (the embeddings already exist).
3. Import real STEP/OCCT graphs to replace synthetic geometry.
4. Add minibatching + a proper train/val/test protocol as datasets grow.
