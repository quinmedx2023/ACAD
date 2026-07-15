# BRep Dataset V1

## What Exists

`data/synthetic-v1` is a real on-disk BRep dataset. It is generated from the
current Rust `BrepGraph` fixtures, not imported from industrial STEP files yet.
Each record stores exact graph topology, sampled face/edge geometry, graph
labels, face labels, and edge labels.

Layout:

```text
data/synthetic-v1/
  dataset.json
  manifest.jsonl
  graphs/*.json
  labels/*.json
```

Current corpus:

| Field | Value |
|-------|-------|
| Records | 36 |
| Classes | `box`, `cylinder`, `plate_with_holes` |
| Train / val | 27 / 9 |
| Files | 74 |
| Face labels | `bottom`, `bottom_cap`, `cylinder_side`, `hole_wall`, `outer_side`, `side`, `top`, `top_cap` |
| Edge labels | `cap_edge`, `convex_line`, `hole_edge` |

## Format

`dataset.json` records dataset-level metadata.

`manifest.jsonl` has one JSON object per sample. New writers include
per-record label histograms so samplers and diagnostics do not need to open
every `labels/*.json` file:

```json
{"id":"box_000000","split":"train","class_id":0,"class_name":"box","graph_path":"graphs/box_000000.json","labels_path":"labels/box_000000.json","stats":{"faces":6,"edges":12,"coedges":24,"face_adjacencies":12},"face_label_counts":{"bottom":1,"side":4,"top":1},"edge_label_counts":{"convex_line":12}}
```

The `face_label_counts` and `edge_label_counts` fields are optional for backward
compatibility. Readers fall back to loading the label JSON when they are absent.

Each graph JSON is a serialized `BrepGraph`:

- faces with `SurfaceKind`, area, centroid, normal, and optional UV sampled `SurfaceGeometry`;
- edges with `CurveKind`, length, midpoint, convexity, and optional sampled `CurveGeometry`;
- coedges with face/edge ownership and mate links;
- face adjacency triples.

Each label JSON stores graph, face, and edge labels.

Supported manifest splits are `train`, `val`, and `test`. Synthetic datasets
still generate `train`/`val`; Fusion cleanup preserves the official
`train_test.json` names, so official Fusion test rows remain `test`.

## Commands

Generate:

```powershell
cargo run -p acad-brep-candle-train -- dataset --out data/synthetic-v1 --samples-per-class 12 --val-fraction 0.25
```

Inspect:

```powershell
cargo run -p acad-brep-candle-train -- inspect --data data/synthetic-v1
```

Train from disk:

```powershell
cargo run -p acad-brep-candle-train -- train --data data/synthetic-v1 --epochs 30 --rounds 2 --hidden 48 --save target/dataset-v1-hybrid-brep-encoder.safetensors
```

Latest local training result:

```text
epochs: 30
train_samples: 27
val_samples: 9
final_loss: 0.061632
train_accuracy: 100.00%
val_accuracy: 100.00%
val_macro_f1: 1.0000
```

## Next Dataset Work

1. Add harder generated classes: slots, bosses, counterbores, stepped blocks, blind holes.
2. Add face/edge supervised heads in the Candle model and train against these labels.
3. Add a Truck/OCCT importer that writes the same dataset format from STEP/BRep files.
