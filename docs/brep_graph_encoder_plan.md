# BRep Graph Encoder Plan

## Objective

Build a trainable BRep Graph Encoder for agentic CAD workflows. The encoder should turn exact CAD topology and geometry into embeddings that help tools and LLM agents recognize features, select faces/edges, diagnose kernel failures, and propose repair actions.

## Current Workspace Mapping

| Crate | Role |
|-------|------|
| `acad-brep-graph` | Kernel-neutral BRep graph schema and synthetic fixtures. |
| `acad-brep-encoder` | Deterministic tensorization and graph-level feature boundary. |
| `acad-brep-candle-train` | Candle training loop for the first synthetic baseline. |
| `acad-cli` | Smoke-test executable for the non-ML graph/encoder path. |

## Stage 0: Deterministic Baseline

Status: scaffolded.

- Represent BRep entities as `Face`, `Edge`, `Coedge`, and `FaceAdjacency`.
- Encode face features: surface kind, area, centroid, normal.
- Encode edge features: curve kind, convexity, length, midpoint.
- Pool graph features into a fixed vector for the first Candle classifier.
- Keep this layer independent of Truck, OCCT, and Candle.

Success criteria:

- Synthetic box, cylinder, and plate-with-hole graphs validate.
- Encoded graph features have stable dimensions.
- `acad-cli` can run without ML dependencies.

## Stage 1: Pooled Candle Baseline

Status: superseded (kept as a lightweight reference).

The original baseline pooled deterministic features into a fixed vector and
trained a small Candle MLP. It served its purpose — verifying Candle training,
backprop, optimizer, and safetensors export — but discarded topology and real
geometry, so 100% accuracy on it mostly meant "learned to count faces/edges".
The `DeterministicGraphEncoder` + `pooled_graph_features` path remains available
for cheap experiments, but the trained model is now the hybrid encoder below.

## Stage 2: Hybrid Geometry + Topology Encoder

Status: **implemented** in `acad-brep-candle-train` (Candle-native).

The encoder combines UV-Net geometry sampling with BRepNet coedge message
passing:

```text
face UV grid  -> conv2d front-end -\
face categorical/scalar ------------> face_in -\
edge curve grid -> conv1d front-end \            \
edge categorical/scalar -------------> edge_in --> coedge init
                                                    |
   T rounds of coedge <-> face / edge / mate message passing (index_add)
                                                    |
   mean-pool face + edge embeddings per graph -> classifier head
```

Implemented pieces:

- **Geometry channels**: `SurfaceGeometry` (UV grid of points/normals/mask) and
  `CurveGeometry` (1D grid of points/tangents) live on the graph; conv2d/conv1d
  front-ends turn them into per-node geometry embeddings.
- **Coedge message passing**: coedges are the primitive. Each round updates the
  coedge from its incident face, edge, and mate, then scatter-mean-aggregates
  coedges back into faces and edges, all with residual + layer norm.
- **Ragged batching**: `BrepBatch` concatenates nodes across graphs and offsets
  the coedge index arrays; a single forward pass covers the minibatch.
- **Scatter-add aggregation**: built on `Tensor::index_add` (Candle's native op).
- **Readouts**: graph embedding for classification, plus per-face and per-edge
  embeddings returned for downstream selection / repair heads.

Evaluation methodology now includes a held-out validation split, accuracy, and
macro-F1, and checkpoints save/load for deployment inference (`load_encoder`).

Remaining Stage 2 work:

1. Ordered coedge loops (`next`/`previous`) for full BRepNet topological walks.
2. Contrastive/self-supervised objectives on real STEP/OCCT imports.
3. Face-level and edge-level supervised heads on the existing embeddings.

## Stage 3: Data Pipeline

Synthetic sources:

- Programmatic primitives from `acad-brep-graph`.
- Later: Truck-generated feature histories.
- Later: OCCT-generated STEP/BRep fixtures.

Real sources:

- STEP imports through OCCT sidecar.
- ABC/Fusion-style datasets after license review.
- Internal tool-call traces from successful agent sessions.

Labels:

- Graph-level: part family, manufacturing process, failure class.
- Face-level: planar pocket, boss, hole wall, fillet face, datum face.
- Edge-level: convex/concave/smooth, fillet candidate, chamfer candidate.
- Action-level: suggested repair operation after kernel failure.

## Stage 4: Evaluation

Track metrics beyond loss:

- Graph classification accuracy.
- Face/edge selection F1.
- Feature recognition F1.
- Kernel repair success rate.
- Valid solid rate after agent repair.
- STEP export success rate.

The important production metric is not model accuracy alone. The key metric is whether the LLM/tool loop produces a valid BRep with fewer retries.

## Stage 5: Agent Integration

Use the encoder as a tool, not as the source of geometry truth:

```text
LLM planner
  -> CAD tool calls
  -> Truck/OCCT execution
  -> BRepGraph extraction
  -> Candle encoder inference
  -> validator + repair hints
  -> revised tool calls
```

Tool outputs should include:

- candidate faces/edges by score;
- graph embedding for retrieval;
- predicted feature labels;
- repair hint class and confidence.

## Risks

| Risk | Mitigation |
|------|------------|
| Synthetic graphs do not match real kernel topology | Introduce Truck/OCCT importers early and compare distributions. |
| Pooled baseline overfits simple counts | Treat it only as a Candle smoke test; move to message passing in Stage 2. |
| Direct BRep generation is invalid | Keep generation as FeatureGraph/tool calls; use BRep encoder for understanding and repair. |
| Candle dependency/API churn | Pin Candle versions in `Cargo.toml`; keep deterministic encoder independent. |

## Near-Term Tasks

1. ~~Verify Candle dependencies build locally.~~ done
2. ~~Add safetensors save/load smoke test.~~ done (`load_encoder` + round-trip test)
3. ~~Add a real `GraphBatch` type with node offsets.~~ done (`BrepBatch`)
4. ~~Add first message-passing layer in Candle.~~ done (coedge hetero message passing)
5. ~~Add UV-grid geometry sampling to the schema.~~ done (`SurfaceGeometry`/`CurveGeometry`)
6. Add importer boundary traits for Truck/OCCT extraction.
7. Add ordered coedge loops for full BRepNet `next`/`previous` walks.
8. Add face-level / edge-level supervised heads on the existing embeddings.
