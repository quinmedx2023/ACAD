# Findings & Decisions

## Requirements
- Generate a Rust Cargo monorepo in `C:\repositories\ACAD`.
- Generate a BRep Graph Encoder plan.
- Align the plan with the prior direction: tools + LLM, BRep graph understanding, Truck/OCCT/Candle later.
- User updated requirement: use Candle for training now.

## Research Findings
- The repository currently contains only `.git`; there is no existing source tree to preserve.
- `git status --short` is blocked by Git safe.directory ownership checks in this sandbox user.
- The first scaffold should avoid external dependencies so it can be checked without registry/network access.
- Candle training is best isolated in a dedicated crate because it needs external dependencies while the graph schema and deterministic encoder can remain dependency-light.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Create `acad-brep-graph` | Holds the canonical graph data model and validation primitives. |
| Create `acad-brep-encoder` | Converts BRep graph data into deterministic tensor-like arrays for model ingestion. |
| Create `acad-cli` | Gives the workspace a small executable smoke test. |
| Put the detailed roadmap in `docs/brep_graph_encoder_plan.md` | Keeps project planning separate from task progress logs. |
| Create `acad-brep-candle-train` | Provides a real Candle training loop over synthetic graph features. |
| Use pooled graph features for the first Candle model | Good smoke test for the data/training path; real message passing comes next. |
| Default Fusion face smoke to uniform sampling without class weights | On the current 512/128 real Fusion face-label smoke, uniform/unweighted outperformed class weights and naive face-balanced graph sampling. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Git ownership protection prevents status checks | Continue without changing global config; mention in delivery. |
| Naive face-balanced sampling hurt short-run validation metrics | Keep it as an opt-in coverage diagnostic; default to uniform train/val sampling until a face-budgeted sampler is implemented. |

## Resources
- `task_plan.md`: task phase tracking.
- `findings.md`: persistent discoveries and decisions.
- `progress.md`: chronological work and test log.
- Candle project docs: https://github.com/huggingface/candle
- Candle API docs: https://docs.rs/candle-core/latest/candle_core/
- Candle NN API docs: https://docs.rs/candle-nn/latest/candle_nn/

## Visual/Browser Findings
- No visual/browser artifacts used for this task.
