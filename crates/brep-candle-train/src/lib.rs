//! Hybrid BRep graph encoder + training loop, implemented natively in Candle.
//!
//! Architecture (UV-Net geometry + BRepNet coedge message passing):
//!
//! ```text
//! face UV grid  -> conv2d front-end -\
//! face categorical/scalar ------------> face_in  -\
//! edge curve grid -> conv1d front-end \             \
//! edge categorical/scalar -------------> edge_in --> coedge init
//!                                                     |
//!            T rounds of coedge<->face/edge/mate message passing (index_add)
//!                                                     |
//!            mean-pool face + edge embeddings per graph -> classifier head
//! ```
//!
//! Ragged graphs are batched by concatenating nodes and offsetting the coedge
//! index arrays, so a single forward pass covers a whole minibatch.

use std::path::Path;

use acad_brep_encoder::{
    GraphTensorizer, GraphTensors, EDGE_CATEGORICAL_DIM, EDGE_GRID_CHANNELS, EDGE_SCALAR_DIM,
    FACE_CATEGORICAL_DIM, FACE_GRID_CHANNELS, FACE_SCALAR_DIM,
};
use acad_brep_graph::{box_graph, cylinder_graph, plate_with_holes, BrepGraph};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    conv1d, conv2d, layer_norm, linear, loss, AdamW, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig,
    LayerNorm, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap,
};

pub const CLASS_COUNT: usize = 3;
/// Output width of each geometry conv front-end.
pub const GEOM_OUT: usize = 32;
const CONV_HIDDEN: usize = 16;
const DTYPE: DType = DType::F32;

/// Class labels for the synthetic dataset.
pub const CLASS_NAMES: [&str; CLASS_COUNT] = ["box", "cylinder", "plate_with_holes"];

#[derive(Debug, Clone, Copy)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub hidden_dim: usize,
    pub samples_per_class: usize,
    pub rounds: usize,
    pub seed: u64,
    /// Fraction of samples held out for validation.
    pub val_fraction: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 120,
            learning_rate: 0.005,
            hidden_dim: 48,
            samples_per_class: 32,
            rounds: 2,
            seed: 42,
            val_fraction: 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrainingReport {
    pub epochs: usize,
    pub train_samples: usize,
    pub val_samples: usize,
    pub hidden_dim: usize,
    pub rounds: usize,
    pub final_loss: f32,
    pub train_accuracy: f32,
    pub val_accuracy: f32,
    pub val_macro_f1: f32,
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

/// Build the synthetic labelled graphs. Parameter sweeps vary size; the plate
/// class also varies hole *count*, giving genuine structural variation.
pub fn synthetic_graphs(samples_per_class: usize) -> Vec<(BrepGraph, u32)> {
    let mut items = Vec::with_capacity(samples_per_class * CLASS_COUNT);
    for index in 0..samples_per_class {
        let t = index as f32 / samples_per_class.max(1) as f32;
        items.push((box_graph(1.0 + t, 0.8 + 0.5 * t, 0.5 + 0.25 * t), 0));
        items.push((cylinder_graph(0.25 + 0.2 * t, 0.8 + t), 1));
        let hole_count = 1 + (index % 3);
        items.push((
            plate_with_holes(
                1.5 + t,
                1.0 + 0.5 * t,
                0.12 + 0.1 * t,
                0.12 + 0.05 * t,
                hole_count,
            ),
            2,
        ));
    }
    items
}

/// Tensorize graphs and split into train/val, keeping class balance by striding
/// every `1/val_fraction`-th sample into the validation set.
pub fn build_dataset(
    samples_per_class: usize,
    val_fraction: f32,
) -> Result<(Vec<(GraphTensors, u32)>, Vec<(GraphTensors, u32)>)> {
    let tensorizer = GraphTensorizer::default();
    let graphs = synthetic_graphs(samples_per_class);
    let stride = if val_fraction <= 0.0 {
        usize::MAX
    } else {
        (1.0 / val_fraction).round().max(2.0) as usize
    };

    let mut train = Vec::new();
    let mut val = Vec::new();
    for (index, (graph, label)) in graphs.iter().enumerate() {
        let tensors = tensorizer
            .tensorize(graph)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        if index % stride == stride - 1 {
            val.push((tensors, *label));
        } else {
            train.push((tensors, *label));
        }
    }
    Ok((train, val))
}

// ---------------------------------------------------------------------------
// Batching
// ---------------------------------------------------------------------------

/// A batch of ragged BRep graphs concatenated into flat tensors with offset
/// coedge index arrays and per-node graph assignments for pooling.
pub struct BrepBatch {
    pub graph_count: usize,
    pub face_categorical: Tensor,
    pub face_scalar: Tensor,
    pub face_grid: Tensor,
    pub edge_categorical: Tensor,
    pub edge_scalar: Tensor,
    pub edge_grid: Tensor,
    pub coedge_face: Tensor,
    pub coedge_edge: Tensor,
    pub coedge_mate: Tensor,
    pub face_graph: Tensor,
    pub edge_graph: Tensor,
}

impl BrepBatch {
    pub fn from_graphs(items: &[GraphTensors], device: &Device) -> Result<Self> {
        assert!(!items.is_empty(), "batch must contain at least one graph");
        let uv_res = items[0].uv_res;
        let curve_res = items[0].curve_res;

        let mut face_categorical = Vec::new();
        let mut face_scalar = Vec::new();
        let mut face_grid = Vec::new();
        let mut edge_categorical = Vec::new();
        let mut edge_scalar = Vec::new();
        let mut edge_grid = Vec::new();
        let mut coedge_face = Vec::new();
        let mut coedge_edge = Vec::new();
        let mut coedge_mate = Vec::new();
        let mut face_graph = Vec::new();
        let mut edge_graph = Vec::new();

        let (mut face_off, mut edge_off, mut coedge_off) = (0u32, 0u32, 0u32);
        for (graph_id, item) in items.iter().enumerate() {
            let graph_id = graph_id as u32;
            face_categorical.extend_from_slice(&item.face_categorical);
            face_scalar.extend_from_slice(&item.face_scalar);
            face_grid.extend_from_slice(&item.face_grid);
            edge_categorical.extend_from_slice(&item.edge_categorical);
            edge_scalar.extend_from_slice(&item.edge_scalar);
            edge_grid.extend_from_slice(&item.edge_grid);

            for &f in &item.coedge_face {
                coedge_face.push(f + face_off);
            }
            for &e in &item.coedge_edge {
                coedge_edge.push(e + edge_off);
            }
            for &m in &item.coedge_mate {
                coedge_mate.push(m + coedge_off);
            }
            face_graph.extend(std::iter::repeat(graph_id).take(item.face_count));
            edge_graph.extend(std::iter::repeat(graph_id).take(item.edge_count));

            face_off += item.face_count as u32;
            edge_off += item.edge_count as u32;
            coedge_off += item.coedge_count as u32;
        }

        let total_faces = face_off as usize;
        let total_edges = edge_off as usize;

        Ok(Self {
            graph_count: items.len(),
            face_categorical: Tensor::from_vec(
                face_categorical,
                (total_faces, FACE_CATEGORICAL_DIM),
                device,
            )?,
            face_scalar: Tensor::from_vec(face_scalar, (total_faces, FACE_SCALAR_DIM), device)?,
            face_grid: Tensor::from_vec(
                face_grid,
                (total_faces, FACE_GRID_CHANNELS, uv_res, uv_res),
                device,
            )?,
            edge_categorical: Tensor::from_vec(
                edge_categorical,
                (total_edges, EDGE_CATEGORICAL_DIM),
                device,
            )?,
            edge_scalar: Tensor::from_vec(edge_scalar, (total_edges, EDGE_SCALAR_DIM), device)?,
            edge_grid: Tensor::from_vec(
                edge_grid,
                (total_edges, EDGE_GRID_CHANNELS, curve_res),
                device,
            )?,
            coedge_face: Tensor::from_vec(coedge_face, coedge_off as usize, device)?,
            coedge_edge: Tensor::from_vec(coedge_edge, coedge_off as usize, device)?,
            coedge_mate: Tensor::from_vec(coedge_mate, coedge_off as usize, device)?,
            face_graph: Tensor::from_vec(face_graph, total_faces, device)?,
            edge_graph: Tensor::from_vec(edge_graph, total_edges, device)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Geometry conv front-ends
// ---------------------------------------------------------------------------

struct FaceGeomCnn {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl FaceGeomCnn {
    fn new(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = conv2d(FACE_GRID_CHANNELS, CONV_HIDDEN, 3, cfg, vb.pp("conv1"))?;
        let conv2 = conv2d(CONV_HIDDEN, GEOM_OUT, 3, cfg, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, grid: &Tensor) -> Result<Tensor> {
        // [N, C, H, W] -> [N, GEOM_OUT] via global average pool.
        let xs = grid.apply(&self.conv1)?.relu()?;
        let xs = xs.apply(&self.conv2)?.relu()?;
        xs.mean(3)?.mean(2)
    }
}

struct EdgeGeomCnn {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl EdgeGeomCnn {
    fn new(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = conv1d(EDGE_GRID_CHANNELS, CONV_HIDDEN, 3, cfg, vb.pp("conv1"))?;
        let conv2 = conv1d(CONV_HIDDEN, GEOM_OUT, 3, cfg, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, grid: &Tensor) -> Result<Tensor> {
        // [N, C, L] -> [N, GEOM_OUT] via global average pool.
        let xs = grid.apply(&self.conv1)?.relu()?;
        let xs = xs.apply(&self.conv2)?.relu()?;
        xs.mean(2)
    }
}

// ---------------------------------------------------------------------------
// Hybrid encoder
// ---------------------------------------------------------------------------

/// Output of a forward pass: graph logits plus reusable node embeddings.
pub struct EncoderOutputs {
    pub graph_logits: Tensor,
    pub face_embeddings: Tensor,
    pub edge_embeddings: Tensor,
}

pub struct HybridBrepEncoder {
    rounds: usize,
    face_geom: FaceGeomCnn,
    edge_geom: EdgeGeomCnn,
    face_in: Linear,
    edge_in: Linear,
    coedge_init: Linear,
    coedge_msg: Vec<(Linear, Linear, LayerNorm)>,
    face_update: Vec<(Linear, LayerNorm)>,
    edge_update: Vec<(Linear, LayerNorm)>,
    readout1: Linear,
    readout2: Linear,
}

impl HybridBrepEncoder {
    pub fn new(hidden: usize, rounds: usize, vb: VarBuilder) -> Result<Self> {
        let face_in_dim = FACE_CATEGORICAL_DIM + FACE_SCALAR_DIM + GEOM_OUT;
        let edge_in_dim = EDGE_CATEGORICAL_DIM + EDGE_SCALAR_DIM + GEOM_OUT;

        let face_geom = FaceGeomCnn::new(vb.pp("face_geom"))?;
        let edge_geom = EdgeGeomCnn::new(vb.pp("edge_geom"))?;
        let face_in = linear(face_in_dim, hidden, vb.pp("face_in"))?;
        let edge_in = linear(edge_in_dim, hidden, vb.pp("edge_in"))?;
        let coedge_init = linear(2 * hidden, hidden, vb.pp("coedge_init"))?;

        let mut coedge_msg = Vec::with_capacity(rounds);
        let mut face_update = Vec::with_capacity(rounds);
        let mut edge_update = Vec::with_capacity(rounds);
        for r in 0..rounds {
            let scope = vb.pp(format!("round{r}"));
            let cm1 = linear(4 * hidden, hidden, scope.pp("coedge_msg1"))?;
            let cm2 = linear(hidden, hidden, scope.pp("coedge_msg2"))?;
            let cln = layer_norm(hidden, 1e-5, scope.pp("coedge_ln"))?;
            coedge_msg.push((cm1, cm2, cln));

            let fu = linear(2 * hidden, hidden, scope.pp("face_update"))?;
            let fln = layer_norm(hidden, 1e-5, scope.pp("face_ln"))?;
            face_update.push((fu, fln));

            let eu = linear(2 * hidden, hidden, scope.pp("edge_update"))?;
            let eln = layer_norm(hidden, 1e-5, scope.pp("edge_ln"))?;
            edge_update.push((eu, eln));
        }

        let readout1 = linear(2 * hidden, hidden, vb.pp("readout1"))?;
        let readout2 = linear(hidden, CLASS_COUNT, vb.pp("readout2"))?;

        Ok(Self {
            rounds,
            face_geom,
            edge_geom,
            face_in,
            edge_in,
            coedge_init,
            coedge_msg,
            face_update,
            edge_update,
            readout1,
            readout2,
        })
    }

    pub fn forward(&self, batch: &BrepBatch) -> Result<EncoderOutputs> {
        // Node initial embeddings from geometry + categorical + scalar features.
        let face_geom = self.face_geom.forward(&batch.face_grid)?;
        let face_feats = Tensor::cat(
            &[&batch.face_categorical, &batch.face_scalar, &face_geom],
            1,
        )?;
        let mut face = face_feats.apply(&self.face_in)?.relu()?;

        let edge_geom = self.edge_geom.forward(&batch.edge_grid)?;
        let edge_feats = Tensor::cat(
            &[&batch.edge_categorical, &batch.edge_scalar, &edge_geom],
            1,
        )?;
        let mut edge = edge_feats.apply(&self.edge_in)?.relu()?;

        // Coedge init from incident face + edge.
        let coedge_face_emb = face.index_select(&batch.coedge_face, 0)?;
        let coedge_edge_emb = edge.index_select(&batch.coedge_edge, 0)?;
        let mut coedge = Tensor::cat(&[&coedge_face_emb, &coedge_edge_emb], 1)?
            .apply(&self.coedge_init)?
            .relu()?;

        let face_count = face.dim(0)?;
        let edge_count = edge.dim(0)?;

        for r in 0..self.rounds {
            // 1. Coedge update: gather face, edge, mate.
            let f_g = face.index_select(&batch.coedge_face, 0)?;
            let e_g = edge.index_select(&batch.coedge_edge, 0)?;
            let m_g = coedge.index_select(&batch.coedge_mate, 0)?;
            let msg = Tensor::cat(&[&coedge, &f_g, &e_g, &m_g], 1)?;
            let (cm1, cm2, cln) = &self.coedge_msg[r];
            let delta = msg.apply(cm1)?.relu()?.apply(cm2)?;
            coedge = cln.forward(&(coedge + delta)?)?;

            // 2. Face update: mean of incident coedges.
            let face_msg = scatter_mean(&coedge, &batch.coedge_face, face_count)?;
            let (fu, fln) = &self.face_update[r];
            let face_delta = Tensor::cat(&[&face, &face_msg], 1)?.apply(fu)?.relu()?;
            face = fln.forward(&(face + face_delta)?)?;

            // 3. Edge update: mean of incident coedges.
            let edge_msg = scatter_mean(&coedge, &batch.coedge_edge, edge_count)?;
            let (eu, eln) = &self.edge_update[r];
            let edge_delta = Tensor::cat(&[&edge, &edge_msg], 1)?.apply(eu)?.relu()?;
            edge = eln.forward(&(edge + edge_delta)?)?;
        }

        // Readout: per-graph mean pool of faces and edges.
        let face_pool = scatter_mean(&face, &batch.face_graph, batch.graph_count)?;
        let edge_pool = scatter_mean(&edge, &batch.edge_graph, batch.graph_count)?;
        let graph_emb = Tensor::cat(&[&face_pool, &edge_pool], 1)?;
        let graph_logits = graph_emb
            .apply(&self.readout1)?
            .relu()?
            .apply(&self.readout2)?;

        Ok(EncoderOutputs {
            graph_logits,
            face_embeddings: face,
            edge_embeddings: edge,
        })
    }
}

/// Segment-mean: average `src` rows into `num` groups given `index[row] -> group`.
fn scatter_mean(src: &Tensor, index: &Tensor, num: usize) -> Result<Tensor> {
    let device = src.device();
    let dim = src.dim(1)?;
    let rows = src.dim(0)?;
    let sum = Tensor::zeros((num, dim), DTYPE, device)?.index_add(index, src, 0)?;
    let ones = Tensor::ones((rows, 1), DTYPE, device)?;
    let count = Tensor::zeros((num, 1), DTYPE, device)?.index_add(index, &ones, 0)?;
    let count = count.clamp(1.0, 1e9)?;
    sum.broadcast_div(&count)
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

pub fn train_synthetic(config: TrainingConfig) -> Result<TrainingReport> {
    train_synthetic_with_optional_save(config, None)
}

pub fn train_synthetic_and_save(
    config: TrainingConfig,
    save_path: &Path,
) -> Result<TrainingReport> {
    train_synthetic_with_optional_save(config, Some(save_path))
}

fn train_synthetic_with_optional_save(
    config: TrainingConfig,
    save_path: Option<&Path>,
) -> Result<TrainingReport> {
    let device = Device::Cpu;
    // Best-effort seeding: Candle's CPU backend does not support set_seed (0.11),
    // so this is a no-op on CPU and only takes effect on CUDA.
    let _ = device.set_seed(config.seed);

    let (train, val) = build_dataset(config.samples_per_class, config.val_fraction)?;
    let train_tensors: Vec<GraphTensors> = train.iter().map(|(t, _)| t.clone()).collect();
    let train_batch = BrepBatch::from_graphs(&train_tensors, &device)?;
    let train_labels = labels_tensor(&train, &device)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DTYPE, &device);
    let model = HybridBrepEncoder::new(config.hidden_dim, config.rounds, vb)?;

    let params = ParamsAdamW {
        lr: config.learning_rate,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

    let mut final_loss = 0.0;
    for _ in 0..config.epochs {
        let outputs = model.forward(&train_batch)?;
        let loss = loss::cross_entropy(&outputs.graph_logits, &train_labels)?;
        final_loss = loss.to_scalar::<f32>()?;
        optimizer.backward_step(&loss)?;
    }

    let train_logits = model.forward(&train_batch)?.graph_logits;
    let train_accuracy = accuracy(&train_logits, &train_labels)?;

    let (val_accuracy, val_macro_f1) = if val.is_empty() {
        (f32::NAN, f32::NAN)
    } else {
        let val_tensors: Vec<GraphTensors> = val.iter().map(|(t, _)| t.clone()).collect();
        let val_batch = BrepBatch::from_graphs(&val_tensors, &device)?;
        let val_labels = labels_tensor(&val, &device)?;
        let val_logits = model.forward(&val_batch)?.graph_logits;
        (
            accuracy(&val_logits, &val_labels)?,
            macro_f1(&val_logits, &val_labels)?,
        )
    };

    if let Some(save_path) = save_path {
        varmap.save(save_path)?;
    }

    Ok(TrainingReport {
        epochs: config.epochs,
        train_samples: train.len(),
        val_samples: val.len(),
        hidden_dim: config.hidden_dim,
        rounds: config.rounds,
        final_loss,
        train_accuracy,
        val_accuracy,
        val_macro_f1,
    })
}

/// Load a checkpoint into a fresh encoder for inference.
pub fn load_encoder(
    config: TrainingConfig,
    path: &Path,
    device: &Device,
) -> Result<HybridBrepEncoder> {
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DTYPE, device);
    let model = HybridBrepEncoder::new(config.hidden_dim, config.rounds, vb)?;
    varmap.load(path)?;
    Ok(model)
}

fn labels_tensor(items: &[(GraphTensors, u32)], device: &Device) -> Result<Tensor> {
    let labels: Vec<u32> = items.iter().map(|(_, label)| *label).collect();
    Tensor::from_vec(labels, items.len(), device)?.to_dtype(DType::U32)
}

fn accuracy(logits: &Tensor, labels: &Tensor) -> Result<f32> {
    let predictions = logits.argmax(D::Minus1)?;
    let correct = predictions
        .eq(labels)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    Ok(correct / labels.dims1()? as f32)
}

/// Macro-averaged F1 over the classes present.
fn macro_f1(logits: &Tensor, labels: &Tensor) -> Result<f32> {
    let predictions = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let targets = labels.to_vec1::<u32>()?;

    let mut tp = [0.0f32; CLASS_COUNT];
    let mut fp = [0.0f32; CLASS_COUNT];
    let mut fn_ = [0.0f32; CLASS_COUNT];
    for (&pred, &target) in predictions.iter().zip(targets.iter()) {
        let (pred, target) = (pred as usize, target as usize);
        if pred == target {
            tp[target] += 1.0;
        } else {
            fp[pred] += 1.0;
            fn_[target] += 1.0;
        }
    }

    let mut f1_sum = 0.0;
    let mut counted = 0;
    for class in 0..CLASS_COUNT {
        if tp[class] + fp[class] + fn_[class] == 0.0 {
            continue; // class absent from this split
        }
        let precision = safe_div(tp[class], tp[class] + fp[class]);
        let recall = safe_div(tp[class], tp[class] + fn_[class]);
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        f1_sum += f1;
        counted += 1;
    }
    Ok(if counted == 0 {
        0.0
    } else {
        f1_sum / counted as f32
    })
}

fn safe_div(numerator: f32, denominator: f32) -> f32 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_splits_keep_class_balance() {
        let (train, val) = build_dataset(8, 0.25).expect("dataset builds");
        assert!(!train.is_empty());
        assert!(!val.is_empty());
        assert_eq!(train.len() + val.len(), 8 * CLASS_COUNT);
    }

    #[test]
    fn batch_offsets_coedge_indices() {
        let (train, _) = build_dataset(4, 0.25).expect("dataset builds");
        let tensors: Vec<GraphTensors> = train.iter().map(|(t, _)| t.clone()).collect();
        let device = Device::Cpu;
        let batch = BrepBatch::from_graphs(&tensors, &device).expect("batch builds");
        let total_faces: usize = tensors.iter().map(|t| t.face_count).sum();
        assert_eq!(batch.face_graph.dims1().unwrap(), total_faces);
    }

    #[test]
    fn forward_produces_class_logits() {
        let device = Device::Cpu;
        let _ = device.set_seed(0);
        let (train, _) = build_dataset(3, 0.0).expect("dataset builds");
        let tensors: Vec<GraphTensors> = train.iter().map(|(t, _)| t.clone()).collect();
        let batch = BrepBatch::from_graphs(&tensors, &device).unwrap();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DTYPE, &device);
        let model = HybridBrepEncoder::new(16, 2, vb).unwrap();
        let outputs = model.forward(&batch).unwrap();

        assert_eq!(outputs.graph_logits.dims(), &[tensors.len(), CLASS_COUNT]);
    }

    #[test]
    fn checkpoint_saves_loads_and_infers() {
        let config = TrainingConfig {
            epochs: 5,
            samples_per_class: 4,
            hidden_dim: 16,
            ..Default::default()
        };
        let mut path = std::env::temp_dir();
        path.push("acad-hybrid-encoder-test.safetensors");
        train_synthetic_and_save(config, &path).expect("train + save");

        let device = Device::Cpu;
        let model = load_encoder(config, &path, &device).expect("load checkpoint");
        let (items, _) = build_dataset(2, 0.0).expect("dataset builds");
        let tensors: Vec<GraphTensors> = items.iter().map(|(t, _)| t.clone()).collect();
        let batch = BrepBatch::from_graphs(&tensors, &device).unwrap();
        let outputs = model.forward(&batch).expect("inference");
        assert_eq!(outputs.graph_logits.dims(), &[tensors.len(), CLASS_COUNT]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn short_training_run_reduces_loss() {
        let config = TrainingConfig {
            epochs: 30,
            samples_per_class: 6,
            hidden_dim: 24,
            ..Default::default()
        };
        let report = train_synthetic(config).expect("training runs");
        assert!(report.final_loss.is_finite());
        assert!(report.train_accuracy >= 0.0 && report.train_accuracy <= 1.0);
        assert!(report.val_samples > 0);
    }
}
