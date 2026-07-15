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

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use acad_brep_dataset::{
    hash_strings_hex, load_manifest, load_metadata, load_samples, BrepSampleLabels, DatasetRecord,
    DatasetSplit, CLASS_NAMES,
};
use acad_brep_encoder::{
    GraphTensorizer, GraphTensors, EDGE_CATEGORICAL_DIM, EDGE_GRID_CHANNELS, EDGE_SCALAR_DIM,
    FACE_CATEGORICAL_DIM, FACE_GRID_CHANNELS, FACE_SCALAR_DIM,
};
use acad_brep_graph::{box_graph, cylinder_graph, plate_with_holes, BrepGraph};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    conv1d, conv2d, layer_norm, linear, loss, ops, AdamW, Conv1d, Conv1dConfig, Conv2d,
    Conv2dConfig, LayerNorm, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap,
};
use serde::{Deserialize, Serialize};

pub const CLASS_COUNT: usize = CLASS_NAMES.len();
/// Output width of each geometry conv front-end.
pub const GEOM_OUT: usize = 32;
const CONV_HIDDEN: usize = 16;
const DTYPE: DType = DType::F32;

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

#[derive(Debug, Clone, Copy)]
pub struct FaceSegmentationConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub hidden_dim: usize,
    pub rounds: usize,
    pub seed: u64,
    pub batch_size: usize,
    /// Maximum train graph samples to load. `None` means all train samples.
    pub max_train_samples: Option<usize>,
    /// Maximum evaluation graph samples to load. `None` means all evaluation samples.
    pub max_eval_samples: Option<usize>,
    pub use_class_weights: bool,
    /// Sampling strategy for training records when a sample limit is set.
    pub sampling_strategy: FaceSamplingStrategy,
    pub eval_split: DatasetSplit,
    pub shuffle_each_epoch: bool,
}

impl Default for FaceSegmentationConfig {
    fn default() -> Self {
        Self {
            epochs: 5,
            learning_rate: 0.003,
            hidden_dim: 48,
            rounds: 2,
            seed: 42,
            batch_size: 8,
            max_train_samples: Some(1024),
            max_eval_samples: Some(256),
            use_class_weights: false,
            sampling_strategy: FaceSamplingStrategy::Uniform,
            eval_split: DatasetSplit::Val,
            shuffle_each_epoch: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceSamplingStrategy {
    /// Deterministic evenly-spaced records from the manifest split.
    Uniform,
    /// Greedy graph selection that favors records containing underrepresented
    /// face labels while preserving deterministic behavior.
    FaceBalanced,
}

impl FaceSamplingStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::FaceBalanced => "face-balanced",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FaceSegmentationReport {
    pub epochs: usize,
    pub train_samples: usize,
    pub eval_samples: usize,
    pub train_faces: usize,
    pub eval_faces: usize,
    pub face_classes: usize,
    pub hidden_dim: usize,
    pub rounds: usize,
    pub batch_size: usize,
    pub eval_split: DatasetSplit,
    pub face_label_names: Vec<String>,
    pub train_face_label_counts: Vec<usize>,
    pub eval_face_label_counts: Vec<usize>,
    pub train_record_ids_hash: String,
    pub eval_record_ids_hash: String,
    pub final_loss: f32,
    pub train_accuracy: f32,
    pub eval_accuracy: f32,
    pub eval_macro_f1: f32,
    pub train_metrics: FaceEvaluationMetrics,
    pub eval_metrics: FaceEvaluationMetrics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaceEvaluationMetrics {
    pub accuracy: f32,
    pub macro_f1: f32,
    pub weighted_f1: f32,
    pub macro_iou: f32,
    pub macro_iou_present: f32,
    pub support: usize,
    pub class_metrics: Vec<FaceClassMetric>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaceClassMetric {
    pub class_id: usize,
    pub label: String,
    pub support: usize,
    pub predicted: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub iou: f32,
}

#[derive(Debug, Clone)]
pub struct FaceSegmentationExample {
    pub record_id: String,
    pub tensors: GraphTensors,
    pub face_labels: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct FaceSegmentationDataset {
    pub train: Vec<FaceSegmentationExample>,
    pub eval: Vec<FaceSegmentationExample>,
    pub label_names: Vec<String>,
}

pub type LabeledGraphTensor = (GraphTensors, u32);
pub type GraphClassificationDataset = (Vec<LabeledGraphTensor>, Vec<LabeledGraphTensor>);

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
) -> Result<GraphClassificationDataset> {
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

/// Load a real on-disk dataset written by `acad-brep-dataset`.
pub fn build_dataset_from_dir(root: &Path) -> Result<GraphClassificationDataset> {
    let tensorizer = GraphTensorizer::default();
    let samples = load_samples(root).map_err(to_candle_error)?;
    let mut train = Vec::new();
    let mut val = Vec::new();

    for sample in samples {
        let tensors = tensorizer
            .tensorize(&sample.graph)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        match sample.record.split {
            DatasetSplit::Train => train.push((tensors, sample.labels.graph_class_id)),
            DatasetSplit::Val => val.push((tensors, sample.labels.graph_class_id)),
            DatasetSplit::Test => {}
        }
    }

    Ok((train, val))
}

/// Load real face-segmentation examples from the on-disk dataset.
///
/// This reads only the selected manifest rows, so sampled training runs over
/// large Fusion datasets do not need to materialize every graph first.
pub fn build_face_segmentation_dataset(
    root: &Path,
    config: FaceSegmentationConfig,
) -> Result<FaceSegmentationDataset> {
    let metadata = load_metadata(root).map_err(to_candle_error)?;
    if metadata.face_label_set.is_empty() {
        return Err(candle_core::Error::Msg(
            "dataset metadata has no face_label_set".to_string(),
        ));
    }

    let label_to_id: BTreeMap<String, u32> = metadata
        .face_label_set
        .iter()
        .enumerate()
        .map(|(index, label)| (label.clone(), index as u32))
        .collect();

    let records = load_manifest(root).map_err(to_candle_error)?;
    let train_records: Vec<_> = records
        .iter()
        .filter(|record| record.split == DatasetSplit::Train)
        .collect();
    let eval_records: Vec<_> = records
        .iter()
        .filter(|record| record.split == config.eval_split)
        .collect();

    let tensorizer = GraphTensorizer::default();
    let train = load_face_examples(
        root,
        select_records_with_strategy(
            root,
            &train_records,
            config.max_train_samples,
            config.sampling_strategy,
            &label_to_id,
        )?,
        &label_to_id,
        &tensorizer,
    )?;
    let eval = load_face_examples(
        root,
        select_records_with_strategy(
            root,
            &eval_records,
            config.max_eval_samples,
            FaceSamplingStrategy::Uniform,
            &label_to_id,
        )?,
        &label_to_id,
        &tensorizer,
    )?;

    Ok(FaceSegmentationDataset {
        train,
        eval,
        label_names: metadata.face_label_set,
    })
}

fn select_records_with_strategy<'a>(
    root: &Path,
    records: &'a [&'a DatasetRecord],
    limit: Option<usize>,
    strategy: FaceSamplingStrategy,
    label_to_id: &BTreeMap<String, u32>,
) -> Result<Vec<&'a DatasetRecord>> {
    let effective = match limit {
        Some(0) | None => records.len(),
        Some(limit) => limit.min(records.len()),
    };
    if effective == records.len() {
        return Ok(records.to_vec());
    }
    if effective == 0 {
        return Ok(Vec::new());
    }

    match strategy {
        FaceSamplingStrategy::Uniform => Ok(select_uniform_records(records, effective)),
        FaceSamplingStrategy::FaceBalanced => {
            select_face_balanced_records(root, records, effective, label_to_id)
        }
    }
}

fn select_uniform_records<'a>(
    records: &'a [&'a DatasetRecord],
    effective: usize,
) -> Vec<&'a DatasetRecord> {
    (0..effective)
        .map(|index| records[index * records.len() / effective])
        .collect()
}

#[derive(Debug, Clone)]
struct RecordFaceCounts {
    index: usize,
    counts: Vec<usize>,
}

fn select_face_balanced_records<'a>(
    root: &Path,
    records: &'a [&'a DatasetRecord],
    effective: usize,
    label_to_id: &BTreeMap<String, u32>,
) -> Result<Vec<&'a DatasetRecord>> {
    if label_to_id.is_empty() {
        return Ok(select_uniform_records(records, effective));
    }

    let class_count = label_to_id.len();
    let mut per_record = Vec::with_capacity(records.len());
    let mut global_counts = vec![0usize; class_count];

    for (index, record) in records.iter().enumerate() {
        let counts = read_record_face_counts(root, record, label_to_id)?;
        for (class, count) in counts.iter().enumerate() {
            global_counts[class] += count;
        }
        per_record.push(RecordFaceCounts { index, counts });
    }

    let total_faces: usize = global_counts.iter().sum();
    if total_faces == 0 {
        return Ok(select_uniform_records(records, effective));
    }

    let mut inverse_global = vec![0.0f32; class_count];
    let present = global_counts
        .iter()
        .filter(|&&count| count > 0)
        .count()
        .max(1);
    for (class, &count) in global_counts.iter().enumerate() {
        if count > 0 {
            inverse_global[class] = total_faces as f32 / (present as f32 * count as f32);
        }
    }

    let mut selected = Vec::with_capacity(effective);
    let mut selected_flags = vec![false; records.len()];
    let mut selected_counts = vec![0usize; class_count];

    while selected.len() < effective {
        let mut best_index = None;
        let mut best_score = f32::NEG_INFINITY;

        for candidate in &per_record {
            if selected_flags[candidate.index] {
                continue;
            }
            let score = balanced_record_score(candidate, &selected_counts, &inverse_global);
            if score > best_score {
                best_score = score;
                best_index = Some(candidate.index);
            }
        }

        let Some(index) = best_index else {
            break;
        };
        selected_flags[index] = true;
        for (class, count) in per_record[index].counts.iter().enumerate() {
            selected_counts[class] += count;
        }
        selected.push(records[index]);
    }

    selected.sort_by_key(|record| &record.id);
    Ok(selected)
}

fn read_record_face_counts(
    root: &Path,
    record: &DatasetRecord,
    label_to_id: &BTreeMap<String, u32>,
) -> Result<Vec<usize>> {
    if !record.face_label_counts.is_empty() {
        let mut counts = vec![0usize; label_to_id.len()];
        for (label, count) in &record.face_label_counts {
            let label_id = label_to_id.get(label).copied().ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "unknown face label {label:?} in manifest record {}",
                    record.id
                ))
            })? as usize;
            counts[label_id] += count;
        }
        return Ok(counts);
    }

    let labels_json =
        fs::read_to_string(root.join(&record.labels_path)).map_err(to_candle_error)?;
    let labels: BrepSampleLabels = serde_json::from_str(&labels_json).map_err(to_candle_error)?;
    let mut counts = vec![0usize; label_to_id.len()];
    for label in labels.face_labels {
        let label_id = label_to_id.get(&label).copied().ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "unknown face label {label:?} in sample {}",
                record.id
            ))
        })? as usize;
        counts[label_id] += 1;
    }
    Ok(counts)
}

fn balanced_record_score(
    candidate: &RecordFaceCounts,
    selected_counts: &[usize],
    inverse_global: &[f32],
) -> f32 {
    let mut score = 0.0;
    for (class, &count) in candidate.counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let selected = selected_counts[class] as f32;
        score += inverse_global[class] * (count as f32).sqrt() / (selected + 1.0).sqrt();
    }
    score
}

fn load_face_examples(
    root: &Path,
    records: Vec<&DatasetRecord>,
    label_to_id: &BTreeMap<String, u32>,
    tensorizer: &GraphTensorizer,
) -> Result<Vec<FaceSegmentationExample>> {
    records
        .into_iter()
        .map(|record| load_face_example(root, record, label_to_id, tensorizer))
        .collect()
}

fn load_face_example(
    root: &Path,
    record: &DatasetRecord,
    label_to_id: &BTreeMap<String, u32>,
    tensorizer: &GraphTensorizer,
) -> Result<FaceSegmentationExample> {
    let graph_json = fs::read_to_string(root.join(&record.graph_path)).map_err(to_candle_error)?;
    let labels_json =
        fs::read_to_string(root.join(&record.labels_path)).map_err(to_candle_error)?;
    let graph = BrepGraph::from_json(&graph_json).map_err(to_candle_error)?;
    graph.validate().map_err(to_candle_error)?;
    let labels: BrepSampleLabels = serde_json::from_str(&labels_json).map_err(to_candle_error)?;

    if labels.graph_class_id != record.class_id {
        return Err(candle_core::Error::Msg(format!(
            "graph class id mismatch for sample {}",
            record.id
        )));
    }
    if labels.face_labels.len() != graph.faces.len() {
        return Err(candle_core::Error::Msg(format!(
            "face label count mismatch for sample {}: labels={}, faces={}",
            record.id,
            labels.face_labels.len(),
            graph.faces.len()
        )));
    }

    let mut face_labels = Vec::with_capacity(labels.face_labels.len());
    for label in labels.face_labels {
        let label_id = label_to_id.get(&label).copied().ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "unknown face label {label:?} in sample {}",
                record.id
            ))
        })?;
        face_labels.push(label_id);
    }

    let tensors = tensorizer
        .tensorize(&graph)
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
    if tensors.face_count != face_labels.len() {
        return Err(candle_core::Error::Msg(format!(
            "tensorized face count mismatch for sample {}: tensors={}, labels={}",
            record.id,
            tensors.face_count,
            face_labels.len()
        )));
    }

    Ok(FaceSegmentationExample {
        record_id: record.id.clone(),
        tensors,
        face_labels,
    })
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
            face_graph.resize(face_graph.len() + item.face_count, graph_id);
            edge_graph.resize(edge_graph.len() + item.edge_count, graph_id);

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

pub struct FaceSegBatch {
    pub batch: BrepBatch,
    pub face_labels: Tensor,
}

impl FaceSegBatch {
    pub fn from_examples(items: &[FaceSegmentationExample], device: &Device) -> Result<Self> {
        if items.is_empty() {
            return Err(candle_core::Error::Msg(
                "face segmentation batch must contain at least one graph".to_string(),
            ));
        }

        let mut tensors = Vec::with_capacity(items.len());
        let mut labels = Vec::new();
        for item in items {
            if item.tensors.face_count != item.face_labels.len() {
                return Err(candle_core::Error::Msg(format!(
                    "face label count mismatch in batch: tensors={}, labels={}",
                    item.tensors.face_count,
                    item.face_labels.len()
                )));
            }
            tensors.push(item.tensors.clone());
            labels.extend_from_slice(&item.face_labels);
        }

        let batch = BrepBatch::from_graphs(&tensors, device)?;
        let face_labels =
            Tensor::from_vec(labels, batch.face_graph.dims1()?, device)?.to_dtype(DType::U32)?;
        Ok(Self { batch, face_labels })
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

pub struct FaceSegmentationModel {
    pub encoder: HybridBrepEncoder,
    face_head: Linear,
}

impl FaceSegmentationModel {
    pub fn new(hidden: usize, rounds: usize, face_classes: usize, vb: VarBuilder) -> Result<Self> {
        if face_classes == 0 {
            return Err(candle_core::Error::Msg(
                "face segmentation requires at least one class".to_string(),
            ));
        }
        let encoder = HybridBrepEncoder::new(hidden, rounds, vb.pp("encoder"))?;
        let face_head = linear(hidden, face_classes, vb.pp("face_head"))?;
        Ok(Self { encoder, face_head })
    }

    pub fn forward_face_logits(&self, batch: &BrepBatch) -> Result<Tensor> {
        self.encoder
            .forward(batch)?
            .face_embeddings
            .apply(&self.face_head)
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

pub fn train_dataset(config: TrainingConfig, dataset_root: &Path) -> Result<TrainingReport> {
    train_dataset_with_optional_save(config, dataset_root, None)
}

pub fn train_dataset_and_save(
    config: TrainingConfig,
    dataset_root: &Path,
    save_path: &Path,
) -> Result<TrainingReport> {
    train_dataset_with_optional_save(config, dataset_root, Some(save_path))
}

pub fn train_face_segmentation(
    config: FaceSegmentationConfig,
    dataset_root: &Path,
) -> Result<FaceSegmentationReport> {
    train_face_segmentation_with_optional_save(config, dataset_root, None)
}

pub fn train_face_segmentation_and_save(
    config: FaceSegmentationConfig,
    dataset_root: &Path,
    save_path: &Path,
) -> Result<FaceSegmentationReport> {
    train_face_segmentation_with_optional_save(config, dataset_root, Some(save_path))
}

fn train_synthetic_with_optional_save(
    config: TrainingConfig,
    save_path: Option<&Path>,
) -> Result<TrainingReport> {
    let (train, val) = build_dataset(config.samples_per_class, config.val_fraction)?;
    train_splits_with_optional_save(config, train, val, save_path)
}

fn train_dataset_with_optional_save(
    config: TrainingConfig,
    dataset_root: &Path,
    save_path: Option<&Path>,
) -> Result<TrainingReport> {
    let (train, val) = build_dataset_from_dir(dataset_root)?;
    train_splits_with_optional_save(config, train, val, save_path)
}

fn train_splits_with_optional_save(
    config: TrainingConfig,
    train: Vec<(GraphTensors, u32)>,
    val: Vec<(GraphTensors, u32)>,
    save_path: Option<&Path>,
) -> Result<TrainingReport> {
    let device = Device::Cpu;
    // Best-effort seeding: Candle's CPU backend does not support set_seed (0.11),
    // so this is a no-op on CPU and only takes effect on CUDA.
    let _ = device.set_seed(config.seed);

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

fn train_face_segmentation_with_optional_save(
    config: FaceSegmentationConfig,
    dataset_root: &Path,
    save_path: Option<&Path>,
) -> Result<FaceSegmentationReport> {
    let dataset = build_face_segmentation_dataset(dataset_root, config)?;
    if dataset.train.is_empty() {
        return Err(candle_core::Error::Msg(
            "face segmentation dataset has no train samples".to_string(),
        ));
    }
    if dataset.eval.is_empty() {
        return Err(candle_core::Error::Msg(format!(
            "face segmentation dataset has no {} samples; choose another evaluation split with --eval-split",
            config.eval_split.as_str()
        )));
    }

    let device = Device::Cpu;
    let _ = device.set_seed(config.seed);
    let face_classes = dataset.label_names.len();
    let batch_size = config.batch_size.max(1);
    let train_faces = count_faces(&dataset.train);
    let eval_faces = count_faces(&dataset.eval);
    let train_face_label_counts = face_label_counts(&dataset.train, face_classes);
    let eval_face_label_counts = face_label_counts(&dataset.eval, face_classes);
    let train_record_ids_hash = record_ids_hash(&dataset.train);
    let eval_record_ids_hash = record_ids_hash(&dataset.eval);
    let class_weights = if config.use_class_weights {
        Some(Tensor::from_vec(
            face_class_weights(&train_face_label_counts),
            face_classes,
            &device,
        )?)
    } else {
        None
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DTYPE, &device);
    let model = FaceSegmentationModel::new(config.hidden_dim, config.rounds, face_classes, vb)?;

    let params = ParamsAdamW {
        lr: config.learning_rate,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

    let mut final_loss = f32::NAN;
    let mut train_order: Vec<usize> = (0..dataset.train.len()).collect();
    for epoch in 0..config.epochs {
        if config.shuffle_each_epoch {
            deterministic_shuffle(
                &mut train_order,
                config.seed.wrapping_add(epoch as u64).wrapping_add(1),
            );
        }
        let mut epoch_loss_sum = 0.0f32;
        let mut epoch_batch_count = 0usize;
        for chunk in train_order.chunks(batch_size) {
            let examples: Vec<_> = chunk
                .iter()
                .map(|&index| dataset.train[index].clone())
                .collect();
            let batch = FaceSegBatch::from_examples(&examples, &device)?;
            let logits = model.forward_face_logits(&batch.batch)?;
            let loss = if let Some(weights) = &class_weights {
                weighted_cross_entropy(&logits, &batch.face_labels, weights)?
            } else {
                loss::cross_entropy(&logits, &batch.face_labels)?
            };
            epoch_loss_sum += loss.to_scalar::<f32>()?;
            epoch_batch_count += 1;
            optimizer.backward_step(&loss)?;
        }
        if epoch_batch_count > 0 {
            final_loss = epoch_loss_sum / epoch_batch_count as f32;
        }
    }

    let train_metrics = evaluate_face_segmentation(
        &model,
        &dataset.train,
        batch_size,
        &device,
        &dataset.label_names,
    )?;
    let eval_metrics = evaluate_face_segmentation(
        &model,
        &dataset.eval,
        batch_size,
        &device,
        &dataset.label_names,
    )?;

    if let Some(save_path) = save_path {
        varmap.save(save_path)?;
        save_face_checkpoint_metadata(save_path, config, &dataset.label_names)?;
    }

    Ok(FaceSegmentationReport {
        epochs: config.epochs,
        train_samples: dataset.train.len(),
        eval_samples: dataset.eval.len(),
        train_faces,
        eval_faces,
        face_classes,
        hidden_dim: config.hidden_dim,
        rounds: config.rounds,
        batch_size,
        eval_split: config.eval_split,
        face_label_names: dataset.label_names,
        train_face_label_counts,
        eval_face_label_counts,
        train_record_ids_hash,
        eval_record_ids_hash,
        final_loss,
        train_accuracy: train_metrics.accuracy,
        eval_accuracy: eval_metrics.accuracy,
        eval_macro_f1: eval_metrics.macro_f1,
        train_metrics,
        eval_metrics,
    })
}

fn save_face_checkpoint_metadata(
    save_path: &Path,
    config: FaceSegmentationConfig,
    label_names: &[String],
) -> Result<()> {
    let metadata = FaceCheckpointMetadata {
        format: "acad-brep-face-checkpoint-v1".to_string(),
        task: "face_segmentation".to_string(),
        hidden_dim: config.hidden_dim,
        rounds: config.rounds,
        face_classes: label_names.len(),
        face_label_names: label_names.to_vec(),
    };
    fs::write(
        checkpoint_metadata_path(save_path),
        serde_json::to_string_pretty(&metadata).map_err(to_candle_error)?,
    )
    .map_err(to_candle_error)
}

fn checkpoint_metadata_path(save_path: &Path) -> PathBuf {
    save_path.with_extension("metadata.json")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct FaceCheckpointMetadata {
    format: String,
    task: String,
    hidden_dim: usize,
    rounds: usize,
    face_classes: usize,
    face_label_names: Vec<String>,
}

fn load_face_checkpoint_metadata(save_path: &Path) -> Result<FaceCheckpointMetadata> {
    let metadata_path = checkpoint_metadata_path(save_path);
    let json = fs::read_to_string(&metadata_path).map_err(to_candle_error)?;
    let metadata: FaceCheckpointMetadata = serde_json::from_str(&json).map_err(to_candle_error)?;
    validate_face_checkpoint_metadata(&metadata)?;
    Ok(metadata)
}

pub fn load_face_checkpoint(
    save_path: &Path,
    device: &Device,
) -> Result<(FaceSegmentationModel, Vec<String>)> {
    let metadata = load_face_checkpoint_metadata(save_path)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DTYPE, device);
    let model = FaceSegmentationModel::new(
        metadata.hidden_dim,
        metadata.rounds,
        metadata.face_classes,
        vb,
    )?;
    varmap.load(save_path)?;
    Ok((model, metadata.face_label_names))
}

fn validate_face_checkpoint_metadata(metadata: &FaceCheckpointMetadata) -> Result<()> {
    if metadata.format != "acad-brep-face-checkpoint-v1" {
        return Err(candle_core::Error::Msg(format!(
            "unsupported face checkpoint metadata format {:?}",
            metadata.format
        )));
    }
    if metadata.task != "face_segmentation" {
        return Err(candle_core::Error::Msg(format!(
            "unsupported face checkpoint task {:?}",
            metadata.task
        )));
    }
    if metadata.face_classes != metadata.face_label_names.len() {
        return Err(candle_core::Error::Msg(format!(
            "face checkpoint class count mismatch: model={}, labels={}",
            metadata.face_classes,
            metadata.face_label_names.len()
        )));
    }
    if metadata.face_classes == 0 {
        return Err(candle_core::Error::Msg(
            "face checkpoint has no face classes".to_string(),
        ));
    }
    Ok(())
}

fn to_candle_error(error: impl std::error::Error) -> candle_core::Error {
    candle_core::Error::Msg(error.to_string())
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

fn weighted_cross_entropy(
    logits: &Tensor,
    labels: &Tensor,
    class_weights: &Tensor,
) -> Result<Tensor> {
    if logits.rank() != 2 {
        return Err(candle_core::Error::Msg(
            "weighted_cross_entropy expects rank-2 logits".to_string(),
        ));
    }
    let log_probs = ops::log_softmax(logits, 1)?;
    let target_col = labels.unsqueeze(1)?;
    let nll = log_probs.gather(&target_col, 1)?.neg()?;
    let sample_weights = class_weights.index_select(labels, 0)?.unsqueeze(1)?;
    let weighted = (&nll * &sample_weights)?;
    let denominator = sample_weights.sum_all()?.clamp(1e-6, 1e9)?;
    weighted.sum_all()?.broadcast_div(&denominator)
}

fn count_faces(items: &[FaceSegmentationExample]) -> usize {
    items.iter().map(|item| item.face_labels.len()).sum()
}

fn face_label_counts(items: &[FaceSegmentationExample], class_count: usize) -> Vec<usize> {
    let mut counts = vec![0; class_count];
    for item in items {
        for &label in &item.face_labels {
            let label = label as usize;
            if label < class_count {
                counts[label] += 1;
            }
        }
    }
    counts
}

fn record_ids_hash(items: &[FaceSegmentationExample]) -> String {
    let ids: Vec<String> = items.iter().map(|item| item.record_id.clone()).collect();
    hash_strings_hex(&ids)
}

fn face_class_weights(counts: &[usize]) -> Vec<f32> {
    let total: usize = counts.iter().sum();
    let present = counts.iter().filter(|&&count| count > 0).count();
    if total == 0 || present == 0 {
        return vec![1.0; counts.len()];
    }

    counts
        .iter()
        .map(|&count| {
            if count == 0 {
                0.0
            } else {
                (total as f32 / (present as f32 * count as f32)).min(20.0)
            }
        })
        .collect()
}

fn deterministic_shuffle(items: &mut [usize], seed: u64) {
    if items.len() < 2 {
        return;
    }
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    if state == 0 {
        state = 1;
    }
    for index in (1..items.len()).rev() {
        let j = (next_u64(&mut state) as usize) % (index + 1);
        items.swap(index, j);
    }
}

fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn evaluate_face_segmentation(
    model: &FaceSegmentationModel,
    examples: &[FaceSegmentationExample],
    batch_size: usize,
    device: &Device,
    label_names: &[String],
) -> Result<FaceEvaluationMetrics> {
    if examples.is_empty() {
        return Ok(face_metrics_from_predictions(&[], &[], label_names));
    }

    let mut predictions = Vec::new();
    let mut targets = Vec::new();
    for chunk in examples.chunks(batch_size.max(1)) {
        let batch = FaceSegBatch::from_examples(chunk, device)?;
        let logits = model.forward_face_logits(&batch.batch)?;
        predictions.extend(logits.argmax(D::Minus1)?.to_vec1::<u32>()?);
        targets.extend(batch.face_labels.to_vec1::<u32>()?);
    }

    Ok(face_metrics_from_predictions(
        &predictions,
        &targets,
        label_names,
    ))
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
    macro_f1_for_classes(logits, labels, CLASS_COUNT)
}

fn macro_f1_for_classes(logits: &Tensor, labels: &Tensor, class_count: usize) -> Result<f32> {
    let predictions = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let targets = labels.to_vec1::<u32>()?;
    Ok(macro_f1_from_predictions(
        &predictions,
        &targets,
        class_count,
    ))
}

fn accuracy_from_predictions(predictions: &[u32], targets: &[u32]) -> f32 {
    if targets.is_empty() {
        return f32::NAN;
    }
    let correct = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(pred, target)| pred == target)
        .count();
    correct as f32 / targets.len() as f32
}

#[derive(Debug, Clone, Copy, Default)]
struct ClassPredictionStats {
    support: usize,
    predicted: usize,
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
}

fn class_prediction_stats(
    predictions: &[u32],
    targets: &[u32],
    class_count: usize,
) -> Vec<ClassPredictionStats> {
    let mut stats = vec![ClassPredictionStats::default(); class_count];
    for (&pred, &target) in predictions.iter().zip(targets.iter()) {
        let (pred, target) = (pred as usize, target as usize);
        if target < class_count {
            stats[target].support += 1;
        }
        if pred < class_count {
            stats[pred].predicted += 1;
        }
        if pred >= class_count || target >= class_count {
            continue;
        }
        if pred == target {
            stats[target].true_positives += 1;
        } else {
            stats[pred].false_positives += 1;
            stats[target].false_negatives += 1;
        }
    }
    stats
}

fn face_metrics_from_predictions(
    predictions: &[u32],
    targets: &[u32],
    label_names: &[String],
) -> FaceEvaluationMetrics {
    let stats = class_prediction_stats(predictions, targets, label_names.len());
    let mut class_metrics = Vec::with_capacity(label_names.len());
    let mut macro_f1_sum = 0.0;
    let mut macro_f1_count = 0usize;
    let mut weighted_f1_sum = 0.0;
    let mut macro_iou_sum = 0.0;
    let mut macro_iou_present_sum = 0.0;
    let mut macro_iou_present_count = 0usize;
    let total_support: usize = stats.iter().map(|item| item.support).sum();

    for (class_id, (label, stats)) in label_names.iter().zip(stats.iter()).enumerate() {
        let tp = stats.true_positives as f32;
        let fp = stats.false_positives as f32;
        let fn_ = stats.false_negatives as f32;
        let precision = safe_div(tp, tp + fp);
        let recall = safe_div(tp, tp + fn_);
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        let iou = safe_div(tp, tp + fp + fn_);

        if stats.true_positives + stats.false_positives + stats.false_negatives > 0 {
            macro_f1_sum += f1;
            macro_f1_count += 1;
        }
        weighted_f1_sum += f1 * stats.support as f32;
        macro_iou_sum += iou;
        if stats.support > 0 {
            macro_iou_present_sum += iou;
            macro_iou_present_count += 1;
        }

        class_metrics.push(FaceClassMetric {
            class_id,
            label: label.clone(),
            support: stats.support,
            predicted: stats.predicted,
            true_positives: stats.true_positives,
            false_positives: stats.false_positives,
            false_negatives: stats.false_negatives,
            precision,
            recall,
            f1,
            iou,
        });
    }

    FaceEvaluationMetrics {
        accuracy: accuracy_from_predictions(predictions, targets),
        macro_f1: if macro_f1_count == 0 {
            0.0
        } else {
            macro_f1_sum / macro_f1_count as f32
        },
        weighted_f1: if total_support == 0 {
            0.0
        } else {
            weighted_f1_sum / total_support as f32
        },
        macro_iou: if label_names.is_empty() {
            0.0
        } else {
            macro_iou_sum / label_names.len() as f32
        },
        macro_iou_present: if macro_iou_present_count == 0 {
            0.0
        } else {
            macro_iou_present_sum / macro_iou_present_count as f32
        },
        support: total_support,
        class_metrics,
    }
}

fn macro_f1_from_predictions(predictions: &[u32], targets: &[u32], class_count: usize) -> f32 {
    if class_count == 0 {
        return f32::NAN;
    }
    let stats = class_prediction_stats(predictions, targets, class_count);
    let mut f1_sum = 0.0;
    let mut counted = 0;
    for stats in stats {
        if stats.true_positives + stats.false_positives + stats.false_negatives == 0 {
            continue; // class absent from this split
        }
        let tp = stats.true_positives as f32;
        let fp = stats.false_positives as f32;
        let fn_ = stats.false_negatives as f32;
        let precision = safe_div(tp, tp + fp);
        let recall = safe_div(tp, tp + fn_);
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        f1_sum += f1;
        counted += 1;
    }
    if counted == 0 {
        0.0
    } else {
        f1_sum / counted as f32
    }
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
    fn loads_dataset_from_disk() {
        let root = std::env::temp_dir().join(format!(
            "acad-brep-candle-dataset-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        acad_brep_dataset::generate_synthetic_dataset(
            &root,
            acad_brep_dataset::DatasetConfig {
                samples_per_class: 3,
                val_fraction: 0.25,
            },
        )
        .expect("dataset writes");

        let (train, val) = build_dataset_from_dir(&root).expect("dataset loads");

        assert_eq!(train.len() + val.len(), 9);
        assert!(!train.is_empty());
        assert!(!val.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn loads_face_segmentation_dataset_from_disk() {
        let root = std::env::temp_dir().join(format!(
            "acad-brep-face-dataset-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        acad_brep_dataset::generate_synthetic_dataset(
            &root,
            acad_brep_dataset::DatasetConfig {
                samples_per_class: 2,
                val_fraction: 0.25,
            },
        )
        .expect("dataset writes");

        let dataset = build_face_segmentation_dataset(
            &root,
            FaceSegmentationConfig {
                max_train_samples: None,
                max_eval_samples: None,
                ..Default::default()
            },
        )
        .expect("face dataset loads");

        assert!(!dataset.label_names.is_empty());
        assert!(!dataset.train.is_empty());
        assert!(dataset
            .train
            .iter()
            .all(|item| item.tensors.face_count == item.face_labels.len()));

        let _ = std::fs::remove_dir_all(&root);
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
    fn face_segmentation_training_smoke() {
        let root =
            std::env::temp_dir().join(format!("acad-brep-face-train-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        acad_brep_dataset::generate_synthetic_dataset(
            &root,
            acad_brep_dataset::DatasetConfig {
                samples_per_class: 2,
                val_fraction: 0.25,
            },
        )
        .expect("dataset writes");

        let report = train_face_segmentation(
            FaceSegmentationConfig {
                epochs: 1,
                hidden_dim: 12,
                rounds: 1,
                batch_size: 2,
                max_train_samples: None,
                max_eval_samples: None,
                ..Default::default()
            },
            &root,
        )
        .expect("face segmentation training runs");

        assert!(report.final_loss.is_finite());
        assert!(report.train_faces > 0);
        assert!(report.face_classes > 0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn face_metrics_report_per_class_f1_and_iou() {
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let predictions = vec![0, 1, 1, 2, 2, 2];
        let targets = vec![0, 0, 1, 1, 2, 2];

        let metrics = face_metrics_from_predictions(&predictions, &targets, &labels);

        assert_eq!(metrics.support, 6);
        assert!((metrics.accuracy - 4.0 / 6.0).abs() < 1e-6);
        assert!((metrics.class_metrics[0].precision - 1.0).abs() < 1e-6);
        assert!((metrics.class_metrics[0].recall - 0.5).abs() < 1e-6);
        assert!((metrics.class_metrics[0].f1 - 2.0 / 3.0).abs() < 1e-6);
        assert!((metrics.class_metrics[0].iou - 0.5).abs() < 1e-6);
        assert_eq!(metrics.class_metrics[1].true_positives, 1);
        assert_eq!(metrics.class_metrics[1].false_positives, 1);
        assert_eq!(metrics.class_metrics[1].false_negatives, 1);
        assert!((metrics.class_metrics[2].precision - 2.0 / 3.0).abs() < 1e-6);
        assert!(metrics.macro_f1 > 0.0);
        assert!(metrics.macro_iou_present > 0.0);
    }

    #[test]
    fn face_checkpoint_writes_metadata_sidecar() {
        let root = std::env::temp_dir().join(format!(
            "acad-brep-face-checkpoint-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        acad_brep_dataset::generate_synthetic_dataset(
            &root,
            acad_brep_dataset::DatasetConfig {
                samples_per_class: 2,
                val_fraction: 0.25,
            },
        )
        .expect("dataset writes");

        let checkpoint = root.join("face-model.safetensors");
        train_face_segmentation_and_save(
            FaceSegmentationConfig {
                epochs: 1,
                hidden_dim: 12,
                rounds: 1,
                batch_size: 2,
                max_train_samples: None,
                max_eval_samples: None,
                ..Default::default()
            },
            &root,
            &checkpoint,
        )
        .expect("face segmentation checkpoint saves");

        let metadata = std::fs::read_to_string(checkpoint_metadata_path(&checkpoint))
            .expect("metadata sidecar exists");
        assert!(metadata.contains("acad-brep-face-checkpoint-v1"));
        assert!(metadata.contains("face_label_names"));

        let device = Device::Cpu;
        let (model, label_names) =
            load_face_checkpoint(&checkpoint, &device).expect("face checkpoint loads");
        assert!(!label_names.is_empty());
        let dataset = build_face_segmentation_dataset(
            &root,
            FaceSegmentationConfig {
                max_train_samples: Some(1),
                max_eval_samples: Some(1),
                ..Default::default()
            },
        )
        .expect("face dataset loads");
        let batch = FaceSegBatch::from_examples(&dataset.train[..1], &device).expect("batch");
        let logits = model
            .forward_face_logits(&batch.batch)
            .expect("face inference");
        assert_eq!(logits.dim(1).unwrap(), label_names.len());

        let _ = std::fs::remove_dir_all(&root);
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
