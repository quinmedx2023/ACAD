//! On-disk BRep dataset format.
//!
//! The dataset is intentionally plain JSON so Rust, Python, and future
//! Truck/OCCT importers can all write the same format:
//!
//! ```text
//! dataset.json
//! manifest.jsonl
//! graphs/<sample_id>.json
//! labels/<sample_id>.json
//! ```

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use acad_brep_graph::{
    box_graph, cylinder_graph, plate_with_holes, BrepGraph, CurveKind, GraphError, GraphStats,
    SurfaceKind,
};
use serde::{Deserialize, Serialize};

pub const DATASET_VERSION: &str = "acad-brep-dataset-v1";
pub const MANIFEST_FILE: &str = "manifest.jsonl";
pub const DATASET_FILE: &str = "dataset.json";
pub const GRAPHS_DIR: &str = "graphs";
pub const LABELS_DIR: &str = "labels";

pub const CLASS_NAMES: [&str; 3] = ["box", "cylinder", "plate_with_holes"];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DatasetConfig {
    pub samples_per_class: usize,
    pub val_fraction: f32,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            samples_per_class: 32,
            val_fraction: 0.25,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub version: String,
    pub records: usize,
    pub samples_per_class: usize,
    pub val_fraction: String,
    pub classes: Vec<String>,
    pub face_label_set: Vec<String>,
    pub edge_label_set: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSplit {
    Train,
    Val,
    Test,
}

impl DatasetSplit {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Val => "val",
            Self::Test => "test",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetRecord {
    pub id: String,
    pub split: DatasetSplit,
    pub class_id: u32,
    pub class_name: String,
    pub graph_path: String,
    pub labels_path: String,
    pub stats: GraphStats,
    #[serde(default)]
    pub face_label_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub edge_label_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrepSampleLabels {
    pub graph_class_id: u32,
    pub graph_class_name: String,
    pub face_labels: Vec<String>,
    pub edge_labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BrepDatasetSample {
    pub record: DatasetRecord,
    pub graph: BrepGraph,
    pub labels: BrepSampleLabels,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub metadata: DatasetMetadata,
    pub split_counts: BTreeMap<String, usize>,
    pub class_counts: BTreeMap<String, usize>,
    pub face_label_counts: BTreeMap<String, usize>,
    pub edge_label_counts: BTreeMap<String, usize>,
}

pub fn generate_synthetic_dataset(
    root: impl AsRef<Path>,
    config: DatasetConfig,
) -> Result<DatasetSummary, DatasetError> {
    let root = root.as_ref();
    let graphs_dir = root.join(GRAPHS_DIR);
    let labels_dir = root.join(LABELS_DIR);
    fs::create_dir_all(&graphs_dir)?;
    fs::create_dir_all(&labels_dir)?;

    let mut records = Vec::new();
    let mut face_label_set = BTreeMap::new();
    let mut edge_label_set = BTreeMap::new();
    let stride = validation_stride(config.val_fraction);

    for (class_id, &class_name) in CLASS_NAMES.iter().enumerate() {
        for index in 0..config.samples_per_class {
            let sample_index = class_id * config.samples_per_class + index;
            let split = if sample_index % stride == stride - 1 {
                DatasetSplit::Val
            } else {
                DatasetSplit::Train
            };
            let id = format!("{class_name}_{index:06}");
            let graph = synthetic_graph(class_id as u32, index, config.samples_per_class);
            let labels = label_graph(class_id as u32, class_name, &graph);
            for label in &labels.face_labels {
                face_label_set.insert(label.clone(), ());
            }
            for label in &labels.edge_labels {
                edge_label_set.insert(label.clone(), ());
            }

            let stats = graph.validate()?;
            if labels.face_labels.len() != stats.faces {
                return Err(DatasetError::InvalidLabels {
                    id,
                    reason: "face label count does not match graph".to_string(),
                });
            }
            if labels.edge_labels.len() != stats.edges {
                return Err(DatasetError::InvalidLabels {
                    id,
                    reason: "edge label count does not match graph".to_string(),
                });
            }

            let graph_path = format!("{GRAPHS_DIR}/{id}.json");
            let labels_path = format!("{LABELS_DIR}/{id}.json");
            fs::write(root.join(&graph_path), graph.to_json()?)?;
            fs::write(
                root.join(&labels_path),
                serde_json::to_string_pretty(&labels)?,
            )?;

            records.push(DatasetRecord {
                id,
                split,
                class_id: class_id as u32,
                class_name: class_name.to_string(),
                graph_path,
                labels_path,
                stats,
                face_label_counts: count_strings(&labels.face_labels),
                edge_label_counts: count_strings(&labels.edge_labels),
            });
        }
    }

    let metadata = DatasetMetadata {
        version: DATASET_VERSION.to_string(),
        records: records.len(),
        samples_per_class: config.samples_per_class,
        val_fraction: format!("{:.6}", config.val_fraction),
        classes: CLASS_NAMES.iter().map(|name| (*name).to_string()).collect(),
        face_label_set: face_label_set.into_keys().collect(),
        edge_label_set: edge_label_set.into_keys().collect(),
    };

    fs::write(
        root.join(DATASET_FILE),
        serde_json::to_string_pretty(&metadata)?,
    )?;
    write_manifest(root.join(MANIFEST_FILE), &records)?;
    summarize_records(root, metadata, records)
}

pub fn load_metadata(root: impl AsRef<Path>) -> Result<DatasetMetadata, DatasetError> {
    let json = fs::read_to_string(root.as_ref().join(DATASET_FILE))?;
    Ok(serde_json::from_str(&json)?)
}

pub fn load_manifest(root: impl AsRef<Path>) -> Result<Vec<DatasetRecord>, DatasetError> {
    let file = fs::File::open(root.as_ref().join(MANIFEST_FILE))?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        records.push(serde_json::from_str(line)?);
    }
    Ok(records)
}

pub fn load_samples(root: impl AsRef<Path>) -> Result<Vec<BrepDatasetSample>, DatasetError> {
    let root = root.as_ref();
    let records = load_manifest(root)?;
    records
        .into_iter()
        .map(|record| {
            let graph_json = fs::read_to_string(root.join(&record.graph_path))?;
            let labels_json = fs::read_to_string(root.join(&record.labels_path))?;
            let graph = BrepGraph::from_json(&graph_json)?;
            let labels: BrepSampleLabels = serde_json::from_str(&labels_json)?;
            graph.validate()?;
            if labels.graph_class_id != record.class_id {
                return Err(DatasetError::InvalidLabels {
                    id: record.id.clone(),
                    reason: "graph class id does not match manifest".to_string(),
                });
            }
            if labels.face_labels.len() != graph.faces.len() {
                return Err(DatasetError::InvalidLabels {
                    id: record.id.clone(),
                    reason: "face label count does not match graph".to_string(),
                });
            }
            if labels.edge_labels.len() != graph.edges.len() {
                return Err(DatasetError::InvalidLabels {
                    id: record.id.clone(),
                    reason: "edge label count does not match graph".to_string(),
                });
            }
            Ok(BrepDatasetSample {
                record,
                graph,
                labels,
            })
        })
        .collect()
}

pub fn summarize_dataset(root: impl AsRef<Path>) -> Result<DatasetSummary, DatasetError> {
    let root = root.as_ref();
    let metadata = load_metadata(root)?;
    let records = load_manifest(root)?;
    summarize_records(root, metadata, records)
}

fn summarize_records(
    root: &Path,
    metadata: DatasetMetadata,
    records: Vec<DatasetRecord>,
) -> Result<DatasetSummary, DatasetError> {
    let mut split_counts = BTreeMap::new();
    let mut class_counts = BTreeMap::new();
    let mut face_label_counts = BTreeMap::new();
    let mut edge_label_counts = BTreeMap::new();

    for record in &records {
        *split_counts
            .entry(record.split.as_str().to_string())
            .or_insert(0) += 1;
        *class_counts.entry(record.class_name.clone()).or_insert(0) += 1;

        if !record.face_label_counts.is_empty() && !record.edge_label_counts.is_empty() {
            add_counts(&mut face_label_counts, &record.face_label_counts);
            add_counts(&mut edge_label_counts, &record.edge_label_counts);
        } else {
            let labels_json = fs::read_to_string(root.join(&record.labels_path))?;
            let labels: BrepSampleLabels = serde_json::from_str(&labels_json)?;
            for label in labels.face_labels {
                *face_label_counts.entry(label).or_insert(0) += 1;
            }
            for label in labels.edge_labels {
                *edge_label_counts.entry(label).or_insert(0) += 1;
            }
        }
    }

    Ok(DatasetSummary {
        metadata,
        split_counts,
        class_counts,
        face_label_counts,
        edge_label_counts,
    })
}

fn write_manifest(path: PathBuf, records: &[DatasetRecord]) -> Result<(), DatasetError> {
    let mut file = fs::File::create(path)?;
    for record in records {
        writeln!(file, "{}", serde_json::to_string(record)?)?;
    }
    Ok(())
}

fn count_strings(values: &[String]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for value in values {
        *counts.entry(value.clone()).or_insert(0) += 1;
    }
    counts
}

fn add_counts(target: &mut BTreeMap<String, usize>, counts: &BTreeMap<String, usize>) {
    for (label, count) in counts {
        *target.entry(label.clone()).or_insert(0) += count;
    }
}

fn validation_stride(val_fraction: f32) -> usize {
    if val_fraction <= 0.0 {
        usize::MAX
    } else {
        (1.0 / val_fraction).round().max(2.0) as usize
    }
}

fn synthetic_graph(class_id: u32, index: usize, samples_per_class: usize) -> BrepGraph {
    let t = index as f32 / samples_per_class.max(1) as f32;
    match class_id {
        0 => box_graph(1.0 + t, 0.8 + 0.5 * t, 0.5 + 0.25 * t),
        1 => cylinder_graph(0.25 + 0.2 * t, 0.8 + t),
        2 => plate_with_holes(
            1.5 + t,
            1.0 + 0.5 * t,
            0.12 + 0.1 * t,
            0.12 + 0.05 * t,
            1 + (index % 3),
        ),
        _ => unreachable!("unknown synthetic class id"),
    }
}

fn label_graph(class_id: u32, class_name: &str, graph: &BrepGraph) -> BrepSampleLabels {
    BrepSampleLabels {
        graph_class_id: class_id,
        graph_class_name: class_name.to_string(),
        face_labels: graph
            .faces
            .iter()
            .map(|face| label_face(class_name, face.surface, face.normal.z))
            .collect(),
        edge_labels: graph
            .edges
            .iter()
            .map(|edge| label_edge(class_name, edge.curve))
            .collect(),
    }
}

fn label_face(class_name: &str, surface: SurfaceKind, normal_z: f32) -> String {
    match (class_name, surface) {
        ("cylinder", SurfaceKind::Cylinder) => "cylinder_side".to_string(),
        ("plate_with_holes", SurfaceKind::Cylinder) => "hole_wall".to_string(),
        (_, SurfaceKind::Cylinder) => "cylindrical_face".to_string(),
        ("cylinder", SurfaceKind::Plane) if normal_z > 0.5 => "top_cap".to_string(),
        ("cylinder", SurfaceKind::Plane) if normal_z < -0.5 => "bottom_cap".to_string(),
        ("plate_with_holes", SurfaceKind::Plane) if normal_z > 0.5 => "top".to_string(),
        ("plate_with_holes", SurfaceKind::Plane) if normal_z < -0.5 => "bottom".to_string(),
        ("plate_with_holes", SurfaceKind::Plane) => "outer_side".to_string(),
        ("box", SurfaceKind::Plane) if normal_z > 0.5 => "top".to_string(),
        ("box", SurfaceKind::Plane) if normal_z < -0.5 => "bottom".to_string(),
        ("box", SurfaceKind::Plane) => "side".to_string(),
        _ => "other_face".to_string(),
    }
}

fn label_edge(class_name: &str, curve: CurveKind) -> String {
    match (class_name, curve) {
        ("plate_with_holes", CurveKind::Circle) => "hole_edge".to_string(),
        ("cylinder", CurveKind::Circle) => "cap_edge".to_string(),
        (_, CurveKind::Circle) => "smooth_circle".to_string(),
        (_, CurveKind::Line) => "convex_line".to_string(),
        _ => "other_edge".to_string(),
    }
}

#[derive(Debug)]
pub enum DatasetError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Graph(GraphError),
    InvalidLabels { id: String, reason: String },
}

impl fmt::Display for DatasetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Json(error) => write!(formatter, "JSON error: {error}"),
            Self::Graph(error) => write!(formatter, "invalid BRep graph: {error}"),
            Self::InvalidLabels { id, reason } => {
                write!(formatter, "invalid labels for sample {id}: {reason}")
            }
        }
    }
}

impl Error for DatasetError {}

impl From<std::io::Error> for DatasetError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for DatasetError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<GraphError> for DatasetError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_split_preserves_test() {
        let json = serde_json::to_string(&DatasetSplit::Test).expect("split serializes");
        assert_eq!(json, "\"test\"");
        let split: DatasetSplit = serde_json::from_str(&json).expect("split deserializes");
        assert_eq!(split, DatasetSplit::Test);
        assert_eq!(split.as_str(), "test");
    }

    #[test]
    fn generates_loads_and_summarizes_dataset() {
        let root =
            std::env::temp_dir().join(format!("acad-brep-dataset-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);

        let summary = generate_synthetic_dataset(
            &root,
            DatasetConfig {
                samples_per_class: 3,
                val_fraction: 0.25,
            },
        )
        .expect("dataset writes");
        assert_eq!(summary.metadata.records, 9);
        assert_eq!(summary.class_counts["box"], 3);

        let samples = load_samples(&root).expect("dataset loads");
        assert_eq!(samples.len(), 9);
        assert!(samples
            .iter()
            .all(|sample| sample.labels.face_labels.len() == sample.graph.faces.len()));
        assert!(samples
            .iter()
            .all(|sample| sample.labels.edge_labels.len() == sample.graph.edges.len()));

        let loaded_summary = summarize_dataset(&root).expect("dataset summarizes");
        assert_eq!(loaded_summary.metadata.records, 9);

        let _ = fs::remove_dir_all(&root);
    }
}
