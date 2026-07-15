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
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use acad_brep_graph::{
    box_graph, cylinder_graph, plate_with_holes, BrepGraph, CurveKind, GraphError, GraphStats,
    SurfaceKind,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const DATASET_VERSION: &str = "acad-brep-dataset-v1";
pub const MANIFEST_FILE: &str = "manifest.jsonl";
pub const DATASET_FILE: &str = "dataset.json";
pub const HARNESS_FILE: &str = "harness.json";
pub const HARNESS_VERSION: &str = "acad-brep-dataset-harness-v1";
pub const FINGERPRINT_HASH_ALGORITHM: &str = "sha256";
pub const GRAPHS_DIR: &str = "graphs";
pub const LABELS_DIR: &str = "labels";

pub const CLASS_NAMES: [&str; 3] = ["box", "cylinder", "plate_with_holes"];

type LabelCounts = BTreeMap<String, usize>;

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

#[derive(Debug, Clone, PartialEq)]
pub struct HarnessConfig {
    pub validation_percent: u8,
    pub validation_seed: u64,
    pub rare_count_threshold: usize,
    pub rare_fraction_threshold: f32,
    pub split_file: Option<PathBuf>,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            validation_percent: 10,
            validation_seed: 42,
            rare_count_threshold: 1000,
            rare_fraction_threshold: 0.005,
            split_file: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetHarnessReport {
    pub harness_version: String,
    pub dataset_version: String,
    pub manifest_hash: String,
    pub manifest_hash_algorithm: String,
    pub split_file_hash: Option<String>,
    pub split_policy: HarnessSplitPolicy,
    pub record_counts_by_split: BTreeMap<String, usize>,
    pub face_label_counts_by_split: BTreeMap<String, BTreeMap<String, usize>>,
    pub edge_label_counts_by_split: BTreeMap<String, BTreeMap<String, usize>>,
    pub graph_face_stats_by_split: BTreeMap<String, CountStats>,
    pub rare_face_label_policy: RareLabelPolicy,
    pub rare_face_labels: Vec<String>,
    pub labels_missing_from_train_inner: Vec<String>,
    pub labels_missing_from_val_inner: Vec<String>,
    pub train_val_label_drift: LabelDriftReport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessSplitPolicy {
    pub validation_source: String,
    pub validation_percent: u8,
    pub validation_seed: u64,
    pub test_policy: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CountStats {
    pub count: usize,
    pub total: usize,
    pub min: usize,
    pub p50: usize,
    pub p90: usize,
    pub p95: usize,
    pub p99: usize,
    pub max: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RareLabelPolicy {
    pub count_lt: usize,
    pub fraction_lt: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelDriftReport {
    pub total_variation: Option<f32>,
    pub labels: BTreeMap<String, LabelDrift>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarnessRecordSplit {
    TrainInner,
    ValInner,
    TestFinal,
}

impl HarnessRecordSplit {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TrainInner => "train_inner",
            Self::ValInner => "val_inner",
            Self::TestFinal => "test_final",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelDrift {
    pub train_fraction: f32,
    pub val_fraction: f32,
    pub train_to_val_ratio: Option<f32>,
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

pub fn inspect_dataset_harness(
    root: impl AsRef<Path>,
    config: HarnessConfig,
) -> Result<DatasetHarnessReport, DatasetError> {
    let root = root.as_ref();
    let metadata = load_metadata(root)?;
    if metadata.version != DATASET_VERSION {
        return Err(DatasetError::InvalidDataset {
            reason: format!(
                "unsupported dataset version {:?}; expected {DATASET_VERSION:?}",
                metadata.version
            ),
        });
    }

    let records = load_manifest(root)?;
    let manifest_hash = hash_file_hex(root.join(MANIFEST_FILE))?;
    let split_file_hash = config
        .split_file
        .as_deref()
        .map(hash_file_hex)
        .transpose()?;
    let has_manifest_val = records
        .iter()
        .any(|record| record.split == DatasetSplit::Val);
    let has_manifest_test = records
        .iter()
        .any(|record| record.split == DatasetSplit::Test);
    let validation_source = if has_manifest_val {
        "manifest_val"
    } else {
        "hash_from_manifest_train"
    };

    let mut record_counts_by_split = BTreeMap::new();
    let mut face_label_counts_by_split: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut edge_label_counts_by_split: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut face_counts_by_split: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    for record in &records {
        let split = harness_split_name(record, has_manifest_val, &config);
        *record_counts_by_split.entry(split.clone()).or_insert(0) += 1;
        face_counts_by_split
            .entry(split.clone())
            .or_default()
            .push(record.stats.faces);

        let (face_counts, edge_counts) = record_label_counts(root, record)?;
        add_counts(
            face_label_counts_by_split.entry(split.clone()).or_default(),
            &face_counts,
        );
        add_counts(
            edge_label_counts_by_split.entry(split).or_default(),
            &edge_counts,
        );
    }

    let mut graph_face_stats_by_split = BTreeMap::new();
    for (split, counts) in face_counts_by_split {
        graph_face_stats_by_split.insert(split, count_stats(counts));
    }

    let train_counts = face_label_counts_by_split
        .get("train_inner")
        .cloned()
        .unwrap_or_default();
    let val_counts = face_label_counts_by_split
        .get("val_inner")
        .cloned()
        .unwrap_or_default();
    let global_counts = sum_nested_counts(&face_label_counts_by_split);
    let total_faces: usize = global_counts.values().sum();
    let rare_face_labels = rare_labels(
        &global_counts,
        total_faces,
        config.rare_count_threshold,
        config.rare_fraction_threshold,
    );
    let labels_missing_from_train_inner = missing_labels(&global_counts, &train_counts);
    let labels_missing_from_val_inner = missing_labels(&global_counts, &val_counts);
    let train_val_label_drift = label_drift_report(&train_counts, &val_counts);

    Ok(DatasetHarnessReport {
        harness_version: HARNESS_VERSION.to_string(),
        dataset_version: metadata.version,
        manifest_hash,
        manifest_hash_algorithm: FINGERPRINT_HASH_ALGORITHM.to_string(),
        split_file_hash,
        split_policy: HarnessSplitPolicy {
            validation_source: validation_source.to_string(),
            validation_percent: if has_manifest_val {
                0
            } else {
                config.validation_percent
            },
            validation_seed: config.validation_seed,
            test_policy: if has_manifest_test {
                "manifest_test_reserved".to_string()
            } else {
                "no_manifest_test".to_string()
            },
        },
        record_counts_by_split,
        face_label_counts_by_split,
        edge_label_counts_by_split,
        graph_face_stats_by_split,
        rare_face_label_policy: RareLabelPolicy {
            count_lt: config.rare_count_threshold,
            fraction_lt: config.rare_fraction_threshold,
        },
        rare_face_labels,
        labels_missing_from_train_inner,
        labels_missing_from_val_inner,
        train_val_label_drift,
    })
}

pub fn write_dataset_harness(
    root: impl AsRef<Path>,
    config: HarnessConfig,
    out: Option<&Path>,
) -> Result<DatasetHarnessReport, DatasetError> {
    let root = root.as_ref();
    let report = inspect_dataset_harness(root, config)?;
    let out = out
        .map(Path::to_path_buf)
        .unwrap_or_else(|| root.join(HARNESS_FILE));
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

pub fn manifest_hash(root: impl AsRef<Path>) -> Result<String, DatasetError> {
    hash_file_hex(root.as_ref().join(MANIFEST_FILE))
}

pub fn hash_strings_hex(values: &[String]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.as_bytes());
        hash.update([0]);
    }
    format!("{:x}", hash.finalize())
}

pub fn hash_file_hex(path: impl AsRef<Path>) -> Result<String, DatasetError> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
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

pub fn manifest_has_validation(records: &[DatasetRecord]) -> bool {
    records
        .iter()
        .any(|record| record.split == DatasetSplit::Val)
}

pub fn harness_record_split(
    record: &DatasetRecord,
    has_manifest_val: bool,
    config: &HarnessConfig,
) -> HarnessRecordSplit {
    match record.split {
        DatasetSplit::Train if !has_manifest_val => {
            if config.validation_percent > 0
                && id_percent(&record.id, config.validation_seed) < config.validation_percent
            {
                HarnessRecordSplit::ValInner
            } else {
                HarnessRecordSplit::TrainInner
            }
        }
        DatasetSplit::Train => HarnessRecordSplit::TrainInner,
        DatasetSplit::Val => HarnessRecordSplit::ValInner,
        DatasetSplit::Test => HarnessRecordSplit::TestFinal,
    }
}

fn harness_split_name(
    record: &DatasetRecord,
    has_manifest_val: bool,
    config: &HarnessConfig,
) -> String {
    harness_record_split(record, has_manifest_val, config)
        .as_str()
        .to_string()
}

fn record_label_counts(
    root: &Path,
    record: &DatasetRecord,
) -> Result<(LabelCounts, LabelCounts), DatasetError> {
    if !record.face_label_counts.is_empty() && !record.edge_label_counts.is_empty() {
        return Ok((
            record.face_label_counts.clone(),
            record.edge_label_counts.clone(),
        ));
    }

    let labels_json = fs::read_to_string(root.join(&record.labels_path))?;
    let labels: BrepSampleLabels = serde_json::from_str(&labels_json)?;
    Ok((
        count_strings(&labels.face_labels),
        count_strings(&labels.edge_labels),
    ))
}

fn count_stats(mut values: Vec<usize>) -> CountStats {
    if values.is_empty() {
        return CountStats {
            count: 0,
            total: 0,
            min: 0,
            p50: 0,
            p90: 0,
            p95: 0,
            p99: 0,
            max: 0,
        };
    }
    values.sort_unstable();
    let total = values.iter().sum();
    CountStats {
        count: values.len(),
        total,
        min: values[0],
        p50: percentile(&values, 50),
        p90: percentile(&values, 90),
        p95: percentile(&values, 95),
        p99: percentile(&values, 99),
        max: values[values.len() - 1],
    }
}

fn percentile(sorted: &[usize], percentile: usize) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) * percentile).div_ceil(100);
    sorted[index.min(sorted.len() - 1)]
}

fn sum_nested_counts(
    splits: &BTreeMap<String, BTreeMap<String, usize>>,
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for split_counts in splits.values() {
        add_counts(&mut counts, split_counts);
    }
    counts
}

fn rare_labels(
    counts: &BTreeMap<String, usize>,
    total: usize,
    count_threshold: usize,
    fraction_threshold: f32,
) -> Vec<String> {
    counts
        .iter()
        .filter_map(|(label, &count)| {
            let fraction = if total == 0 {
                0.0
            } else {
                count as f32 / total as f32
            };
            (count < count_threshold || fraction < fraction_threshold).then(|| label.clone())
        })
        .collect()
}

fn missing_labels(
    global_counts: &BTreeMap<String, usize>,
    split_counts: &BTreeMap<String, usize>,
) -> Vec<String> {
    global_counts
        .keys()
        .filter(|label| !split_counts.contains_key(*label))
        .cloned()
        .collect()
}

fn label_drift_report(
    train_counts: &BTreeMap<String, usize>,
    val_counts: &BTreeMap<String, usize>,
) -> LabelDriftReport {
    let train_total: usize = train_counts.values().sum();
    let val_total: usize = val_counts.values().sum();
    let mut labels = BTreeMap::new();
    let mut tv = 0.0f32;

    for label in train_counts.keys().chain(val_counts.keys()) {
        if labels.contains_key(label) {
            continue;
        }
        let train_fraction = fraction(*train_counts.get(label).unwrap_or(&0), train_total);
        let val_fraction = fraction(*val_counts.get(label).unwrap_or(&0), val_total);
        tv += (train_fraction - val_fraction).abs();
        labels.insert(
            label.clone(),
            LabelDrift {
                train_fraction,
                val_fraction,
                train_to_val_ratio: if val_fraction == 0.0 {
                    None
                } else {
                    Some(train_fraction / val_fraction)
                },
            },
        );
    }

    LabelDriftReport {
        total_variation: (train_total > 0 && val_total > 0).then_some(tv * 0.5),
        labels,
    }
}

fn fraction(count: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        count as f32 / total as f32
    }
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

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn id_percent(id: &str, seed: u64) -> u8 {
    let mut hash = fnv1a_update(FNV_OFFSET, &seed.to_le_bytes());
    hash = fnv1a_update(hash, id.as_bytes());
    (hash % 100) as u8
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
    InvalidDataset { reason: String },
    InvalidLabels { id: String, reason: String },
}

impl fmt::Display for DatasetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Json(error) => write!(formatter, "JSON error: {error}"),
            Self::Graph(error) => write!(formatter, "invalid BRep graph: {error}"),
            Self::InvalidDataset { reason } => write!(formatter, "invalid dataset: {reason}"),
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

    #[test]
    fn writes_dataset_harness_report() {
        let root =
            std::env::temp_dir().join(format!("acad-brep-harness-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        generate_synthetic_dataset(
            &root,
            DatasetConfig {
                samples_per_class: 4,
                val_fraction: 0.25,
            },
        )
        .expect("dataset writes");

        let report = write_dataset_harness(
            &root,
            HarnessConfig {
                rare_count_threshold: 2,
                rare_fraction_threshold: 0.01,
                ..Default::default()
            },
            None,
        )
        .expect("harness writes");

        assert_eq!(report.harness_version, HARNESS_VERSION);
        assert_eq!(report.dataset_version, DATASET_VERSION);
        assert_eq!(report.manifest_hash.len(), 64);
        assert_eq!(report.manifest_hash_algorithm, FINGERPRINT_HASH_ALGORITHM);
        assert_eq!(report.split_policy.validation_source, "manifest_val");
        assert!(report.record_counts_by_split.contains_key("train_inner"));
        assert!(report.record_counts_by_split.contains_key("val_inner"));
        assert!(report
            .train_val_label_drift
            .total_variation
            .is_some_and(|value| value >= 0.0));
        assert!(root.join(HARNESS_FILE).is_file());

        let loaded: DatasetHarnessReport =
            serde_json::from_str(&fs::read_to_string(root.join(HARNESS_FILE)).unwrap()).unwrap();
        assert_eq!(loaded.manifest_hash, report.manifest_hash);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn harness_hash_split_carves_validation_from_train_and_preserves_test() {
        let records = vec![
            DatasetRecord {
                id: "a".to_string(),
                split: DatasetSplit::Train,
                class_id: 0,
                class_name: "part".to_string(),
                graph_path: "graphs/a.json".to_string(),
                labels_path: "labels/a.json".to_string(),
                stats: GraphStats {
                    faces: 1,
                    edges: 0,
                    coedges: 0,
                    face_adjacencies: 0,
                },
                face_label_counts: BTreeMap::from([("rare".to_string(), 1)]),
                edge_label_counts: BTreeMap::from([("edge".to_string(), 0)]),
            },
            DatasetRecord {
                id: "b".to_string(),
                split: DatasetSplit::Train,
                class_id: 0,
                class_name: "part".to_string(),
                graph_path: "graphs/b.json".to_string(),
                labels_path: "labels/b.json".to_string(),
                stats: GraphStats {
                    faces: 3,
                    edges: 0,
                    coedges: 0,
                    face_adjacencies: 0,
                },
                face_label_counts: BTreeMap::from([("common".to_string(), 3)]),
                edge_label_counts: BTreeMap::from([("edge".to_string(), 0)]),
            },
            DatasetRecord {
                id: "c".to_string(),
                split: DatasetSplit::Test,
                class_id: 0,
                class_name: "part".to_string(),
                graph_path: "graphs/c.json".to_string(),
                labels_path: "labels/c.json".to_string(),
                stats: GraphStats {
                    faces: 2,
                    edges: 0,
                    coedges: 0,
                    face_adjacencies: 0,
                },
                face_label_counts: BTreeMap::from([("common".to_string(), 2)]),
                edge_label_counts: BTreeMap::from([("edge".to_string(), 0)]),
            },
        ];
        let config = HarnessConfig {
            validation_percent: 100,
            validation_seed: 7,
            rare_count_threshold: 2,
            rare_fraction_threshold: 0.0,
            split_file: None,
        };

        assert_eq!(harness_split_name(&records[0], false, &config), "val_inner");
        assert_eq!(
            harness_split_name(&records[2], false, &config),
            "test_final"
        );

        let mut split_counts = BTreeMap::new();
        for record in &records {
            let split = harness_split_name(record, false, &config);
            let (face_counts, _) = record_label_counts(Path::new("."), record).unwrap();
            add_counts(split_counts.entry(split).or_default(), &face_counts);
        }
        let global = sum_nested_counts(&split_counts);
        assert_eq!(rare_labels(&global, 6, 2, 0.0), vec!["rare"]);

        let empty = BTreeMap::new();
        let drift = label_drift_report(
            split_counts.get("train_inner").unwrap_or(&empty),
            split_counts.get("val_inner").unwrap_or(&empty),
        );
        assert!(drift.total_variation.is_none());
    }
}
