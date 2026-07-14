//! Deterministic BRep graph encoding boundary.
//!
//! Two encoders live here, both free of any ML runtime:
//!
//! * [`DeterministicGraphEncoder`] pools hand-crafted features into a fixed
//!   vector (used by the CLI smoke test and as a lightweight baseline).
//! * [`GraphTensorizer`] emits the ragged, channel-major arrays a hybrid
//!   UV-Net/BRepNet encoder consumes: per-face UV grids, per-edge curve grids,
//!   categorical/scalar node features, and coedge topology index arrays.

use std::error::Error;
use std::fmt;

use acad_brep_graph::{BrepGraph, Convexity, CurveKind, GraphError, SurfaceKind};

pub const FACE_FEATURE_DIM: usize = SurfaceKind::COUNT + 7;
pub const EDGE_FEATURE_DIM: usize = CurveKind::COUNT + Convexity::COUNT + 4;
pub const GRAPH_FEATURE_DIM: usize = FACE_FEATURE_DIM + EDGE_FEATURE_DIM + 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncoderConfig {
    pub length_scale: f32,
    pub area_scale: f32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            length_scale: 100.0,
            area_scale: 10_000.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphEncoding {
    pub face_count: usize,
    pub edge_count: usize,
    pub face_feature_dim: usize,
    pub edge_feature_dim: usize,
    pub face_features: Vec<f32>,
    pub edge_features: Vec<f32>,
    pub face_edge_face: Vec<[u32; 3]>,
}

impl GraphEncoding {
    pub fn validate_shape(&self) -> Result<(), EncodingShapeError> {
        let expected_face_values = self.face_count * self.face_feature_dim;
        if self.face_features.len() != expected_face_values {
            return Err(EncodingShapeError::FaceFeatures {
                expected: expected_face_values,
                actual: self.face_features.len(),
            });
        }

        let expected_edge_values = self.edge_count * self.edge_feature_dim;
        if self.edge_features.len() != expected_edge_values {
            return Err(EncodingShapeError::EdgeFeatures {
                expected: expected_edge_values,
                actual: self.edge_features.len(),
            });
        }

        Ok(())
    }
}

pub fn pooled_graph_features(encoding: &GraphEncoding) -> Vec<f32> {
    let mut features = Vec::with_capacity(GRAPH_FEATURE_DIM);
    push_mean_features(
        &mut features,
        &encoding.face_features,
        encoding.face_count,
        encoding.face_feature_dim,
    );
    push_mean_features(
        &mut features,
        &encoding.edge_features,
        encoding.edge_count,
        encoding.edge_feature_dim,
    );

    let adjacency_count = encoding.face_edge_face.len();
    features.push(encoding.face_count as f32 / 128.0);
    features.push(encoding.edge_count as f32 / 256.0);
    features.push(adjacency_count as f32 / 512.0);
    features.push(if encoding.face_count == 0 {
        0.0
    } else {
        adjacency_count as f32 / encoding.face_count as f32
    });

    features
}

pub trait BrepGraphEncoder {
    fn encode(&self, graph: &BrepGraph) -> Result<GraphEncoding, EncoderError>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DeterministicGraphEncoder {
    config: EncoderConfig,
}

impl DeterministicGraphEncoder {
    pub fn new(config: EncoderConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> EncoderConfig {
        self.config
    }
}

impl BrepGraphEncoder for DeterministicGraphEncoder {
    fn encode(&self, graph: &BrepGraph) -> Result<GraphEncoding, EncoderError> {
        graph.validate().map_err(EncoderError::InvalidGraph)?;

        let mut face_features = Vec::with_capacity(graph.faces.len() * FACE_FEATURE_DIM);
        for face in &graph.faces {
            push_one_hot(&mut face_features, SurfaceKind::COUNT, face.surface.index());
            face_features.push(scale(face.area, self.config.area_scale));
            face_features.push(scale(face.centroid.x, self.config.length_scale));
            face_features.push(scale(face.centroid.y, self.config.length_scale));
            face_features.push(scale(face.centroid.z, self.config.length_scale));
            face_features.push(face.normal.x);
            face_features.push(face.normal.y);
            face_features.push(face.normal.z);
        }

        let mut edge_features = Vec::with_capacity(graph.edges.len() * EDGE_FEATURE_DIM);
        for edge in &graph.edges {
            push_one_hot(&mut edge_features, CurveKind::COUNT, edge.curve.index());
            push_one_hot(&mut edge_features, Convexity::COUNT, edge.convexity.index());
            edge_features.push(scale(edge.length, self.config.length_scale));
            edge_features.push(scale(edge.midpoint.x, self.config.length_scale));
            edge_features.push(scale(edge.midpoint.y, self.config.length_scale));
            edge_features.push(scale(edge.midpoint.z, self.config.length_scale));
        }

        let face_edge_face = graph
            .face_adjacency
            .iter()
            .map(|adjacency| {
                Ok([
                    to_u32(adjacency.left)?,
                    to_u32(adjacency.edge)?,
                    to_u32(adjacency.right)?,
                ])
            })
            .collect::<Result<Vec<_>, EncoderError>>()?;

        let encoding = GraphEncoding {
            face_count: graph.faces.len(),
            edge_count: graph.edges.len(),
            face_feature_dim: FACE_FEATURE_DIM,
            edge_feature_dim: EDGE_FEATURE_DIM,
            face_features,
            edge_features,
            face_edge_face,
        };
        encoding
            .validate_shape()
            .map_err(EncoderError::InvalidEncodingShape)?;

        Ok(encoding)
    }
}

// ---------------------------------------------------------------------------
// Geometry-aware tensorization for the hybrid UV-Net/BRepNet encoder.
// ---------------------------------------------------------------------------

/// Point (3) + normal (3) + trimming mask (1) channels per face UV sample.
pub const FACE_GRID_CHANNELS: usize = 7;
/// Point (3) + tangent (3) channels per edge curve sample.
pub const EDGE_GRID_CHANNELS: usize = 6;
/// One-hot surface kind.
pub const FACE_CATEGORICAL_DIM: usize = SurfaceKind::COUNT;
/// area, centroid.{x,y,z}, normal.{x,y,z}.
pub const FACE_SCALAR_DIM: usize = 7;
/// One-hot curve kind + one-hot convexity.
pub const EDGE_CATEGORICAL_DIM: usize = CurveKind::COUNT + Convexity::COUNT;
/// length, midpoint.{x,y,z}.
pub const EDGE_SCALAR_DIM: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorizerConfig {
    pub length_scale: f32,
    pub area_scale: f32,
    pub uv_res: usize,
    pub curve_res: usize,
}

impl Default for TensorizerConfig {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            area_scale: 1.0,
            uv_res: acad_brep_graph::DEFAULT_UV_RES,
            curve_res: acad_brep_graph::DEFAULT_CURVE_RES,
        }
    }
}

/// Ragged, framework-neutral tensors for a single BRep graph. Channel-major grid
/// layouts (`[node, channel, ...]`) map directly onto a conv front-end.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphTensors {
    pub face_count: usize,
    pub edge_count: usize,
    pub coedge_count: usize,
    pub uv_res: usize,
    pub curve_res: usize,
    /// `[face_count, FACE_CATEGORICAL_DIM]`
    pub face_categorical: Vec<f32>,
    /// `[face_count, FACE_SCALAR_DIM]`
    pub face_scalar: Vec<f32>,
    /// `[face_count, FACE_GRID_CHANNELS, uv_res, uv_res]`
    pub face_grid: Vec<f32>,
    /// `[edge_count, EDGE_CATEGORICAL_DIM]`
    pub edge_categorical: Vec<f32>,
    /// `[edge_count, EDGE_SCALAR_DIM]`
    pub edge_scalar: Vec<f32>,
    /// `[edge_count, EDGE_GRID_CHANNELS, curve_res]`
    pub edge_grid: Vec<f32>,
    /// `[coedge_count]` face incident to each coedge.
    pub coedge_face: Vec<u32>,
    /// `[coedge_count]` edge incident to each coedge.
    pub coedge_edge: Vec<u32>,
    /// `[coedge_count]` mate coedge; self-index when a coedge has no mate.
    pub coedge_mate: Vec<u32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GraphTensorizer {
    config: TensorizerConfig,
}

impl GraphTensorizer {
    pub fn new(config: TensorizerConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> TensorizerConfig {
        self.config
    }

    pub fn tensorize(&self, graph: &BrepGraph) -> Result<GraphTensors, EncoderError> {
        graph.validate().map_err(EncoderError::InvalidGraph)?;
        let uv_res = self.config.uv_res;
        let curve_res = self.config.curve_res;
        let uv_samples = uv_res * uv_res;

        let mut face_categorical = Vec::with_capacity(graph.faces.len() * FACE_CATEGORICAL_DIM);
        let mut face_scalar = Vec::with_capacity(graph.faces.len() * FACE_SCALAR_DIM);
        let mut face_grid = Vec::with_capacity(graph.faces.len() * FACE_GRID_CHANNELS * uv_samples);
        for (index, face) in graph.faces.iter().enumerate() {
            push_one_hot(
                &mut face_categorical,
                SurfaceKind::COUNT,
                face.surface.index(),
            );
            face_scalar.push(scale(face.area, self.config.area_scale));
            face_scalar.push(scale(face.centroid.x, self.config.length_scale));
            face_scalar.push(scale(face.centroid.y, self.config.length_scale));
            face_scalar.push(scale(face.centroid.z, self.config.length_scale));
            face_scalar.push(face.normal.x);
            face_scalar.push(face.normal.y);
            face_scalar.push(face.normal.z);

            match &face.geometry {
                Some(geometry) => {
                    if geometry.u_res != uv_res || geometry.v_res != uv_res {
                        return Err(EncoderError::GridResolutionMismatch {
                            node: index,
                            expected: uv_res,
                            actual_u: geometry.u_res,
                            actual_v: geometry.v_res,
                        });
                    }
                    push_channel(&mut face_grid, &geometry.points, |p| {
                        [
                            scale(p.x, self.config.length_scale),
                            scale(p.y, self.config.length_scale),
                            scale(p.z, self.config.length_scale),
                        ]
                    });
                    push_channel(&mut face_grid, &geometry.normals, |n| [n.x, n.y, n.z]);
                    face_grid.extend_from_slice(&geometry.mask);
                }
                None => {
                    face_grid.extend(std::iter::repeat(0.0).take(FACE_GRID_CHANNELS * uv_samples));
                }
            }
        }

        let mut edge_categorical = Vec::with_capacity(graph.edges.len() * EDGE_CATEGORICAL_DIM);
        let mut edge_scalar = Vec::with_capacity(graph.edges.len() * EDGE_SCALAR_DIM);
        let mut edge_grid = Vec::with_capacity(graph.edges.len() * EDGE_GRID_CHANNELS * curve_res);
        for (index, edge) in graph.edges.iter().enumerate() {
            push_one_hot(&mut edge_categorical, CurveKind::COUNT, edge.curve.index());
            push_one_hot(
                &mut edge_categorical,
                Convexity::COUNT,
                edge.convexity.index(),
            );
            edge_scalar.push(scale(edge.length, self.config.length_scale));
            edge_scalar.push(scale(edge.midpoint.x, self.config.length_scale));
            edge_scalar.push(scale(edge.midpoint.y, self.config.length_scale));
            edge_scalar.push(scale(edge.midpoint.z, self.config.length_scale));

            match &edge.geometry {
                Some(geometry) => {
                    if geometry.res != curve_res {
                        return Err(EncoderError::GridResolutionMismatch {
                            node: index,
                            expected: curve_res,
                            actual_u: geometry.res,
                            actual_v: geometry.res,
                        });
                    }
                    push_channel(&mut edge_grid, &geometry.points, |p| {
                        [
                            scale(p.x, self.config.length_scale),
                            scale(p.y, self.config.length_scale),
                            scale(p.z, self.config.length_scale),
                        ]
                    });
                    push_channel(&mut edge_grid, &geometry.tangents, |t| [t.x, t.y, t.z]);
                }
                None => {
                    edge_grid.extend(std::iter::repeat(0.0).take(EDGE_GRID_CHANNELS * curve_res));
                }
            }
        }

        let coedge_count = graph.coedges.len();
        let mut coedge_face = Vec::with_capacity(coedge_count);
        let mut coedge_edge = Vec::with_capacity(coedge_count);
        let mut coedge_mate = Vec::with_capacity(coedge_count);
        for (index, coedge) in graph.coedges.iter().enumerate() {
            coedge_face.push(to_u32(coedge.face)?);
            coedge_edge.push(to_u32(coedge.edge)?);
            coedge_mate.push(to_u32(coedge.mate.unwrap_or(index))?);
        }

        Ok(GraphTensors {
            face_count: graph.faces.len(),
            edge_count: graph.edges.len(),
            coedge_count,
            uv_res,
            curve_res,
            face_categorical,
            face_scalar,
            face_grid,
            edge_categorical,
            edge_scalar,
            edge_grid,
            coedge_face,
            coedge_edge,
            coedge_mate,
        })
    }
}

/// Push three interleaved channels (x, y, z), one full channel at a time, so the
/// grid ends up channel-major (`[..., channel, sample]`).
fn push_channel<T, F>(output: &mut Vec<f32>, values: &[T], project: F)
where
    F: Fn(&T) -> [f32; 3],
{
    for axis in 0..3 {
        for value in values {
            output.push(project(value)[axis]);
        }
    }
}

fn push_one_hot(values: &mut Vec<f32>, width: usize, active: usize) {
    for index in 0..width {
        values.push(if index == active { 1.0 } else { 0.0 });
    }
}

fn scale(value: f32, denominator: f32) -> f32 {
    if denominator == 0.0 {
        value
    } else {
        value / denominator
    }
}

fn push_mean_features(output: &mut Vec<f32>, values: &[f32], count: usize, dim: usize) {
    if count == 0 {
        output.extend(std::iter::repeat(0.0).take(dim));
        return;
    }

    let start_len = output.len();
    output.extend(std::iter::repeat(0.0).take(dim));
    for row in values.chunks_exact(dim) {
        for (index, value) in row.iter().enumerate() {
            output[start_len + index] += value;
        }
    }
    for value in &mut output[start_len..start_len + dim] {
        *value /= count as f32;
    }
}

fn to_u32(value: usize) -> Result<u32, EncoderError> {
    u32::try_from(value).map_err(|_| EncoderError::IndexOverflow { value })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncoderError {
    InvalidGraph(GraphError),
    InvalidEncodingShape(EncodingShapeError),
    IndexOverflow {
        value: usize,
    },
    GridResolutionMismatch {
        node: usize,
        expected: usize,
        actual_u: usize,
        actual_v: usize,
    },
}

impl fmt::Display for EncoderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(error) => write!(formatter, "invalid BRep graph: {error}"),
            Self::InvalidEncodingShape(error) => {
                write!(formatter, "invalid encoding shape: {error}")
            }
            Self::IndexOverflow { value } => {
                write!(formatter, "graph index {value} does not fit in u32")
            }
            Self::GridResolutionMismatch {
                node,
                expected,
                actual_u,
                actual_v,
            } => {
                write!(
                    formatter,
                    "node {node} grid resolution {actual_u}x{actual_v} does not match expected {expected}"
                )
            }
        }
    }
}

impl Error for EncoderError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodingShapeError {
    FaceFeatures { expected: usize, actual: usize },
    EdgeFeatures { expected: usize, actual: usize },
}

impl fmt::Display for EncodingShapeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FaceFeatures { expected, actual } => {
                write!(
                    formatter,
                    "face feature values: expected {expected}, got {actual}"
                )
            }
            Self::EdgeFeatures { expected, actual } => {
                write!(
                    formatter,
                    "edge feature values: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl Error for EncodingShapeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use acad_brep_graph::sample_box_graph;

    #[test]
    fn encodes_sample_box_with_stable_shapes() {
        let graph = sample_box_graph();
        let encoder = DeterministicGraphEncoder::default();

        let encoding = encoder.encode(&graph).expect("box should encode");

        assert_eq!(encoding.face_count, 6);
        assert_eq!(encoding.edge_count, 12);
        assert_eq!(encoding.face_feature_dim, FACE_FEATURE_DIM);
        assert_eq!(encoding.edge_feature_dim, EDGE_FEATURE_DIM);
        assert_eq!(encoding.face_features.len(), 6 * FACE_FEATURE_DIM);
        assert_eq!(encoding.edge_features.len(), 12 * EDGE_FEATURE_DIM);
        assert_eq!(encoding.face_edge_face.len(), 12);
        assert_eq!(encoding.face_features[SurfaceKind::Plane.index()], 1.0);
    }

    #[test]
    fn pooled_features_have_stable_shape() {
        let graph = sample_box_graph();
        let encoder = DeterministicGraphEncoder::default();
        let encoding = encoder.encode(&graph).expect("box should encode");

        let features = pooled_graph_features(&encoding);

        assert_eq!(features.len(), GRAPH_FEATURE_DIM);
    }

    #[test]
    fn tensorizer_emits_channel_major_grids() {
        let graph = sample_box_graph();
        let tensorizer = GraphTensorizer::default();
        let tensors = tensorizer.tensorize(&graph).expect("box should tensorize");

        let uv_samples = tensors.uv_res * tensors.uv_res;
        assert_eq!(tensors.face_count, 6);
        assert_eq!(tensors.edge_count, 12);
        assert_eq!(tensors.coedge_count, 24);
        assert_eq!(
            tensors.face_categorical.len(),
            tensors.face_count * FACE_CATEGORICAL_DIM
        );
        assert_eq!(
            tensors.face_scalar.len(),
            tensors.face_count * FACE_SCALAR_DIM
        );
        assert_eq!(
            tensors.face_grid.len(),
            tensors.face_count * FACE_GRID_CHANNELS * uv_samples
        );
        assert_eq!(
            tensors.edge_grid.len(),
            tensors.edge_count * EDGE_GRID_CHANNELS * tensors.curve_res
        );
        assert_eq!(tensors.coedge_face.len(), 24);
        assert_eq!(tensors.coedge_mate.len(), 24);
    }

    #[test]
    fn tensorizer_records_mates() {
        let graph = sample_box_graph();
        let tensors = GraphTensorizer::default()
            .tensorize(&graph)
            .expect("box should tensorize");
        // Every synthetic coedge is mated, so no entry should self-reference.
        for (index, &mate) in tensors.coedge_mate.iter().enumerate() {
            assert_ne!(mate as usize, index);
        }
    }
}
