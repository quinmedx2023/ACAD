//! Core BRep graph data model.
//!
//! This crate keeps the graph representation independent from any one CAD
//! kernel. Importers can map Truck, OCCT, STEP, or synthetic fixtures into this
//! structure before model encoding.
//!
//! In addition to topology (faces, edges, coedges, adjacency) the model carries
//! optional sampled *geometry*: a UV grid of points/normals per face and a 1D
//! grid of points/tangents per edge. This mirrors the UV-Net representation and
//! lets a neural encoder consume real surface/curve shape rather than only
//! hand-crafted scalars.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

pub type FaceId = usize;
pub type EdgeId = usize;
pub type CoedgeId = usize;

/// Default UV grid resolution used by the synthetic fixtures.
pub const DEFAULT_UV_RES: usize = 6;
/// Default curve grid resolution used by the synthetic fixtures.
pub const DEFAULT_CURVE_RES: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurfaceKind {
    Plane,
    Cylinder,
    Cone,
    Sphere,
    Torus,
    Nurbs,
    Unknown,
}

impl SurfaceKind {
    pub const COUNT: usize = 7;

    pub const fn index(self) -> usize {
        match self {
            Self::Plane => 0,
            Self::Cylinder => 1,
            Self::Cone => 2,
            Self::Sphere => 3,
            Self::Torus => 4,
            Self::Nurbs => 5,
            Self::Unknown => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveKind {
    Line,
    Circle,
    Ellipse,
    Spline,
    Unknown,
}

impl CurveKind {
    pub const COUNT: usize = 5;

    pub const fn index(self) -> usize {
        match self {
            Self::Line => 0,
            Self::Circle => 1,
            Self::Ellipse => 2,
            Self::Spline => 3,
            Self::Unknown => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Convexity {
    Concave,
    Convex,
    Smooth,
    Unknown,
}

impl Convexity {
    pub const COUNT: usize = 4;

    pub const fn index(self) -> usize {
        match self {
            Self::Concave => 0,
            Self::Convex => 1,
            Self::Smooth => 2,
            Self::Unknown => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Orientation {
    Forward,
    Reversed,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    pub fn scale(self, factor: f32) -> Self {
        Self::new(self.x * factor, self.y * factor, self.z * factor)
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Self {
        let length = self.length();
        if length <= f32::EPSILON {
            Self::new(0.0, 0.0, 1.0)
        } else {
            self.scale(1.0 / length)
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

/// Two orthonormal in-plane axes for a (near) unit normal.
fn tangent_basis(normal: Vec3) -> (Vec3, Vec3) {
    let normal = normal.normalized();
    let helper = if normal.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let tangent = helper.cross(normal).normalized();
    let bitangent = normal.cross(tangent).normalized();
    (tangent, bitangent)
}

/// Sampled surface geometry on a UV grid (row-major: `u` outer, `v` inner).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceGeometry {
    pub u_res: usize,
    pub v_res: usize,
    /// Sampled positions, length `u_res * v_res`.
    pub points: Vec<Vec3>,
    /// Surface normals at each sample, length `u_res * v_res`.
    pub normals: Vec<Vec3>,
    /// Trimming mask in `[0, 1]`; `1.0` inside the trimmed face, length `u_res * v_res`.
    pub mask: Vec<f32>,
}

impl SurfaceGeometry {
    pub fn sample_count(&self) -> usize {
        self.u_res * self.v_res
    }

    pub fn is_shaped(&self) -> bool {
        let n = self.sample_count();
        self.points.len() == n && self.normals.len() == n && self.mask.len() == n
    }

    fn is_finite(&self) -> bool {
        self.points.iter().all(|p| p.is_finite())
            && self.normals.iter().all(|n| n.is_finite())
            && self.mask.iter().all(|m| m.is_finite())
    }
}

/// Sampled curve geometry on a 1D grid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveGeometry {
    pub res: usize,
    /// Sampled positions along the curve, length `res`.
    pub points: Vec<Vec3>,
    /// Unit tangents at each sample, length `res`.
    pub tangents: Vec<Vec3>,
}

impl CurveGeometry {
    pub fn is_shaped(&self) -> bool {
        self.points.len() == self.res && self.tangents.len() == self.res
    }

    fn is_finite(&self) -> bool {
        self.points.iter().all(|p| p.is_finite()) && self.tangents.iter().all(|t| t.is_finite())
    }
}

/// Sample a rectangular planar patch centered at `centroid` with the given normal.
pub fn sample_plane(
    centroid: Vec3,
    normal: Vec3,
    half_u: f32,
    half_v: f32,
    u_res: usize,
    v_res: usize,
) -> SurfaceGeometry {
    let (tangent, bitangent) = tangent_basis(normal);
    let normal = normal.normalized();
    let mut points = Vec::with_capacity(u_res * v_res);
    let mut normals = Vec::with_capacity(u_res * v_res);
    let mut mask = Vec::with_capacity(u_res * v_res);
    for iu in 0..u_res {
        let u = grid_coord(iu, u_res) * half_u;
        for iv in 0..v_res {
            let v = grid_coord(iv, v_res) * half_v;
            let point = centroid + tangent.scale(u) + bitangent.scale(v);
            points.push(point);
            normals.push(normal);
            mask.push(1.0);
        }
    }
    SurfaceGeometry {
        u_res,
        v_res,
        points,
        normals,
        mask,
    }
}

/// Sample a cylindrical side patch of the given `radius`/`height` about the +Z axis
/// centered at `center`. `u` sweeps the angle, `v` sweeps the height.
pub fn sample_cylinder(
    center: Vec3,
    radius: f32,
    height: f32,
    u_res: usize,
    v_res: usize,
) -> SurfaceGeometry {
    let mut points = Vec::with_capacity(u_res * v_res);
    let mut normals = Vec::with_capacity(u_res * v_res);
    let mut mask = Vec::with_capacity(u_res * v_res);
    for iu in 0..u_res {
        let theta = (iu as f32 / u_res as f32) * std::f32::consts::TAU;
        let (sin, cos) = theta.sin_cos();
        for iv in 0..v_res {
            let z = grid_coord(iv, v_res) * (height * 0.5);
            let point = center + Vec3::new(radius * cos, radius * sin, z);
            points.push(point);
            normals.push(Vec3::new(cos, sin, 0.0));
            mask.push(1.0);
        }
    }
    SurfaceGeometry {
        u_res,
        v_res,
        points,
        normals,
        mask,
    }
}

/// Sample canonical curve geometry for a synthetic edge.
///
/// The synthetic fixtures do not carry real curve endpoints, so lines are
/// sampled along the local +X axis through `midpoint` and circles are sampled in
/// the local XY plane. This is a documented simplification: it gives the geometry
/// CNN a consistent, class-informative signal (a circle grid differs from a line
/// grid) without requiring full vertex geometry.
pub fn sample_curve(kind: CurveKind, length: f32, midpoint: Vec3, res: usize) -> CurveGeometry {
    let mut points = Vec::with_capacity(res);
    let mut tangents = Vec::with_capacity(res);
    match kind {
        CurveKind::Circle | CurveKind::Ellipse => {
            let radius = (length / std::f32::consts::TAU).max(f32::EPSILON);
            for i in 0..res {
                let theta = (i as f32 / res as f32) * std::f32::consts::TAU;
                let (sin, cos) = theta.sin_cos();
                points.push(midpoint + Vec3::new(radius * cos, radius * sin, 0.0));
                tangents.push(Vec3::new(-sin, cos, 0.0));
            }
        }
        _ => {
            for i in 0..res {
                let t = grid_coord(i, res) * (length * 0.5);
                points.push(midpoint + Vec3::new(t, 0.0, 0.0));
                tangents.push(Vec3::new(1.0, 0.0, 0.0));
            }
        }
    }
    CurveGeometry {
        res,
        points,
        tangents,
    }
}

/// Grid coordinate in `[-1, 1]` for sample `i` of `res` (single-sample grids map to 0).
fn grid_coord(i: usize, res: usize) -> f32 {
    if res <= 1 {
        0.0
    } else {
        (i as f32 / (res - 1) as f32) * 2.0 - 1.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Face {
    pub surface: SurfaceKind,
    pub area: f32,
    pub centroid: Vec3,
    pub normal: Vec3,
    #[serde(default)]
    pub geometry: Option<SurfaceGeometry>,
}

impl Face {
    pub fn new(surface: SurfaceKind, area: f32, centroid: Vec3, normal: Vec3) -> Self {
        Self {
            surface,
            area,
            centroid,
            normal,
            geometry: None,
        }
    }

    pub fn with_geometry(mut self, geometry: SurfaceGeometry) -> Self {
        self.geometry = Some(geometry);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub curve: CurveKind,
    pub length: f32,
    pub midpoint: Vec3,
    pub convexity: Convexity,
    #[serde(default)]
    pub geometry: Option<CurveGeometry>,
}

impl Edge {
    pub fn new(curve: CurveKind, length: f32, midpoint: Vec3, convexity: Convexity) -> Self {
        Self {
            curve,
            length,
            midpoint,
            convexity,
            geometry: None,
        }
    }

    pub fn with_geometry(mut self, geometry: CurveGeometry) -> Self {
        self.geometry = Some(geometry);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coedge {
    pub edge: EdgeId,
    pub face: FaceId,
    pub orientation: Orientation,
    /// The paired coedge on the same edge belonging to the adjacent face.
    #[serde(default)]
    pub mate: Option<CoedgeId>,
}

impl Coedge {
    pub const fn new(edge: EdgeId, face: FaceId, orientation: Orientation) -> Self {
        Self {
            edge,
            face,
            orientation,
            mate: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaceAdjacency {
    pub left: FaceId,
    pub edge: EdgeId,
    pub right: FaceId,
}

impl FaceAdjacency {
    pub const fn new(left: FaceId, edge: EdgeId, right: FaceId) -> Self {
        Self { left, edge, right }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BrepGraph {
    pub faces: Vec<Face>,
    pub edges: Vec<Edge>,
    pub coedges: Vec<Coedge>,
    pub face_adjacency: Vec<FaceAdjacency>,
}

impl BrepGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_face(&mut self, face: Face) -> FaceId {
        let id = self.faces.len();
        self.faces.push(face);
        id
    }

    pub fn add_edge(&mut self, edge: Edge) -> EdgeId {
        let id = self.edges.len();
        self.edges.push(edge);
        id
    }

    pub fn add_coedge(&mut self, coedge: Coedge) -> CoedgeId {
        let id = self.coedges.len();
        self.coedges.push(coedge);
        id
    }

    pub fn add_face_adjacency(&mut self, adjacency: FaceAdjacency) {
        self.face_adjacency.push(adjacency);
    }

    /// Serialize to pretty JSON for Python/interop consumers.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON produced by [`BrepGraph::to_json`].
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn validate(&self) -> Result<GraphStats, GraphError> {
        for (index, face) in self.faces.iter().enumerate() {
            if !face.area.is_finite() || face.area < 0.0 {
                return Err(GraphError::InvalidFaceMetric {
                    face: index,
                    field: "area",
                });
            }
            if !face.centroid.is_finite() {
                return Err(GraphError::InvalidFaceMetric {
                    face: index,
                    field: "centroid",
                });
            }
            if !face.normal.is_finite() {
                return Err(GraphError::InvalidFaceMetric {
                    face: index,
                    field: "normal",
                });
            }
            if let Some(geometry) = &face.geometry {
                if !geometry.is_shaped() || !geometry.is_finite() {
                    return Err(GraphError::InvalidFaceMetric {
                        face: index,
                        field: "geometry",
                    });
                }
            }
        }

        for (index, edge) in self.edges.iter().enumerate() {
            if !edge.length.is_finite() || edge.length < 0.0 {
                return Err(GraphError::InvalidEdgeMetric {
                    edge: index,
                    field: "length",
                });
            }
            if !edge.midpoint.is_finite() {
                return Err(GraphError::InvalidEdgeMetric {
                    edge: index,
                    field: "midpoint",
                });
            }
            if let Some(geometry) = &edge.geometry {
                if !geometry.is_shaped() || !geometry.is_finite() {
                    return Err(GraphError::InvalidEdgeMetric {
                        edge: index,
                        field: "geometry",
                    });
                }
            }
        }

        for (index, coedge) in self.coedges.iter().enumerate() {
            if coedge.edge >= self.edges.len() {
                return Err(GraphError::InvalidCoedgeEdge {
                    coedge: index,
                    edge: coedge.edge,
                });
            }
            if coedge.face >= self.faces.len() {
                return Err(GraphError::InvalidCoedgeFace {
                    coedge: index,
                    face: coedge.face,
                });
            }
            if let Some(mate) = coedge.mate {
                if mate >= self.coedges.len() {
                    return Err(GraphError::InvalidCoedgeMate {
                        coedge: index,
                        mate,
                    });
                }
            }
        }

        for (index, adjacency) in self.face_adjacency.iter().enumerate() {
            if adjacency.left >= self.faces.len() {
                return Err(GraphError::InvalidAdjacencyFace {
                    adjacency: index,
                    face: adjacency.left,
                });
            }
            if adjacency.right >= self.faces.len() {
                return Err(GraphError::InvalidAdjacencyFace {
                    adjacency: index,
                    face: adjacency.right,
                });
            }
            if adjacency.edge >= self.edges.len() {
                return Err(GraphError::InvalidAdjacencyEdge {
                    adjacency: index,
                    edge: adjacency.edge,
                });
            }
        }

        Ok(GraphStats {
            faces: self.faces.len(),
            edges: self.edges.len(),
            coedges: self.coedges.len(),
            face_adjacencies: self.face_adjacency.len(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphStats {
    pub faces: usize,
    pub edges: usize,
    pub coedges: usize,
    pub face_adjacencies: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    InvalidFaceMetric { face: FaceId, field: &'static str },
    InvalidEdgeMetric { edge: EdgeId, field: &'static str },
    InvalidCoedgeEdge { coedge: CoedgeId, edge: EdgeId },
    InvalidCoedgeFace { coedge: CoedgeId, face: FaceId },
    InvalidCoedgeMate { coedge: CoedgeId, mate: CoedgeId },
    InvalidAdjacencyFace { adjacency: usize, face: FaceId },
    InvalidAdjacencyEdge { adjacency: usize, edge: EdgeId },
}

impl fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFaceMetric { face, field } => {
                write!(formatter, "face {face} has invalid {field}")
            }
            Self::InvalidEdgeMetric { edge, field } => {
                write!(formatter, "edge {edge} has invalid {field}")
            }
            Self::InvalidCoedgeEdge { coedge, edge } => {
                write!(formatter, "coedge {coedge} references missing edge {edge}")
            }
            Self::InvalidCoedgeFace { coedge, face } => {
                write!(formatter, "coedge {coedge} references missing face {face}")
            }
            Self::InvalidCoedgeMate { coedge, mate } => {
                write!(formatter, "coedge {coedge} references missing mate {mate}")
            }
            Self::InvalidAdjacencyFace { adjacency, face } => {
                write!(
                    formatter,
                    "face adjacency {adjacency} references missing face {face}"
                )
            }
            Self::InvalidAdjacencyEdge { adjacency, edge } => {
                write!(
                    formatter,
                    "face adjacency {adjacency} references missing edge {edge}"
                )
            }
        }
    }
}

impl Error for GraphError {}

pub fn sample_box_graph() -> BrepGraph {
    box_graph(1.0, 1.0, 1.0)
}

pub fn box_graph(width: f32, height: f32, depth: f32) -> BrepGraph {
    let mut graph = BrepGraph::new();
    let width = width.max(0.001);
    let height = height.max(0.001);
    let depth = depth.max(0.001);
    let half_width = width * 0.5;
    let half_height = height * 0.5;
    let half_depth = depth * 0.5;

    // (surface, area, centroid, normal, half_u, half_v)
    let faces = [
        (
            height * depth,
            Vec3::new(half_width, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            half_height,
            half_depth,
        ),
        (
            height * depth,
            Vec3::new(-half_width, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            half_height,
            half_depth,
        ),
        (
            width * depth,
            Vec3::new(0.0, half_height, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            half_width,
            half_depth,
        ),
        (
            width * depth,
            Vec3::new(0.0, -half_height, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            half_width,
            half_depth,
        ),
        (
            width * height,
            Vec3::new(0.0, 0.0, half_depth),
            Vec3::new(0.0, 0.0, 1.0),
            half_width,
            half_height,
        ),
        (
            width * height,
            Vec3::new(0.0, 0.0, -half_depth),
            Vec3::new(0.0, 0.0, -1.0),
            half_width,
            half_height,
        ),
    ];

    for (area, centroid, normal, half_u, half_v) in faces {
        let geometry = sample_plane(
            centroid,
            normal,
            half_u,
            half_v,
            DEFAULT_UV_RES,
            DEFAULT_UV_RES,
        );
        graph.add_face(
            Face::new(SurfaceKind::Plane, area, centroid, normal).with_geometry(geometry),
        );
    }

    for length in [
        height, height, depth, depth, height, height, depth, depth, width, width, width, width,
    ] {
        graph.add_edge(line_edge(length, Vec3::zero()));
    }

    let adjacencies = [
        (0, 0, 2),
        (0, 1, 3),
        (0, 2, 4),
        (0, 3, 5),
        (1, 4, 2),
        (1, 5, 3),
        (1, 6, 4),
        (1, 7, 5),
        (2, 8, 4),
        (2, 9, 5),
        (3, 10, 4),
        (3, 11, 5),
    ];

    for (left, edge, right) in adjacencies {
        add_bidirectional_adjacency(&mut graph, left, edge, right);
    }

    graph
}

pub fn cylinder_graph(radius: f32, height: f32) -> BrepGraph {
    let mut graph = BrepGraph::new();
    let radius = radius.max(0.001);
    let height = height.max(0.001);
    let half_height = height * 0.5;
    let cap_area = std::f32::consts::PI * radius * radius;
    let circle_length = 2.0 * std::f32::consts::PI * radius;

    let side = graph.add_face(
        Face::new(
            SurfaceKind::Cylinder,
            circle_length * height,
            Vec3::zero(),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .with_geometry(sample_cylinder(
            Vec3::zero(),
            radius,
            height,
            DEFAULT_UV_RES,
            DEFAULT_UV_RES,
        )),
    );
    let top = graph.add_face(
        Face::new(
            SurfaceKind::Plane,
            cap_area,
            Vec3::new(0.0, 0.0, half_height),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .with_geometry(sample_plane(
            Vec3::new(0.0, 0.0, half_height),
            Vec3::new(0.0, 0.0, 1.0),
            radius,
            radius,
            DEFAULT_UV_RES,
            DEFAULT_UV_RES,
        )),
    );
    let bottom = graph.add_face(
        Face::new(
            SurfaceKind::Plane,
            cap_area,
            Vec3::new(0.0, 0.0, -half_height),
            Vec3::new(0.0, 0.0, -1.0),
        )
        .with_geometry(sample_plane(
            Vec3::new(0.0, 0.0, -half_height),
            Vec3::new(0.0, 0.0, -1.0),
            radius,
            radius,
            DEFAULT_UV_RES,
            DEFAULT_UV_RES,
        )),
    );

    let top_edge = graph.add_edge(circle_edge(circle_length, Vec3::new(0.0, 0.0, half_height)));
    let bottom_edge = graph.add_edge(circle_edge(
        circle_length,
        Vec3::new(0.0, 0.0, -half_height),
    ));

    add_bidirectional_adjacency(&mut graph, side, top_edge, top);
    add_bidirectional_adjacency(&mut graph, side, bottom_edge, bottom);

    graph
}

pub fn plate_with_hole_graph(
    width: f32,
    height: f32,
    thickness: f32,
    hole_radius: f32,
) -> BrepGraph {
    plate_with_holes(width, height, thickness, hole_radius, 1)
}

/// A rectangular plate perforated by `hole_count` through-holes. Structural
/// variation (hole count) makes the class non-trivial for a topology-aware model.
pub fn plate_with_holes(
    width: f32,
    height: f32,
    thickness: f32,
    hole_radius: f32,
    hole_count: usize,
) -> BrepGraph {
    let mut graph = BrepGraph::new();
    let width = width.max(0.001);
    let height = height.max(0.001);
    let thickness = thickness.max(0.001);
    let hole_count = hole_count.max(1);
    let max_radius = width.min(height) * 0.45 / hole_count as f32;
    let hole_radius = hole_radius.clamp(0.001, max_radius);
    let half_width = width * 0.5;
    let half_height = height * 0.5;
    let half_thickness = thickness * 0.5;
    let hole_area = std::f32::consts::PI * hole_radius * hole_radius;
    let major_area = (width * height - hole_area * hole_count as f32).max(0.001);
    let circle_length = 2.0 * std::f32::consts::PI * hole_radius;

    let top = graph.add_face(
        Face::new(
            SurfaceKind::Plane,
            major_area,
            Vec3::new(0.0, 0.0, half_thickness),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .with_geometry(sample_plane(
            Vec3::new(0.0, 0.0, half_thickness),
            Vec3::new(0.0, 0.0, 1.0),
            half_width,
            half_height,
            DEFAULT_UV_RES,
            DEFAULT_UV_RES,
        )),
    );
    let bottom = graph.add_face(
        Face::new(
            SurfaceKind::Plane,
            major_area,
            Vec3::new(0.0, 0.0, -half_thickness),
            Vec3::new(0.0, 0.0, -1.0),
        )
        .with_geometry(sample_plane(
            Vec3::new(0.0, 0.0, -half_thickness),
            Vec3::new(0.0, 0.0, -1.0),
            half_width,
            half_height,
            DEFAULT_UV_RES,
            DEFAULT_UV_RES,
        )),
    );
    let side_pos_x = graph.add_face(planar_side(
        height * thickness,
        Vec3::new(half_width, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        half_height,
        half_thickness,
    ));
    let side_neg_x = graph.add_face(planar_side(
        height * thickness,
        Vec3::new(-half_width, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        half_height,
        half_thickness,
    ));
    let side_pos_y = graph.add_face(planar_side(
        width * thickness,
        Vec3::new(0.0, half_height, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        half_width,
        half_thickness,
    ));
    let side_neg_y = graph.add_face(planar_side(
        width * thickness,
        Vec3::new(0.0, -half_height, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        half_width,
        half_thickness,
    ));

    // Shell edges: top loop (0..4), bottom loop (4..8), verticals (8..12).
    for length in [width, height, width, height, width, height, width, height] {
        graph.add_edge(line_edge(length, Vec3::zero()));
    }
    for _ in 0..4 {
        graph.add_edge(line_edge(thickness, Vec3::zero()));
    }

    let shell_adjacencies = [
        (top, 0, side_pos_y),
        (top, 1, side_pos_x),
        (top, 2, side_neg_y),
        (top, 3, side_neg_x),
        (bottom, 4, side_pos_y),
        (bottom, 5, side_pos_x),
        (bottom, 6, side_neg_y),
        (bottom, 7, side_neg_x),
        (side_pos_y, 8, side_pos_x),
        (side_pos_x, 9, side_neg_y),
        (side_neg_y, 10, side_neg_x),
        (side_neg_x, 11, side_pos_y),
    ];
    for (left, edge, right) in shell_adjacencies {
        add_bidirectional_adjacency(&mut graph, left, edge, right);
    }

    // Holes laid out along X so their centers differ.
    for k in 0..hole_count {
        let offset = if hole_count == 1 {
            0.0
        } else {
            let t = k as f32 / (hole_count - 1) as f32;
            (t * 2.0 - 1.0) * half_width * 0.6
        };
        let center = Vec3::new(offset, 0.0, 0.0);
        let hole = graph.add_face(
            Face::new(
                SurfaceKind::Cylinder,
                circle_length * thickness,
                center,
                Vec3::new(1.0, 0.0, 0.0),
            )
            .with_geometry(sample_cylinder(
                center,
                hole_radius,
                thickness,
                DEFAULT_UV_RES,
                DEFAULT_UV_RES,
            )),
        );
        let top_edge = graph.add_edge(circle_edge(
            circle_length,
            Vec3::new(offset, 0.0, half_thickness),
        ));
        let bottom_edge = graph.add_edge(circle_edge(
            circle_length,
            Vec3::new(offset, 0.0, -half_thickness),
        ));
        add_bidirectional_adjacency(&mut graph, top, top_edge, hole);
        add_bidirectional_adjacency(&mut graph, bottom, bottom_edge, hole);
    }

    graph
}

fn planar_side(area: f32, centroid: Vec3, normal: Vec3, half_u: f32, half_v: f32) -> Face {
    let geometry = sample_plane(
        centroid,
        normal,
        half_u,
        half_v,
        DEFAULT_UV_RES,
        DEFAULT_UV_RES,
    );
    Face::new(SurfaceKind::Plane, area, centroid, normal).with_geometry(geometry)
}

fn line_edge(length: f32, midpoint: Vec3) -> Edge {
    Edge::new(CurveKind::Line, length, midpoint, Convexity::Convex).with_geometry(sample_curve(
        CurveKind::Line,
        length,
        midpoint,
        DEFAULT_CURVE_RES,
    ))
}

fn circle_edge(length: f32, midpoint: Vec3) -> Edge {
    Edge::new(CurveKind::Circle, length, midpoint, Convexity::Smooth).with_geometry(sample_curve(
        CurveKind::Circle,
        length,
        midpoint,
        DEFAULT_CURVE_RES,
    ))
}

fn add_bidirectional_adjacency(graph: &mut BrepGraph, left: FaceId, edge: EdgeId, right: FaceId) {
    graph.add_face_adjacency(FaceAdjacency::new(left, edge, right));
    let first = graph.add_coedge(Coedge::new(edge, left, Orientation::Forward));
    let second = graph.add_coedge(Coedge::new(edge, right, Orientation::Reversed));
    graph.coedges[first].mate = Some(second);
    graph.coedges[second].mate = Some(first);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_box_graph_is_valid() {
        let graph = sample_box_graph();
        let stats = graph.validate().expect("sample box should validate");

        assert_eq!(stats.faces, 6);
        assert_eq!(stats.edges, 12);
        assert_eq!(stats.coedges, 24);
        assert_eq!(stats.face_adjacencies, 12);
    }

    #[test]
    fn faces_and_edges_carry_geometry() {
        let graph = sample_box_graph();
        for face in &graph.faces {
            let geometry = face.geometry.as_ref().expect("face geometry present");
            assert!(geometry.is_shaped());
            assert_eq!(geometry.sample_count(), DEFAULT_UV_RES * DEFAULT_UV_RES);
        }
        for edge in &graph.edges {
            let geometry = edge.geometry.as_ref().expect("edge geometry present");
            assert!(geometry.is_shaped());
        }
    }

    #[test]
    fn coedges_are_mated() {
        let graph = sample_box_graph();
        for (index, coedge) in graph.coedges.iter().enumerate() {
            let mate = coedge.mate.expect("coedge has a mate");
            assert_eq!(graph.coedges[mate].mate, Some(index));
            assert_eq!(graph.coedges[mate].edge, coedge.edge);
        }
    }

    #[test]
    fn multi_hole_plate_has_more_faces() {
        let one = plate_with_holes(4.0, 2.0, 0.25, 0.2, 1);
        let three = plate_with_holes(4.0, 2.0, 0.25, 0.2, 3);
        assert_eq!(one.faces.len() + 2, three.faces.len());
        three.validate().expect("multi-hole plate should validate");
    }

    #[test]
    fn json_round_trips() {
        let graph = sample_box_graph();
        let json = graph.to_json().expect("serialize");
        let restored = BrepGraph::from_json(&json).expect("deserialize");
        assert_eq!(graph, restored);
    }

    #[test]
    fn validation_rejects_missing_face_references() {
        let mut graph = BrepGraph::new();
        graph.add_edge(Edge::new(
            CurveKind::Line,
            1.0,
            Vec3::zero(),
            Convexity::Unknown,
        ));
        graph.add_coedge(Coedge::new(0, 42, Orientation::Forward));

        let error = graph.validate().expect_err("missing face should fail");
        assert!(matches!(
            error,
            GraphError::InvalidCoedgeFace {
                coedge: 0,
                face: 42
            }
        ));
    }

    #[test]
    fn synthetic_training_fixtures_are_valid() {
        for graph in [
            box_graph(2.0, 1.0, 0.5),
            cylinder_graph(0.5, 2.0),
            plate_with_hole_graph(4.0, 2.0, 0.25, 0.4),
            plate_with_holes(4.0, 2.0, 0.25, 0.2, 3),
        ] {
            graph.validate().expect("fixture should validate");
        }
    }
}
