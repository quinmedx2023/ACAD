use acad_brep_encoder::{BrepGraphEncoder, DeterministicGraphEncoder};
use acad_brep_graph::sample_box_graph;

fn main() {
    let graph = sample_box_graph();
    let encoder = DeterministicGraphEncoder::default();
    let encoding = encoder
        .encode(&graph)
        .expect("sample box graph should encode");

    println!("ACAD BRep graph encoder smoke test");
    println!("faces: {}", encoding.face_count);
    println!("edges: {}", encoding.edge_count);
    println!("face_feature_dim: {}", encoding.face_feature_dim);
    println!("edge_feature_dim: {}", encoding.edge_feature_dim);
    println!("face_edge_face triples: {}", encoding.face_edge_face.len());
}
