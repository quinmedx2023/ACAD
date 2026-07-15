#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepGProp.hxx>
#include <GProp_GProps.hxx>
#include <GeomAbs_CurveType.hxx>
#include <GeomAbs_SurfaceType.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <NCollection_IndexedDataMap.hxx>
#include <NCollection_List.hxx>
#include <STEPControl_Reader.hxx>
#include <TopAbs_Orientation.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <gp_Pnt.hxx>
#include <gp_Vec.hxx>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using ShapeList = NCollection_List<TopoDS_Shape>;
using EdgeFaceMap = NCollection_IndexedDataMap<TopoDS_Shape, ShapeList, TopTools_ShapeMapHasher>;

constexpr int kUvRes = 6;
constexpr int kCurveRes = 6;

struct Args {
    fs::path raw;
    fs::path out;
    std::optional<std::size_t> limit;
    double val_fraction = 0.25;
    bool allow_boundary = false;
};

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct SurfaceGeometry {
    int u_res = kUvRes;
    int v_res = kUvRes;
    std::vector<Vec3> points;
    std::vector<Vec3> normals;
    std::vector<double> mask;
};

struct CurveGeometry {
    int res = kCurveRes;
    std::vector<Vec3> points;
    std::vector<Vec3> tangents;
};

struct FaceJson {
    std::string surface;
    double area = 0.0;
    Vec3 centroid;
    Vec3 normal;
    SurfaceGeometry geometry;
};

struct EdgeJson {
    std::string curve;
    double length = 0.0;
    Vec3 midpoint;
    std::string convexity = "Unknown";
    CurveGeometry geometry;
};

struct CoedgeJson {
    int edge = 0;
    int face = 0;
    std::string orientation = "Forward";
    std::optional<int> mate;
};

struct FaceAdjacencyJson {
    int left = 0;
    int edge = 0;
    int right = 0;
};

struct GraphJson {
    std::vector<FaceJson> faces;
    std::vector<EdgeJson> edges;
    std::vector<CoedgeJson> coedges;
    std::vector<FaceAdjacencyJson> face_adjacency;
};

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

double finite(double value) {
    if (!std::isfinite(value)) {
        return 0.0;
    }
    return value;
}

Vec3 vec3(const gp_Pnt& p) {
    return {finite(p.X()), finite(p.Y()), finite(p.Z())};
}

Vec3 normalized(const gp_Vec& v, Vec3 fallback) {
    const double mag = v.Magnitude();
    if (!std::isfinite(mag) || mag <= 1e-12) {
        return fallback;
    }
    return {finite(v.X() / mag), finite(v.Y() / mag), finite(v.Z() / mag)};
}

std::pair<double, double> finite_range(double a, double b) {
    a = finite(a);
    b = finite(b);
    if (std::abs(a) > 1e6 || std::abs(b) > 1e6) {
        return {-1.0, 1.0};
    }
    if (std::abs(a - b) < 1e-9) {
        return {a - 0.5, b + 0.5};
    }
    return {a, b};
}

double grid_t(int i, int n) {
    return n <= 1 ? 0.5 : static_cast<double>(i) / static_cast<double>(n - 1);
}

double lerp(double a, double b, double t) {
    return a + (b - a) * t;
}

std::string surface_kind(GeomAbs_SurfaceType type) {
    switch (type) {
        case GeomAbs_Plane: return "Plane";
        case GeomAbs_Cylinder: return "Cylinder";
        case GeomAbs_Cone: return "Cone";
        case GeomAbs_Sphere: return "Sphere";
        case GeomAbs_Torus: return "Torus";
        case GeomAbs_BSplineSurface: return "Nurbs";
        default: return "Unknown";
    }
}

std::string curve_kind(GeomAbs_CurveType type) {
    switch (type) {
        case GeomAbs_Line: return "Line";
        case GeomAbs_Circle: return "Circle";
        case GeomAbs_Ellipse: return "Ellipse";
        case GeomAbs_BSplineCurve: return "Spline";
        default: return "Unknown";
    }
}

Vec3 normal_at(BRepAdaptor_Surface& surface, double u, double v, Vec3 fallback) {
    try {
        gp_Pnt p;
        gp_Vec du;
        gp_Vec dv;
        surface.D1(u, v, p, du, dv);
        return normalized(du.Crossed(dv), fallback);
    } catch (...) {
        return fallback;
    }
}

Vec3 tangent_at(BRepAdaptor_Curve& curve, double t) {
    try {
        gp_Pnt p;
        gp_Vec d;
        curve.D1(t, p, d);
        return normalized(d, {1.0, 0.0, 0.0});
    } catch (...) {
        return {1.0, 0.0, 0.0};
    }
}

SurfaceGeometry sample_surface(BRepAdaptor_Surface& surface, Vec3 fallback_normal) {
    const auto [u1, u2] = finite_range(surface.FirstUParameter(), surface.LastUParameter());
    const auto [v1, v2] = finite_range(surface.FirstVParameter(), surface.LastVParameter());
    SurfaceGeometry geom;
    geom.points.reserve(kUvRes * kUvRes);
    geom.normals.reserve(kUvRes * kUvRes);
    geom.mask.reserve(kUvRes * kUvRes);
    for (int iu = 0; iu < kUvRes; ++iu) {
        const double u = lerp(u1, u2, grid_t(iu, kUvRes));
        for (int iv = 0; iv < kUvRes; ++iv) {
            const double v = lerp(v1, v2, grid_t(iv, kUvRes));
            geom.points.push_back(vec3(surface.Value(u, v)));
            geom.normals.push_back(normal_at(surface, u, v, fallback_normal));
            geom.mask.push_back(1.0);
        }
    }
    return geom;
}

CurveGeometry sample_curve(BRepAdaptor_Curve& curve) {
    const auto [t1, t2] = finite_range(curve.FirstParameter(), curve.LastParameter());
    CurveGeometry geom;
    geom.points.reserve(kCurveRes);
    geom.tangents.reserve(kCurveRes);
    for (int i = 0; i < kCurveRes; ++i) {
        const double t = lerp(t1, t2, grid_t(i, kCurveRes));
        geom.points.push_back(vec3(curve.Value(t)));
        geom.tangents.push_back(tangent_at(curve, t));
    }
    return geom;
}

FaceJson face_to_json(const TopoDS_Face& face) {
    BRepAdaptor_Surface surface(face);
    GProp_GProps props;
    BRepGProp::SurfaceProperties(face, props);
    const auto [u1, u2] = finite_range(surface.FirstUParameter(), surface.LastUParameter());
    const auto [v1, v2] = finite_range(surface.FirstVParameter(), surface.LastVParameter());
    Vec3 normal = normal_at(surface, (u1 + u2) * 0.5, (v1 + v2) * 0.5, {0.0, 0.0, 1.0});
    FaceJson out;
    out.surface = surface_kind(surface.GetType());
    out.area = finite(props.Mass());
    out.centroid = vec3(props.CentreOfMass());
    out.normal = normal;
    out.geometry = sample_surface(surface, normal);
    return out;
}

EdgeJson edge_to_json(const TopoDS_Edge& edge) {
    BRepAdaptor_Curve curve(edge);
    GProp_GProps props;
    BRepGProp::LinearProperties(edge, props);
    EdgeJson out;
    out.curve = curve_kind(curve.GetType());
    out.length = finite(props.Mass());
    out.midpoint = vec3(props.CentreOfMass());
    out.geometry = sample_curve(curve);
    return out;
}

int find_index_ignore_orientation(const TopTools_IndexedMapOfShape& map, const TopoDS_Shape& shape) {
    TopoDS_Shape needle = shape;
    needle.Orientation(TopAbs_FORWARD);
    for (int i = 1; i <= map.Extent(); ++i) {
        TopoDS_Shape candidate = map(i);
        candidate.Orientation(TopAbs_FORWARD);
        if (candidate.IsSame(needle)) {
            return i - 1;
        }
    }
    return -1;
}

GraphJson read_step_graph(const fs::path& path, bool allow_boundary) {
    STEPControl_Reader reader;
    IFSelect_ReturnStatus status = reader.ReadFile(path.string().c_str());
    if (status != IFSelect_RetDone) {
        throw std::runtime_error("STEP import failed");
    }
    reader.TransferRoots();
    TopoDS_Shape shape = reader.OneShape();

    TopTools_IndexedMapOfShape face_map;
    TopTools_IndexedMapOfShape edge_map;
    EdgeFaceMap edge_faces;
    TopExp::MapShapes(shape, TopAbs_FACE, face_map);
    TopExp::MapShapes(shape, TopAbs_EDGE, edge_map);
    TopExp::MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
    if (face_map.IsEmpty() || edge_map.IsEmpty()) {
        throw std::runtime_error("empty topology");
    }

    GraphJson graph;
    graph.faces.reserve(face_map.Extent());
    graph.edges.reserve(edge_map.Extent());
    for (int i = 1; i <= face_map.Extent(); ++i) {
        graph.faces.push_back(face_to_json(TopoDS::Face(face_map(i))));
    }
    for (int i = 1; i <= edge_map.Extent(); ++i) {
        graph.edges.push_back(edge_to_json(TopoDS::Edge(edge_map(i))));
    }

    std::vector<std::vector<int>> coedges_by_edge(graph.edges.size());
    for (int edge_i = 1; edge_i <= edge_map.Extent(); ++edge_i) {
        const int edge_index = edge_i - 1;
        const TopoDS_Shape& edge = edge_map(edge_i);
        std::set<int> incident_faces;
        if (edge_faces.Contains(edge)) {
            const ShapeList& faces = edge_faces.FindFromKey(edge);
            for (ShapeList::Iterator it(faces); it.More(); it.Next()) {
                int face_index = find_index_ignore_orientation(face_map, it.Value());
                if (face_index >= 0) {
                    incident_faces.insert(face_index);
                }
            }
        }
        if (incident_faces.size() != 2 && !allow_boundary) {
            std::ostringstream message;
            message << "edge " << edge_index << " has " << incident_faces.size()
                    << " incident faces";
            throw std::runtime_error(message.str());
        }
        for (int face_index : incident_faces) {
            const int coedge_index = static_cast<int>(graph.coedges.size());
            graph.coedges.push_back({edge_index, face_index, "Forward", std::nullopt});
            coedges_by_edge[edge_index].push_back(coedge_index);
        }
        if (incident_faces.size() >= 2) {
            auto it = incident_faces.begin();
            int left = *it++;
            int right = *it;
            graph.face_adjacency.push_back({left, edge_index, right});
        }
    }

    for (const auto& ids : coedges_by_edge) {
        if (ids.size() == 2) {
            graph.coedges[ids[0]].mate = ids[1];
            graph.coedges[ids[1]].mate = ids[0];
        }
    }
    return graph;
}

std::vector<int> parse_segmentation(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open .seg file");
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<int> values;
    std::string token;
    for (char ch : text) {
        if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-') {
            token.push_back(ch);
        } else if (!token.empty()) {
            values.push_back(std::stoi(token));
            token.clear();
        }
    }
    if (!token.empty()) {
        values.push_back(std::stoi(token));
    }
    return values;
}

std::string edge_label(const EdgeJson& edge) {
    if (edge.curve == "Circle") {
        return "circle_edge";
    }
    if (edge.curve == "Line") {
        return "line_edge";
    }
    return "other_edge";
}

std::string lower_ext(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

std::string sanitize_id(const fs::path& relative) {
    fs::path no_ext = relative;
    no_ext.replace_extension();
    std::string raw = no_ext.generic_string();
    std::string out;
    for (char ch : raw) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            out.push_back(ch);
        } else if (!out.empty() && out.back() != '_') {
            out.push_back('_');
        }
    }
    if (!out.empty() && out.back() == '_') {
        out.pop_back();
    }
    return out.empty() ? "sample" : out;
}

std::string split_by_index(std::size_t index, double val_fraction) {
    if (val_fraction <= 0.0) {
        return "train";
    }
    const auto stride = std::max<std::size_t>(2, static_cast<std::size_t>(std::round(1.0 / val_fraction)));
    return index % stride == stride - 1 ? "val" : "train";
}

std::vector<fs::path> collect_steps(const fs::path& raw, std::optional<std::size_t> limit) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(raw)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string ext = lower_ext(entry.path());
        if (ext == ".step" || ext == ".stp") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (limit && files.size() > *limit) {
        files.resize(*limit);
    }
    return files;
}

std::unordered_map<std::string, fs::path> index_seg_files(const fs::path& raw) {
    std::unordered_map<std::string, fs::path> out;
    for (const auto& entry : fs::recursive_directory_iterator(raw)) {
        if (entry.is_regular_file() && lower_ext(entry.path()) == ".seg") {
            out.emplace(entry.path().stem().string(), entry.path());
        }
    }
    return out;
}

void write_vec3(std::ostream& out, const Vec3& value) {
    out << "{\"x\":" << value.x << ",\"y\":" << value.y << ",\"z\":" << value.z << "}";
}

void write_vec3_array(std::ostream& out, const std::vector<Vec3>& values) {
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ",";
        write_vec3(out, values[i]);
    }
    out << "]";
}

void write_graph(const fs::path& path, const GraphJson& graph) {
    std::ofstream out(path);
    out << "{\n\"faces\":[";
    for (std::size_t i = 0; i < graph.faces.size(); ++i) {
        const FaceJson& f = graph.faces[i];
        if (i) out << ",";
        out << "{\"surface\":\"" << f.surface << "\",\"area\":" << f.area << ",\"centroid\":";
        write_vec3(out, f.centroid);
        out << ",\"normal\":";
        write_vec3(out, f.normal);
        out << ",\"geometry\":{\"u_res\":" << f.geometry.u_res << ",\"v_res\":"
            << f.geometry.v_res << ",\"points\":";
        write_vec3_array(out, f.geometry.points);
        out << ",\"normals\":";
        write_vec3_array(out, f.geometry.normals);
        out << ",\"mask\":[";
        for (std::size_t j = 0; j < f.geometry.mask.size(); ++j) {
            if (j) out << ",";
            out << f.geometry.mask[j];
        }
        out << "]}}";
    }
    out << "],\n\"edges\":[";
    for (std::size_t i = 0; i < graph.edges.size(); ++i) {
        const EdgeJson& e = graph.edges[i];
        if (i) out << ",";
        out << "{\"curve\":\"" << e.curve << "\",\"length\":" << e.length << ",\"midpoint\":";
        write_vec3(out, e.midpoint);
        out << ",\"convexity\":\"" << e.convexity << "\",\"geometry\":{\"res\":"
            << e.geometry.res << ",\"points\":";
        write_vec3_array(out, e.geometry.points);
        out << ",\"tangents\":";
        write_vec3_array(out, e.geometry.tangents);
        out << "}}";
    }
    out << "],\n\"coedges\":[";
    for (std::size_t i = 0; i < graph.coedges.size(); ++i) {
        const CoedgeJson& c = graph.coedges[i];
        if (i) out << ",";
        out << "{\"edge\":" << c.edge << ",\"face\":" << c.face
            << ",\"orientation\":\"" << c.orientation << "\",\"mate\":";
        if (c.mate) {
            out << *c.mate;
        } else {
            out << "null";
        }
        out << "}";
    }
    out << "],\n\"face_adjacency\":[";
    for (std::size_t i = 0; i < graph.face_adjacency.size(); ++i) {
        const FaceAdjacencyJson& a = graph.face_adjacency[i];
        if (i) out << ",";
        out << "{\"left\":" << a.left << ",\"edge\":" << a.edge << ",\"right\":" << a.right << "}";
    }
    out << "]\n}\n";
}

void write_labels(const fs::path& path, const std::vector<int>& seg, const GraphJson& graph) {
    std::ofstream out(path);
    out << "{\n\"graph_class_id\":0,\n\"graph_class_name\":\"fusion_part\",\n\"face_labels\":[";
    for (std::size_t i = 0; i < seg.size(); ++i) {
        if (i) out << ",";
        out << "\"segment_" << seg[i] << "\"";
    }
    out << "],\n\"edge_labels\":[";
    for (std::size_t i = 0; i < graph.edges.size(); ++i) {
        if (i) out << ",";
        out << "\"" << edge_label(graph.edges[i]) << "\"";
    }
    out << "]\n}\n";
}

void write_json_string_array(std::ostream& out, const std::set<std::string>& values) {
    out << "[";
    bool first = true;
    for (const auto& value : values) {
        if (!first) out << ",";
        first = false;
        out << "\"" << json_escape(value) << "\"";
    }
    out << "]";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--raw") {
            args.raw = next("--raw");
        } else if (arg == "--out") {
            args.out = next("--out");
        } else if (arg == "--limit") {
            args.limit = static_cast<std::size_t>(std::stoull(next("--limit")));
        } else if (arg == "--val-fraction") {
            args.val_fraction = std::stod(next("--val-fraction"));
        } else if (arg == "--allow-boundary") {
            args.allow_boundary = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: occt_cleaner --raw DIR --out DIR [--limit N] [--val-fraction F] [--allow-boundary]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (args.raw.empty() || args.out.empty()) {
        throw std::runtime_error("--raw and --out are required");
    }
    return args;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        fs::create_directories(args.out / "graphs");
        fs::create_directories(args.out / "labels");

        std::vector<fs::path> steps = collect_steps(args.raw, args.limit);
        auto seg_files = index_seg_files(args.raw);
        if (steps.empty()) {
            throw std::runtime_error("no STEP files found");
        }

        std::ofstream manifest(args.out / "manifest.jsonl");
        std::ofstream skipped(args.out / "skipped.jsonl");
        std::set<std::string> face_label_set;
        std::set<std::string> edge_label_set;
        std::map<std::string, int> split_counts;
        int records = 0;
        int skipped_count = 0;

        for (std::size_t i = 0; i < steps.size(); ++i) {
            const fs::path& step = steps[i];
            std::string id = sanitize_id(fs::relative(step, args.raw));
            auto seg_it = seg_files.find(step.stem().string());
            if (seg_it == seg_files.end()) {
                skipped << "{\"id\":\"" << json_escape(id) << "\",\"reason\":\"missing .seg\"}\n";
                ++skipped_count;
                continue;
            }

            try {
                GraphJson graph = read_step_graph(step, args.allow_boundary);
                std::vector<int> seg = parse_segmentation(seg_it->second);
                if (seg.size() != graph.faces.size()) {
                    throw std::runtime_error("seg face count mismatch");
                }
                for (int label : seg) {
                    face_label_set.insert("segment_" + std::to_string(label));
                }
                for (const auto& edge : graph.edges) {
                    edge_label_set.insert(edge_label(edge));
                }

                const std::string graph_path = "graphs/" + id + ".json";
                const std::string labels_path = "labels/" + id + ".json";
                write_graph(args.out / graph_path, graph);
                write_labels(args.out / labels_path, seg, graph);

                std::string split = split_by_index(i, args.val_fraction);
                split_counts[split] += 1;
                manifest << "{\"id\":\"" << json_escape(id) << "\",\"split\":\"" << split
                         << "\",\"class_id\":0,\"class_name\":\"fusion_part\",\"graph_path\":\""
                         << json_escape(graph_path) << "\",\"labels_path\":\""
                         << json_escape(labels_path) << "\",\"stats\":{\"faces\":"
                         << graph.faces.size() << ",\"edges\":" << graph.edges.size()
                         << ",\"coedges\":" << graph.coedges.size()
                         << ",\"face_adjacencies\":" << graph.face_adjacency.size() << "}}\n";
                ++records;
            } catch (const std::exception& error) {
                skipped << "{\"id\":\"" << json_escape(id) << "\",\"reason\":\""
                        << json_escape(error.what()) << "\"}\n";
                ++skipped_count;
            }
        }

        std::ofstream metadata(args.out / "dataset.json");
        metadata << "{\n\"version\":\"acad-brep-dataset-v1\",\n\"records\":" << records
                 << ",\n\"samples_per_class\":" << records << ",\n\"val_fraction\":\""
                 << args.val_fraction << "\",\n\"classes\":[\"fusion_part\"],\n\"face_label_set\":";
        write_json_string_array(metadata, face_label_set);
        metadata << ",\n\"edge_label_set\":";
        write_json_string_array(metadata, edge_label_set);
        metadata << "\n}\n";

        std::cout << "records: " << records << "\n";
        std::cout << "skipped: " << skipped_count << "\n";
        for (const auto& [split, count] : split_counts) {
            std::cout << split << ": " << count << "\n";
        }
        return records > 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "occt_cleaner: " << error.what() << "\n";
        return 2;
    }
}
