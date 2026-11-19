#include "cfl_detail.h"
#include "../bc/periodic.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace aero {
namespace {
using cfl_detail::Entry;
std::size_t extent(std::uint64_t n) {
    if (n >= static_cast<std::uint64_t>((std::numeric_limits<std::ptrdiff_t>::max)()))
        throw std::length_error("CFL grid exceeds addressable range");
    return static_cast<std::size_t>(n);
}
double magnitude(CvVector s) {
    const double area = std::hypot(std::hypot(s.x, s.y), s.z);
    if (!(area > 0 && std::isfinite(area)))
        throw std::invalid_argument("invalid CFL face area");
    return area;
}
struct Edge {
    std::uint64_t a, b;
    CvVector area;
};
}
// Постоянная геометрия для выбора шага. Между шагами меняются только скорости волн.
struct CflTimestep::Impl {
    // Границы строк таблицы «узел → площадки», а entries хранит номера и площади этих площадок.
    std::vector<std::size_t> offsets;
    std::vector<Entry> entries;
    std::vector<double> volumes;
    std::size_t edge_count = 0, boundary_count = 0;
    // В режиме CUDA владеет копией таблицы и рабочими буферами поиска минимума на карте.
    void* gpu = nullptr;

    cfl_detail::Grid view() const { return {volumes.size(), offsets.data(), entries.data(), volumes.data()}; }
    template <class ReadEdge>
    void prepare(std::size_t nodes, std::size_t edges, std::size_t pieces, const double* v,
                 const CvBoundary* boundary, ReadEdge read_edge, int device) {
        if (device < -1 || !nodes || !v || (pieces && !boundary))
            throw std::invalid_argument("invalid CFL grid");
        if (pieces > entries.max_size() || edges > (entries.max_size() - pieces) / 2)
            throw std::length_error("too many CFL faces");
        edge_count = edges;
        boundary_count = pieces;
        volumes.assign(v, v + nodes);
        for (double volume : volumes)
            if (!(volume > 0 && std::isfinite(volume)))
                throw std::invalid_argument("invalid CFL control volume");

        // Считаем длину строки каждого узла. Внутреннее ребро попадёт в две строки, внешняя площадка — в одну.
        offsets.assign(nodes + 1, 0);
        for (std::size_t e = 0; e < edges; ++e) {
            const Edge edge = read_edge(e);
            if (edge.a >= nodes || edge.b >= nodes)
                throw std::invalid_argument("invalid CFL edge endpoint");
            ++offsets[edge.a + 1];
            ++offsets[edge.b + 1];
        }
        for (std::size_t b = 0; b < pieces; ++b) {
            if (boundary[b].node >= nodes)
                throw std::invalid_argument("invalid CFL boundary node");
            ++offsets[boundary[b].node + 1];
        }
        for (std::size_t i = 0; i < nodes; ++i) {
            if (!offsets[i + 1])
                throw std::invalid_argument("isolated CFL node");
            offsets[i + 1] += offsets[i];
        }
        // Смещения уже накоплены. Отдельный курсор позволяет заполнить строки, сохраняя их начала для расчётного цикла.
        auto cursor = offsets;
        entries.resize(offsets.back());
        for (std::size_t e = 0; e < edges; ++e) {
            const Edge edge = read_edge(e);
            // Модуль вектора площадки вычисляем один раз при подготовке и используем для обеих сторон.
            const Entry face = {e, magnitude(edge.area)};
            entries[cursor[edge.a]++] = face;
            entries[cursor[edge.b]++] = face; // Даже периодическая петля входит в строку узла дважды.
        }
        for (std::size_t b = 0; b < pieces; ++b)
            entries[cursor[boundary[b].node]++] = {edges + b, magnitude(boundary[b].area)};
        if (device >= 0) {
#ifdef AERO_WITH_CUDA
            gpu = cfl_detail::upload(view(), device);
#else
            throw std::invalid_argument("CUDA is not enabled in timestep build");
#endif
        }
    }
    ~Impl() {
#ifdef AERO_WITH_CUDA
        if (gpu)
            cfl_detail::release(gpu);
#endif
    }
};

CflTimestep::CflTimestep(const MeshView& mesh, const CvView& cv, int device) : impl_(new Impl) {
    if (mesh.node_count != cv.node_count || mesh.edge_count != cv.edge_count ||
        mesh.dimension != cv.dimension ||
        (cv.edge_count && (!mesh.edge_begin || !mesh.edge_end || !cv.edge_areas)))
        throw std::invalid_argument("incompatible mesh and CV geometry");
    impl_->prepare(
        extent(cv.node_count), extent(cv.edge_count), extent(cv.boundary_count), cv.volumes, cv.boundary,
        [&](std::size_t i) -> Edge { return {mesh.edge_begin[i], mesh.edge_end[i], cv.edge_areas[i]}; },
        device);
}
CflTimestep::CflTimestep(const PeriodicView& cv, int device) : impl_(new Impl) {
    if (cv.edge_count && !cv.edges)
        throw std::invalid_argument("missing CFL periodic edges");
    impl_->prepare(
        extent(cv.node_count), extent(cv.edge_count), extent(cv.boundary_count), cv.volumes, cv.boundary,
        [&](std::size_t i) -> Edge {
            const PeriodicEdge& edge = cv.edges[i];
            return {edge.begin, edge.end, edge.area};
        },
        device);
}
CflTimestep::~CflTimestep() {}
CflStep CflTimestep::compute(double cfl, WaveSpeeds speeds, void* stream) const {
    const Impl& p = *impl_;
    if (!(cfl > 0 && std::isfinite(cfl)) || speeds.edge_count != p.edge_count ||
        speeds.boundary_count != p.boundary_count || (speeds.edge_count && !speeds.edge) ||
        (speeds.boundary_count && !speeds.boundary))
        throw std::invalid_argument("invalid CFL or wave-speed arrays");
    CflStep result = cfl_detail::unconstrained();
#ifdef AERO_WITH_CUDA
    if (p.gpu)
        result = cfl_detail::reduce(p.gpu, speeds, cfl, stream);
    else
#endif
    {
        if (stream)
            throw std::invalid_argument("CUDA stream supplied to CPU timestep");
        const auto grid = p.view();
        result = cfl_detail::host_min(grid.nodes,
                                      [&](size_t n) { return cfl_detail::local(grid, speeds, cfl, n); });
    }
    if (result.dt < 0)
        throw std::domain_error("invalid wave speed or CFL sum at node " + std::to_string(result.node));
    if (result.node == cfl_detail::unconstrained().node)
        throw std::domain_error("no finite global CFL step");
    return result;
}
}
