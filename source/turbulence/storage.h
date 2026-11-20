#pragma once
#include "sa.h"
#include <tuple>
#include "../aero/mpi_periodic.h"

namespace aero {
namespace turbulence {
// Расстояния относятся к исходным геометрическим граням. При периодической склейке
// несколько координат могут соответствовать одной расчётной неизвестной.
std::vector<double> wall_distances(const MeshPartition &, const PeriodicPartition *, const BoundaryPatches &,
                                   const std::vector<PeriodicPair> &, const Processes &, MeshIndex nodes);
struct Storage {
    std::vector<double> w, distance, work;
    std::vector<View::Material> nodal_material;
    std::vector<RowRange> diffusion_rows, face_rows;
    std::vector<FaceTerm> face_terms;
    std::vector<double> incoming;
    unsigned face_stride = 1, diffusion_stride = 1;

    // На GPU соседние потоки читают одну позицию соседних строк подряд. Для
    // сильно различающихся длин оставляем обычные строки CSR, без большого дополнения.
    template <class T> static unsigned arrange(bool host, std::vector<RowRange> &rows, std::vector<T> &terms) {
        if (host || rows.empty())
            return 1;
        constexpr size_t lanes = 32;
        const auto padded = [&](size_t i) {
            size_t longest = 0;
            for (size_t j = i; j < std::min(i + lanes, rows.size()); ++j)
                longest = std::max(longest, static_cast<size_t>(rows[j].end - rows[j].first));
            return longest * lanes;
        };
        size_t total = 0;
        for (size_t i = 0; i < rows.size(); i += lanes)
            total += padded(i);
        if (total > terms.size() + terms.size() / 2)
            return 1;
        std::vector<T> packed(total);
        size_t base = 0;
        for (size_t i = 0; i < rows.size(); i += lanes) {
            const auto block = padded(i);
            for (size_t j = i; j < std::min(i + lanes, rows.size()); ++j) {
                auto cursor = base + j - i;
                const auto row = rows[j];
                for (auto k = row.first; k < row.end; ++k, cursor += lanes)
                    packed[cursor] = terms[k];
                rows[j] = {base + j - i, cursor};
            }
            base += block;
        }
        terms.swap(packed);
        return lanes;
    }

    template <class Grid> void prepare_faces(const Grid &grid, View &view, bool host) {
        for (MeshIndex i = 0; i < grid.nodes; ++i) {
            const auto first = face_terms.size();
            for (auto k = grid.offsets[i]; k < grid.offsets[i + 1]; ++k) {
                const auto link = grid.incidence[k];
                const auto f = link.face();
                const auto &face = grid.surfaces[f];
                const bool interior = f < grid.edges;
                FaceTerm term{link.encoded, interior ? (face.begin == i ? face.end : face.begin) : i, {}};
                const double weight = link.outward_flux(interior ? .5 : 1.);
                for (unsigned d = 0; d < grid.dim; ++d)
                    term.area[d] = weight * face.area[d];
                face_terms.push_back(term);
            }
            face_rows.push_back({first, face_terms.size()});
        }
        face_stride = arrange(host, face_rows, face_terms);
        view.face_stride = face_stride;
        if (host) {
            incoming.resize(face_terms.size());
            view.face_terms = face_terms.data();
            view.face_rows = face_rows.data();
            view.incoming = incoming.data();
        }
    }
    // На повторяющихся ячейках строки часто совпадают до сдвига номеров. Сравниваем
    // все коэффициенты точно. На неструктурированной сетке остаётся обычная раскладка.
    bool share_diffusion_rows() {
        struct Compare {
            bool operator()(const std::vector<DiffusionTerm> &a, const std::vector<DiffusionTerm> &b) const {
                return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
                                                    [](const DiffusionTerm &x, const DiffusionTerm &y) {
                                                        return std::tie(x.element, x.neighbour, x.geometry) <
                                                               std::tie(y.element, y.neighbour, y.geometry);
                                                    });
            }
        };
        std::map<std::vector<DiffusionTerm>, RowRange, Compare> dictionary;
        std::vector<RowRange> rows;
        std::vector<DiffusionTerm> shared, key;
        std::vector<MeshIndex> anchors;
        const auto limit = diffusion_terms.size() / 2;
        for (MeshIndex i = 0; i < diffusion_rows.size(); ++i) {
            const auto row = diffusion_rows[i];
            const auto anchor = row.first < row.end ? diffusion_terms[row.first].element : 0;
            key.clear();
            for (auto p = row.first; p < row.end; ++p) {
                const auto &term = diffusion_terms[p];
                // Индексы беззнаковые: обратное сложение восстанавливает и номера меньше базового.
                key.push_back({term.element - anchor, term.neighbour - i, term.geometry});
            }
            auto it = dictionary.lower_bound(key);
            if (it == dictionary.end() || Compare{}(key, it->first)) {
                if (shared.size() + key.size() > limit)
                    return false;
                const RowRange span{shared.size(), shared.size() + key.size()};
                shared.insert(shared.end(), key.begin(), key.end());
                if (dictionary.size() >= 4096)
                    dictionary.clear();
                it = dictionary.emplace(key, span).first;
            }
            rows.push_back(it->second);
            anchors.push_back(anchor);
        }
        diffusion_rows.swap(rows);
        diffusion_terms.swap(shared);
        diffusion_anchors.swap(anchors);
        return true;
    }
    std::vector<MeshIndex> diffusion_anchors;
    std::vector<DiffusionTerm> diffusion_terms;
    std::vector<double> element_mean, molecular_mean;
    MeshIndex element_count = 0;
    // Нумерация соседей и интегралы AES постоянны. На стадии нужны только номера
    // двух узлов и средний коэффициент переноса элемента. Матрицу напряжений газа не читаем.
    template <class Aes> void prepare_diffusion(const Aes &aes, View &view, bool host) {
        element_count = aes.cell_count;
        diffusion_rows.reserve(aes.row_count);
        const double scale = (aes.law.gamma - 1.) * aes.law.prandtl / (aes.law.gamma * sigma);
        for (MeshIndex node = 0; node < aes.row_count; ++node) {
            const auto first = diffusion_terms.size();
            for (auto pos = aes.offsets[node]; pos < aes.offsets[node + 1]; ++pos) {
                const auto entry = aes.incidence[pos], e = entry / 8;
                const unsigned i = static_cast<unsigned>(entry % 8);
                const auto &cell = aes.cells[e];
                for (unsigned j = 0; j < cell.count; ++j)
                    if (cell.node[j] != node) {
                        const double heat = aes.stencil.metric(cell, i, j).heat;
                        if (heat != 0.)
                            diffusion_terms.push_back({e, cell.node[j], -heat * scale});
                    }
            }
            diffusion_rows.push_back({first, diffusion_terms.size()});
        }
        if (!share_diffusion_rows())
            diffusion_stride = arrange(host, diffusion_rows, diffusion_terms);
        if (host) {
            element_mean.resize(element_count);
            molecular_mean.resize(element_count);
            view.diffusion = {
                diffusion_rows.data(), diffusion_anchors.empty() ? nullptr : diffusion_anchors.data(),
                diffusion_stride,      diffusion_terms.data(),
                molecular_mean.data(), element_mean.data()};
        }
    }
    // Рабочая вязкость живёт отдельно от компонент газа. При выключенной модели массивы пусты.
    void prepare(View &v, const Settings &settings, double reynolds, double prandtl, bool host) {
        v.active = true;
        v.ambient = ambient_chi(settings.nut_nu_amb) / reynolds;
        if (!std::isfinite(v.ambient))
            throw std::invalid_argument("SA: freestream working viscosity is out of range");
        v.pr_ratio = prandtl / settings.prandtl;
        w.assign(distance.size(), v.ambient);
        for (size_t i = 0; i < w.size(); ++i)
            if (distance[i] < 0.)
                w[i] = 0.;
        if (host && !w.empty()) {
            nodal_material.resize(w.size());
            v.nodal_material = nodal_material.data();
            work.resize(6 * w.size());
            v.w = w.data();
            v.inverse_distance = distance.data();
            bind_work(v, work.data(), w.size());
        }
    }
    static void bind_work(View &v, double *data, size_t n) {
        if (!n || !data)
            return;
        v.old = data;
        v.previous = data + n;
        v.backup = data + 2 * n;
        v.rate = data + 3 * n;
        v.diagonal = data + 4 * n;
        v.rotation = data + 5 * n;
    }
    template <class Device> void upload(View &v, Device &device) const {
        v.nodal_material = device.template allocate<View::Material>(w.size());
        v.face_terms = device.upload(face_terms);
        v.face_rows = device.upload(face_rows);
        v.incoming = device.template allocate<double>(face_terms.size());
        v.face_stride = face_stride;
        v.diffusion = {device.upload(diffusion_rows),
                       device.upload(diffusion_anchors),
                       diffusion_stride,
                       device.upload(diffusion_terms),
                       device.template allocate<double>(element_count),
                       device.template allocate<double>(element_count)};
        v.w = device.upload(w);
        v.inverse_distance = device.upload(distance);
        bind_work(v, device.template allocate<double>(6 * w.size()), w.size());
    }
};
}
}
