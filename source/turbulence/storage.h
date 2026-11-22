#pragma once
#include "sa.h"
#include "../parallel/parallel_for.h"
#include "../aero/mpi_periodic.h"

namespace aero {
namespace viscosity { struct Operator; }
namespace turbulence {
// Расстояния относятся к исходным геометрическим граням. При периодической склейке
// несколько координат могут соответствовать одной расчётной неизвестной.
std::vector<double> wall_distances(const MeshPartition &, const PeriodicPartition *, const BoundaryPatches &,
                                   const std::vector<PeriodicPair> &, const Processes &, MeshIndex nodes);
struct Storage {
    std::vector<double> w, distance, work;
    bool implicit_work=false;
    std::vector<View::Material> nodal_material;
    std::vector<RowRange> diffusion_rows, face_rows;
    std::vector<FaceTerm> face_terms;
    std::vector<AxisFace> axis_faces;
    std::vector<std::uint32_t> face_orientations, face_neighbours;
    std::vector<double> face_areas;
    std::vector<double> incoming;
    std::vector<LineVertex> line_vertices;
    std::vector<RowRange> line_segments;
    void prepare_lines(const std::vector<MeshIndex>& owned);
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
        // Отдельная группа из 32 строк пишет только в свой диапазон packed.
        const size_t groups = (rows.size() + lanes - 1) / lanes;
        std::vector<size_t> offsets(groups + 1);
        for (size_t group = 0; group < groups; ++group)
            offsets[group+1] = offsets[group] + padded(group*lanes);
        if (offsets.back() > terms.size() + terms.size() / 2)
            return 1;
        std::vector<T> packed(offsets.back());
        parallel_for(groups, [&](size_t group) {
            const auto i = group * lanes, base = offsets[group];
            for (size_t j = i; j < std::min(i + lanes, rows.size()); ++j) {
                auto cursor = base + j - i;
                const auto row = rows[j];
                for (auto k = row.first; k < row.end; ++k, cursor += lanes)
                    packed[cursor] = terms[k];
                rows[j] = {base + j - i, cursor};
            }
        }, 32);
        terms.swap(packed);
        return lanes;
    }

    template <class Grid> void prepare_faces(const Grid &grid, View &view, bool host) {
        // В списке инцидентности уже есть место для каждого вклада, включая граничные.
        face_terms.resize(grid.nodes ? grid.offsets[grid.nodes] : 0);
        face_rows.resize(grid.nodes);
        parallel_for(grid.nodes, [&](MeshIndex i) {
            for (auto k = grid.offsets[i]; k < grid.offsets[i + 1]; ++k) {
                const auto link = grid.incidence[k];
                const auto f = link.face();
                const auto &face = grid.surfaces[f];
                const bool interior = f < grid.edges;
                FaceTerm term{link.encoded, interior ? (face.begin == i ? face.end : face.begin) : i, {}};
                const double weight = link.outward_flux(interior ? .5 : 1.);
                for (unsigned d = 0; d < grid.dim; ++d)
                    term.area[d] = weight * face.area[d];
                face_terms[k] = term;
            }
            face_rows[i] = {grid.offsets[i], grid.offsets[i+1]};
        });
        face_stride = arrange(host, face_rows, face_terms);
        // Компактная запись допустима только при точном нуле двух компонент.
        // Уже одна косая площадка сохраняет общий обход для всей подобласти.
        bool axial=true;
        for (const auto& f:face_terms) {
            unsigned components=0;
            for (double value:f.area) components+=value!=0.;
            if (components>1 || f.encoded>(UINT32_MAX>>2) || f.neighbour>UINT32_MAX) {
                axial=false;
                break;
            }
        }
        if (axial) {
            axis_faces.resize(face_terms.size());
            parallel_for(face_terms.size(), [&](size_t i) {
                const auto& f = face_terms[i];
                const unsigned axis=f.area[1]!=0. ? 1 : (f.area[2]!=0. ? 2 : 0);
                axis_faces[i] = {static_cast<std::uint32_t>(4*f.encoded+axis),
                                      static_cast<std::uint32_t>(f.neighbour),f.area[axis]};
            });
            std::vector<FaceTerm>().swap(face_terms);
        }
        if (!host && !axial && !face_terms.empty() &&
            std::all_of(face_terms.begin(),face_terms.end(),[](const FaceTerm& f) {
                return f.encoded<=UINT32_MAX && f.neighbour<=UINT32_MAX;
            })) {
            const auto count=face_terms.size();
            face_orientations.resize(count);
            face_neighbours.resize(count);
            face_areas.resize(grid.dim*count);
            parallel_for(count, [&](MeshIndex i) {
                const auto& f=face_terms[i];
                face_orientations[i]=static_cast<std::uint32_t>(f.encoded);
                face_neighbours[i]=static_cast<std::uint32_t>(f.neighbour);
                for (unsigned d=0; d<grid.dim; ++d) face_areas[d*count+i]=f.area[d];
            });
            std::vector<FaceTerm>().swap(face_terms);
        }
        view.face_stride = face_stride;
        if (host) {
            incoming.resize(axial ? axis_faces.size() : face_terms.size());
            view.axis_faces=axis_faces.empty() ? nullptr : axis_faces.data();
            view.face_terms = face_terms.data();
            view.face_rows = face_rows.data();
            view.incoming = incoming.data();
        }
    }
    // На повторяющихся ячейках строки часто совпадают до сдвига номеров. Сравниваем
    // все коэффициенты точно. На неструктурированной сетке остаётся обычная раскладка.
    bool share_diffusion_rows();
    std::vector<MeshIndex> diffusion_anchors;
    std::vector<DiffusionTerm> diffusion_terms;
    std::vector<LocalDiffusionTerm> local_terms;
    std::vector<DiffusionTile> diffusion_tiles;
    std::vector<double> pair_weights;
    std::vector<MeshIndex> shape_index;
    std::vector<ScalarContribution> vertex_contributions;
    std::vector<DiffusionCell> cell_rows;
    std::vector<DiffusionTriple> triples;
    // Проверка выполняется один раз на сетку. Остальные топологии и большие
    // локальные номера по-прежнему используют полный 64-битный список пар.
    bool pack_three_neighbours(size_t node_count);
    // Несколько пар элемента имеют одну среднюю вязкость. На CPU храним
    // полную строку подряд, на GPU чередуем элементы 32 соседних строк.
    bool prepare_element_pairs(const viscosity::Operator& aes,double scale);
    template<unsigned lanes> bool pack_dense_rows(MeshIndex possible,std::vector<DiffusionBlock<lanes>>& blocks);
    std::vector<double> element_mean, molecular_mean;
    MeshIndex element_count = 0;
    // Нумерация соседей и интегралы AES постоянны. На стадии нужны только номера
    // двух узлов и средний коэффициент переноса элемента. Матрицу напряжений газа не читаем.
    void prepare_diffusion(const viscosity::Operator& aes, View& view, bool host, bool implicit, const std::vector<MeshIndex>* steady_nodes);
    // Рабочая вязкость живёт отдельно от компонент газа. При выключенной модели массивы пусты.
    void prepare(View &v, const Settings &settings, double reynolds, double prandtl, bool host, bool implicit) {
        implicit_work=implicit;
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
            work.resize((implicit_work ? 8 : 6) * w.size());
            v.w = w.data();
            v.inverse_distance = distance.data();
            bind_work(v, work.data(), w.size());
        }
    }
    void bind_work(View &v, double *data, size_t n) const {
        if (!n || !data)
            return;
        v.old = data;
        v.previous = data + n;
        v.backup = data + 2 * n;
        v.rate = data + 3 * n;
        v.diagonal = data + 4 * n;
        v.rotation = data + 5 * n;
        if (implicit_work) {
            v.last_correction=data+6*n;
            v.plain_update=data+7*n;
        }
    }
    template <class Device> void upload(View &v, Device &device) const {
        v.nodal_material = device.template allocate<View::Material>(w.size());
        v.face_terms = device.upload(face_terms);
        v.axis_faces = device.upload(axis_faces);
        v.face_columns={device.upload(face_orientations),device.upload(face_neighbours),
                        device.upload(face_areas),face_orientations.size()};
        v.lines={device.upload(line_vertices),device.upload(line_segments),line_segments.size()};
        v.face_rows = device.upload(face_rows);
        v.incoming = device.template allocate<double>(implicit_work ? std::max(face_orientations.size(), std::max(axis_faces.size(),face_terms.size())) : 0);
        v.face_stride = face_stride;
        v.diffusion = {device.upload(diffusion_rows),
                       device.upload(diffusion_anchors),
                       diffusion_stride,
                       device.upload(diffusion_terms), device.upload(local_terms), device.upload(diffusion_tiles), nullptr, nullptr, nullptr, nullptr,
                       device.upload(triples),
                       device.template allocate<double>(element_count),
                       device.template allocate<double>(element_count), v.diffusion.short_blocks};
        v.w = device.upload(w);
        v.inverse_distance = device.upload(distance);
        bind_work(v, device.template allocate<double>((implicit_work ? 8 : 6) * w.size()), w.size());
    }
};
}
}
