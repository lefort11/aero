#pragma once
#include "sa.h"
#include "face_geometry.h"
#include "../parallel/parallel_for.h"
#include "../aero/mpi_periodic.h"

namespace aero {
namespace viscosity { struct Operator; }
namespace turbulence {
// Расстояния относятся к исходным геометрическим граням. При периодической склейке
// несколько координат могут соответствовать одной расчётной неизвестной.
std::vector<double> wall_distances(const MeshPartition &, const PeriodicPartition *, const BoundaryPatches &,
                                   const std::vector<PeriodicPair> &, const Processes &, MeshIndex nodes);
// Владеет полями SA, расстояниями до стенки и упакованными диффузионными связями.
struct Storage {
    // Рабочая вязкость, обратные расстояния до стенки и временные поля.
    std::vector<double> w, distance, work;
    // Нужны дополнительные массивы неявного шага SA.
    bool implicit_work=false;
    // Узловой кэш молекулярной и турбулентной вязкости.
    std::vector<View::Material> nodal_material;
    // Диапазоны диффузионных связей и конвективных граней.
    std::vector<RowRange> diffusion_rows, face_rows;
    // Полные геометрические записи конвекции.
    std::vector<FaceTerm> face_terms;
    // Упакованные грани с осевой нормалью.
    std::vector<AxisFace> axis_faces;
    // Отдельные столбцы ориентаций и номеров соседей.
    std::vector<std::uint32_t> face_orientations, face_neighbours;
    // Меры площадок столбцовой раскладки.
    std::vector<double> face_areas;
    // Входящие коэффициенты конвекции для неявных итераций.
    std::vector<double> incoming;
    // Узлы цепочек сглаживания.
    std::vector<LineVertex> line_vertices;
    // Диапазоны независимых участков цепочек.
    std::vector<RowRange> line_segments;
    // Строит короткие цепочки узлов для сглаживания уравнения SA.
    void prepare_lines(const viscosity::Operator& aes,double scale,const std::vector<MeshIndex>& owned);
    // Шаг чтения элементов строки конвекции.
    unsigned face_stride = 1;
    MeshIndex gas_face_entries=0; // Длина исходного списка: по одному вхождению грани на узел.

    // На GPU соседние потоки читают одну позицию соседних строк подряд. Для
    // сильно различающихся длин оставляем обычные строки CSR, без большого дополнения.
    template <class T> static unsigned arrange(bool host, std::vector<RowRange> &rows, std::vector<T> &terms) {
        if (host || rows.empty())
            return 1;
        constexpr size_t lanes = 32;
        // Округляет размер массива до границы группы совместного доступа GPU.
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

    // Подготавливает представление граней конвективного переноса SA.
    template <class Grid> void prepare_faces(const Grid &grid, View &view, bool host) {
        // В списке инцидентности уже есть место для каждого вклада, включая граничные.
        gas_face_entries=grid.nodes ? grid.offsets[grid.nodes] : 0;
        face_terms.resize(gas_face_entries);
        face_rows.resize(grid.nodes);
        parallel_for(grid.nodes, [&](MeshIndex i) {
            for (auto k = grid.offsets[i]; k < grid.offsets[i + 1]; ++k) {
                const auto link = grid.incidence[k];
                face_terms[k]=make_face_term(grid.surfaces[link.face()],link,i,grid.edges,grid.dim);
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
    // Опорные индексы для локальной нумерации диффузионных связей.
    std::vector<MeshIndex> diffusion_anchors;
    // Диффузионные члены в общем представлении.
    std::vector<DiffusionTerm> diffusion_terms;
    // Диффузионные члены в локальной нумерации.
    std::vector<LocalDiffusionTerm> local_terms;
    // Таблица геометрических весов пар.
    std::vector<double> pair_weights;
    // Номер общей формы по элементу.
    std::vector<MeshIndex> shape_index;
    // Временные вклады диффузии по вершинам.
    std::vector<ScalarContribution> vertex_contributions;
    // Строки диффузии для восьмиузловых ячеек.
    std::vector<DiffusionCell> cell_rows;
    // Компактные тройки диффузионных связей.
    std::vector<DiffusionTriple> triples;
    // Проверка выполняется один раз на сетку. Остальные топологии и большие
    // локальные номера по-прежнему используют полный 64-битный список пар.
    bool pack_three_neighbours(size_t node_count);
    // Несколько пар элемента имеют одну среднюю вязкость. На CPU храним
    // полную строку подряд, на GPU чередуем элементы 32 соседних строк.
    bool prepare_element_pairs(const viscosity::Operator& aes,double scale);
    // Упаковывает плотные диффузионные строки для регулярного доступа.
    bool pack_dense_rows(MeshIndex possible);
    // Средние по элементам коэффициенты рабочей и молекулярной вязкости.
    std::vector<double> element_mean, molecular_mean;
    // Число элементов, по которым рассчитываются средние.
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
            work.resize((implicit_work ? 7 : 6) * w.size());
            v.w = w.data();
            v.inverse_distance = distance.data();
            bind_work(v, work.data(), w.size());
        }
    }
    // Размечает общий рабочий буфер SA и связывает его с представлением оператора.
    void bind_work(View &v, double *data, size_t n) const {
        if (!n || !data)
            return;
        v.old = data;
        v.previous = data + n;
        v.backup = data + 2 * n;
        v.rate = data + 3 * n;
        v.diagonal = data + 4 * n;
        v.rotation = data + 5 * n;
        if (implicit_work) v.plain_update=data+6*n;
    }
    // Переносит подготовленные геометрию и коэффициенты SA на GPU.
    template <class Device> void upload(View &v, Device &device,bool share_faces=false) const {
        v.nodal_material = device.template allocate<View::Material>(w.size());
        v.lines={device.upload(line_vertices),device.upload(line_segments),line_segments.size()};
        // Скаляры входящего потока нужны только между итерациями SA при неявном шаге.
        MeshIndex incoming_count=gas_face_entries;
        if (share_faces) {
            v.face_rows=nullptr;
            v.face_terms=nullptr;
            v.axis_faces=nullptr;
            v.face_columns={};
            v.face_stride=1;
        } else {
            v.face_terms=device.upload(face_terms);
            v.axis_faces=device.upload(axis_faces);
            v.face_columns={device.upload(face_orientations),device.upload(face_neighbours),
                            device.upload(face_areas),face_orientations.size()};
            v.face_rows=device.upload(face_rows);
            v.face_stride=face_stride;
            incoming_count=std::max(face_orientations.size(),std::max(axis_faces.size(),face_terms.size()));
        }
        // Когда газ и SA читают один граф граней, массовый поток уже лежит на
        // карте. Из него восстанавливаем входящую скорость на каждой итерации.
        v.incoming=device.template allocate<double>(implicit_work && !share_faces ? incoming_count : 0);
        // Связность и постоянные коэффициенты уже загружены вязким оператором.
        // Для SA на карте дополнительно нужны только средние вязкости элементов.
        v.diffusion = {};
        v.diffusion.molecular_mean = device.template allocate<double>(element_count);
        v.diffusion.element_mean = device.template allocate<double>(element_count);
        v.w = device.upload(w);
        v.inverse_distance = device.upload(distance);
        bind_work(v, device.template allocate<double>((implicit_work ? 7 : 6) * w.size()), w.size());
    }
};
}
}
