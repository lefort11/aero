#pragma once
#include "device_execution.h"
#include "device_volume_coordinates.h"
#include "../viscosity/transport.h"

namespace aero {
namespace device_work {
namespace affine_aes {
using Geometry = viscosity::AffineHex;
using viscosity::Gradient;
using Team = Kokkos::TeamPolicy<Execution>;
// На каждый элемент назначены 32 исполнителя: восемь вершин для каждой из четырёх компонент.
struct Workspace {
    // Узловые значения для обработки гексаэдра.
    FixedMatrix<double,4,8> value;
    // Средние компоненты градиента.
    FixedMatrix<double,3,3> mean;
    // Отклонения и соответствующие вклады в вязкую силу.
    FixedMatrix<double,3,4> departure, force;
};
// Командное ядро вязких градиентов аффинного гексаэдра.
struct Apply {
    // Метрики аффинной геометрии.
    const Geometry *geometry;
    // Таблица ячеек в выбранном формате индексов.
    viscosity::CellTable cells;
    // Газовые величины на узлах.
    const viscosity::Gas *gas;
    // Выходной массив градиентов.
    Gradient *gradients;
    const double *pair_heat; // Исходные коэффициенты теплопроводности, общие с диффузией SA.
    // Начальная позиция вершин обрабатываемой плитки.
    MeshIndex first_vertex;
    // Первая ячейка текущей плитки.
    MeshIndex first_cell;

    // При постоянном поле приращения от первого узла точно нулевые.
    KOKKOS_INLINE_FUNCTION static double mean_gradient(const Geometry &g, const double *v, unsigned axis) {
        double sum = 0.;
        for (unsigned i = 1; i < 8; ++i)
            sum += g.basis(axis, i) * (v[i] - v[0]);
        return sum;
    }
    // Вычисляет отклонение значения вершины от аффинного продолжения базового тетраэдра.
    KOKKOS_INLINE_FUNCTION static double departure(const Geometry &g, const double *v, unsigned r) {
        double result = v[4 + r] - v[0];
        for (unsigned k = 1; k < 4; ++k)
            result -= g.interpolation(r, k) * (v[k] - v[0]);
        return result;
    }
    // В маленьком блоке связаны все три компоненты скорости. Для температуры уравнение скалярное.
    KOKKOS_INLINE_FUNCTION static double force(const Geometry &g, const Workspace &w, unsigned component,
                                               unsigned r) {
        double result = 0.;
        for (unsigned s = 0; s < 4; ++s)
            for (unsigned d = 0; d < 3; ++d)
                result += g.stress[Geometry::upper_index(3 * r + component, 3 * s + d, 12)] * w.departure(d, s);
        return result;
    }
    // Распределение по узлам сопряжено интерполяции: четыре веса суммируются в единицу.
    KOKKOS_INLINE_FUNCTION static double scatter(const Geometry &g, const double *f, unsigned i) {
        if (i >= 4)
            return f[i - 4];
        double result = 0.;
        for (unsigned r = 0; r < 4; ++r)
            result -= g.interpolation(r, i) * f[r];
        return result;
    }
    // Вычисляет вклад средних напряжений в поток выбранной вершины.
    KOKKOS_INLINE_FUNCTION double mean_flux(const Geometry &g, const Workspace &w, unsigned m, unsigned i) const {
        double sum = 0.;
        const double dilatation = (2. / 3.) * (w.mean(0, 0) + w.mean(1, 1) + w.mean(2, 2));
        for (unsigned d = 0; d < 3; ++d) {
            const double stress = w.mean(m, d) + w.mean(d, m) - (d == m ? dilatation : 0.);
            sum += g.basis(d, i) * stress;
        }
        return g.volume * sum;
    }
    // Коэффициенты теплопроводности нужны также уравнению SA. Берём готовые
    // значения из одной общей таблицы и суммируем в исходном порядке вершин.
    KOKKOS_INLINE_FUNCTION double heat_flux(const Geometry &g, const viscosity::CellView &cell,
                                            const double *temperature, unsigned i) const {
        const unsigned local = g.node[i];
        double sum = 0.;
        for (unsigned j = 0; j < 8; ++j)
            if (j != local)
                sum += pair_heat[cell.pair_index(local, j)] * (temperature[g.ordered[j]] - temperature[i]);
        return sum;
    }
    // Совместно вычисляет градиенты одного гексаэдра потоками команды GPU.
    KOKKOS_INLINE_FUNCTION void operator()(const Team::member_type &team) const {
        const MeshIndex e = first_cell + static_cast<unsigned>(team.league_rank());
        const unsigned tid = team.team_rank(), i = tid % 8, m = tid / 8;
        const auto &g = geometry[e];
        const auto &cell = cells[e];
        const auto local = g.node[i];
        auto &w = *static_cast<Workspace *>(team.team_shmem().get_shmem(sizeof(Workspace)));
        const auto &point = gas[cell.node[local]];
        w.value(m, i) = m == 3 ? point.pressure_over_density : point.velocity[m];
        team.team_barrier();
        if (m < 3 && i < 3)
            w.mean(m, i) = mean_gradient(g, w.value.row(m), i);
        if (m < 3 && i >= 4)
            w.departure(m, i - 4) = departure(g, w.value.row(m), i - 4);
        team.team_barrier();
        if (m < 3 && i < 4)
            w.force(m, i) = force(g, w, m, i);
        team.team_barrier();
        auto &out = gradients[cell.first_vertex - first_vertex + local];
        if (m == 3)
            out.heat = heat_flux(g, cell, w.value.row(3), i);
        else
            out.stress[m] = mean_flux(g, w, m, i) + scatter(g, w.force.row(m), i);
    }
};
// Kokkos выделяет рабочие числа только на время обработки одного элемента.
inline void gradients(const Execution &exec, const viscosity::Operator &g, Gradient *output, MeshIndex first_cell,
                      MeshIndex cells, MeshIndex first_vertex) {
    if (g.tetra.cells) {
        coordinate_aes::gradients(exec,g,output,first_cell,cells,first_vertex);
        return;
    }
    const auto policy =
        Team(exec, static_cast<int>(cells), 32).set_scratch_size(0, Kokkos::PerTeam(sizeof(Workspace)));
    Kokkos::parallel_for("Aero: affine AES", policy,
                         Apply{g.affine, g.cells, g.gas, output, g.stencil.pair_heat, first_vertex, first_cell});
}
// Одни и те же границы плиток нужны при сборке невязки и матрицы. Потребитель
// ставит все операции текущей плитки в очередь Kokkos до повторной записи её буферов.
template<class Calculate>
inline void for_tiles(const viscosity::Operator& g,Calculate calculate) {
    for (MeshIndex t=0;t<g.gradient_tile_count;++t) calculate(g.host_gradient_tiles[t]);
}
// Рабочий массив вмещает одну группу гексаэдров. Сначала все её градиенты
// вычисляются на карте, затем потребитель добавляет вклады в невязку или матрицу.
// Последовательность запусков в одном исполнителе Kokkos защищает от перезаписи.
template<class Accumulate>
inline void for_gradient_tiles(const Execution& exec, const viscosity::Operator& g, Accumulate accumulate) {
    for_tiles(g,[&](viscosity::GradientTile tile) {
        gradients(exec, g, g.gradient_cache, tile.first_cell, tile.cells, tile.first_vertex);
        accumulate(tile);
    });
}
}
}
}
