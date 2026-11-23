#ifndef AERO_PARTITION_KERNELS_H
#define AERO_PARTITION_KERNELS_H
#include "../parallel/kernel_math.h"
#include "../residual/residual_detail.h"
#include "../explicit/explicit_stage.h"
#include "../mesh/mesh_index.h"
#include "../residual/flux_cfl.h"

namespace aero {
namespace viscosity { struct Operator; }
namespace partition_detail {
using residual_detail::Index;
// Указатели этого представления действительны на устройстве, выбранном данным MPI-процессом.
struct Grid {
    residual_detail::Geometry geometry;
    const MeshIndex *computed, *owned;
    Index computed_count, owned_count;
    const double* volumes;
    double *initial, *state, *original;
    EulerFluxResult* flux;
#ifdef AERO_HAS_TURBULENCE
    const viscosity::Operator* filtered_gas; // На цельной сетке готовим материал при обновлении газа.
#endif
    Index* bad;
    Index *failed_count, *failed_nodes; // Список недопустимых собственных узлов для отката на GPU.
};
// Пространственный оператор хранит интегральные потоки на гранях. Складываем их
// с теми же знаками и вязкими членами, что и при обновлении решения.
struct PressureRate {
    residual_detail::Geometry geometry;
    const EulerFluxResult* flux;
    const double *volumes, *state;
    const MeshIndex* owned;
    double* values; // По одному результату на собственный узел процесса.

    AERO_HOST_DEVICE double at(Index node) const {
        double residual[5];
        if (!residual_detail::assemble_node_residual(geometry, node, flux, residual)) {
            return HUGE_VAL;
        }
        const unsigned dim = geometry.dim;
        // В Linear последняя компонента состояния уже является пульсацией давления.
        double pressure_balance = residual[dim + 1];
        if (!geometry.options.background.active()) {
            // В Euler p = (gamma-1)*(E - |m|²/(2*rho)). Дифференцируем по
            // консервативным переменным в состоянии именно записываемого слоя.
            const double* u = geometry.node_row(state, node);
            double speed_squared = 0.;
            for (unsigned axis = 0; axis < dim; ++axis) {
                const double velocity = u[1 + axis] / u[0];
                pressure_balance -= velocity * residual[1 + axis];
                speed_squared += velocity * velocity;
            }
            pressure_balance = (geometry.options.gamma - 1.) *
                               (pressure_balance + .5 * speed_squared * residual[0]);
        }
        // Невязка положительна при оттоке. Правая часть для узла равна -R/V.
        return -pressure_balance / volumes[node];
    }
    AERO_HOST_DEVICE void operator()(Index row) const {
        values[row] = at(owned ? owned[row] : row);
    }
};
// Нумерация полосы общая для наблюдателей и их соседей. Только владелец
// вычисляет невязку узла. На другие ранги уходит один признак состояния.
struct PressureSite {
    MeshIndex row, node;
    bool monitor, owner;
};
struct ObservePressure {
    PressureRate rate;
    const PressureSite* sites;
    PressureTrend* history;
    double* signals;
    double reference;
    bool first;
    AERO_HOST_DEVICE void operator()(Index i) const {
        const auto site = sites[i];
        signals[i] = 0.;
        if (!site.monitor || !site.owner) return;
        EulerState gas{};
        const auto& g = rate.geometry;
        const double rhs = rate.at(site.node);
        if (euler_state(g.dim,g.node_row(rate.state,site.node),g.options.gamma,&gas)!=EulerStatus::Success ||
            !(::fabs(rhs)<=max_finite_double)) { signals[i]=-1.; return; }
        if (first) { history[i] = {gas.pressure,rhs,0.,0,0,false,gas.pressure,rhs}; return; }
        signals[i] = history[i].observe(gas.pressure,rhs,reference);
    }
};
struct UpdatePressureCaps {
    const PressureSite* sites;
    const MeshIndex *offsets,*neighbours;
    const double *signals,*steps;
    double* caps;
    residual_detail::FluxCfl gas;
    double fraction,courant;
    bool first;
    AERO_HOST_DEVICE void operator()(Index i, MeshIndex& event) const {
        const auto site=sites[i];
        if (!site.owner) return;
        // Ошибка имеет приоритет над уведомлением об уменьшении шага.
        if (signals[i]<0.) { event=0; return; }
        if (signals[i]==2. && i+1<event) event=i+1;
        if (first) { caps[i]=0.; return; }
        double nearby=0.;
        for (auto k=offsets[i]; k<offsets[i+1]; ++k)
            nearby=::fmax(nearby,signals[neighbours[k]]);
        if (caps[i]==0. && nearby!=2. && signals[i]!=2.) return;
        caps[i]=pressure_step_cap(caps[i],steps[site.node]*fraction,signals[i],nearby,
                                  gas.local(site.row,courant).dt);
    }
};
// Тот же критерий плотности и давления используется на обеих архитектурах.
AERO_HOST_DEVICE inline bool admissible_node(const residual_detail::Geometry& geometry, const double* state, Index node) {
    EulerState gas;
    return euler_state(geometry.dim, geometry.node_row(state, node), geometry.options.gamma, &gas) == EulerStatus::Success;
}
// Обновляет один собственный узел прямо в массиве состояния. k — его номер среди узлов процесса.
template <unsigned fixed_width = 0>
AERO_HOST_DEVICE inline bool update_node(const residual_detail::Geometry& geometry,
                                         const EulerFluxResult* flux, const double* volumes,
                                         const MeshIndex* owned, double* state, double* original, Index k,
                                         explicit_detail::Stage stage, double dt, bool save_original) {
    const Index node = owned ? owned[k] : k;
    const unsigned width = fixed_width ? fixed_width : geometry.components();
    double residual[5];
    // Сначала собираем всю строку выходящих потоков из готового буфера стадии.
    if (!residual_detail::assemble_node_residual<fixed_width>(geometry, node, flux, residual))
        return false;

    for (unsigned c = 0; c < width; ++c) {
        // В поле есть и узлы соседей. В компактном снимке позиция k считается только среди собственных узлов.
        double& current = state[node * width + c];
        const Index saved_entry = k * width + c;
        if (save_original)
            original[saved_entry] = current;
        // На стадиях со смешиванием читаем U_n. Для Эйлера буфер снимка может отсутствовать.
        const double step_start = stage.uses_step_start() ? original[saved_entry] : current;
        if (!explicit_detail::advance_component(stage, step_start, current, residual[c], volumes[node], dt,
                                                current))
            return false;
    }
    // Без списка owned обрабатываем всё поле. Пока строка в кэше, готовим газ для следующей стадии.
    // В разделённой сетке сначала должен закончиться обмен состояниями с соседями.
    if (!owned && geometry.packed.gas) {
        residual_detail::decode_nodal_gas(geometry.dim, state, geometry.options.gamma, node, geometry.packed.gas);
        if (!geometry.options.retry_nonphysical && !(geometry.packed.gas[node].density > 0.)) return false;
    } else if (!geometry.options.retry_nonphysical && !geometry.options.background.active()) {
        // Даже на последней стадии проверяем допустимость принятого состояния.
        EulerState gas;
        if (euler_state(geometry.dim, state + node * width, geometry.options.gamma, &gas) != EulerStatus::Success)
            return false;
    }
    return true;
}
// Вид передаваемой записи. Обычное поле имеет width последовательных чисел.
 // Для потока к ним добавляется оценка скорости, хранящаяся отдельно в EulerFluxResult.
struct RecordField {
    unsigned width;
    double* values;
    EulerFluxResult* flux;
    AERO_HOST_DEVICE unsigned stride() const { return width + unsigned(flux != nullptr); }
    AERO_HOST_DEVICE void copy_record(MeshIndex i, MeshIndex k, const double* input, double* output) const {
        double* field = flux ? flux[i].flux : values + i * width;
        const auto offset = k * stride();
        for (unsigned v = 0; v < width; ++v) {
            if (output) output[offset + v] = field[v];
            else field[v] = input[offset + v];
        }
        if (flux) {
            if (output) output[offset + width] = flux[i].max_wave_speed;
            else flux[i].max_wave_speed = input[offset + width];
        }
    }
};
void check_newton_nodes(Grid grid);
void update(Grid grid, explicit_detail::Stage stage, double dt, bool save_original);
void transfer(RecordField field, const MeshIndex* ids, Index count, const double* input, double* output);
}
}
#endif
