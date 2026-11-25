#ifndef AERO_PARTITION_KERNELS_H
#define AERO_PARTITION_KERNELS_H
#include "../parallel/kernel_math.h"
#include "../residual/residual_detail.h"
#include "../explicit/explicit_stage.h"
#include "../mesh/mesh_index.h"

namespace aero {
namespace viscosity { struct Operator; }
namespace turbulence { struct View; }
namespace partition_detail {
using residual_detail::Index;
// Указатели этого представления действительны на устройстве, выбранном данным MPI-процессом.
struct Grid {
    // Связность и метрики для обхода контрольных объёмов.
    residual_detail::Geometry geometry;
    // Списки вычисляемых узлов и узлов во владении ранга.
    const MeshIndex *computed, *owned;
    // Число элементов в каждом из двух списков.
    Index computed_count, owned_count;
    // Объёмы узловых контрольных объёмов.
    const double* volumes;
    // Начальные данные, текущее состояние и сохранённое поле шага.
    double *initial, *state, *original;
    // Потоки газа, индексируемые по площадке.
    EulerFluxResult* flux;
#ifdef AERO_HAS_TURBULENCE
    const viscosity::Operator* filtered_gas; // На цельной сетке готовим материал при обновлении газа.
#endif
    // Рабочая память для регистрации недопустимых состояний.
    Index* bad;
    Index *failed_count, *failed_nodes; // Список недопустимых собственных узлов для отката на GPU.
};
// Пространственный оператор хранит интегральные потоки на гранях. Складываем их
// с теми же знаками и вязкими членами, что и при обновлении решения.
struct PressureRate {
    // Геометрия для суммирования потоков вокруг узла.
    residual_detail::Geometry geometry;
    // Потоки через площадки на текущем поле.
    const EulerFluxResult* flux;
    // Объём и консервативное поле для пересчёта скорости изменения давления.
    const double *volumes, *state;
    // Отображение собственной строки в узел подобласти.
    const MeshIndex* owned;
    double* values; // По одному результату на собственный узел процесса.

    // Вычисляет производную давления одного собственного узла из собранной невязки.
    AERO_HOST_DEVICE void operator()(Index row) const {
        const Index node = owned ? owned[row] : row;
        double residual[5];
        if (!residual_detail::assemble_node_residual(geometry, node, flux, residual)) {
            values[row] = HUGE_VAL; // Процесс сообщит номер недопустимого узла до обмена.
            return;
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
        values[row] = -pressure_balance / volumes[node];
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

// При совмещении с подготовкой материала GPU обрабатывает всю сетку в прямой нумерации.
template<unsigned width, bool fused_sa = false>
KOKKOS_FORCEINLINE_FUNCTION bool advance_owned_node(const residual_detail::Geometry& geometry,
        const EulerFluxResult* flux, const double* volumes, const MeshIndex* owned, double* state,
        double* original, Index k, explicit_detail::Stage stage, double dt, bool save,
        const viscosity::Operator* material, const turbulence::View* sa) {
#ifdef AERO_HAS_TURBULENCE
    if (fused_sa) sa = material ? &material->sa : nullptr;
    if (sa && (!fused_sa || sa->active))
        sa->advance_explicit(fused_sa || !owned ? k : owned[k],stage,dt,save);
#else
    (void)sa;
#endif
    if (!update_node<width>(geometry,flux,volumes,owned,state,original,k,stage,dt,save)) return false;
#ifdef AERO_HAS_VISCOSITY
    if (material) material->prepare_gas(k,state,geometry.packed.gas);
#else
    (void)material;
#endif
    return true;
}
// Вид передаваемой записи. Обычное поле имеет width последовательных чисел.
 // Для потока к ним добавляется оценка скорости, хранящаяся отдельно в EulerFluxResult.
struct RecordField {
    // Число компонент в записываемой строке.
    unsigned width;
    // Плоский буфер компонент поля.
    double* values;
    // Буфер результатов вычисления потоков.
    EulerFluxResult* flux;
    // Возвращает число значений записи, включая скорость волны для потоков.
    AERO_HOST_DEVICE unsigned stride() const { return width + unsigned(flux != nullptr); }
    // Переносит одну запись состояния или потока между индексированным полем и буфером обмена.
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
// Запускает обновление только собственных степеней свободы MPI-процесса.
void check_newton_nodes(Grid grid);
// Запускает GPU-обновление собственных узлов выбранной явной стадией.
void update(Grid grid, explicit_detail::Stage stage, double dt, bool save_original);
// Переносит выбранные узловые записи между полем и файлом по заданным смещениям.
void transfer(RecordField field, const MeshIndex* ids, Index count, const double* input, double* output);
}
}
#endif
