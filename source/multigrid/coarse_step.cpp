#include "coarse_step.h"
#include "../implicit/bdf1_linear_system.h"
#include "../implicit/bicgstab_vectors.h"
#include "../parallel/parallel_for.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_api.h"
#endif
#include <cmath>

namespace aero {
namespace multigrid {
// Готовит операторы переноса и рабочие векторы грубосеточной коррекции.
CoarseStep::CoarseStep(const residual_detail::Geometry& host, const residual_detail::Geometry& active,
                       MeshIndex rows, int device, const implicit::Partition& partition)
    : partition_(partition), graph_(host, rows, partition.host_owned), device_(device),
      size_(graph_.count()*host.components()), linear_gas_(active.options.background.active()) {
    transfer_.offsets = graph_.offsets.data();
    transfer_.members = graph_.members.data();
    transfer_.owned = partition.owned;
    transfer_.width = active.components();
    transfer_.gamma = active.options.gamma;
#ifdef AERO_HAS_VISCOSITY
    transfer_.wall = active.options.wall;
#endif
    // Грубый оператор применяем как Pᵀ J P. Вторая матрица и буфер для обмена
    // грубыми блоками не требуются даже при CSGS и компактном вязком якобиане.
    const MeshIndex stride = std::max<MeshIndex>(1,size_);
    const MeshIndex fine_size = rows*host.components();
    if (fine_size > host_.max_size() || stride > (host_.max_size()-fine_size)/(3+2*Directions))
        throw std::length_error("Steady multigrid: coarse vectors too large");
    residual_ = workspace_.allocate(device_,host_,stride*(3+2*Directions)+fine_size);
    solution_ = residual_+stride;
    transfer_.inverse_mass = solution_+stride;
    directions_ = transfer_.inverse_mass+stride;
    images_ = directions_+Directions*stride;
    transfer_.smoother = images_+Directions*stride;
#ifdef AERO_WITH_KOKKOS_GPU
    if (device_ >= 0) {
        gpu::DeviceScope scope(device_);
        gpu::checked(scope.status);
        transfer_.offsets = workspace_.arrays.upload(graph_.offsets);
        transfer_.members = workspace_.arrays.upload(graph_.members);
    }
#endif
}

// Выполняет ограничение или продление вектора на CPU либо GPU.
void CoarseStep::transfer(Transfer::Operation op,double* coarse) {
    transfer_.operation = op;
    transfer_.coarse = coarse;
    partition_.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (device_ >= 0) {
            device_work::check(device_work::aero_device_multigrid_transfer(device_,size_,transfer_));
            return;
        }
#endif
        parallel_for(size_,transfer_);
    });
}

// Выполняет выбранное поэлементное обновление рабочих векторов коррекции.
void CoarseStep::update(VectorUpdate::Operation op,double* target,const double* source,double factor) {
    const VectorUpdate work{op,target,source,transfer_.inverse_mass,factor};
    partition_.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (device_ >= 0) {
            device_work::check(device_work::aero_device_multigrid_update(device_,size_,work));
            return;
        }
#endif
        parallel_for(size_,work);
    });
}

// Вычисляет глобальное скалярное произведение для подбора грубой поправки.
double CoarseStep::dot(const double* x,const double* y) const {
    double value = 0.;
    partition_.local([&] {
        if (!size_) return;
#ifdef AERO_WITH_KOKKOS_GPU
        if (device_ >= 0) {
            device_work::check(device_work::aero_device_krylov_reduce(device_,size_,x,y,&value));
            return;
        }
#endif
        host_parallel::ensure_runtime();
        value = implicit::krylov_reduce(host_parallel::Execution(),size_,x,y);
    });
    return partition_.reduce ? partition_.reduce(value,false) : value;
}

// Решает малую систему для коэффициентов направления коррекции.
bool CoarseStep::solve(implicit::BdfLinearSystem& matrix) {
    update(VectorUpdate::Clear,solution_);
    double squared_norm[Directions] = {};
    bool advanced = false;
    for (unsigned k = 0; k < Directions; ++k) {
        double* direction = directions_+k*size_;
        double* image = images_+k*size_;
        update(VectorUpdate::ScaleResidual,direction,residual_);
        transfer(Transfer::Prolong,direction);
        // apply обновляет удалённые компоненты P e. Поэтому на границе двух
        // процессов в произведение входят те же блоки, что в исходном якобиане.
        matrix.apply(transfer_.fine,transfer_.product);
        transfer(Transfer::RestrictProduct,image);
        // Ортогонализуем образы направлений. Тот же коэффициент переносим на
        // сами направления, сохраняя соотношение image = (Pᵀ J P) direction.
        for (unsigned j = 0; j < k; ++j) {
            const double projection = dot(image,images_+j*size_)/squared_norm[j];
            update(VectorUpdate::Add,image,images_+j*size_,-projection);
            update(VectorUpdate::Add,direction,directions_+j*size_,-projection);
        }
        squared_norm[k] = dot(image,image);
        if (!(squared_norm[k] > 0.) || !std::isfinite(squared_norm[k])) break;
        const double length = dot(residual_,image)/squared_norm[k];
        if (!std::isfinite(length) || length == 0.) break;
        update(VectorUpdate::Add,solution_,direction,length);
        update(VectorUpdate::Add,residual_,image,-length);
        advanced = true;
    }
    return advanced;
}

// Применяет заданную долю найденного направления к пробному полю.
void CoarseStep::set_fraction(Transfer::Operation trial,double fraction) {
    transfer_.fraction = fraction;
    transfer(trial,solution_);
    if (partition_.share_state) partition_.share_state(transfer_.state);
}

// Один проход существующего BJ/CSGS по дефекту после грубой поправки.
void CoarseStep::smooth(implicit::BdfLinearSystem& matrix) {
    transfer(Transfer::Prolong,solution_);
    matrix.apply(transfer_.fine,transfer_.product);
    transfer(Transfer::SmoothingRhs,solution_);
    matrix.predict(transfer_.fine,transfer_.smoother);
}

// Оценивает долю коррекции с учётом предыдущего успешного направления.
double CoarseStep::history_fraction(implicit::BdfLinearSystem& matrix) {
    // В пограничном слое даже слабый скачок постоянной поправки между двумя
    // агрегатами может сильно изменить вязкий поток. Изменение уже принятого
    // поля v даёт дополнительное направление, непрерывное на стыках агрегатов.
    transfer(Transfer::HistoryDirection,nullptr);
    matrix.apply(transfer_.fine,transfer_.product);
    transfer(Transfer::RestrictProduct,images_);
    // Минимум ||R b − a R J v||₂. Перед проверкой полной невязки разрешаем
    // продолжение не более чем на два принятых изменения газового поля.
    const double denominator = dot(images_,images_);
    if (!(denominator > 0.) || !std::isfinite(denominator)) return 0.;
    const double length = dot(residual_,images_)/denominator;
    return std::isfinite(length) ? std::max(0.,std::min(2.,length)) : 0.;
}

// Проверяет пробное направление по полной невязке и принимает только пригодную коррекцию.
double CoarseStep::try_direction(Transfer::Operation trial,double fraction,double accepted_norm,
                                 const std::function<double()>& trial_norm) {
    // Все пробы начинаются от одного поля после обычного мелкосеточного шага.
    transfer(Transfer::SaveField,solution_);
    try {
        for (unsigned attempt = 0; attempt < 4; ++attempt, fraction *= .5) {
            set_fraction(trial,fraction);
            // Проверка до расчёта вязкости и SA: недопустимое пробное давление
            // отклоняет лишь грубую поправку, а не принятый псевдошаг.
            if (!linear_gas_) {
                transfer(Transfer::CheckGas,images_);
                if (dot(images_,images_) != 0.) continue;
            }
            const double value = trial_norm();
            if (std::isfinite(value) && value < accepted_norm) return fraction;
        }
    } catch (...) {
        // Даже при исключении в вычислении потока состояние остаётся принятым.
        set_fraction(trial,0.);
        throw;
    }
    set_fraction(trial,0.);
    return 0.;
}

// Возвращает принятую долю поправки. После отказа поле и потоки восстановлены.
double CoarseStep::correct(implicit::BdfLinearSystem& matrix,double* state,const double* volumes,double dt,
                          double* negative_residual,double* product,double accepted_norm,
                          const std::function<double()>& trial_norm,const std::function<void()>& restore_flux,
                          const double* start) {
    transfer_.state = state;
    transfer_.fine = negative_residual;
    transfer_.product = product;
    transfer_.volumes = volumes;
    transfer_.steps = SteadyNodeSteps::active(partition_.steady_steps);
    transfer_.dt = dt;
    transfer_.start = start;
    used_history_ = false;
    transfer(Transfer::RestrictResidual,residual_);
    bool tried = false;
    if (start) {
        const double length = history_fraction(matrix);
        if (length > 0.) {
            tried = true;
            const double fraction = try_direction(Transfer::HistoryTrial,length,accepted_norm,trial_norm);
            if (fraction > 0.) { used_history_ = true; return fraction; }
        }
    }
    if (solve(matrix)) {
        smooth(matrix);
        tried = true;
        const double fraction = try_direction(Transfer::TrialField,1.,accepted_norm,trial_norm);
        if (fraction > 0.) return fraction;
    }
    if (tried) restore_flux();
    return 0.;
}
}
}
