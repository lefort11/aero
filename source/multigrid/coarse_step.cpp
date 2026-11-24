#include "coarse_step.h"
#include "../implicit/bdf1_linear_system.h"
#include "../implicit/bicgstab_vectors.h"
#include "../parallel/parallel_for.h"
#include "../parallel/upload_compact.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_api.h"
#endif
#include <cmath>
#include <limits>

namespace aero {
namespace multigrid {
namespace {
// Каждый свободный вектор содержит целое число грубых направлений; остатки не пересекаются.
unsigned borrowed_directions(MeshIndex stride,MeshIndex fine_size,const implicit::KrylovScratch& scratch) {
    if (scratch.count && fine_size>std::numeric_limits<MeshIndex>::max()/scratch.count)
        throw std::length_error("Steady multigrid: borrowed storage size overflow");
    const auto capacity=scratch.contiguous ? (MeshIndex(scratch.count)*fine_size)/stride
                                          : MeshIndex(scratch.count)*(fine_size/stride);
    return static_cast<unsigned>(std::min<MeshIndex>(8,capacity));
}
// После направлений и их образов может остаться целый мелкосеточный вектор.
// Только тогда J*v занимает хвост, а переданное поле product сохраняет сглаживатель
// между пробами. Остальные режимы продолжают выделять отдельный сглаживатель.
bool borrow_product(MeshIndex stride,MeshIndex fine_size,const implicit::KrylovScratch& scratch,bool persistent) {
    return persistent && scratch.contiguous && fine_size && scratch.count &&
           MeshIndex(borrowed_directions(stride,fine_size,scratch))*stride<=MeshIndex(scratch.count-1)*fine_size;
}
// Таблица агрегатов подчиняется той же проверке диапазона, что геометрия сетки.
MeshIndex index_bytes(const std::vector<MeshIndex>& values) {
    const bool narrow=values.empty() || *std::max_element(values.begin(),values.end())<=UINT32_MAX;
    return values.size()*(narrow ? sizeof(std::uint32_t) : sizeof(MeshIndex));
}
}
// Оценка повторяет раскладку конструктора, включая индексы переноса и минимальный пустой вектор.
MeshIndex CoarseStep::storage_bytes(const Aggregates& graph,MeshIndex rows,unsigned width,const implicit::KrylovScratch& scratch,
                                   bool product_persistent) {
    const auto limit=std::numeric_limits<MeshIndex>::max();
    if (!width || rows>limit/width || graph.count()>limit/width || graph.count()>limit/2)
        throw std::length_error("Steady multigrid: field size overflow");
    const auto stride=std::max<MeshIndex>(1,graph.count()*width),fine_size=rows*width;
    const auto borrowed=borrowed_directions(stride,fine_size,scratch);
    const auto smoother=borrow_product(stride,fine_size,scratch,product_persistent) ? 0 : fine_size;
    const auto masses=std::max<MeshIndex>(1,2*graph.count());
    if (masses>limit-smoother || stride>(limit-smoother-masses)/(2+2*Directions-borrowed))
        throw std::length_error("Steady multigrid: vector size overflow");
    const auto values=stride*(2+2*Directions-borrowed)+smoother+masses;
    const auto graph_bytes=index_bytes(graph.offsets)+index_bytes(graph.members);
    if (values>(limit-graph_bytes)/sizeof(double))
        throw std::length_error("Steady multigrid: memory size overflow");
    return values*sizeof(double)+graph_bytes;
}
// Готовит операторы переноса и рабочие векторы грубосеточной коррекции.
CoarseStep::CoarseStep(const residual_detail::Geometry& host, const residual_detail::Geometry& active,
                       MeshIndex rows, int device, const implicit::Partition& partition)
    : partition_(partition), graph_(partition.coarse_graph ? partition.coarse_graph :
            std::make_shared<Aggregates>(host,rows,partition.host_owned)), device_(device),
      size_(graph_->count()*host.components()), linear_gas_(active.options.background.active()) {
    transfer_.offsets = graph_->offsets.data();
    transfer_.members = graph_->members.data();
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
    const auto& scratch=partition.krylov_scratch;
    const auto borrowed=device>=0 ? borrowed_directions(stride,fine_size,scratch) : 0;
    const bool share=device>=0 && borrow_product(stride,fine_size,scratch,partition.coarse_product_persistent);
    const auto masses=std::max<MeshIndex>(1,2*graph_->count());
    residual_ = workspace_.allocate(device_,host_,stride*(2+2*Directions-borrowed)+masses+(share ? 0 : fine_size));
    solution_ = residual_+stride;
    transfer_.inverse_mass = solution_+stride;
    double* remaining=transfer_.inverse_mass+masses;
    if (share) borrowed_product_=scratch.field[0]+borrowed*stride;
    else { transfer_.smoother=remaining; remaining+=fine_size; }
    const auto per_vector=fine_size/stride;
    for (unsigned i=0;i<2*Directions;++i) {
        double* field=i<borrowed ? (scratch.contiguous ? scratch.field[0]+i*stride
                                   : scratch.field[i/per_vector]+(i%per_vector)*stride) : remaining;
        if (i>=borrowed) remaining+=stride;
        if (i<Directions) directions_[i]=field;
        else images_[i-Directions]=field;
    }
#ifdef AERO_WITH_KOKKOS_GPU
    if (device_ >= 0) {
        gpu::DeviceScope scope(device_);
        gpu::checked(scope.status);
        transfer_.offsets = upload_indices(workspace_.arrays,graph_->offsets.size(),graph_->offsets.data());
        transfer_.members = upload_indices(workspace_.arrays,graph_->members.size(),graph_->members.data());
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
    const VectorUpdate work{op,target,source,transfer_.inverse_mass,factor,transfer_.width};
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
        double* direction = directions_[k];
        double* image = images_[k];
        update(VectorUpdate::ScaleResidual,direction,residual_);
        transfer(Transfer::Prolong,direction);
        // apply обновляет удалённые компоненты P e. Поэтому на границе двух
        // процессов в произведение входят те же блоки, что в исходном якобиане.
        matrix.apply(transfer_.fine,transfer_.product);
        transfer(Transfer::RestrictProduct,image);
        // Ортогонализуем образы направлений. Тот же коэффициент переносим на
        // сами направления, сохраняя соотношение image = (Pᵀ J P) direction.
        for (unsigned j = 0; j < k; ++j) {
            const double projection = dot(image,images_[j])/squared_norm[j];
            update(VectorUpdate::Add,image,images_[j],-projection);
            update(VectorUpdate::Add,direction,directions_[j],-projection);
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
    transfer(Transfer::RestrictProduct,images_[0]);
    // Минимум ||R b − a R J v||₂. Перед проверкой полной невязки разрешаем
    // продолжение не более чем на два принятых изменения газового поля.
    const double denominator = dot(images_[0],images_[0]);
    if (!(denominator > 0.) || !std::isfinite(denominator)) return 0.;
    const double length = dot(residual_,images_[0])/denominator;
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
                transfer(Transfer::CheckGas,images_[0]);
                if (dot(images_[0],images_[0]) != 0.) continue;
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
    if (borrowed_product_) {
        // Пробы могут уничтожить весь Krylov, включая J*v. Этот образ всегда
        // вычисляется заново перед чтением. Сглаживатель лежит вне этого буфера.
        transfer_.product=borrowed_product_;
        transfer_.smoother=product;
    }
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
