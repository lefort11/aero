#pragma once
#include "scalar_system.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_work.h"
#endif
#include "../implicit/krylov_operations.h"
#include "../implicit/ordered_sweep.h"
#include "../implicit/newton_predictor.h"
#include "../implicit/time_prediction.h"
#include "../implicit/prediction_execution.h"
#include "../implicit/workspace.h"

namespace aero { namespace turbulence {
// Владеет только численными коэффициентами SA и недостающими рабочими векторами.
// Газовый RHS можно занять до обязательного пересчёта невязки пары полей.
class ScalarSolver : public implicit::KrylovOperations<ScalarSolver> {
    implicit::Partition part_; // Обмен одного скаляра и общие редукции рангов.
    implicit::Workspace buffers_; // Единственный владелец выделений на карте.
    std::vector<double> host_; // Такое же рабочее пространство на CPU.
    std::vector<implicit::LinearCoefficient> coefficients_; // Внедиагональные значения CPU.
    ScalarOperation row_{}; // Текущее представление матрицы и уравнения.
    implicit::ScalarTopology graph_; // Разметка газовой матрицы и свободный прямой вектор.
    double *rhs_, *halo_; // Правая часть и полноразмерный обменный вектор MPI.

    implicit::PredictionHistory history_{implicit::KrylovScratch{}}; // История в собственных свободных векторах.
    bool first_stage_=true; // Экстраполяция допустима только один раз на физическом слое.
    unsigned age_=0; // Число принятых слоёв с момента сборки матрицы.
    bool reuse_=false; // На первом согласовании можно перенести пространственную матрицу.
    double* previous_steps_=nullptr; // Узловые dt для обновления временной диагонали.
    double leading_=0.; // Временной коэффициент последней сборки.
    // Выполняет построчную операцию на исполнителе полей.
    void launch(ScalarAction action,const double* input,double* output,const double* halo=nullptr,double fraction=1.) const {
        auto op=row_;
        op.fraction=fraction; op.action=action; op.input=input; op.halo=halo; op.output=output;
#ifdef AERO_WITH_KOKKOS_GPU
        if (device>=0) {
            device_work::check(device_work::aero_device_scalar_rows(op,count));
            return;
        }
#endif
        parallel_for(count,op);
    }
public:
    const int device; // Номер CUDA/HIP устройства, -1 для CPU.
    const MeshIndex count; // Число собственных скалярных неизвестных.
    bool ready=false; // Матрица соответствует текущей линеаризации.
    implicit::KrylovVectors work{}; // Векторы общего алгоритма BiCGSTAB.
    // Передаёт обмены единому решателю.
    const implicit::Partition& partition() const { return part_; }
    // Размещает поля единожды на сетку. Первый участок берётся из свободной газовой RHS.
    ScalarSolver(const implicit::ScalarTopology& graph,implicit::Partition part,int id,MeshIndex rows,
                 MeshIndex nodes,double* gas_rhs,unsigned width,bool cache_steps,const std::function<void(double*)>& exchange)
        : part_(std::move(part)),graph_(graph),device(id),count(rows) {
        const bool distributed=bool(part_.extend_vector);
        const auto n=std::max<MeshIndex>(1,rows);
        double* buf=buffers_.allocate(device,host_,(8-width+unsigned(cache_steps))*n+(distributed ? nodes : 0));
        double* v[6];
        for (unsigned j=0;j<6;++j) v[j]=j<width && rows ? gas_rhs+j*rows :
                                              (j<width ? buf : buf+(j-width)*n);
        implicit::KrylovScratch history_space;
        work={v[0],v[1],v[2],v[3],v[4],v[5]};
        // Вектор z занят и при одной подстановке предиктора, когда линейного решения нет.
        for (unsigned j=width;j<6;++j)
            if (v[j]!=work.z) history_space.field[history_space.count++]=v[j];
        history_=implicit::PredictionHistory(history_space);
        auto& m=row_.matrix;
        m.rows=graph.rows; m.columns=graph.columns; m.sweep=graph.sweep;
        m.colored=graph.offsets!=nullptr; m.owned=part_.owned;
        m.diagonal=buf+(6-width)*n; rhs_=m.diagonal+n; previous_steps_=cache_steps ? rhs_+n : nullptr; halo_=rhs_+n*(1+unsigned(cache_steps));
#ifdef AERO_WITH_KOKKOS_GPU
        if (device>=0) {
            gpu::DeviceScope scope(device);
            gpu::checked(scope.status);
            m.neighbours=buffers_.arrays.allocate<implicit::LinearCoefficient>(graph.slots);
        } else
#endif
        { coefficients_.resize(graph.slots); m.neighbours=coefficients_.data(); }
        if (distributed) part_.extend_vector=[this,exchange](const double* compact) {
            part_.local([&]{launch(ScalarAction::Extend,compact,halo_);});
            exchange(halo_); return halo_;
        };
    }
    // Обновляет поля уравнения без копий и пересчитывает матрицу по запросу Ньютона.
    void prepare(const Step& step,GasFaces faces,bool simplified,bool assemble) {
        row_.physics={step,faces,simplified};
        if (assemble || !ready) { part_.local([&]{launch(ScalarAction::Assemble,nullptr,nullptr);}); ready=true; }
    }
    // Открывает шаг с тем же интервалом обновления якобиана, что у газа.
    void begin_step(unsigned interval) {
        first_stage_=true;
        reuse_=ready && previous_steps_ && age_<interval;
    }
    // Первая сборка нового физического слоя при необходимости меняет только массу.
    void prepare_stage(const Step& step,GasFaces faces,const implicit::NewtonSettings& settings) {
        const bool assemble=!ready || !settings.simplified_newton() || (first_stage_ && !reuse_);
        prepare(step,faces,settings.simplified_jacobian,assemble);
        if (assemble) {
            age_=0;
            if (previous_steps_) part_.local([&]{launch(ScalarAction::SaveStep,nullptr,previous_steps_);});
        } else if (first_stage_ && reuse_) {
            row_.previous_leading=leading_;
            part_.local([&]{launch(ScalarAction::UpdateTime,nullptr,previous_steps_);});
        }
        leading_=step.leading;
    }
    // Сохраняет начало принятого слоя для того же полиномиального предиктора.
    void commit(double dt,bool history_allowed) {
        ++age_;
        if (!history_allowed) { history_.clear(); return; }
        if (auto* dest=history_.append(dt)) part_.local([&]{launch(ScalarAction::SaveLayer,nullptr,dest);});
    }
    // Полное матричное произведение служит также независимой проверкой b−Ax.
    void apply(const double* x,double* y) {
        const double* extended=part_.extend_vector ? part_.extend_vector(x) : x;
        part_.local([&]{launch(ScalarAction::Product,x,y,extended);});
    }
    // Применяет выбранный в конфигурации Якоби или CSGS.
    void predict(const double* b,double* x) {
        if (!graph_.offsets) { part_.local([&]{launch(ScalarAction::Jacobi,b,x);}); return; }
        auto sweep=graph_.sweep;
        sweep.diagonal={count,1,nullptr,nullptr,row_.matrix.diagonal,nullptr};
        sweep.blocks=nullptr; sweep.neighbour_values=row_.matrix.neighbours;
        implicit::ordered_sweep(sweep,*graph_.offsets,device,part_,graph_.forward,b,x);
    }
    // Правое предобусловливание общего алгоритма BiCGSTAB.
    void apply_preconditioned(const double* x,double* image,bool) { predict(x,work.z); apply(work.z,image); }
    // Общий предиктор пробует матричную или временную поправку, проверяя полный дефект.
    template<class Refresh>
    double initial_guess(double norm,double reference,const implicit::NewtonSettings& settings,
                         bool spatial,bool local,bool history_allowed,const Refresh& refresh) {
        const bool use_history=history_allowed && !local && settings.prediction_order && !settings.without_predictor;
        const bool extrapolate=first_stage_ && use_history;
        first_stage_=false;
        if (!use_history) history_.clear();
        if (settings.without_predictor) return norm;
        if (extrapolate) {
            const auto& step=row_.physics.step;
            implicit::TimePrediction trial(part_.owned,1,step.transport.sa.w,step.transport.volume,step.local_steps,step.dt);
            history_.prepare(step.dt,settings.prediction_order,trial);
            if (trial.count) {
                part_.local([&]{implicit::apply_rows(device,count,trial);});
                const double probe=refresh();
                if (probe<.8*norm) norm=probe;
                else {
                    part_.local([&]{launch(ScalarAction::RestoreLayer,nullptr,nullptr);});
                    norm=refresh();
                }
            }
        }
        typedef implicit::NewtonUpdate Update;
        const auto change=[&](Update::Operation op,double fraction) {
            if (op==Update::UndoPrediction || op==Update::RelaxPrediction) {
                part_.local([&]{launch(op==Update::UndoPrediction ? ScalarAction::UndoPrediction :
                                       ScalarAction::RelaxPrediction,nullptr,nullptr,nullptr,fraction);});
                return;
            }
            part_.local([&]{launch(ScalarAction::Rhs,nullptr,rhs_);});
            if (spatial) predict(rhs_,work.z);
            else {
                // Правая часть SA уже умножена на локальный шаг. Временная диагональ
                // равна старшему коэффициенту BDF, одинаковому для всех строк.
                part_.local([&]{launch(ScalarAction::Temporal,rhs_,work.z);});
            }
            part_.local([&]{launch(op==Update::FilterTemporal ? ScalarAction::FilterPrediction :
                                   ScalarAction::Predict,work.z,nullptr,nullptr,fraction);});
        };
        double measured=norm;
        const auto evaluate=[&]{measured=refresh();};
        return implicit::predict_initial_field(norm,reference,settings,change,[] {},[&]{return measured;},
            evaluate,[&]{evaluate();return std::isfinite(measured);},true,use_history,false,{},false,
            spatial && local ? 1u : 24u);
    }
    // Решает линеаризованное уравнение с газовыми настройками линейного решателя.
    implicit::LinearResult correct(const implicit::NewtonSettings& settings,double residual,bool filtered,bool shifted) {
        history_.clear(); // BiCGSTAB использует все свои рабочие поля.
        auto* delta=row_.physics.step.transport.sa.plain_update;
        part_.local([&]{launch(ScalarAction::Rhs,nullptr,rhs_);});
        implicit::LinearTolerance tolerance;
        tolerance.relative=settings.linear_relative; tolerance.max_iterations=settings.linear_max_iterations;
        const auto result=implicit::solve_bicgstab(*this,rhs_,delta,tolerance,residual,shifted ? rhs_ : nullptr);
        if (result.status==implicit::LinearStatus::Converged)
            part_.local([&]{launch(filtered ? ScalarAction::FilterNewton : ScalarAction::Accept,delta,nullptr);});
        return result;
    }
};
} }
