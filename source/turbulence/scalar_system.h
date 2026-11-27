#pragma once
#include "sa_jacobian.h"
#include "../implicit/scalar_topology.h"

namespace aero { namespace turbulence {
// Индексы раздела заимствованы у газовой матрицы. Только численные коэффициенты свои.
struct ScalarRows {
    const MeshIndex* rows; // Смещения CSR для блочного Якоби.
    implicit::IndexList columns; // Номера полных узлов CSR.
    implicit::CsgsSweep sweep; // Упакованная топология при CSGS.
    bool colored; // Коэффициенты записываются сразу в порядок проходов.
    const MeshIndex* owned; // Номера собственных узлов; пусто для одного ранга.
    double* diagonal; // Диагональ неявной матрицы.
    implicit::LinearCoefficient* neighbours; // Один коэффициент на связь.
    // Переводит собственную строку в номер узла.
    AERO_HOST_DEVICE MeshIndex node(MeshIndex row) const { return owned ? owned[row] : row; }
    // Выполняет операцию над каждым внедиагональным коэффициентом строки.
    template<class Visit> AERO_HOST_DEVICE void visit(MeshIndex index,const Visit& visit_entry) const {
        if (colored) {
            const auto span=sweep.row_range(index);
            for (unsigned pass=0;pass<2;++pass) {
                auto slot=span.first[pass];
                for (MeshIndex j=0;j<span.count[pass];++j,slot+=sweep.lanes) {
                    const auto key=sweep.neighbours[slot];
                    visit_entry(slot,(key&1) ? key/2 : node(key/2));
                }
            }
        } else {
            for (auto slot=rows[index]+1;slot<rows[index+1];++slot) visit_entry(slot,columns[slot]);
        }
    }
};
// При сборке повторные рёбра и периодические копии дают сумму в одном столбце.
struct ScalarWriter {
    const ScalarRows& matrix; // Коэффициенты и структура.
    MeshIndex index,row,node; // Номер в раскраске, компактная строка и полный узел.
    // Отсутствующие в компактном якобиане дальние соседи остаются в полной невязке.
    AERO_HOST_DEVICE void add(MeshIndex column,double value) const {
        if (column==node) { matrix.diagonal[row]+=value; return; }
        matrix.visit(index,[&](MeshIndex slot,MeshIndex other) {
            if (other==column) matrix.neighbours[slot]+=static_cast<implicit::LinearCoefficient>(value);
        });
    }
};
// Операции над полями одного блока. Режим задаётся перед построчным запуском.
enum class ScalarAction { Assemble, Rhs, Product, Temporal, Jacobi, Extend, Accept, FilterNewton, Predict, FilterPrediction, RelaxPrediction, UndoPrediction, SaveLayer, RestoreLayer, SaveStep, UpdateTime };
// Аргументы единого ядра CPU/GPU. Поля решения не покидают активный исполнитель.
struct ScalarOperation {
    SaJacobianRow physics; // Геометрия и состояние в точке линеаризации.
    ScalarRows matrix; // Скаляры в раскладке газового графа.
    ScalarAction action; // Операция текущего запуска.
    const double* input; // Вектор решения либо поправки.
    const double* halo; // Полный входной вектор вместе с обменными узлами.
    double* output; // Выходной вектор запуска.
    double fraction; // Доля демпфирования или пробной поправки.
    double previous_leading; // Временная диагональ предыдущего слоя при повторном использовании матрицы.
    // Собирает или применяет одну собственную строку.
    AERO_HOST_DEVICE void operator()(MeshIndex index) const {
        const auto row=matrix.colored ? matrix.sweep.sequence[index] : index;
        const auto i=matrix.node(row);
        const auto& step=physics.step;
        const auto& sa=step.transport.sa;
        switch (action) {
        case ScalarAction::Assemble:
            matrix.diagonal[row]=0.;
            matrix.visit(index,[&](MeshIndex slot,MeshIndex){matrix.neighbours[slot]=0.;});
            if (sa.w[i]==0. && step.defect(i)>0.) matrix.diagonal[row]=1.;
            else physics.assemble(row,ScalarWriter{matrix,index,row,i});
            break;
        case ScalarAction::Rhs:
            output[row]=sa.zero_working_viscosity(i) ? -sa.w[i] :
                         (sa.w[i]==0. ? ::fmax(0.,-step.defect(i)) : -step.defect(i));
            break;
        case ScalarAction::Product: {
            double sum=matrix.diagonal[row]*input[row];
            matrix.visit(index,[&](MeshIndex slot,MeshIndex other){sum+=matrix.neighbours[slot]*halo[other];});
            output[row]=sum;
            break;
        }
        case ScalarAction::Temporal: output[row]=input[row]/step.leading; break;
        case ScalarAction::Jacobi: output[row]=input[row]/matrix.diagonal[row]; break;
        case ScalarAction::Extend: output[i]=input[row]; break;
        case ScalarAction::SaveStep: output[row]=step.local_steps.at(i,step.dt); break;
        case ScalarAction::UpdateTime: {
            const double h=step.local_steps.at(i,step.dt), ratio=h/output[row];
            if (!sa.zero_working_viscosity(i)) {
                matrix.diagonal[row]=step.leading+(matrix.diagonal[row]-previous_leading)*ratio;
                if (ratio!=1.) matrix.visit(index,[&](MeshIndex slot,MeshIndex){
                    matrix.neighbours[slot]=static_cast<implicit::LinearCoefficient>(matrix.neighbours[slot]*ratio);
                });
            }
            output[row]=h;
            break;
        }
        case ScalarAction::RestoreLayer: sa.w[i]=sa.old[i]; break;
        case ScalarAction::SaveLayer: output[row]=sa.old[i]; break;
        case ScalarAction::UndoPrediction: sa.w[i]=sa.plain_update[row]; break;
        case ScalarAction::RelaxPrediction:
            sa.w[i]=sa.plain_update[row]+fraction*(sa.w[i]-sa.plain_update[row]); break;
        case ScalarAction::Predict:
        case ScalarAction::FilterPrediction: {
            const double before=sa.w[i];
            const double delta=action==ScalarAction::Predict ? input[row] :
                CorrectionFilter::change(input[row],fraction ? before-sa.plain_update[row] : 0.);
            sa.plain_update[row]=before;
            sa.w[i]=sa.zero_working_viscosity(i) ? 0. : ::fmax(0.,before+delta);
            break;
        }
        case ScalarAction::FilterNewton: {
            const double before=sa.w[i];
            sa.w[i]=sa.zero_working_viscosity(i) ? 0. : ::fmax(0.,before+CorrectionFilter::weight()*input[row]);
            sa.plain_update[row]=CorrectionFilter::shift(sa.w[i]-before);
            break;
        }
        case ScalarAction::Accept: sa.w[i]=sa.zero_working_viscosity(i) ? 0. : ::fmax(0.,sa.w[i]+input[row]); break;
        }
    }
};
} }
