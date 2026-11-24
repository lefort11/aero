#ifndef AERO_IMPLICIT_SPATIAL_JACOBIAN_DETAIL_H
#define AERO_IMPLICIT_SPATIAL_JACOBIAN_DETAIL_H
#include "../timestep/node_step.h"
#include "boundary_jacobian.h"
#include "roe_edge_jacobian.h"
#include "lf_edge_jacobian.h"
#include "block_layout.h"
#include "../residual/residual_detail.h"

namespace aero {
namespace implicit {
// Адреса ссылаются на память выбранного исполнителя. Первая запись каждой
// строки CSR — её диагональ. Несколько рёбер между узлами делят один блок.
struct SpatialBlocks {
    // Связность и геометрия для вычисления производных потоков.
    residual_detail::Geometry grid;
    const MeshIndex* rows;       // Смещения строк в массиве номеров столбцов.
    IndexList columns;           // Номер узла для каждого плотного блока.
    IndexList neighbour;         // Индекс блока соседа для каждого вхождения ребра.
    const MeshIndex* computed;   // Индексы площадок, поток которых вычисляет процесс.
    // Размеры наборов площадок и строк матрицы.
    MeshIndex face_count, row_count;
#ifdef AERO_HAS_VISCOSITY
    bool viscous;                // Блоки AES уже заполнены перед добавлением производных конвективных потоков.
#endif
    // Шаги для узловой временной диагонали.
    NodeSteps local_steps;
    bool roe;                    // Выбор численного потока для матрицы поправки.
    bool simplified;             // Фиксировать диссипацию LF или Роу при дифференцировании.
    // Возвращает узел диагонального блока указанной строки.
    AERO_HOST_DEVICE MeshIndex node(MeshIndex row) const { return columns[rows[row]]; }
    // Переводит позицию вычисляемой площадки в её геометрический номер.
    AERO_HOST_DEVICE MeshIndex face(MeshIndex k) const { return computed ? computed[k] : k; }
    double* values;             // Единственный массив коэффициентов пространственной матрицы.
    BlockLayout layout;         // Адреса коэффициентов в CSR или в порядке проходов CSGS.
    double* spatial_diagonal;    // Чистые пространственные коэффициенты A_kk для повторного использования.
    bool direct_faces; // На тесной карте одного процесса не храним по две матрицы на каждое ребро.
    // Поле в точке линеаризации пространственного оператора.
    const double* assembly_state;
    // Начальные данные для дифференцирования граничных условий.
    const double* assembly_initial;
    double* face_values;        // Для внутреннего ребра два блока, для границы один.

    // Вклад производной по времени одинаков при первой сборке и при смене шага.
    // В steady берём шаг данного узла, при глобальном интегрировании — общий dt.
    AERO_HOST_DEVICE bool time_mass(MeshIndex i, const double* volumes, double dt, double& mass) const {
        mass=volumes[i]/local_steps.at(i,dt);
        return mass>0. && euler_detail::finite(mass);
    }
    // Возвращает число коэффициентов квадратного блока газовых компонент.
    AERO_HOST_DEVICE unsigned stride() const { return (grid.dim + 2) * (grid.dim + 2); }
    // Возвращает адрес блока производных, соответствующего площадке.
    AERO_HOST_DEVICE double* face_block(MeshIndex f) const {
        return face_values + (f < grid.edges ? 2*f : grid.edges + f) * stride();
    }
    // Матрица постоянного линейного потока задана своим действием на единичные
    // векторы. Здесь нет вычитания близких потоков или подбора приращения.
    AERO_HOST_DEVICE bool linear_blocks(MeshIndex f, double* out) const {
        const auto& s = grid.surfaces[f];
        const unsigned width = grid.components();
        double normal[3] = {};
        if (euler_unit_direction(grid.dim,s.area,normal) != EulerStatus::Success) return false;
        const LinearFace face(grid.options.background,grid.dim,normal);
        const bool boundary = f >= grid.edges;
        for (unsigned k=0; k<width; ++k) {
            double unit[5] = {}, zero[5] = {}, physical[5];
            unit[k] = 1.;
            EulerFluxResult flux;
            if (boundary) {
                if (!linear_boundary_flux(face,grid.patches.by_patch[s.patch],grid.patches.free_rules,
                                          unit,nullptr,true,flux)) return false;
            } else {
                face.physical(unit,physical);
                if (!face.numerical(roe,grid.options.upw_coeff,unit,zero,physical,zero,flux,grid.options.entropy_fix)) return false;
            }
            for (unsigned r=0; r<width; ++r) out[r*width+k] = s.measure*flux.flux[r];
            if (!boundary) {
                if (!face.numerical(roe,grid.options.upw_coeff,zero,unit,zero,physical,flux,grid.options.entropy_fix)) return false;
                for (unsigned r=0; r<width; ++r) out[stride()+r*width+k] = s.measure*flux.flux[r];
            }
        }
        return true;
    }
    // Каждая площадка единожды вычисляет свои производные и пишет только в свою память.
    AERO_HOST_DEVICE bool face_derivatives(MeshIndex f,const double* u,const double* initial,double* out) const {
        const auto& s = grid.surfaces[f];
        const double* left = grid.node_row(u, s.begin);
        if (grid.options.background.active()) return linear_blocks(f,out);
        if (f >= grid.edges)
            return boundary_detail::block(grid.dim, grid.patches.by_patch[s.patch], grid.patches.free_rules,
                    left, grid.boundary_initial(initial, f), s.area, grid.options.gamma, out, grid.patches.profile(f-grid.edges));
        EdgeFluxJacobian edge;
        const euler_detail::FaceStates states = {grid.dim, left, grid.node_row(u, s.end), s.area,
                                                 grid.options.gamma, grid.options.upw_coeff};
        const bool valid = roe ? roe_edge_jacobian(states, s.measure, edge, simplified,grid.options.entropy_fix)
                               : lf_edge_jacobian(states, s.measure, edge, simplified);
        if (!valid)
            return false;
        const unsigned count = stride();
        for (unsigned k = 0; k < count; ++k) {
            out[k] = edge.left[k];
            out[count + k] = edge.right[k];
        }
        return true;
    }
    // Вычисляет производные потока площадки по состояниям её узлов.
    AERO_HOST_DEVICE bool differentiate(MeshIndex f,const double* u,const double* initial) const {
        return face_derivatives(f,u,initial,face_block(f));
    }
    // Каждый исполнитель владеет одной строкой плотного блока (GPU) или всем блоком (CPU).
    // Одна и та же формула обрабатывает обычное ребро, границу и кратные связи.
    template<unsigned Width, unsigned Equations,bool direct=false>
    AERO_HOST_DEVICE bool assemble_equations(MeshIndex row, unsigned first, const double* volumes, double dt) const {
        const MeshIndex i = node(row);
        const unsigned stride = Width*Width;
        const unsigned offset = first*Width;
        const unsigned count = Equations*Width;
        double diagonal[count] = {};
#ifdef AERO_HAS_VISCOSITY
        const bool add_to_existing = viscous;
        if (viscous) {
            const auto diffusion = layout.view(values, rows[row], Width, true);
            for (unsigned k = 0; k < count; ++k) diagonal[k] = diffusion[offset+k];
        }
#else
        const bool add_to_existing = false;
#endif
        for (MeshIndex p=grid.offsets[i]; p<grid.offsets[i+1]; ++p) {
            const auto entry = grid.incidence[p];
            const auto f=entry.face();
            double local[direct ? 2*Width*Width : 1];
            if (direct && !face_derivatives(f,assembly_state,assembly_initial,local)) return false;
            // У каждой строки свой буфер в прямом режиме. В обычном режиме два конца
            // читают общую запись вычисленной грани. Ориентация и порядок суммы одинаковы.
            const double* pair=direct ? local : face_block(f);
            if (f >= grid.edges) {
                const double* boundary=pair+offset;
                for (unsigned k=0; k<count; ++k) diagonal[k] += boundary[k];
                continue;
            }
            const auto slot = neighbour[p];
            if (!slot) continue;
            // Младший бит ориентации выбирает производную по своему узлу.
            // Сосед получит тот же блок с обратным знаком: так сохраняется баланс потока.
            const unsigned side=static_cast<unsigned>(entry.encoded&1);
            const double* own=pair+side*stride+offset;
            const double* other=pair+(side^1u)*stride+offset;
            const auto block = layout.view(values, slot/2, Width, false);
            for (unsigned k=0; k<count; ++k) {
                diagonal[k] += entry.outward_flux(own[k]);
                const double contribution = entry.outward_flux(other[k]);
                if (add_to_existing || (slot&1)) block.add(offset+k, contribution);
                else block.set(offset+k, contribution);
                // Производные грани уже проверены. Переполнение возможно только при сложении связей.
                if (((slot&1) || block.neighbours) && !euler_detail::finite(block[offset+k])) return false;
            }
        }
#ifdef AERO_HAS_VISCOSITY
        if (viscous) {
            const bool wall = grid.options.wall && grid.options.wall[i];
            for (unsigned r = 0; r < Equations; ++r) {
                // Невязка импульса на неподвижной стенке тождественно нулевая.
                // Обнуляем всю пространственную строку уже после консервативной сборки.
                const bool fixed = wall && first+r > 0 && first+r+1 < Width;
                if (fixed) for (unsigned c = 0; c < Width; ++c) diagonal[r*Width+c] = 0.;
                for (MeshIndex p = rows[row]+1; p < rows[row+1]; ++p) {
                    const auto block = layout.view(values, p, Width, false);
                    for (unsigned c = 0; c < Width; ++c) {
                        const unsigned k = (first+r)*Width+c;
                        if (fixed) block.set(k, 0.);
                        else if (!euler_detail::finite(block[k])) return false;
                    }
                }
            }
        }
#endif
        if (volumes) {
            double mass;
            if (!time_mass(i,volumes,dt,mass)) return false;
            for (unsigned r = 0; r < Equations; ++r) {
                const unsigned k = first + r;
                double& coefficient = diagonal[r*Width + k];
                if (spatial_diagonal) spatial_diagonal[row*Width + k] = coefficient;
                coefficient += mass;
            }
        }
        const auto output = layout.view(values, rows[row], Width, true);
        for (unsigned k=0; k<count; ++k) {
            if (!euler_detail::finite(diagonal[k])) return false;
            output.set(offset+k, diagonal[k]);
        }
        return true;
    }

    // При смене шага меняются только главные элементы блока. Берём сохранённое
    // пространственное значение напрямую, чтобы не вычитать старое большое V/dt.
    AERO_HOST_DEVICE bool update_time_diagonal(MeshIndex row, const double* volumes, double dt) const {
        const unsigned width = grid.components();
        double mass;
        if (!time_mass(node(row),volumes,dt,mass)) return false;
        const auto block = layout.view(values, rows[row], width, true);
        for (unsigned k = 0; k < width; ++k) {
            const double value = spatial_diagonal[row*width + k] + mass;
            if (!euler_detail::finite(value)) return false;
            block.set(k*width + k, value);
        }
        return true;
    }

    // Собирает одну строку блоков якобиана из потоковых вкладов.
    AERO_HOST_DEVICE bool assemble_row(MeshIndex row, const double* volumes, double dt) const {
        switch (grid.dim) {
        case 1: return assemble_equations<3,3>(row,0,volumes,dt);
        case 2: return assemble_equations<4,4>(row,0,volumes,dt);
        default: return assemble_equations<5,5>(row,0,volumes,dt);
        }
    }

    // Фиксированный размер убирает внутренние циклы по компонентам при компиляции.
    // Equations определяет, сколько уравнений строки CSR считает один поток.
    template<unsigned Width, unsigned Equations,bool direct=false>
    AERO_HOST_DEVICE void multiply_equations(MeshIndex row, unsigned first, const double* x, double* y) const {
        double sum[Equations] = {};
        for (MeshIndex p=rows[row]; p<rows[row+1]; ++p) {
            const auto block = layout.view(values, p, Width, p==rows[row]);
            const double* v = x+columns[p]*Width;
            for (unsigned r=0; r<Equations; ++r)
                for (unsigned c=0; c<Width; ++c) sum[r] += block[(first+r)*Width+c]*v[c];
        }
        for (unsigned r=0; r<Equations; ++r) y[row*Width+first+r] = sum[r];
    }
    // Вычисляет произведение одной блочной строки матрицы на вектор.
    AERO_HOST_DEVICE void multiply_row(MeshIndex row, const double* x, double* y) const {
        switch (grid.dim) {
        case 1: return multiply_equations<3,3>(row,0,x,y);
        case 2: return multiply_equations<4,4>(row,0,x,y);
        default: return multiply_equations<5,5>(row,0,x,y);
        }
    }

};
}
}
#endif
