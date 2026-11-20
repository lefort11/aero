#ifndef AERO_IMPLICIT_SPATIAL_JACOBIAN_DETAIL_H
#define AERO_IMPLICIT_SPATIAL_JACOBIAN_DETAIL_H
#include "boundary_jacobian.h"
#include "roe_edge_jacobian.h"
#include "lf_edge_jacobian.h"
#include "../residual/residual_detail.h"

namespace aero {
namespace implicit {
// Адреса ссылаются на память выбранного исполнителя. Первая запись каждой
// строки CSR — её диагональ. Несколько рёбер между узлами делят один блок.
struct SpatialBlocks {
    residual_detail::Geometry grid;
    const MeshIndex* rows;       // Смещения строк в массиве номеров столбцов.
    const MeshIndex* columns;    // Номер узла для каждого плотного блока.
    const MeshIndex* neighbour;  // Индекс блока соседа для каждого вхождения ребра.
    const MeshIndex* computed;   // Индексы площадок, поток которых вычисляет процесс.
    MeshIndex face_count, row_count;
#ifdef AERO_HAS_VISCOSITY
    bool viscous;                // Блоки AES уже заполнены перед добавлением производных конвективных потоков.
#endif
    bool roe;                    // Выбор численного потока для матрицы поправки.
    bool simplified;             // Фиксировать диссипацию LF или Роу при дифференцировании.
    AERO_HOST_DEVICE MeshIndex node(MeshIndex row) const { return columns[rows[row]]; }
    AERO_HOST_DEVICE MeshIndex face(MeshIndex k) const { return computed ? computed[k] : k; }
    double* values;             // Блоки подряд, внутри блока — по строкам.
    double* spatial_diagonal;    // Чистые пространственные коэффициенты A_kk для повторного использования.
    double* face_values;        // Для внутреннего ребра два блока, для границы один.

    AERO_HOST_DEVICE unsigned stride() const { return (grid.dim + 2) * (grid.dim + 2); }
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
                if (!face.numerical(roe,grid.options.upw_coeff,unit,zero,physical,zero,flux)) return false;
            }
            for (unsigned r=0; r<width; ++r) out[r*width+k] = s.measure*flux.flux[r];
            if (!boundary) {
                if (!face.numerical(roe,grid.options.upw_coeff,zero,unit,zero,physical,flux)) return false;
                for (unsigned r=0; r<width; ++r) out[stride()+r*width+k] = s.measure*flux.flux[r];
            }
        }
        return true;
    }
    // Каждая площадка единожды вычисляет свои производные и пишет только в свою память.
    AERO_HOST_DEVICE bool differentiate(MeshIndex f, const double* u, const double* initial) const {
        const auto& s = grid.surfaces[f];
        const double* left = grid.node_row(u, s.begin);
        double* out = face_block(f);
        if (grid.options.background.active()) return linear_blocks(f,out);
        if (f >= grid.edges)
            return boundary_detail::block(grid.dim, grid.patches.by_patch[s.patch], grid.patches.free_rules,
                    left, initial ? grid.node_row(initial, s.begin) : nullptr, s.area, grid.options.gamma, out);
        EdgeFluxJacobian edge;
        const euler_detail::FaceStates states = {grid.dim, left, grid.node_row(u, s.end), s.area,
                                                 grid.options.gamma, grid.options.upw_coeff};
        const bool valid = roe ? roe_edge_jacobian(states, s.measure, edge, simplified)
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
    // Каждый исполнитель владеет одной строкой плотного блока (GPU) или всем блоком (CPU).
    // Одна и та же формула обрабатывает обычное ребро, границу и кратные связи.
    template<unsigned Width, unsigned Equations>
    AERO_HOST_DEVICE bool assemble_equations(MeshIndex row, unsigned first, const double* volumes, double dt) const {
        const MeshIndex i = node(row);
        const unsigned stride = Width*Width;
        const unsigned offset = first*Width;
        const unsigned count = Equations*Width;
        double diagonal[count] = {};
#ifdef AERO_HAS_VISCOSITY
        const bool add_to_existing = viscous;
        if (viscous) {
            const double* diffusion = values+rows[row]*stride+offset;
            for (unsigned k = 0; k < count; ++k) diagonal[k] = diffusion[k];
        }
#else
        const bool add_to_existing = false;
#endif
        for (MeshIndex p=grid.offsets[i]; p<grid.offsets[i+1]; ++p) {
            const auto entry = grid.incidence[p];
            const auto f = entry.face();
            if (f >= grid.edges) {
                const double* boundary = face_values+(grid.edges+f)*stride+offset;
                for (unsigned k=0; k<count; ++k) diagonal[k] += boundary[k];
                continue;
            }
            const auto slot = neighbour[p];
            if (!slot) continue;
            // Младший бит ориентации выбирает производную по своему узлу.
            // Сосед получит тот же блок с обратным знаком: так сохраняется баланс потока.
            const double* own = face_values+entry.encoded*stride+offset;
            const double* other = face_values+(entry.encoded^1)*stride+offset;
            double* block = values+(slot/2)*stride+offset;
            for (unsigned k=0; k<count; ++k) {
                diagonal[k] += entry.outward_flux(own[k]);
                const double contribution = entry.outward_flux(other[k]);
                block[k] = (add_to_existing || (slot&1)) ? block[k]+contribution : contribution;
                // Производные грани уже проверены. Переполнение возможно только при сложении связей.
                if ((slot&1) && !euler_detail::finite(block[k])) return false;
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
                    double* equation = values+p*stride+(first+r)*Width;
                    for (unsigned c = 0; c < Width; ++c) {
                        if (fixed) equation[c] = 0.;
                        else if (!euler_detail::finite(equation[c])) return false;
                    }
                }
            }
        }
#endif
        if (volumes) {
            const double mass = volumes[i]/dt;
            if (!(mass>0.) || !euler_detail::finite(mass)) return false;
            for (unsigned r = 0; r < Equations; ++r) {
                const unsigned k = first + r;
                double& coefficient = diagonal[r*Width + k];
                if (spatial_diagonal) spatial_diagonal[row*Width + k] = coefficient;
                coefficient += mass;
            }
        }
        double* output = values+rows[row]*stride+offset;
        for (unsigned k=0; k<count; ++k) {
            if (!euler_detail::finite(diagonal[k])) return false;
            output[k] = diagonal[k];
        }
        return true;
    }

    // При смене шага меняются только главные элементы блока. Берём сохранённое
    // пространственное значение напрямую, чтобы не вычитать старое большое V/dt.
    AERO_HOST_DEVICE bool update_time_diagonal(MeshIndex row, const double* volumes, double dt) const {
        const unsigned width = grid.components();
        const double mass = volumes[node(row)] / dt;
        if (!(mass > 0.) || !euler_detail::finite(mass)) return false;
        double* block = values + rows[row] * stride();
        for (unsigned k = 0; k < width; ++k) {
            const double value = spatial_diagonal[row*width + k] + mass;
            if (!euler_detail::finite(value)) return false;
            block[k*width + k] = value;
        }
        return true;
    }

    AERO_HOST_DEVICE bool assemble_row(MeshIndex row, const double* volumes, double dt) const {
        switch (grid.dim) {
        case 1: return assemble_equations<3,3>(row,0,volumes,dt);
        case 2: return assemble_equations<4,4>(row,0,volumes,dt);
        default: return assemble_equations<5,5>(row,0,volumes,dt);
        }
    }

    // Фиксированный размер убирает внутренние циклы по компонентам при компиляции.
    // Equations определяет, сколько уравнений строки CSR считает один поток.
    template<unsigned Width, unsigned Equations>
    AERO_HOST_DEVICE void multiply_equations(MeshIndex row, unsigned first, const double* x, double* y) const {
        double sum[Equations] = {};
        for (MeshIndex p=rows[row]; p<rows[row+1]; ++p) {
            const double* block = values+p*Width*Width+first*Width;
            const double* v = x+columns[p]*Width;
            for (unsigned r=0; r<Equations; ++r)
                for (unsigned c=0; c<Width; ++c) sum[r] += block[r*Width+c]*v[c];
        }
        for (unsigned r=0; r<Equations; ++r) y[row*Width+first+r] = sum[r];
    }
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
