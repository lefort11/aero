#include "grad.h"
#include "volume_simplices.h"
#include "../mesh/mesh.h"
#include <algorithm>
#include <cmath>

namespace aero {
namespace viscosity {
namespace {
// Сохраняем только 28 различных пар восьми вершин. Для каждой пары подряд лежат 9 производных.
using PairProducts = FixedTensor<double,28,3,3>;
// AES использует след произведения градиентов для теплопроводности и полную
// матрицу для напряжений. Оба направления пары получаются из одной матрицы.
PairMetric make_metric(const PairProducts& products, unsigned pair_id, unsigned dim, double gamma, double prandtl, double (&diffusion)[2]) {
    PairMetric pair{};
    double trace = 0.;
    for (unsigned m = 0; m < dim; ++m) trace += products(pair_id,m,m);
    const double thermal = trace * gamma / prandtl;
    pair.heat = thermal / (gamma - 1.);
    diffusion[0] = diffusion[1] = std::abs(thermal);
    for (unsigned m = 0; m < dim; ++m)
        for (unsigned n = 0; n < dim; ++n)
            pair.stress(m, n) = PairMetric::stress_entry(m,n,trace,products(pair_id,m,n),products(pair_id,n,m));
    // Для прямой пары берём максимум сумм модулей по строкам, для обратной — по столбцам.
    for (unsigned m = 0; m < dim; ++m) {
        double row = 0., column = 0.;
        for (unsigned n = 0; n < dim; ++n) {
            row += std::abs(pair.stress(m, n));
            column += std::abs(pair.stress(n, m));
        }
        diffusion[0] = std::max(diffusion[0], row);
        diffusion[1] = std::max(diffusion[1], column);
    }
    return pair;
}
struct Geometry {
    const Element& element;
    unsigned dim;
    PairProducts products;
    MeanGradient* mean;

    // Инвертируем матрицу рёбер симплекса. Столбцы обратной матрицы — градиенты
    // линейных базисных функций; первая функция дополняет остальные до единицы.
    void simplex(double weight, unsigned a, unsigned b, unsigned c = 0, unsigned d = 0) {
        const unsigned vertex[] = {a,b,c,d};
        FixedMatrix<double,3,6> matrix{};
        FixedMatrix<double,4,3> gradient{};
        for (unsigned i = 0; i < dim; ++i) {
            for (unsigned j = 0; j < dim; ++j)
                matrix(i, j) = element.position[vertex[i+1]][j] - element.position[a][j];
            matrix(i, dim+i) = 1.;
        }
        double determinant = 1.;
        for (unsigned j = 0; j < dim; ++j) {
            unsigned pivot = j;
            for (unsigned i = j+1; i < dim; ++i)
                if (std::abs(matrix(i, j)) > std::abs(matrix(pivot, j))) pivot = i;
            if (pivot != j) std::swap_ranges(matrix.row(j),matrix.row(j)+2*dim,matrix.row(pivot));
            const double scale = matrix(j, j);
            if (!scale) throw MeshError("AES: degenerate simplex in cell " + std::to_string(element.id));
            determinant *= scale;
            for (unsigned k = 0; k < 2*dim; ++k) matrix(j, k) /= scale;
            for (unsigned i = 0; i < dim; ++i) {
                if (i == j) continue;
                const double factor = matrix(i, j);
                for (unsigned k = 0; k < 2*dim; ++k) matrix(i, k) -= factor*matrix(j, k);
            }
        }
        for (unsigned i = 1; i <= dim; ++i)
            for (unsigned j = 0; j < dim; ++j) {
                gradient(i, j) = matrix(j, dim+i-1);
                gradient(0, j) -= gradient(i, j);
            }
        const double volume = std::abs(determinant) / (dim == 3 ? 6. : dim == 2 ? 2. : 1.);
        if (mean) {
            mean->volume += weight*volume;
            for (unsigned i=0;i<=dim;++i) for (unsigned j=0;j<dim;++j)
                mean->basis[vertex[i]][j] += weight*volume*gradient(i, j);
        }
        if (!(volume > 0. && std::isfinite(volume))) throw MeshError("AES: invalid simplex volume");
        for (unsigned i = 0; i <= dim; ++i)
            for (unsigned j = 0; j <= dim; ++j) {
                // Для обратного направления достаточно транспонировать эту же матрицу.
                if (vertex[i] >= vertex[j]) continue;
                const auto pair_id=Cell::pair_offset(8,vertex[i],vertex[j]);
                for (unsigned m = 0; m < dim; ++m)
                    for (unsigned n = 0; n < dim; ++n)
                        products(pair_id,m,n) += weight*volume*gradient(i, m)*gradient(j, n);
            }
    }
    // Каждая тройка вершин четырёхугольника входит в одно из двух разбиений.
    void quadrilateral() {
        for (unsigned omit = 0; omit < 4; ++omit) {
            unsigned v[3], n = 0;
            for (unsigned i = 0; i < 4; ++i) if (omit != i) v[n++] = i;
            simplex(.5, v[0], v[1], v[2]);
        }
    }
    void build() {
        const auto n = element.count;
        if (dim==3) {
            if (for_volume_tetrahedra(element,[&](double weight,unsigned a,unsigned b,unsigned c,unsigned d) {
                simplex(weight,a,b,c,d);
            })) return;
        } else {
            if (n == dim+1) { simplex(1.,0,1,2,3); return; }
            if (dim == 2 && n == 4) { quadrilateral(); return; }
        }
        throw MeshError("AES: unsupported topology in cell " + std::to_string(element.id));
    }
};
}

void GradientBuilder::prepare(const Element& element, unsigned dim, double gamma, double prandtl,
                             Cell& cell, std::vector<PairMetric>& pairs, MeanGradient* mean) {
    // Координаты element отсчитаны от первой вершины. Перенесённая копия
    // имеет тот же ключ, если все числа и все исходные рёбра совпадают точно.
    std::array<double,32> key{};
    for (unsigned i = 0; i < cell.count; ++i) {
        key[4*i] = element.neighbours[i];
        for (unsigned j = 0; j < dim; ++j) key[4*i+1+j] = element.position[i][j];
    }
    auto found = shapes_.find(key);
    if (found != shapes_.end() && mean && !found->second.mean.volume) {
        shapes_.erase(found);
        found=shapes_.end();
    }
    if (found == shapes_.end()) {
        Shape shape{{},pairs.size(),{}};
        Geometry geometry{element,dim,{},mean ? &shape.mean : nullptr};
        geometry.build();
        if (mean) for (auto& vertex : shape.mean.basis)
            for (double& weight : vertex) weight /= shape.mean.volume;
        for (unsigned i = 0; i < cell.count; ++i) for (unsigned j = i+1; j < cell.count; ++j) {
            double diffusion[2];
            pairs.push_back(make_metric(geometry.products,Cell::pair_offset(8,i,j),dim,gamma,prandtl,diffusion));
            shape.diffusion[i] += diffusion[0];
            shape.diffusion[j] += diffusion[1];
        }
        if (shapes_.size() >= 4096) shapes_.clear();
        found = shapes_.emplace(key,shape).first;
    }
    if (mean) *mean = found->second.mean;
    cell.first_pair = found->second.first_pair;
    std::copy_n(found->second.diffusion.begin(),cell.count,cell.diffusion);
}
}
}
