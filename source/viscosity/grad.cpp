#include "grad.h"
#include "../mesh/mesh.h"
#include <algorithm>
#include <cmath>

namespace aero {
namespace viscosity {
namespace {
using Tensor = std::array<std::array<double,3>,3>;
// AES использует след произведения градиентов для теплопроводности и полную
// матрицу для напряжений. Оба направления пары получаются из одной матрицы.
PairMetric make_metric(const Tensor& product, unsigned dim, double gamma, double prandtl, double (&diffusion)[2]) {
    PairMetric pair{};
    double trace = 0.;
    for (unsigned m = 0; m < dim; ++m) trace += product[m][m];
    const double thermal = trace * gamma / prandtl;
    pair.heat = thermal / (gamma - 1.);
    diffusion[0] = diffusion[1] = std::abs(thermal);
    for (unsigned m = 0; m < dim; ++m)
        for (unsigned n = 0; n < dim; ++n)
            pair.stress[m][n] = (m == n ? trace : 0.) + product[n][m] - (2./3.)*product[m][n];
    // Для прямой пары берём максимум сумм модулей по строкам, для обратной — по столбцам.
    for (unsigned m = 0; m < dim; ++m) {
        double row = 0., column = 0.;
        for (unsigned n = 0; n < dim; ++n) {
            row += std::abs(pair.stress[m][n]);
            column += std::abs(pair.stress[n][m]);
        }
        diffusion[0] = std::max(diffusion[0], row);
        diffusion[1] = std::max(diffusion[1], column);
    }
    return pair;
}
struct Geometry {
    const Element& element;
    unsigned dim;
    Tensor product[8][8];
    MeanGradient* mean;

    // Инвертируем матрицу рёбер симплекса. Столбцы обратной матрицы — градиенты
    // линейных базисных функций; первая функция дополняет остальные до единицы.
    void simplex(double weight, unsigned a, unsigned b, unsigned c = 0, unsigned d = 0) {
        const unsigned vertex[] = {a,b,c,d};
        double matrix[3][6]{}, gradient[4][3]{};
        for (unsigned i = 0; i < dim; ++i) {
            for (unsigned j = 0; j < dim; ++j)
                matrix[i][j] = element.position[vertex[i+1]][j] - element.position[a][j];
            matrix[i][dim+i] = 1.;
        }
        double determinant = 1.;
        for (unsigned j = 0; j < dim; ++j) {
            unsigned pivot = j;
            for (unsigned i = j+1; i < dim; ++i)
                if (std::abs(matrix[i][j]) > std::abs(matrix[pivot][j])) pivot = i;
            if (pivot != j) std::swap(matrix[j], matrix[pivot]);
            const double scale = matrix[j][j];
            if (!scale) throw MeshError("AES: degenerate simplex in cell " + std::to_string(element.id));
            determinant *= scale;
            for (unsigned k = 0; k < 2*dim; ++k) matrix[j][k] /= scale;
            for (unsigned i = 0; i < dim; ++i) {
                if (i == j) continue;
                const double factor = matrix[i][j];
                for (unsigned k = 0; k < 2*dim; ++k) matrix[i][k] -= factor*matrix[j][k];
            }
        }
        for (unsigned i = 1; i <= dim; ++i)
            for (unsigned j = 0; j < dim; ++j) {
                gradient[i][j] = matrix[j][dim+i-1];
                gradient[0][j] -= gradient[i][j];
            }
        const double volume = std::abs(determinant) / (dim == 3 ? 6. : dim == 2 ? 2. : 1.);
        if (mean) {
            mean->volume += weight*volume;
            for (unsigned i=0;i<=dim;++i) for (unsigned j=0;j<dim;++j)
                mean->basis[vertex[i]][j] += weight*volume*gradient[i][j];
        }
        if (!(volume > 0. && std::isfinite(volume))) throw MeshError("AES: invalid simplex volume");
        for (unsigned i = 0; i <= dim; ++i)
            for (unsigned j = 0; j <= dim; ++j) {
                // Для обратного направления достаточно транспонировать эту же матрицу.
                if (vertex[i] >= vertex[j]) continue;
                auto& p = product[vertex[i]][vertex[j]];
                for (unsigned m = 0; m < dim; ++m)
                    for (unsigned n = 0; n < dim; ++n)
                        p[m][n] += weight*volume*gradient[i][m]*gradient[j][n];
            }
    }
    // Каждая тройка из четырёх вершин основания — один треугольник одного из двух
    // разбиений. Для пирамиды добавляем общую вершину и получаем тетраэдры.
    void quadrilateral(unsigned apex) {
        std::vector<unsigned> base;
        for (unsigned i = 0; i < element.count; ++i) if (i != apex) base.push_back(i);
        for (unsigned omit = 0; omit < 4; ++omit) {
            unsigned v[3], n = 0;
            for (unsigned i = 0; i < 4; ++i) if (omit != i) v[n++] = base[i];
            simplex(.5, v[0], v[1], v[2], apex);
        }
    }
    bool prism() {
        // Находим одно треугольное основание и три боковых ребра. Перестановки
        // вершин основания перечисляют ровно шесть разбиений призмы на три тетраэдра.
        for (unsigned a = 0; a < 6; ++a) for (unsigned b = a+1; b < 6; ++b)
            for (unsigned c = b+1; c < 6; ++c) {
                if (!(element.edge(a,b) && element.edge(b,c) && element.edge(c,a))) continue;
                unsigned bottom[] = {a,b,c}, top[6]{};
                for (unsigned i : bottom) {
                    unsigned matches = 0;
                    for (unsigned j = 0; j < 6; ++j)
                        if (j!=a && j!=b && j!=c && element.edge(i,j)) { top[i] = j; ++matches; }
                    if (matches != 1) return false;
                }
                if (!(element.edge(top[a],top[b]) && element.edge(top[b],top[c]) && element.edge(top[c],top[a])))
                    return false;
                do {
                    const unsigned i = bottom[0], j = bottom[1], k = bottom[2];
                    simplex(1./6., i,j,k,top[k]);
                    simplex(1./6., i,j,top[j],top[k]);
                    simplex(1./6., i,top[i],top[j],top[k]);
                } while (std::next_permutation(bottom,bottom+3));
                return true;
            }
        return false;
    }
    bool hexahedron() {
        // Каждая из четырёх пространственных диагоналей — общее ребро шести
        // тетраэдров. Две другие вершины берём с рёбер оставшегося шестиугольника.
        unsigned count = 0;
        for (unsigned a = 0; a < 8; ++a) for (unsigned b = a+1; b < 8; ++b) {
            if (element.edge(a,b) || (element.neighbours[a] & element.neighbours[b])) continue;
            for (unsigned i = 0; i < 8; ++i) for (unsigned j = i+1; j < 8; ++j) {
                if (i==a || i==b || j==a || j==b || !element.edge(i,j)) continue;
                simplex(.25,a,b,i,j);
                ++count;
            }
        }
        return count == 24;
    }
    void build() {
        const auto n = element.count;
        if (n == dim+1) { simplex(1.,0,1,2,3); return; }
        if (dim == 2 && n == 4) { quadrilateral(n); return; }
        if (dim == 3 && n == 5) {
            for (unsigned i = 0; i < n; ++i) {
                if (element.neighbours[i] == (31u ^ (1u<<i))) { quadrilateral(i); return; }
            }
        }
        if (dim == 3 && n == 6 && prism()) return;
        if (dim == 3 && n == 8 && hexahedron()) return;
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
            pairs.push_back(make_metric(geometry.product[i][j],dim,gamma,prandtl,diffusion));
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
