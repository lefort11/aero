#pragma once
#include "../mesh/coordinate.h"
#include "../mesh/mesh_index.h"
#include "../eulersolver/euler.h"
#include "../parallel/fixed_matrix.h"
#include <array>
#include <map>
#include <vector>

namespace aero {
namespace viscosity {
// Для геометрии нужны исходные точки, даже когда периодические вершины разделяют одну
// неизвестную. В MPI пересылаем описание элемента владельцам всех его расчётных узлов.
struct Element {
    MeshIndex id, nodes[8];
    int owners[8];
    unsigned count, neighbours[8]; // Битовая маска исходных рёбер для каждой вершины.
    Coordinate position[8];       // Координаты в базисе уравнений, с масштабом сетки.
    // Возвращает разность координат концов локального ребра элемента.
    bool edge(unsigned a, unsigned b) const { return (neighbours[a] & (1u<<b)) != 0; }
};
// Четыре пространственные диагонали гексаэдра задают четыре триангуляции AES.
// Обходим тетраэдры в одном порядке при сборке таблицы и при вычислении по координатам.
template<class Visit>
inline bool for_hex_tetrahedra(const Element& element, Visit visit) {
    unsigned count=0;
    for (unsigned a=0;a<8;++a) for (unsigned b=a+1;b<8;++b) {
        if (element.edge(a,b) || (element.neighbours[a]&element.neighbours[b])) continue;
        for (unsigned i=0;i<8;++i) for (unsigned j=i+1;j<8;++j) {
            if (i==a || i==b || j==a || j==b || !element.edge(i,j)) continue;
            visit(a,b,i,j);
            ++count;
        }
    }
    return count==24;
}
// Коэффициенты пары постоянны на неподвижной сетке. При обходе в обратном
// направлении матрица напряжений транспонируется, а коэффициент тепла тот же.
struct PairMetric {
    FixedMatrix<double,3,3> stress;
    double heat;
    // Из интегралов P_mn = ∫ ∂mNi ∂nNj получаем производную вязкого напряжения.
    // Здесь одна формула для постоянной таблицы CPU и вычисления по координатам.
    AERO_HOST_DEVICE static double stress_entry(unsigned m,unsigned n,double trace,double p_mn,double p_nm) {
        return (m==n ? trace : 0.)+p_nm-(2./3.)*p_mn;
    }
};
// Вершины элемента, веса усреднения и геометрические коэффициенты диффузии.
struct Cell {
    unsigned count;
    double inverse_count; // Вес вершины при усреднении коэффициентов переноса.
    MeshIndex node[8];
    double diffusion[8]; // Сумма геометрических коэффициентов CFL для каждой вершины.
    MeshIndex first_pair, first_vertex; // Начало таблиц пар и вершин данного элемента.
    // Вычисляет начало строки пар вершин в треугольной упаковке.
    AERO_HOST_DEVICE static unsigned pair_offset(unsigned vertices,unsigned a,unsigned b) {
        return a*(2*vertices-a-1)/2 + b-a-1; // Верхний треугольник без диагонали, a < b.
    }
    // Преобразует пару локальных вершин в индекс упакованного коэффициента.
    AERO_HOST_DEVICE MeshIndex pair_index(unsigned i, unsigned j) const {
        const unsigned a = i < j ? i : j, b = i < j ? j : i;
        return first_pair + pair_offset(count,a,b);
    }
};
// Вклад геометрии одной пары до умножения на коэффициенты переноса.
// Эти же линейные формулы применяются к разности полей и к её производной.
struct Gradient {
    double stress[3], heat;

};
// Разности скорости и отношения давления к плотности между состояниями.
struct Difference {
    double velocity[3], thermal; // Приращения компонент u и величины p/rho.
};
// Вычисляет разности скорости и отношения p/rho между газовыми состояниями.
template<class GasState>
AERO_HOST_DEVICE inline Difference difference(unsigned dim, const GasState& from, const GasState& to) {
    Difference change{};
    change.thermal = to.pressure_over_density - from.pressure_over_density;
    for (unsigned k = 0; k < dim; ++k) change.velocity[k] = to.velocity[k] - from.velocity[k];
    return change;
}
// Умножаем готовую геометрическую матрицу на разности скоростей. Тем же кодом
// якобиан обрабатывает производные этих разностей по консервативным переменным.
template<unsigned dim>
AERO_HOST_DEVICE inline void add_gradient(const PairMetric& pair, bool reverse,
                                           const Difference& change, Gradient& sum) {
    sum.heat += pair.heat * change.thermal;
    for (unsigned m = 0; m < dim; ++m) {
        double stress = 0.;
        for (unsigned n = 0; n < dim; ++n)
            stress += (reverse ? pair.stress(n, m) : pair.stress(m, n)) * change.velocity[n];
        sum.stress[m] += stress;
    }
}
// Одно представление геометрических коэффициентов для невязки и её якобиана.
struct GradientStencil {
    const PairMetric* pairs; // Для элемента его верхний треугольник лежит подряд в памяти.
    const double* pair_heat; // На тесном GPU диффузия SA читает только скаляр теплопереноса.
    // Вычисляет тепловой вклад парного геометрического оператора.
    AERO_HOST_DEVICE double heat(const Cell& cell,unsigned i,unsigned j) const {
        const auto index=cell.pair_index(i,j);
        return pair_heat ? pair_heat[index] : pairs[index].heat;
    }
    // Возвращает геометрический тензор взаимодействия пары вершин.
    AERO_HOST_DEVICE const PairMetric& metric(const Cell& cell, unsigned i, unsigned j) const {
        return pairs[cell.pair_index(i,j)];
    }
    // Оператор AES использует все остальные вершины элемента, включая
    // диагонально противоположные. Сначала суммируем геометрические вклады, затем
    // один раз умножаем на средние mu и mu*u, общие для этих пар.
    template<unsigned d, class GasState>
    AERO_HOST_DEVICE Gradient vertex_gradient(const Cell& cell, unsigned i, const GasState* gas) const {
        Gradient gradient{};
        const auto& here = gas[cell.node[i]];
        // Пары записаны верхним треугольником. До диагонали берём столбец i
        // с транспонированием, после неё подряд читаем строку i в прямом направлении.
        auto index = cell.first_pair;
        if (i != 0) index += i-1;
        for (unsigned j = 0; j < i; ++j) {
            add_gradient<d>(pairs[index], true, difference(d, here, gas[cell.node[j]]), gradient);
            index += cell.count-2-j;
        }
        index = cell.first_pair + i*(2*cell.count-i-1)/2;
        for (unsigned j = i+1; j < cell.count; ++j, ++index)
            add_gradient<d>(pairs[index], false, difference(d, here, gas[cell.node[j]]), gradient);
        return gradient;
    }
    // Вычисляет градиенты величин по геометрическим коэффициентам элемента.
    template<unsigned d, class GasState>
    AERO_HOST_DEVICE void cell_gradients(const Cell& cell, const GasState* gas, Gradient* output) const {
        for (unsigned i = 0; i < cell.count; ++i)
            output[cell.first_vertex+i] = vertex_gradient<d>(cell, i, gas);
    }
};

// На этапе подготовки сетки вычисляет интегралы произведений градиентов
// базисных функций и сохраняет коэффициенты всех пар элемента одним массивом.
struct MeanGradient {
    Coordinate basis[8]; // Средние по объёму производные базисных функций.
    double volume;
};
// Подготавливает геометрические коэффициенты вязких градиентов.
class GradientBuilder {
public:
    // Вычисляет геометрические и транспортные коэффициенты градиента одного элемента.
    void prepare(const Element&, unsigned dimension, double gamma, double prandtl,
                 Cell&, std::vector<PairMetric>&, MeanGradient* mean = nullptr);
private:
    // Общие градиенты и парные коэффициенты элементов одной геометрической формы.
    struct Shape {
        MeanGradient mean;
        MeshIndex first_pair; // Начало коэффициентов для этой формы элемента.
        std::array<double,8> diffusion; // Геометрическая часть вязкого ограничения CFL по вершинам.
    };
    // Точный ключ содержит относительные координаты и связность вершин. Размер
    // временного словаря ограничен также для полностью неструктурированной сетки.
    std::map<std::array<double,32>, Shape> shapes_;
};
}
}
