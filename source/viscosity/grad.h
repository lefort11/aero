#pragma once
#include "../mesh/coordinate.h"
#include "../mesh/mesh_index.h"
#include "../mesh/compact_table.h"
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
    // Индекс элемента и номера его вершин.
    MeshIndex id, nodes[8];
    // Ранги владельцев вершин.
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
    // Геометрические коэффициенты тензора вязких напряжений.
    FixedMatrix<double,3,3> stress;
    // Геометрический коэффициент теплопереноса.
    double heat;
    // Из интегралов P_mn = ∫ ∂mNi ∂nNj получаем производную вязкого напряжения.
    // Здесь одна формула для постоянной таблицы CPU и вычисления по координатам.
    AERO_HOST_DEVICE static double stress_entry(unsigned m,unsigned n,double trace,double p_mn,double p_nm) {
        return (m==n ? trace : 0.)+p_nm-(2./3.)*p_mn;
    }
};
// Вершины элемента, веса усреднения и геометрические коэффициенты диффузии.
struct Cell {
    // Число активных вершин ячейки.
    unsigned count;
    double inverse_count; // Вес вершины при усреднении коэффициентов переноса.
    // Номера вершин в локальном порядке.
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
// Поля записи доступны без копирования массивов в локальную память GPU.
struct CellView {
    // Число доступных узлов представления.
    unsigned count;
    // Обратное число вершин для усреднения.
    double inverse_count;
    // Представление номеров вершин в выбранной разрядности.
    IndexArray<MeshIndex> node;
    // Геометрические коэффициенты диффузии.
    const double* diffusion;
    // Смещения первых пары и вершины в рабочих массивах.
    MeshIndex first_pair,first_vertex;
    // Пустое представление используется только неактивными потоками последней группы.
    AERO_HOST_DEVICE CellView() : count(0),inverse_count(0.),node(),diffusion(nullptr),first_pair(0),first_vertex(0) {}
    // Подключает массивы исходной 64-битной записи без копирования.
    AERO_HOST_DEVICE CellView(const Cell& c) : count(c.count),inverse_count(c.inverse_count),node(c.node),
        diffusion(c.diffusion),first_pair(c.first_pair),first_vertex(c.first_vertex) {}
    // Сохраняет прежнее треугольное расположение коэффициентов пар.
    AERO_HOST_DEVICE MeshIndex pair_index(unsigned i,unsigned j) const {
        return first_pair+Cell::pair_offset(count,i<j ? i : j,i<j ? j : i);
    }
};
// Упаковываются только целые номера; коэффициенты и веса усреднения остаются double.
struct Cell32 {
    // Обратное число вершин и геометрические коэффициенты двойной точности.
    double inverse_count,diffusion[8];
    // Короткие номера вершин, их количество и смещения.
    std::uint32_t node[8],count,first_pair,first_vertex;
    // Представление ячейки для чтения из обоих форматов.
    using View=CellView;
    // Учитывает адреса таблиц пар и вершин, которые могут быть больше числа ячеек.
    static bool fits(const Cell& c) {
        if (c.first_pair>UINT32_MAX || c.first_vertex>UINT32_MAX) return false;
        for (unsigned i=0;i<c.count;++i) if (c.node[i]>UINT32_MAX) return false;
        return true;
    }
    // Неиспользуемые позиции остаются нулевыми и никогда не читаются оператором.
    static Cell32 pack(const Cell& c) {
        Cell32 result{};
        result.count=c.count; result.inverse_count=c.inverse_count;
        result.first_pair=static_cast<std::uint32_t>(c.first_pair);
        result.first_vertex=static_cast<std::uint32_t>(c.first_vertex);
        for (unsigned i=0;i<c.count;++i) {
            result.node[i]=static_cast<std::uint32_t>(c.node[i]); result.diffusion[i]=c.diffusion[i];
        }
        return result;
    }
    // Возвращает представление исходного формата на CPU и больших сетках GPU.
    AERO_HOST_DEVICE static CellView view(const Cell& c) { return CellView(c); }
    // Адреса узлов и коэффициентов указывают непосредственно в постоянную таблицу.
    AERO_HOST_DEVICE static CellView view(const Cell32& c) {
        CellView result;
        result.count=c.count; result.inverse_count=c.inverse_count;
        result.node=IndexArray<MeshIndex>(c.node); result.diffusion=c.diffusion;
        result.first_pair=c.first_pair; result.first_vertex=c.first_vertex;
        return result;
    }
};
using CellTable=CompactTable<Cell,Cell32>;
// Вклад геометрии одной пары до умножения на коэффициенты переноса.
// Эти же линейные формулы применяются к разности полей и к её производной.
struct Gradient {
    // Компоненты вязкого напряжения и член теплопереноса.
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
    AERO_HOST_DEVICE double heat(const CellView& cell,unsigned i,unsigned j) const {
        const auto index=cell.pair_index(i,j);
        return pair_heat ? pair_heat[index] : pairs[index].heat;
    }
    // Возвращает геометрический тензор взаимодействия пары вершин.
    AERO_HOST_DEVICE const PairMetric& metric(const CellView& cell, unsigned i, unsigned j) const {
        return pairs[cell.pair_index(i,j)];
    }
    // Оператор AES использует все остальные вершины элемента, включая
    // диагонально противоположные. Сначала суммируем геометрические вклады, затем
    // один раз умножаем на средние mu и mu*u, общие для этих пар.
    template<unsigned d, class GasState>
    AERO_HOST_DEVICE Gradient vertex_gradient(const CellView& cell, unsigned i, const GasState* gas) const {
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
    AERO_HOST_DEVICE void cell_gradients(const CellView& cell, const GasState* gas, Gradient* output) const {
        for (unsigned i = 0; i < cell.count; ++i)
            output[cell.first_vertex+i] = vertex_gradient<d>(cell, i, gas);
    }
};

// На этапе подготовки сетки вычисляет интегралы произведений градиентов
// базисных функций и сохраняет коэффициенты всех пар элемента одним массивом.
struct MeanGradient {
    Coordinate basis[8]; // Средние по объёму производные базисных функций.
    // Объём элемента для нормировки среднего градиента.
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
        // Метрики среднего градиента данной формы.
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
