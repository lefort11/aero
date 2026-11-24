#pragma once
#include "grad.h"
#include <cmath>
#include <cstdint>

namespace aero {
namespace viscosity {
// Определитель сохраняет ориентацию тетраэдра. Его модуль нужен для интеграла,
// а нормали — для производных четырёх базисных функций в этом же порядке вершин.
struct TetrahedronGeometry {
    Coordinate normal[4];
    double determinant;
    // Вычисляет геометрию тетраэдра по указанным вершинам.
    template<class Points>
    AERO_HOST_DEVICE TetrahedronGeometry(const Points& point,const unsigned char* v) {
        const auto origin=point[v[0]];
        const auto a=point[v[1]]-origin,b=point[v[2]]-origin,c=point[v[3]]-origin;
        normal[1]=cross3(b,c); normal[2]=cross3(c,a); normal[3]=cross3(a,b);
        normal[0]=Coordinate{}-(normal[1]+normal[2]+normal[3]);
        determinant=dot3(a,normal[1]);
    }
    // Возвращает объёмный множитель геометрического симплекса.
    AERO_HOST_DEVICE double factor(double divisor) const { return 1./(divisor*::fabs(determinant)); }
};
// Локальная связность зависит от порядка вершин в исходной ячейке.
// Координаты к этой таблице не привязаны. Одинаковый порядок вершин даёт одну таблицу.
struct VolumePattern {
    FixedMatrix<unsigned char,24,4> nodes;       // Четыре вершины каждого тетраэдра AES.
    FixedMatrix<unsigned char,8,12> incidence;   // Пары (тетраэдр, его вершина) для суммирования в узел.
    unsigned char tetrahedra, degree[8]; // Действительные длины списков, зависят от типа элемента.
    double volume_divisor;            // 6 / вес одного разбиения: от определителя к объёму в AES.
    unsigned char pair_offsets[29];   // Для каждой пары вершин перечислены только общие тетраэдры.
    std::uint16_t pair_terms[24*6];   // Номер тетраэдра и две локальные вершины в нём.

    // Все элементы с одним разбиением используют общий список. Порядок обхода
    // тетраэдров совпадает с порядком интегрирования при построении таблицы CPU.
    void prepare_pairs() {
        unsigned used=0;
        for (unsigned i=0;i<8;++i) for (unsigned j=i+1;j<8;++j) {
            pair_offsets[Cell::pair_offset(8,i,j)]=static_cast<unsigned char>(used);
            for (unsigned t=0;t<tetrahedra;++t) {
                unsigned a=4,b=4;
                for (unsigned k=0;k<4;++k) {
                    if (nodes(t,k)==i) a=k;
                    if (nodes(t,k)==j) b=k;
                }
                if (a<4 && b<4) pair_terms[used++]=static_cast<std::uint16_t>(16*t+4*a+b);
            }
        }
        pair_offsets[28]=static_cast<unsigned char>(used);
    }
};
// Геометрические номера вершин и индекс шаблона объёмного элемента.
struct VolumeCellNodes {
    std::uint32_t vertex[8]; // Номера геометрических узлов, до периодической склейки.
    std::uint32_t pattern;
};
// Обычно геометрический узел и неизвестная имеют один номер. При периодическом
// отождествлении это может нарушаться: тогда координаты читаются по исходной сетке.
struct VolumeVertices {
    const MeshIndex* shared;
    const std::uint32_t* separate;
    // Возвращает элемент по индексу; допустимость индекса обеспечивает вызывающий код.
    AERO_HOST_DEVICE MeshIndex operator[](unsigned i) const { return shared ? shared[i] : separate[i]; }
};
// Вершины одного элемента и индекс его объёмного шаблона.
struct VolumeCellView {
    VolumeVertices vertex;
    std::uint32_t pattern;
};
// Выбирает общую или отдельную таблицу геометрических вершин элементов.
struct VolumeCellTable {
    const VolumeCellNodes* separate; // Геометрическая связность, если номера отличаются.
    const Cell* shared;              // Вершины уже есть в таблице вязкого оператора.
    const std::uint32_t* patterns;   // Для единственного разбиения индекс всегда равен нулю.
    // Проверяет наличие подключённой таблицы геометрических вершин.
    AERO_HOST_DEVICE explicit operator bool() const { return separate || shared; }
    // Возвращает элемент по индексу; допустимость индекса обеспечивает вызывающий код.
    AERO_HOST_DEVICE VolumeCellView operator[](MeshIndex e) const {
        if (shared) return {{shared[e].node,nullptr},patterns ? patterns[e] : 0u};
        return {{nullptr,separate[e].vertex},separate[e].pattern};
    }
};
// Координаты вершин и связность элементов для объёмного оператора AES.
struct VolumeCoordinates {
    const Coordinate* position;       // Исходные координаты всех точек в базисе уравнений.
    VolumeCellTable cells;            // Локальные номера вершин и тип их нумерации.
    const VolumePattern* patterns;
    const MeanGradient* mean;         // Средние производные для фильтра DDES.
    double* volume_factors;           // Постоянный множитель каждого тетраэдра, 24 значения на ячейку.
    bool all_hex;                     // На однородной сетке границы обхода известны при компиляции.
    MeshIndex cached_cells;           // Длина части сетки, для которой выделили таблицу.
};
// Номера геометрических узлов могут отличаться от номеров неизвестных, например,
// при периодической склейке. Здесь всегда берём координаты исходного элемента.
struct VolumePoints {
    const Coordinate* position;
    VolumeVertices vertex;
    // Подключает геометрические координаты к локальной нумерации элемента.
    AERO_HOST_DEVICE VolumePoints(const Coordinate* points,const VolumeCellView* cell)
        : position(points),vertex(cell->vertex) {}
    // Подключает геометрические координаты к локальной нумерации элемента.
    VolumePoints(const Coordinate* points,const VolumeCellNodes* cell)
        : position(points),vertex{nullptr,cell->vertex} {}
    // Возвращает элемент по индексу; допустимость индекса обеспечивает вызывающий код.
    AERO_HOST_DEVICE const Coordinate& operator[](unsigned i) const { return position[vertex[i]]; }
};

typedef FixedMatrix<double,3,3> PairStress;
// Суммируем произведения градиентов по всем тетраэдрам, содержащим пару.
// Для одного направления храним одну матрицу; обратное направление даёт транспонирование.
template<class ReadNormals>
AERO_HOST_DEVICE inline PairStress integrate_pair(const VolumePattern& pattern,unsigned pair, const ReadNormals& read) {
    PairStress products{};
    for (unsigned k=pattern.pair_offsets[pair];k<pattern.pair_offsets[pair+1];++k) {
        const unsigned code=pattern.pair_terms[k];
        Coordinate left,right;
        const double factor=read(code/16,(code/4)%4,code%4,left,right);
        for (unsigned m=0;m<3;++m) for (unsigned n=0;n<3;++n)
            products(m,n)+=factor*left[m]*right[n];
    }
    const double trace=products(0,0)+products(1,1)+products(2,2);
    PairStress stress;
    for (unsigned m=0;m<3;++m) for (unsigned n=0;n<3;++n)
        stress(m,n)=PairMetric::stress_entry(m,n,trace,products(m,n),products(n,m));
    return stress;
}
// Вычисляет парные геометрические множители по нормалям объёмных симплексов.
struct PairNormals {
    VolumePoints points;
    const VolumePattern* pattern;
    const double* factors;
    // Возвращает нормали двух вершин симплекса и соответствующий объёмный множитель.
    AERO_HOST_DEVICE double operator()(unsigned t,unsigned i,unsigned j,Coordinate& left,Coordinate& right) const {
        const TetrahedronGeometry tetra(points,pattern->nodes.row(t));
        left=tetra.normal[i]; right=tetra.normal[j];
        return factors ? factors[t] : tetra.factor(pattern->volume_divisor);
    }
};
// Рабочая плитка парных тензоров напряжений для группы элементов.
struct CoordinatePairTile {
    PairStress* pairs;
    MeshIndex first_cell;
    // Верхний треугольник восьми вершин содержит максимум 28 тензоров по 9 чисел.
    // Этой же оценкой пользуются разбиение сетки на CPU и размещение группы на GPU.
    static MeshIndex capacity(MeshIndex bytes) { return bytes/(28*sizeof(PairStress)); }
    // Возвращает парный тензор из рабочей плитки по локальным индексам.
    AERO_HOST_DEVICE PairStress get(const VolumeCoordinates& g,MeshIndex e,unsigned i,unsigned j) const {
        const unsigned pair=i<j ? Cell::pair_offset(8,i,j) : Cell::pair_offset(8,j,i);
        if (pairs) return pairs[(e-first_cell)*28+pair];
        // Небольшая или распределённая сетка может не иметь временной таблицы пар.
        const auto& cell=g.cells[e];
        const auto& pattern=g.patterns[cell.pattern];
        const auto factors=e<g.cached_cells ? g.volume_factors+24*e : nullptr;
        return integrate_pair(pattern,pair,PairNormals{{g.position,&cell},&pattern,factors});
    }
};
}
}
