#pragma once
#include "volume_coordinates.h"
#include <vector>

namespace aero {
namespace viscosity {
// Интеграл линейной базисной функции зависит от ориентированных площадей
// треугольников. Для каждой вершины сохраняем только ненулевые слагаемые.
// Номера и целые множители общие у ячеек с тем же разбиением и ориентациями.
struct MeanAreaTerm {
    unsigned char a,b;
    signed char multiplier;
};
struct MeanAreaForm {
    MeshIndex first;
    unsigned char offset[9]; // Диапазон слагаемых каждой из восьми вершин.
};
struct MeanAreaTable {
    const MeanAreaForm* forms;
    const MeanAreaTerm* terms;
    const std::uint32_t* cell_ids; // При единственной форме индекс у всех ячеек равен нулю.
};

// Для тетраэдра интеграл градиента N_i равен нормали к противоположной
// грани, умноженной на знак определителя и вес разбиения, делённый на шесть.
// До запуска солвера складываем целые коэффициенты у одинаковых произведений
// r_a × r_b. Для любого положения вершин они остаются теми же.
inline MeanAreaForm append_mean_area_form(const VolumePattern& pattern,std::uint32_t signs,
                                          std::vector<MeanAreaTerm>& terms) {
    FixedTensor<int,8,8,8> coefficient{};
    const unsigned opposite[4][3]={{1,3,2},{0,2,3},{0,3,1},{0,1,2}};
    for (unsigned t=0;t<pattern.tetrahedra;++t) {
        const auto* vertex=pattern.nodes.row(t);
        const int sign=(signs & (std::uint32_t(1)<<t)) ? -1 : 1;
        for (unsigned k=0;k<4;++k) for (unsigned j=0;j<3;++j) {
            // (r_b-r_a) × (r_c-r_a) = r_a×r_b + r_b×r_c + r_c×r_a.
            const auto a=vertex[opposite[k][j]],b=vertex[opposite[k][(j+1)%3]];
            if (a<b) coefficient(vertex[k],a,b)+=sign;
            else coefficient(vertex[k],b,a)-=sign;
        }
    }
    MeanAreaForm form{};
    form.first=terms.size();
    for (unsigned k=0;k<8;++k) {
        form.offset[k]=static_cast<unsigned char>(terms.size()-form.first);
        // При вычислении градиента скорости из всех скоростей вычитается u_0.
        // Слагаемое вершины 0 равно нулю; координаты тоже отсчитаны от x_0.
        if (!k) continue;
        for (unsigned a=1;a<8;++a) for (unsigned b=a+1;b<8;++b) {
            const int value=coefficient(k,a,b);
            if (value) terms.push_back({static_cast<unsigned char>(a),static_cast<unsigned char>(b),
                                        static_cast<signed char>(value)});
        }
    }
    // Не более 7*21=147 слагаемых; все локальные смещения помещаются в один байт.
    form.offset[8]=static_cast<unsigned char>(terms.size()-form.first);
    return form;
}

struct MeanAreaBasis {
    const MeanAreaForm& form;
    const MeanAreaTerm* terms;
    Coordinate relative[8];
    double inverse_volume;
    AERO_HOST_DEVICE MeanAreaBasis(const MeanAreaTable& table,const VolumeCoordinates& geo,
                                   MeshIndex e,unsigned count,double volume)
        : form(table.forms[table.cell_ids ? table.cell_ids[e] : 0]), terms(table.terms+form.first) {
        const auto& cell=geo.cells[e];
        const auto& origin=geo.position[cell.vertex[0]];
        for (unsigned i=1;i<count;++i) relative[i]=geo.position[cell.vertex[i]]-origin;
        inverse_volume=1./(geo.patterns[cell.pattern].volume_divisor*volume);
    }
    // Сумма ориентированных площадей, делённая на фактический объём ячейки,
    // даёт средний градиент базисной функции сразу по трём направлениям.
    AERO_HOST_DEVICE void read(unsigned vertex,double* result) const {
        Coordinate sum{};
        for (unsigned i=form.offset[vertex];i<form.offset[vertex+1];++i) {
            const auto term=terms[i];
            const auto area=cross3(relative[term.a],relative[term.b]);
            for (unsigned j=0;j<3;++j) sum[j]+=term.multiplier*area[j];
        }
        for (unsigned j=0;j<3;++j) result[j]=sum[j]*inverse_volume;
    }
};
}
}
