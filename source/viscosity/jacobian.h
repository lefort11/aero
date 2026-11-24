#ifndef AERO_VISCOUS_JACOBIAN_H
#define AERO_VISCOUS_JACOBIAN_H
#include "transport.h"
#include "../implicit/edge_jacobian.h"
#include "../implicit/block_layout.h"

namespace aero {
namespace viscosity {
// Узел столбца может входить в несколько элементов и встречаться в одном элементе
// несколькими периодическими копиями. Адреса блоков определяются один раз при старте.
struct ColumnElement {
    // Номер ячейки, дающей вклад в столбец.
    MeshIndex cell;
    unsigned source;    // Вершина элемента, консервативное состояние которой варьируем.
    MeshIndex block[8]; // Позиция (строка вершины, столбец источника) в CSR или missing.
};
constexpr MeshIndex missing_block = static_cast<MeshIndex>(-1);
// Для отброшенной связи пропускаем также её вклад в диагональ. Для чужой
// строки MPI (missing_block) компенсация в собственной диагонали сохраняется.
constexpr MeshIndex omitted_pair = missing_block-1;

// Производная по одной компоненте U=(rho,rho*u,E). thermal обозначает p/rho.
// Сначала дифференцируем переход к примитивным переменным, затем оба коэффициента
// элемента: среднюю вязкость и среднее произведение вязкости на скорость.
struct GasDerivative {
    // Производные примитивных переменных по консервативной компоненте.
    Difference primitive;
    // Производные транспортных коэффициентов.
    Material transport;
    // Вычисляет производные примитивных и транспортных величин по компоненте состояния.
    AERO_HOST_DEVICE GasDerivative(unsigned dim, const Law& law, const Gas& gas, unsigned variable)
        : primitive{}, transport{} {
        const double thermal = gas.pressure_over_density;
        const double kinetic = variable == 0 ? .5*euler_detail::dot(dim,gas.velocity,gas.velocity) : 0.;
        const double dp = implicit::edge_detail::pressure_column(variable,dim+1,law.gamma,kinetic,gas);
        primitive.thermal = (dp - (variable == 0 ? thermal : 0.)) / gas.density;
        for (unsigned m = 0; m < dim; ++m)
            primitive.velocity[m] = implicit::edge_detail::velocity_column(variable,m,gas);
        // d ln(mu)/d ln(T) для normal равно 1/2; поправка Сазерленда равна S/(T+S).
        const double theta = law.temperature_scale * thermal;
        const double exponent = .5 + law.sutherland_ratio/(theta+law.sutherland_ratio);
        transport.mu = exponent * gas.mu * (primitive.thermal/thermal);
        for (unsigned m = 0; m < dim; ++m)
            transport.mu_velocity[m] = transport.mu*gas.velocity[m] + gas.mu*primitive.velocity[m];
    }
};

// Диагональ стоит первой, остальные столбцы отсортированы. Одинаковый поиск
// используется при подготовке тензора пары и при записи коэффициентов в матрицу.
struct ColumnGraph {
    // Границы списков столбцов по строкам.
    const MeshIndex* offsets;
    // Номера столбцов в выбранной разрядности.
    implicit::IndexList columns;
    // Находит требуемого соседа в строке разреженной матрицы.
    AERO_HOST_DEVICE MeshIndex find_neighbour(MeshIndex row,MeshIndex column) const {
        MeshIndex first=offsets[row]+1;
        const MeshIndex end=offsets[row+1];
        MeshIndex last=end;
        while (first<last) {
            const MeshIndex middle=first+(last-first)/2;
            if (columns[middle]<column) first=middle+1;
            else last=middle;
        }
        return first!=end && columns[first]==column ? first : omitted_pair;
    }
    // После периодической склейки две вершины могут обозначать одну неизвестную.
    // Для остальных проверяем обе строки, чтобы учитывать и одностороннюю связь.
    AERO_HOST_DEVICE bool contains_pair(MeshIndex a,MeshIndex b) const {
        return a!=b && (find_neighbour(a,b)!=omitted_pair || find_neighbour(b,a)!=omitted_pair);
    }
};

// Собирает производные вязкого оператора по столбцам газового состояния.
struct Jacobian {
    // Вязкий оператор в точке линеаризации.
    Operator aes;
    const MeshIndex* offsets;       // Все вхождения каждой неизвестной в элементы.
    const MeshIndex* row_offsets;    // При размещении всей сетки номера узлов равны номерам строк CSR.
    implicit::IndexList column_nodes; // В компактном режиме находим блок по двум узлам в готовом графе.
    // Элементы, участвующие в заполнении столбцов.
    const ColumnElement* elements;
    Gradient* gradients;            // Геометрическая часть напряжений и теплопереноса по вершинам элемента.
    double* values;                 // Та же CSR-матрица, в которую добавляется конвективный вклад.
    // Количество обрабатываемых значений матрицы.
    MeshIndex value_count;
    implicit::BlockLayout layout;   // Общая с конвекцией адресация коэффициентов Якобиана.
    MeshIndex single_count;         // Длина отдельной области для соседних блоков в mixed-сборке.

    // Градиенты и коэффициенты нужны всем столбцам элемента. Готовим их один раз
    // при фактической сборке матрицы на том состоянии, которое передал Ньютон.
    AERO_HOST_DEVICE void prepare_cell(MeshIndex index) const {
        aes.average_material(index);
        const auto& cell = aes.cells[index];
        if (aes.dim == 3) aes.stencil.cell_gradients<3>(cell, aes.gas, gradients);
        else if (aes.dim == 2) aes.stencil.cell_gradients<2>(cell, aes.gas, gradients);
        else aes.stencil.cell_gradients<1>(cell, aes.gas, gradients);
    }
    // Формирует адреса матричных блоков для производных одного элемента.
    AERO_HOST_DEVICE ColumnGraph column_graph() const { return {row_offsets,column_nodes}; }
    // Обходим столбец: изменение одной компоненты U в одном узле. Потоки CPU
    // и GPU пишут в разные столбцы даже при совпадении строк, атомарные операции не нужны.
    template<unsigned dim>
    AERO_HOST_DEVICE void add_column(const GradientSlice& slice, unsigned variable,
                                     MeshIndex first_vertex,CoordinatePairTile coordinate_pairs = {}) const {
        if (slice.first == slice.end) return;
        const MeshIndex node = slice.node;
        const unsigned width = dim+2;
        GasDerivative derivative(dim,aes.law,aes.gas[node],variable);
#ifdef AERO_HAS_TURBULENCE
        if (aes.eddy_nu) {
            // В матрице Ньютона фиксируем ν_t. Производные ρν_t и работы напряжений
            // остаются в блоке газа; в полной невязке Sigma обновляется при каждой пробе.
            const double nu=aes.eddy_nu[node];
            const double dmu=variable==0 ? nu : 0.;
            derivative.transport.mu+=dmu;
            derivative.transport.extra_heat=dmu*(aes.les_pr_ratio-1.);
            for (unsigned m=0;m<dim;++m)
                derivative.transport.mu_velocity[m]+=dmu*aes.gas[node].velocity[m]+
                    nu*aes.gas[node].density*derivative.primitive.velocity[m];
        }
        if(aes.sa.active) {
            // Газовый блок дифференцируем при фиксированной рабочей вязкости.
            // Изменение nu_tilde учитывается повторным решением уравнения SA на итерации.
            const auto& gas=aes.gas[node];
            const double w=aes.sa.w[node], rho_w=gas.density*w, chi=rho_w/gas.mu;
            const double f=turbulence::fv1(chi), slope=3.*f*(1.-f);
            const double dmut=(variable==0 ? w*(f+slope) : 0.)-chi*slope*derivative.transport.mu;
            derivative.transport.mu+=dmut;
            derivative.transport.extra_heat=dmut*(aes.sa.pr_ratio-1.);
            for(unsigned m=0;m<dim;++m) derivative.transport.mu_velocity[m]+=
                dmut*gas.velocity[m]+rho_w*f*derivative.primitive.velocity[m];
        }
#endif
        for (MeshIndex k=slice.first; k<slice.end; ++k) {
            // При работе на одной карте связь узел–элемент уже есть в AES.
            // Для MPI храним подготовленные адреса, включая строки соседнего процесса.
            const MeshIndex code=column_nodes ? aes.incidence[k] : 0;
            const MeshIndex element=column_nodes ? code/8 : elements[k].cell;
            const unsigned j=column_nodes ? static_cast<unsigned>(code%8) : elements[k].source;
            const auto& cell=aes.cells[element];
            const auto diagonal=column_nodes ? row_offsets[node] : elements[k].block[j];
            double balance[5]{};
            for (unsigned i = 0; i < cell.count; ++i) {
                // После периодической склейки поток между копиями одного узла внутренний.
                if (cell.node[i] == node) continue;
                const auto slot=column_nodes ? column_graph().find_neighbour(cell.node[i],node) : elements[k].block[i];
                if (slot == omitted_pair || (slot == missing_block && diagonal == missing_block)) continue;

                // Правило произведения. Меняются разности u и p/rho на паре i,j,
                // а также усреднённые по ВСЕМ вершинам коэффициенты перед градиентами.
                Gradient direct{};
                if (aes.affine) {
                    // Компактная таблица сразу выдаёт блок в направлении i -> j.
                    const auto pair=aes.affine[element].metric(i,j,aes.stencil.heat(cell,i,j));
                    add_gradient<dim>(pair,false,derivative.primitive,direct);
                } else if (dim==3 && aes.tetra.cells) {
                    const PairMetric pair{coordinate_pairs.get(aes.tetra,element,i,j),aes.stencil.heat(cell,i,j)};
                    add_gradient<dim>(pair,i>j,derivative.primitive,direct);
                } else {
                    add_gradient<dim>(aes.stencil.metric(cell,i,j),i>j,derivative.primitive,direct);
                }
                double field[5]{}, coefficient[5]{};
                aes.material[element].add_flux(dim,direct,field);
                // Рабочий массив начинается с первой вершины текущей группы элементов.
                derivative.transport.add_flux(dim,gradients[cell.first_vertex-first_vertex+i],coefficient);
                for (unsigned r = 1; r < width; ++r) {
                    const double value = field[r] + coefficient[r]*cell.inverse_count;
                    if (slot != missing_block) layout.view(values,slot,width,false).add(r*width+variable,value);
                    balance[r] -= value;
                }
            }
            // Сумма производных внутренней невязки всех строк по U_j равна нулю.
            // В диагонали учитываем также соседние строки, принадлежащие другому рангу.
            if (diagonal != missing_block)
                for (unsigned r = 1; r < width; ++r)
                    layout.view(values,diagonal,width,true).add(r*width+variable,balance[r]);
        }
    }
    // Добавляет производные вязких вкладов в выбранный столбец матрицы.
    AERO_HOST_DEVICE void assemble_column(MeshIndex node, unsigned variable) const {
        // Весь столбец сразу. На одной карте список смежных элементов уже есть в AES.
        const IndexArray<MeshIndex> starts = column_nodes ? aes.offsets : IndexArray<MeshIndex>(offsets);
        const GradientSlice slice{node,starts[node],starts[node+1]};
        if (aes.dim == 3) add_column<3>(slice,variable,0);
        else if (aes.dim == 2) add_column<2>(slice,variable,0);
        else add_column<1>(slice,variable,0);
    }
};
}
}
#endif
