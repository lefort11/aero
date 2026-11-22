#ifndef AERO_VISCOUS_TRANSPORT_H
#define AERO_VISCOUS_TRANSPORT_H
#include "affine_hex.h"
#include "volume_coordinates.h"
#include "mean_area.h"
#include "../parallel/modules.h"
#ifdef AERO_HAS_TURBULENCE
#include "../turbulence/sa.h"
#include "../turbulence/nodal_filter.h"
#include "../turbulence/hex_filter.h"
#endif

namespace aero {
namespace viscosity {
// Внутри шага BDF скаляр SA уточняется при фиксированном газе. Этап указывает,
// какие данные устарели и нужно ли уже собирать газовую невязку.
enum class Evaluation { Full, CurrentMaterial, LesClosure, DdesGasOnly, FilterOnly, BeforeSaIterations, AfterSaUpdate, AfterSaFixedGas, RowsOnly };
// Один план для CPU и GPU. Газ может уже быть подготовлен сборкой конвективного
// потока. Здесь определяем, какие действия выполнить после получения газа.
struct EvaluationPlan {
    bool filter=false;          // Вычислить подсеточный масштаб и завершить фазу.
    bool sa_material=false;     // Обновить узловую вязкость из нового решения SA.
    bool cell_material=false;   // Усреднить свойства газа по вершинам элемента.
    bool rows=false;            // Добавить вязкие потоки в строки невязки.
    bool reuse_gradients=false; // Внутри итераций SA поле газа осталось прежним.

    explicit EvaluationPlan(Evaluation phase) {
#ifdef AERO_HAS_TURBULENCE
        if (phase==Evaluation::DdesGasOnly) return;
        if (phase==Evaluation::LesClosure || phase==Evaluation::FilterOnly) {
            filter=true;
            return;
        }
        sa_material=phase==Evaluation::AfterSaUpdate || phase==Evaluation::AfterSaFixedGas;
#endif
        cell_material=phase!=Evaluation::RowsOnly;
        rows=phase!=Evaluation::BeforeSaIterations;
        reuse_gradients=phase==Evaluation::AfterSaFixedGas;
    }
};

struct Gas {
    double velocity[3], pressure_over_density, density, mu;
};
struct Material {
    double mu, mu_velocity[3]; // Средние по вершинам исходного элемента, формула (48).
#ifdef AERO_HAS_TURBULENCE
    double extra_heat; // Разность коэффициентов тепла и импульса при разных Pr и Pr_t.
#endif
    // Напряжения дают поток импульса, а их работа и теплопроводность — поток энергии.
    AERO_HOST_DEVICE void add_flux(unsigned dim, const Gradient& gradient, double* flux) const {
        double heat = mu;
#ifdef AERO_HAS_TURBULENCE
        heat += extra_heat;
#endif
        flux[dim+1] += heat * gradient.heat;
        for (unsigned m = 0; m < dim; ++m) {
            flux[1+m] += mu * gradient.stress[m];
            flux[dim+1] += mu_velocity[m] * gradient.stress[m];
        }
    }
};
struct Row {
    double residual[5], cfl_rate;
};
// В компактном неявном расчёте газ и вязкие строки занимают соседние части
// буфера потоков. Между сборками этот же участок используется решателем СЛАУ.
struct FlowScratch {
    Gas* gas;
    Row* rows;
};
struct Law {
    double gamma, temperature_scale, inverse_reynolds, prandtl, sutherland_ratio;
    // Масштабы: rho_inf = U_inf = 1. При T = T_inf получаем mu = 1/Re для обоих законов.
    AERO_HOST_DEVICE double dynamic_viscosity(double p_over_rho) const {
        const double theta = temperature_scale * p_over_rho;
        const double normal = ::sqrt(theta) * inverse_reynolds;
        return sutherland_ratio == 0. ? normal : normal * (theta / (theta + sutherland_ratio)) * (1. + sutherland_ratio);
    }
};
// Адреса принадлежат одному исполнителю. Те же три операции выполняются на CPU и в Kokkos на GPU.
struct GradientSlice {
    MeshIndex node,first,end; // Какие вхождения текущего узла принадлежат группе элементов.
};
struct GradientTile {
    MeshIndex first_cell,cells,first_slice,slices,first_vertex; // Начало градиентов группы в общей нумерации.
};
struct Operator {
    unsigned dim;
    Law law;
#ifdef AERO_HAS_TURBULENCE
    turbulence::View sa;
    turbulence::LesSettings les;
    const turbulence::FilterCell* filters;
    const turbulence::FilterExtent* filter_extents; // Для упакованного GPU-фильтра нужны только размеры и координаты.
    const turbulence::FilterLengths* coordinate_filters; // При общей сетке точки уже доступны в tetra.position.
    MeanAreaTable mean_areas; // Интегралы градиентов фильтра по координатам вершин ячейки.
    const MeshIndex* filter_indices;
    // При явном шаге один узел обслуживает все свои элементы. Вес для
    // напряжений, тепла и работы напряжений меняется только вместе с полем узла.
    struct NodalMaterial {
        double mu, extra_heat, mu_velocity[3], scalar_diffusion;
    };
    NodalMaterial* material_nodes;
    const std::uint32_t* material_vertices;
    unsigned vertices_per_cell;
    // В этой таблице GPU одинаковые вершины 32 соседних элементов лежат рядом.
    static constexpr unsigned material_lanes = 32;
    turbulence::HexFilterData hex_filter;
    const std::uint32_t* box_nodes;
    MeshIndex box_stride,box_pitch;
    const double* les_volumes;
    turbulence::LesSample* samples;
    double* eddy_nu; // Значения ν_t в узлах, включая копии с соседних процессов.
    double les_pr_ratio, c_des; // c_des положителен только для гибридной модели SA-DDES.
    bool fixed_ddes_length, skip_ddes_filter;
    double* cell_gradient_norm; // Объёмно-взвешенная норма градиента скорости для защиты слоя DDES.
    const unsigned char* no_slip; // Маска узлов неподвижной стенки AES.
    // При dim < 3 у градиента скорости есть нулевое сингулярное число.
    // Вязкость Sigma тождественно равна нулю на каждом узле при любом масштабе.
    AERO_HOST_DEVICE bool computes_les() const {
        return (dim==3 || c_des>0.) && les.scale!=turbulence::Scale::Off;
    }
    // Вычитаем скорость первой вершины до умножения на градиент базисной функции.
    // Так постоянная скорость даёт точный ноль даже на сильно искажённом элементе.
    // Максимум модуля компоненты нужен только для фильтра с направлением по ротору.
    template<class Geometry>
    AERO_HOST_DEVICE static void read_basis(const Geometry& geometry,unsigned vertex,unsigned dimension,double* basis) {
        for (unsigned j=0;j<dimension;++j) basis[j]=geometry.basis(vertex,j);
    }
    AERO_HOST_DEVICE static void read_basis(const MeanAreaBasis& geometry,unsigned vertex,unsigned,double* basis) {
        geometry.read(vertex,basis);
    }
    template<unsigned fixed_dim,unsigned fixed_count,class Geometry,class Index>
    AERO_HOST_DEVICE double mean_velocity_gradient(const Geometry& geometry,const Index* nodes,unsigned stride,
                                  unsigned count,turbulence::VelocityGradient& grad) const {
        const unsigned dimension=fixed_dim ? fixed_dim : dim;
        const unsigned vertices=fixed_count ? fixed_count : count;
        const bool directional=les.directional();
        const auto& origin=gas[nodes[0]];
        double speed=0.;
        if (directional) for (unsigned i=0;i<dimension;++i) speed=::fmax(speed,::fabs(origin.velocity[i]));
        for (unsigned k=1;k<vertices;++k) {
            const auto& point=gas[nodes[k*stride]];
            double basis[3];
            read_basis(geometry,k,dimension,basis);
            for (unsigned i=0;i<dimension;++i) {
                if (directional) speed=::fmax(speed,::fabs(point.velocity[i]));
                const double jump=point.velocity[i]-origin.velocity[i];
                for (unsigned j=0;j<dimension;++j) grad(i, j)+=jump*basis[j];
            }
        }
        return speed;
    }
    template<bool ddes,unsigned fixed_dim,bool box=false>
    AERO_HOST_DEVICE void sample_les(MeshIndex e) const {
        const auto& cell=cells[e];
        const auto& geometry=filters[filter_indices[e]];
        turbulence::VelocityGradient grad{};
        const double speed=box ? geometry.rectangular_gradient(box_nodes+e*box_pitch,box_stride,gas,les.directional(),grad)
                               : mean_velocity_gradient<fixed_dim,0>(geometry,cell.node,1,cell.count,grad);
        finish_sample<ddes>(e,geometry,box ? 8 : cell.count,fixed_dim ? fixed_dim : dim,grad,speed,
                            geometry.gradient_sum,geometry.volume*(box ? .125 : cell.inverse_count));
    }
    // Средние производные здесь те же, что использованы для линейной части AES.
    // Возвращаем номер вершины в порядке сетки: порядок суммирования модели сохраняется.
    struct AffineBasis {
        const AffineHex& cell;
        AERO_HOST_DEVICE double basis(unsigned i,unsigned axis) const { return cell.basis(axis,cell.ordered[i]); }
    };
    struct TetraBasis {
        const MeanGradient& cell;
        AERO_HOST_DEVICE double basis(unsigned i,unsigned axis) const { return cell.basis[i][axis]; }
    };
    // Фильтр получает те же относительные координаты, что при подготовке AES.
    // Геометрические номера берутся до периодической склейки; прикладная
    // модель читает один и тот же диаметр при обеих раскладках памяти.
    struct CoordinateFilter {
        const turbulence::FilterLengths& shape;
        const Coordinate* positions;
        VolumeCellView cell;
        double vol_length;
        AERO_HOST_DEVICE Coordinate relative(unsigned i) const {
            return positions[cell.vertex[i]]-positions[cell.vertex[0]];
        }
        AERO_HOST_DEVICE Coordinate point(unsigned i) const {
            if (!shape.diagonals) return relative(i);
            const unsigned pair=shape.longest_pairs[i];
            return relative(pair/8)-relative(pair%8);
        }
        AERO_HOST_DEVICE Coordinate long_diagonal(unsigned k) const { return shape.diagonal(*this,k); }
        AERO_HOST_DEVICE double length_squared(turbulence::Scale scale,unsigned count,const Coordinate& axis) const {
            return shape.squared(*this,scale,count,axis);
        }
    };
    template<bool ddes,unsigned fixed_count,class Basis,class Extent>
    AERO_HOST_DEVICE void sample_compact_cell(MeshIndex e,const Basis& basis,const Extent& extent,
                                            const turbulence::FilterLengths& sizes) const {
        turbulence::VelocityGradient gradient{};
        const unsigned count=fixed_count ? fixed_count : cells[e].count;
        const double weight=fixed_count==8 ? .125 : cells[e].inverse_count;
        const double speed=mean_velocity_gradient<3,fixed_count>(basis,cells[e].node,1,count,gradient);
        finish_sample<ddes>(e,extent,count,3,gradient,speed,sizes.gradient_sum,sizes.volume*weight);
    }
    template<bool ddes,unsigned fixed_count>
    AERO_HOST_DEVICE void sample_compact_filter(MeshIndex e) const {
        const auto shape=filter_indices[e];
        if (coordinate_filters) {
            const auto& sizes=coordinate_filters[shape];
            const CoordinateFilter extent{sizes,tetra.position,tetra.cells[e],sizes.vol_length};
            if (mean_areas.forms) {
                const auto count=fixed_count ? fixed_count : cells[e].count;
                const MeanAreaBasis basis(mean_areas,tetra,e,count,sizes.volume);
                sample_compact_cell<ddes,fixed_count>(e,basis,extent,sizes);
            } else sample_compact_cell<ddes,fixed_count>(e,TetraBasis{tetra.mean[shape]},extent,sizes);
        } else {
            const auto& extent=filter_extents[shape];
            sample_compact_cell<ddes,8>(e,AffineBasis{affine[e]},extent,extent);
        }
    }
    // Для компактной таблицы известны все размеры: восемь вершин и три направления.
    // Градиент и формула модели те же, что на смешанной сетке.
    template<bool ddes,class Shape>
    AERO_HOST_DEVICE void sample_hex(MeshIndex e,const Shape* shapes) const {
        const auto& table=hex_filter;
        const auto* nodes=table.nodes+table.first(e,8);
        const auto* column=table.coefficients+table.first(e,turbulence::HexFilterData::Coefficients);
        const turbulence::PackedHexFilter<Shape> geometry{column,table.lanes,shapes,filter_indices,e,
                column[turbulence::HexFilterData::VolumeLength*table.lanes]};
        turbulence::VelocityGradient grad{};
        const double speed=mean_velocity_gradient<3,8>(geometry,nodes,table.lanes,8,grad);
        finish_sample<ddes>(e,geometry,8,3,grad,speed,geometry.value(turbulence::HexFilterData::GradientSum),
                            geometry.value(turbulence::HexFilterData::Volume)*.125);
    }
    template<bool ddes,class Geometry>
    AERO_HOST_DEVICE void finish_sample(MeshIndex e,const Geometry& geometry,unsigned count,unsigned dimension,
                    turbulence::VelocityGradient& grad,double speed,double gradient_sum,double weight) const {
        // До восьми вкладов в производную и две производные в компоненте ротора.
        // Порог масштабируется как скорость/длина и следует за местной точностью данных.
        const double rotation_floor=(16.*DBL_EPSILON*speed)*gradient_sum;
        turbulence::LesSample result;
        if (ddes) {
            // DDES использует только норму первого порядка. Для обычных скоростей
            // сам квадрат нормы уже даёт безопасный масштаб для всех дальнейших операций.
            double sum=0.;
            for (unsigned i=0;i<dimension;++i) for (unsigned j=0;j<dimension;++j) sum+=grad(i, j)*grad(i, j);
            double magnitude=1.;
            if (!(sum>1.e-160 && sum<1.e160)) {
                magnitude=turbulence::normalize_gradient(grad);
                sum=0.;
                for (unsigned i=0;i<dimension;++i) for (unsigned j=0;j<dimension;++j) sum+=grad(i, j)*grad(i, j);
            }
            cell_gradient_norm[e]=weight*(magnitude*::sqrt(sum));
            if (fixed_ddes_length) return;
            result=turbulence::filter_scale(grad,geometry,count,les,magnitude>0. ? rotation_floor/magnitude : 0.);
        } else {
            result=turbulence::evaluate_sigma(grad,geometry,count,les,rotation_floor);
        }
        samples[e]={weight*result.value,weight*result.tilt};
    }
    AERO_HOST_DEVICE turbulence::NodalFilter nodal_filter() const {
        return {offsets,incidence,les_volumes,cell_gradient_norm,samples,no_slip,
                eddy_nu,skip_ddes_filter ? nullptr : sa.ddes,les,c_des,fixed_ddes_length};
    }
#endif
    MeshIndex node_count, cell_count, row_count, vertex_count, neighbour_count;
    const Cell* cells;
    const GradientSlice* gradient_slices; // На GPU сумма в узле продолжается со следующей группы элементов.
    const GradientTile* host_gradient_tiles; // Описание запусков Kokkos читает только CPU.
    MeshIndex gradient_tile_count;
    const AffineHex* affine; // Сокращённый оператор по элементам, когда полная таблица не помещается на GPU.
    VolumeCoordinates tetra;    // При дефиците памяти напряжения восстанавливаются по координатам на карте.
    GradientStencil stencil; // Геометрические данные принадлежат тому же исполнителю, что и поле.
    const MeshIndex *offsets, *incidence;
    Gas* gas;
    Gradient* gradient_cache; // Производные в порядке элементов или вхождений в собственные строки.
    bool element_order;      // Все вершины каждого элемента размещаются подряд.
    Material* material;
    Row* rows;

#ifdef AERO_HAS_TURBULENCE
    // На вершину приходится несколько элементов. Функцию демпфирования вычисляем
    // один раз по текущему полю SA. Материал всех соседних элементов читает результат.
    AERO_HOST_DEVICE void cache_material(MeshIndex node,double turbulent_mu,double pr_ratio,double scalar) const {
        const auto& point=gas[node];
        auto& result=material_nodes[node];
        result.mu=point.mu+turbulent_mu;
        result.extra_heat=turbulent_mu*(pr_ratio-1.);
        for (unsigned i=0; i<dim; ++i) result.mu_velocity[i]=result.mu*point.velocity[i];
        result.scalar_diffusion=scalar;
    }
    template<bool ddes>
    AERO_HOST_DEVICE void gather_filter(MeshIndex node) const {
        nodal_filter().gather<ddes>(node);
        if (!ddes && material_nodes) cache_material(node,gas[node].density*eddy_nu[node],les_pr_ratio,0.);
    }
    AERO_HOST_DEVICE void prepare_sa_material(MeshIndex node) const {
        const auto& g=gas[node];
        const double rho_w=g.density*sa.w[node];
        const double damping=turbulence::fv1(rho_w/g.mu);
        const turbulence::View::Material properties{g.mu/g.density,rho_w*damping,damping};
        sa.nodal_material[node]=properties;
        if (material_nodes) cache_material(node,properties.eddy_mu,sa.pr_ratio,properties.nu+sa.w[node]);
    }
#endif
    AERO_HOST_DEVICE bool needs_gas(Evaluation phase) const {
#ifdef AERO_HAS_TURBULENCE
        // Перед вязким потоком LES уже декодировала это состояние для расчёта ν_t.
        if (computes_les()) return phase==Evaluation::LesClosure || phase==Evaluation::DdesGasOnly;
#endif
        return phase==Evaluation::Full || phase==Evaluation::BeforeSaIterations;
    }
    // При наличии готового газа используем давление и скорости, уже вычисленные для
    // конвективного потока. В экономичной раскладке на GPU этого буфера может не быть.
    AERO_HOST_DEVICE void prepare_gas(MeshIndex node, const double* state, const EulerState* cached) const {
        EulerState decoded{};
        if (!cached && euler_state(dim,state+node*(dim+2),law.gamma,&decoded) != EulerStatus::Success)
            decoded.density = 0.;
        const EulerState& primitive = cached ? cached[node] : decoded;
        auto& value = gas[node];
        if (!(primitive.density > 0.)) {
            value = Gas{};
            value.mu = HUGE_VAL;
            return;
        }
        value.density = primitive.density;
        value.pressure_over_density = primitive.pressure / primitive.density;
        value.mu = law.dynamic_viscosity(value.pressure_over_density);
        for (unsigned k = 0; k < dim; ++k) value.velocity[k] = primitive.velocity[k];
#ifdef AERO_HAS_TURBULENCE
        if (sa.active) prepare_sa_material(node);
#endif
    }
    // Для работы напряжений в уравнении энергии усредняем именно произведение mu*u.
    enum class Mixing { General, Sa, Sigma };
    template<unsigned fixed_dim=0,unsigned fixed_count=0,Mixing model=Mixing::General>
    AERO_HOST_DEVICE void average_material(MeshIndex index) const {
        const auto& cell = cells[index];
        const unsigned dimension=fixed_dim ? fixed_dim : dim;
        const unsigned vertices=fixed_count ? fixed_count : cell.count;
        Material average{};
#ifdef AERO_HAS_TURBULENCE
        double molecular_sum=0., working_sum=0.;
#endif
        for (unsigned k = 0; k < vertices; ++k) {
            const auto& point = gas[cell.node[k]];
            double mu = point.mu;
#ifdef AERO_HAS_TURBULENCE
            if (model==Mixing::Sa || (model==Mixing::General && sa.active)) {
                const double working=sa.w[cell.node[k]];
                const auto properties=sa.nodal_material[cell.node[k]];
                molecular_sum+=properties.nu;
                working_sum+=working;
                const double turbulent_mu=properties.eddy_mu;
                mu += turbulent_mu;
                average.extra_heat += turbulent_mu*(sa.pr_ratio-1.);
            } else if (model==Mixing::Sigma || eddy_nu) {
                const double turbulent_mu=point.density*eddy_nu[cell.node[k]];
                mu+=turbulent_mu;
                average.extra_heat+=turbulent_mu*(les_pr_ratio-1.);
            }
#endif
            average.mu += mu;
            for (unsigned m = 0; m < dimension; ++m) average.mu_velocity[m] += mu * point.velocity[m];
        }
#ifdef AERO_HAS_TURBULENCE
        if (model==Mixing::Sa || (model==Mixing::General && sa.active)) {
            // Газ и SA используют одни вершины. Средние для диффузии SA готовим
            // при том же чтении узлов, что и напряжения и тепловой поток газа.
            sa.diffusion.molecular_mean[index]=molecular_sum*cell.inverse_count;
            sa.diffusion.element_mean[index]=sa.diffusion.molecular_mean[index]+working_sum*cell.inverse_count;
        }
        average.extra_heat *= cell.inverse_count;
#endif
        average.mu *= cell.inverse_count;
        for (unsigned m = 0; m < dimension; ++m) average.mu_velocity[m] *= cell.inverse_count;
        material[index] = average;
    }
#ifdef AERO_HAS_TURBULENCE
    // Среднее берём в исходном порядке вершин. Произведения mu*u уже
    // вычислены при подготовке узла и одинаковы у всех прилегающих элементов.
    template<unsigned count,bool scalar>
    AERO_HOST_DEVICE void average_cached_material(MeshIndex e) const {
        const unsigned lanes=material_lanes;
        const auto* vertices=material_vertices+(e/lanes)*count*lanes+e%lanes;
        Material mean{};
        double diffusivity=0.;
        for (unsigned k=0; k<count; ++k) {
            const auto& point=material_nodes[vertices[k*lanes]];
            mean.mu+=point.mu;
            mean.extra_heat+=point.extra_heat;
            for (unsigned i=0; i<dim; ++i) mean.mu_velocity[i]+=point.mu_velocity[i];
            if (scalar) diffusivity+=point.scalar_diffusion;
        }
        const double weight=1./count;
        mean.mu*=weight;
        mean.extra_heat*=weight;
        for (unsigned i=0; i<dim; ++i) mean.mu_velocity[i]*=weight;
        if (scalar) sa.diffusion.element_mean[e]=weight*diffusivity;
        material[e]=mean;
    }
#endif
    // Каждую строку записывает только владелец узла. Номер вхождения хранит
    // номер элемента и местный номер его вершины (не больше семи).
    enum class Gradients { Direct, Save, Reuse, Element, MixedElement };
    // Соседние исполнители читают общие скорости и коэффициенты одного
    // гексаэдра. Состав и порядок слагаемых совпадают со сборкой по узлам.
    AERO_HOST_DEVICE void prepare_hex_gradient(MeshIndex entry) const {
        const auto& cell = cells[entry/8];
        gradient_cache[entry]=stencil.vertex_gradient<3>(cell,static_cast<unsigned>(entry%8),gas);
    }
    template<unsigned d, Gradients mode>
    AERO_HOST_DEVICE void add_row_range(Row& result,MeshIndex first,MeshIndex end,MeshIndex first_vertex) const {
        for (MeshIndex k=first;k<end;++k) {
            const auto entry = incidence[k];
            const auto e = entry / 8;
            const auto& average = material[e];
            const unsigned i = static_cast<unsigned>(entry % 8);
            // При изменении только рабочей вязкости BDF сохраняет точно тот же
            // газ. Меняются средние коэффициенты элемента, но не производные поля.
            // В гексаэдре код вхождения 8*e+i уже равен адресу градиента.
            // На смешанной сетке берём начало списка фактических вершин элемента.
            const auto gradient=mode==Gradients::Element ? gradient_cache[entry-first_vertex] :
                mode==Gradients::MixedElement ? gradient_cache[cells[e].first_vertex+i-first_vertex] :
                mode==Gradients::Reuse ? gradient_cache[k] : stencil.vertex_gradient<d>(cells[e],i,gas);
            if (mode==Gradients::Save) gradient_cache[k]=gradient;
            average.add_flux(d, gradient, result.residual);
            double diffusion_mu=average.mu;
#ifdef AERO_HAS_TURBULENCE
            diffusion_mu+=::fmax(0.,average.extra_heat);
#endif
            result.cfl_rate += diffusion_mu * cells[e].diffusion[i];
        }
    }
    // Продолжаем сумму, только когда предыдущая группа уже записала часть строки.
    // Деление на плотность относится ко всей строке и выполняется после последней части.
    template<unsigned d, Gradients mode,bool partial>
    AERO_HOST_DEVICE void finish_row(MeshIndex node,MeshIndex first,MeshIndex end,MeshIndex first_vertex) const {
        Row result{};
        if (partial && first!=offsets[node]) result=rows[node];
        add_row_range<d,mode>(result,first,end,first_vertex);
        if (!partial || end==offsets[node+1]) result.cfl_rate*=2./gas[node].density;
        rows[node]=result;
    }
    template<unsigned d, Gradients mode=Gradients::Direct>
    AERO_HOST_DEVICE void assemble_row(MeshIndex node) const {
        finish_row<d,mode,false>(node,offsets[node],offsets[node+1],0);
    }
    AERO_HOST_DEVICE void assemble_slice(const GradientSlice& slice,MeshIndex first_vertex) const {
        if (tetra.cells && !tetra.all_hex)
            finish_row<3,Gradients::MixedElement,true>(slice.node,slice.first,slice.end,first_vertex);
        else
            finish_row<3,Gradients::Element,true>(slice.node,slice.first,slice.end,first_vertex);
    }
    template<Gradients mode=Gradients::Direct>
    AERO_HOST_DEVICE void assemble(MeshIndex node) const {
        if (dim == 3) assemble_row<3,mode>(node);
        else if (dim == 2) assemble_row<2,mode>(node);
        else assemble_row<1,mode>(node);
    }
};
}
}
#endif
