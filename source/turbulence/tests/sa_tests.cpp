#include "../transport.h"
#include "../convection.h"
#include "../../../tests/test_checks.h"
#include <iostream>
#include <vector>
using namespace aero::turbulence;
// Вершины 0 и 2 принадлежат одному периодическому узлу. Его собственная
// пара исключается, остальные связи имеют разные знаки или нулевой вес.
void test_shared_diffusion() {
    aero::viscosity::Cell cell{};
    cell.count=4;
    cell.node[1]=1;
    cell.node[3]=2;
    aero::viscosity::PairMetric pairs[6]{};
    const double heat[]={-2.,17.,1.,0.,3.,-1.};
    for (unsigned p=0;p<6;++p) pairs[p].heat=heat[p];
    const aero::MeshIndex offsets[]={0,2}, incidence[]={0,2};
    double w[]={2.,5.,8.}, mean=2.;
    Transport t{};
    t.cells=&cell;
    t.shared_geometry.pairs=pairs;
    t.heat_to_diffusion=.5;
    t.filter.offsets=offsets;
    t.filter.incidence=incidence;
    t.sa.w=w;
    t.sa.diffusion.element_mean=&mean;
    double rate=0., diagonal=0.;
    t.add_volume_diffusion<false>(0,w[0],rate,diagonal);
    aero_test::require(rate==6. && diagonal==4.,"SA diffusion from signed AES coefficients");
}
// У одного ребра задаём разные приращения слева и справа. Смена знака расхода
// должна выбрать другую сторону, в том числе при локальном понижении порядка.
void test_sa_convection() {
    using aero::ReconstructionIndex;
    const ReconstructionIndex offsets[] = {0,2,4}, nodes[] = {0,1,0,1}, owners[] = {0,1};
    const double weights[] = {-1,1,-.5,.5};
    aero::ReconstructionStencilView stencil{};
    stencil.edge_count = 1;
    stencil.node_count = 2;
    stencil.offsets = offsets;
    stencil.nodes = nodes;
    stencil.weights = weights;
    stencil.row_owner = owners;
    double w[] = {4,6};
    aero::residual_detail::Surface face{};
    face.end = 1;
    aero::EulerFluxResult gas{};
    double result = 0.;
    const SaEdgeConvection edge{stencil,&face,nullptr,&gas,w,&result,1.,0.};
    // Вычисляет эталонный поток SA для проверки конвективного оператора.
    const auto flux = [&](double mass, unsigned char mask) {
        gas.flux[0] = mass;
        face.use_nodal_traces = mask;
        edge(0);
        return result;
    };
    aero_test::require(flux(2.,0)==2. && flux(-2.,0)==1., "SA upwind side and orientation");
    aero_test::require(flux(2.,2)==0. && flux(-2.,2)==1., "SA first order on the left");
    aero_test::require(flux(2.,4)==2. && flux(-2.,4)==0., "SA first order on the right");
    aero_test::require(flux(2.,1)==0. && flux(-2.,1)==0., "SA open boundary nodal stencil");
    w[0] = w[1];
    aero_test::require(flux(2.,0)==0. && flux(-2.,0)==0., "constant SA field");
    w[0] = 0.;
    aero_test::require(flux(2.,0)==0., "no SA outflow from an empty donor");
    w[0] = .125;
    aero_test::require(flux(2.,0)==.25, "bound on additional outflow");

    // Упакованный полный шаблон берёт дальний наклон со стороны донора.
    // Разные значения на двух концах проверяют ориентацию при смене знака потока.
    using namespace aero::residual_detail;
    WenoEdgeTile<5> tile{};
    tile.order[0].flags=9;
    const unsigned anchors[]{2,1,3,0,4};
    for (unsigned row=0; row<5; ++row) {
        const auto first=first_slope_term(row);
        tile.nodes(first,0)=anchors[row];
        tile.nodes(first+1,0)=anchors[row]+1;
        tile.weights(first-row,0)=1.;
    }
    const double profile[]{1.,1.2,1.45,1.8,2.4,3.1};
    face.begin=2; face.end=3; face.use_nodal_traces=0;
    stencil.mp5.alpha=4.;
    const SaEdgeConvection packed{stencil,&face,nullptr,&gas,profile,&result,1.,0.};
    gas.flux[0]=2.;
    packed.packed<5,true>(0,&tile);
    aero_test::require(std::abs(result-aero::mp5::increment(.2,.25,.35,.6,4.))<1.e-14,"packed left SA flux");
    gas.flux[0]=-2.;
    packed.packed<5,true>(0,&tile);
    aero_test::require(std::abs(result-aero::mp5::increment(.7,.6,.35,.25,4.))<1.e-14,"packed right SA flux");
}
// Несимметричный градиент с сингулярными числами 3, 2, 1; ось вихря совпадает с z.
void test_sigma() {
    FilterCell box{};
    box.volume=8.; box.vol_length=2.; box.min_edge=1.; box.max_edge=4.;
    for (unsigned i=0;i<8;++i)
        box.position[i]={{double(i&1),double(i&2),double(i&4)}};
    // Вычисляет физическую вихревую вязкость из тестового состояния SA.
    const auto viscosity=[&](Scale scale) {
        LesSettings settings;
        settings.scale=scale;
        settings.c_sigma=1.;
        VelocityGradient g={{0.,-2.,0.,3.,0.,0.,0.,0.,1.}};
        return evaluate_sigma(g,box,8,settings);
    };
    for (auto v : {std::make_pair(Scale::Max,16.),{Scale::Min,1.},{Scale::Vol,4.},
                  {Scale::Omega,2.},{Scale::OmegaTilde,5./3.},{Scale::Sla,5./3.}}) {
        const auto sample=viscosity(v.first);
        aero_test::require(std::abs(sample.value-v.second/9.)<1.e-14 && sample.tilt==0., "Sigma box filter");
    }
    VelocityGradient planar={{1.,2.,0.,4.,0.,0.,0.,0.,0.}};
    aero_test::require(sigma_rate(planar)==0., "Sigma planar limit");
    VelocityGradient rotated={{1.8,-2.4,0.,1.6,1.2,0.,0.,0.,1.}};
    aero_test::require(std::abs(sigma_rate(rotated)-1./9.)<1.e-14,"Sigma rotated basis");
    aero_test::require(kh_factor(.1,.2)==.2 && kh_factor(.4,.2)==1. &&
                       std::abs(kh_factor(.225,.2)-.6)<1.e-14,"SLA ramp");
}
// Вращаем диагональную матрицу слева и справа: исходные сингулярные числа
// известны и после обоих поворотов. Проверяем в том числе почти кратные корни.
void test_sigma_spectrum() {
    for (unsigned i=0;i<=32768;++i) {
        const double r=i/16384.-1.;
        const double expected=std::cos(std::acos(r)/3.);
        aero_test::require(std::abs(largest_root_cosine(r)-expected)<4.e-16,"Sigma largest root");
    }
    for (double middle : {1., 1.-1.e-9, .99, .7, .31, 1.e-3, 1.e-8}) {
        for (double ratio : {1., 1.-1.e-9, .999, .55, .04, 1.e-10, 0.}) {
            const double low=middle*ratio;
            const double expected=low*(1.-middle)*(middle-low);
            for (unsigned n=1;n<=12;++n) {
                VelocityGradient g={{1.,0.,0.,0.,middle,0.,0.,0.,low}};
                for (unsigned i=0;i<3;++i) {
                    const unsigned j=(i+1)%3;
                    const double c=std::cos(n+i*.7), s=std::sin(n+i*.7);
                    for (unsigned k=0;k<3;++k) {
                        const double a=g(i, k), b=g(j, k);
                        g(i, k)=c*a-s*b;
                        g(j, k)=s*a+c*b;
                    }
                    for (unsigned k=0;k<3;++k) {
                        const double a=g(k, i), b=g(k, j);
                        g(k, i)=c*a+s*b;
                        g(k, j)=c*b-s*a;
                    }
                }
                const double rate=sigma_rate(g);
                aero_test::require(rate>=0. && std::abs(rate-expected)<2.e-13,
                                   "Sigma rotated prescribed singular values");
            }
        }
    }
}

// У вытянутого наклонного гексаэдра сравниваем четыре диагонали со всеми 28
// парами вершин. У искажённого элемента должен сохраниться общий обход пар.
void test_sigma_filter_geometry() {
    FilterCell box{};
    box.volume=4.;
    for (unsigned i=0;i<8;++i) {
        const double x=i&1, y=(i>>1)&1, z=(i>>2)&1;
        box.position[i]={{4.*x+.25*y,.5*y+.125*z,2.*z}};
    }
    FilterCell compact=box;
    compact.prepare_geometry(8);
    aero_test::require(compact.diagonals==4,"Sigma symmetric filter geometry");
    for (unsigned i=1;i<20;++i) {
        aero::Coordinate axis{{std::sin(i),std::cos(i),.3}};
        const double norm=aero::coordinate_length(axis);
        for (double& v : axis) v/=norm;
        for (auto scale : {Scale::Omega,Scale::OmegaTilde,Scale::Sla}) {
            const double expected=box.length_squared(scale,8,axis);
            aero_test::require(std::abs(compact.length_squared(scale,8,axis)-expected)<3.e-14,
                               "Sigma projected diameter");
        }
    }
    for (double width : {1.,4.,1.e6}) {
        FilterCell rectangular{};
        rectangular.volume=2.*width;
        for (unsigned i=0;i<8;++i)
            rectangular.position[i]={{(i&1)*width,double((i>>1)&1),2.*((i>>2)&1)}};
        FilterCell reduced=rectangular;
        reduced.prepare_geometry(8);
        aero_test::require(reduced.rectangular_box,"rectangular hexahedron detection");
        // Смена местных номеров вершин и большая постоянная скорость не должны
        // менять производные линейного поля. Проверяем все девять компонент.
        struct Cell { unsigned node[8]; } cell{};
        // Тестовые компоненты скорости в вершинах элемента.
        struct Gas { double velocity[3]; } gas[8]{};
        const VelocityGradient linear={{1.,.5,-2.,.25,-3.,4.,-2.,2.,1.}};
        for (unsigned k=0;k<8;++k) {
            cell.node[k]=(3*k+1)%8;
            for (unsigned i=0;i<3;++i) {
                gas[cell.node[k]].velocity[i]=4096.;
                for (unsigned j=0;j<3;++j)
                    gas[cell.node[k]].velocity[i]+=linear(i, j)*rectangular.position[k][j];
            }
        }
        VelocityGradient gradient{};
        unsigned nodes[8];
        for (unsigned k=0;k<8;++k) nodes[k]=cell.node[reduced.corner_node[k]];
        const double speed=reduced.rectangular_gradient(nodes,1,gas,true,gradient);
        aero_test::require(speed>=4096.,"rectangular filter velocity scale");
        for (unsigned i=0;i<3;++i) for (unsigned j=0;j<3;++j)
            aero_test::require(std::abs(gradient(i, j)-linear(i, j))<1.e-14,"linear box gradient");
        for (aero::Coordinate axis : {aero::Coordinate{{1.,0.,0.}},aero::Coordinate{{-.2,.3,.4}}}) {
            for (auto scale : {Scale::Omega,Scale::OmegaTilde,Scale::Sla}) {
                const double ref=rectangular.length_squared(scale,8,axis);
                aero_test::require(std::abs(reduced.length_squared(scale,8,axis)-ref)<=1.e-14*ref,
                                   "rectangular filter versus all projected pairs");
            }
        }
    }
    box.position[6][1]+=.03125;
    const FilterCell all_pairs=box;
    box.prepare_geometry(8);
    const aero::MeshIndex shape_id=0;
    // Собираем таблицу в обеих раскладках. Ненулевой номер полосы на GPU
    // позволяет обнаружить ошибку в адресации компонент одной диагонали.
    for (unsigned lanes : {1u,32u}) {
        std::vector<double> coefficients(HexFilterData::Coefficients*lanes);
        double* column=coefficients.data()+lanes-1;
        column[HexFilterData::OtherDiameter*lanes]=box.remaining_length2;
        for (unsigned k=0; k<4; ++k) {
            const auto diagonal=box.long_diagonal(k);
            for (unsigned j=0; j<3; ++j) column[(HexFilterData::Diagonals+3*k+j)*lanes]=diagonal[j];
        }
        const HexFilter packed{column,lanes,&box,&shape_id,0,box.vol_length};
        unsigned shortcuts=0;
        for (unsigned k=1;k<193;++k) {
            aero::Coordinate axis{{std::sin(k),std::cos(.3*k),std::sin(1.3*k)}};
            const double expected=all_pairs.full_transverse_filter(8,axis);
            double filter2;
            if (try_transverse_filter(packed,axis,box.remaining_length2,filter2)) ++shortcuts;
            aero_test::require(packed.length_squared(Scale::Sla,8,axis)==expected,"packed diameter versus all pairs");
            aero_test::require(box.length_squared(Scale::OmegaTilde,8,axis)==expected,"bounded diameter versus all pairs");
        }
        aero_test::require(shortcuts>0 && shortcuts<192,"both projected diameter paths exercised");
    }
    aero_test::require(box.diagonals==0,"Sigma general filter geometry");
    // Вход при min/max имеет нулевую ось по определению. Сжатая таблица
    // должна вернуть выбранную длину ребра, даже когда она отлична от объёмного масштаба.
    box.min_edge=.5; box.max_edge=4.; box.vol_length=1.25;
    const HexFilter packed{nullptr,1,&box,&shape_id,0,box.vol_length};
    for (auto scale : {Scale::Min,Scale::Max,Scale::Vol}) {
        const double expected=box.length_squared(scale,8,aero::Coordinate{});
        aero_test::require(packed.length_squared(scale,8,aero::Coordinate{})==expected,"packed filter min/max/vol");
    }
}
// Сверяем защиту пограничного слоя с формулой Спаларта. При отсутствии стенки
// остаётся сеточный масштаб, при нулевом градиенте у стенки остаётся исходная SA.
void test_ddes() {
    DdesPoint p{.1,3.};
    aero_test::require(p.inverse_length(-1.,.02,kappa)<0.,"DDES no-slip wall");
    aero_test::require(p.inverse_length(0.,.02,kappa)==10.,"DDES no solid wall");
    aero_test::require(p.inverse_length(20.,.02,kappa)==20.,"DDES coarse mesh");
    for (double ratio : {0.,.001,.02,.08,.1,.25,1.,20.}) {
        const double wall=.7, nu=ratio*kappa*kappa*wall*wall*p.gradient_norm;
        const double fd=1.-std::tanh(std::pow(8.*ratio,3.));
        const double exact=1./(wall-fd*(wall-p.grid_length));
        aero_test::require(std::abs(p.inverse_length(1./wall,nu,kappa)-exact)<2.e-14,"DDES shielding");
    }
    FilterCell box{};
    box.volume=2.; box.vol_length=std::cbrt(2.);
    VelocityGradient strain={{1.,1.e-16,0.,0.,2.,0.,0.,0.,3.}};
    LesSettings scale;
    scale.scale=Scale::Sla;
    const auto uncertain=filter_scale(strain,box,0,scale,1.e-14);
    aero_test::require(uncertain.value==box.vol_length*box.vol_length && uncertain.tilt==0.,
                       "Roundoff in curl uses volumetric filter");
    p.gradient_norm=0.;
    aero_test::require(p.inverse_length(2.,.01,kappa)==2.,"DDES stagnant layer");
    p.grid_length=0.;
    aero_test::require(p.inverse_length(0.,.01,kappa)<0.,"DDES zero unshielded SLA scale");
}
// Три узла с разными локальными шагами. Подставляем найденные поправки
// в исходное трёхдиагональное уравнение; справа оставляем заданную невязку SA.
void test_sa_line() {
    const RowRange segments[] = {{0, 3}, {3, 4}};
    const LineVertex vertices[] = {{0,0.,2.}, {1,1.,2.}, {2,2.,0.}, {3,0.,0.}};
    double w[] = {1.,1.,1.,0.}, old[] = {1.,1.,1.,0.}, plain[4]{};
    double dt[] = {.5,1.,2.,.25}, rate[] = {1.,-.2,.3,5.}, diagonal[] = {3.,2.,4.,0.};
    const double volume[] = {1.,1.,1.,1.}, distance[] = {1.,1.,1.,-1.};
    View::Material material[4]{};
    Step step{};
    auto& sa = step.transport.sa;
    sa.lines = {vertices, segments, 2};
    sa.nodal_material = material;
    sa.w = w; sa.old = old; sa.rate = rate; sa.diagonal = diagonal;
    sa.plain_update = plain; sa.inverse_distance = distance;
    step.local_steps = {dt}; step.dt = step.leading = 1.;
    step.transport.volume = volume;
    step.relax_line(0);
    step.relax_line(1);
    for (unsigned i=0; i<3; ++i) {
        // Во второй строке сумма весов равна трём при диагонали два.
        const double scale = i==1 ? 2./3. : 1.;
        double balance = (1.+dt[i]*diagonal[i])*(w[i]-old[i]);
        if (i) balance -= dt[i]*scale*vertices[i].left*(w[i-1]-old[i-1]);
        if (i<2) balance -= dt[i]*scale*vertices[i].right*(w[i+1]-old[i+1]);
        aero_test::require(std::abs(balance - dt[i]*rate[i]) < 2.e-15, "SA line balance");
        aero_test::require(plain[i]==old[i]+dt[i]*rate[i]/(1.+dt[i]*diagonal[i]), "SA line fallback");
    }
    aero_test::require(w[3]==0. && plain[3]==0., "SA line wall");
}

// Запускает проверки модуля и возвращает ненулевой код при ошибке.
int main() {
    try {
        // Для геометрической последовательности поправок получаем точную сумму.
        // Нарастающие поправки и отрицательное пробное значение оставляют обычный шаг.
        aero_test::require(extrapolated_viscosity(1.5,.5,1.)==2.,"implicit SA geometric corrections");
        aero_test::require(extrapolated_viscosity(1.,.75,1.)==1.,"implicit SA increasing extrapolation");
        aero_test::require(extrapolated_viscosity(.1,-.5,-1.)==.1,"implicit SA positive branch");
        test_sa_line();
        test_sa_convection();
        test_shared_diffusion();
        test_sigma();
        test_ddes();
        for (unsigned k=0;k<=20000;++k) {
            const double x=k/1000.;
            aero_test::require(std::abs(DdesPoint::shielding_fraction(x)-std::tanh(x))<=3.e-16,
                               "DDES protection versus hyperbolic tangent");
        }
        test_sigma_spectrum();
        test_sigma_filter_geometry();
        // Половина демпфирования при chi=cv1 даёт независимую проверку обращения.
        aero_test::require(std::abs(ambient_chi(cv1 / 2.) - cv1) < 2.e-14, "SA midpoint viscosity");
        for (double ratio : {0., 1.e-150, .001, .21, 1., 100., 1.e15}) {
            const double chi = ambient_chi(ratio);
            aero_test::require(std::abs(chi * fv1(chi) - ratio) <= 2.e-14 * ratio,
                               "physical eddy viscosity ratio");
        }
        // При нулевой завихренности и бесконечном расстоянии до стенки источник равен нулю.
        aero_test::require(source_rate(.03, .01, 0., 0.) == 0., "uniform freestream");
        const double growth = cb1 * (1. - 1.2 * std::exp(-4.5)) * 2.;
        aero_test::require(std::abs(source_rate(.03, .01, 2., 0.) - growth) < 1.e-14, "free shear production");
        aero_test::require(source_rate(.03, .01, 0., 10.) < 0., "wall destruction");
        // Независимая формула с pow охватывает обе ветви и весь допустимый интервал r.
        // Густая сетка около g=2 проверяет область наибольшей ошибки начальной оценки.
        for (unsigned i = 0; i <= 12000; ++i) {
            const double r = 10. * i / 12000., g = .7 * r + .3 * std::pow(r, 6);
            const double exact = g * std::pow(65. / (64. + std::pow(g, 6)), 1. / 6.);
            aero_test::require(std::abs(wall_function(g) - exact) <= 2.e-15 * exact, "SA wall function");
        }
        std::cout << "SA sources, Sigma spectra and filter geometry, DDES shielding passed.\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
