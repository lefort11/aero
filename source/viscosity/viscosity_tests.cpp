#include "aes.h"
#include "../implicit/spatial_jacobian.h"
#include "../residual/residual_detail.h"
#include "../../qa/test_field.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_arrays.h"
#endif
#include "../../qa/test_checks.h"
#include <cmath>
#include <iostream>
#include <numeric>

namespace {
using namespace aero;
using Vertex = std::array<double,3>;
using Face = std::vector<MeshIndex>;

void near(double value, double expected, const char* reason) {
    aero_test::require(std::abs(value-expected) <= 2.e-12*(1.+std::abs(expected)),reason);
}

// Одна ячейка с косыми осями. Сдвиг не меняет объём, но даёт ненулевые смешанные
// производные базиса, которые нельзя проверить одним лапласианом на прямоугольнике.
MeshPartition element(unsigned dim, const std::vector<Vertex>& coordinates, const std::vector<Face>& faces) {
    MeshData mesh;
    mesh.dimension = dim;
    for (auto point : coordinates) {
        mesh.x.push_back(point[0]+.27*point[1]-.13*point[2]);
        mesh.y.push_back(point[1]+.19*point[2]);
        mesh.z.push_back(point[2]);
    }
    mesh.patch_names.push_back("boundary");
    for (const auto& face : faces) {
        const auto index = mesh.boundary_faces.size();
        mesh.face_nodes.indices.insert(mesh.face_nodes.indices.end(),face.begin(),face.end());
        mesh.face_nodes.offsets.push_back(mesh.face_nodes.indices.size());
        mesh.cell_faces.indices.push_back(index);
        mesh.cell_face_orientation.push_back(dim == 1 && index == 0 ? -1 : 1);
        mesh.boundary_faces.push_back(index);
        mesh.boundary_patch.push_back(0);
    }
    mesh.cell_faces.offsets.push_back(faces.size());
    MeshPartition part;
    part.mesh = Mesh(std::move(mesh));
    part.node_owners.resize(coordinates.size(),0);
    part.nodes.resize(coordinates.size());
    std::iota(part.nodes.begin(),part.nodes.end(),MeshIndex(0));
    return part;
}

// При x = X + s*Y на квадрате вклад противоположной вершины в лапласиан равен s
// для разбиения вдоль этой диагонали и нулю для другой. Среднее AES равно s/2.
// Ни один обход только физических рёбер четырёхугольника эту связь не воспроизведёт.
void check_quadrilateral_diagonal(const Processes& world, const MeshPartition& part) {
    ViscositySettings settings;
    settings.law = "normal";
    settings.reynolds = settings.prandtl = 1.;
    viscosity::Aes aes;
    const double gamma = 1.4;
    aes.prepare(part,nullptr,{},world,settings,gamma,1./std::sqrt(gamma),4,
                [](MeshIndex id,int) { return id; });
    aes.allocate_host(4);
    double state[16]{};
    for (unsigned node = 0; node < 4; ++node) {
        state[4*node] = 1.;
        state[4*node+3] = (node == 0 ? 2. : 1.)/(gamma-1.);
    }
    aes.evaluate(state);
    const double average_mu = (std::sqrt(2.)+3.)/4.;
    near(aes.view().rows[2].residual[3],.27/2.*average_mu*gamma/(gamma-1.),"AES virtual diagonal");
}

// Собираем настоящий CSR на выбранном исполнителе. Эталон — центральная разность
// двух полных невязок (LF первого порядка + AES) при фиксированном начальном поле.
// Проверка каждого столбца охватывает в том числе пары, связанные только вязкостью.
void check_jacobian(viscosity::Aes& aes, const Mesh& mesh, const std::vector<double>& field, int device) {
    using namespace residual_detail;
    const auto topology = mesh.view();
    const MedianDual dual(mesh);
    std::vector<Surface> faces(topology.edge_count);
    for (MeshIndex e = 0; e < topology.edge_count; ++e) {
        faces[e].begin = topology.edge_begin[e];
        faces[e].end = topology.edge_end[e];
        project_surface(faces[e],dual.view().edge_areas[e],topology.dimension,{});
    }
    std::vector<Index> offsets;
    std::vector<OrientedFace> incidence;
    build_incidence(topology.node_count,topology.edge_count,faces,offsets,incidence);
    const auto viscous = aes.view();
    auto options = residual_options("conservative","lf",viscous.law.gamma);
    options.upw_coeff = .65;
    options.diffusion = viscous.rows;
    options.wall = aes.wall.data();
    Geometry host{topology.dimension,topology.node_count,faces.size(),faces.size(),faces.data(),
                  offsets.data(),incidence.data(),{},{},options,{}};
    implicit::Partition partition;
    partition.host_aes = partition.active_aes = viscous;
    auto active = host;
#ifdef AERO_WITH_KOKKOS_GPU
    gpu::DeviceArrays arrays;
    if (device >= 0) {
        arrays.device = device;
        partition.active_aes = aes.upload(arrays,host.nodes);
        active.surfaces = arrays.upload(faces);
        active.offsets = arrays.upload(offsets);
        active.incidence = arrays.upload(incidence);
        active.options.diffusion = partition.active_aes.rows;
        active.options.wall = arrays.upload(aes.wall);
    }
#endif
    implicit::SpatialJacobian matrix(host,active,device,partition);
    aero_test::TestField state(device,field.size()), x(device,field.size()), y(device,field.size());
    state.set(field);
    matrix.assemble(state.data());
    const unsigned width = host.components();
    std::vector<EulerFluxResult> flux(faces.size());
    const auto residual = [&](const std::vector<double>& u) {
        aes.evaluate(u.data());
        cpu_face_fluxes({host,nullptr,faces.size(),u.data(),nullptr,flux.data(),nullptr,nullptr,false});
        std::vector<double> r(u.size());
        for (MeshIndex n = 0; n < host.nodes; ++n)
            aero_test::require(assemble_node_residual(host,n,flux.data(),r.data()+n*width),"invalid residual");
        return r;
    };
    const double h = 2.e-6;
    for (size_t column = 0; column < field.size(); ++column) {
        std::vector<double> unit(field.size(),0.), plus = field, minus = field;
        unit[column] = 1.;
        x.set(unit);
        matrix.apply(x.data(),y.data());
        const auto& product = y.read();
        plus[column] += h;
        minus[column] -= h;
        const auto rp = residual(plus), rm = residual(minus);
        for (unsigned r = 0; r < width; ++r) {
            double sum = 0., norm = 0.;
            for (size_t i = r; i < field.size(); i += width) {
                const double expected = (rp[i]-rm[i])/(2.*h);
                const double error = std::abs(product[i]-expected);
                aero_test::require(error < 2.e-8*(1.+std::abs(expected)),
                    "LF+AES Jacobian finite difference: column " + std::to_string(column) +
                    ", row " + std::to_string(i) + ", error " + std::to_string(error));
                sum += product[i];
                norm += std::abs(product[i]);
            }
            // Закреплённая компонента импульса имеет внешнюю реакцию стенки.
            // Для массы и энергии баланс столбца сохраняется и в этом случае.
            if (!aes.wall.front() || r == 0 || r+1 == width)
                aero_test::require(std::abs(sum) <= 3.e-15*(1.+norm),"nonconservative LF+AES Jacobian column");
        }
    }
}
void check_element(const Processes& world, const MeshPartition& part, double volume, bool sutherland = true) {
    const auto mesh = part.mesh.view();
    const auto d = mesh.dimension, width = d+2;
    const double gamma = 1.4, mach = .75;
    ViscositySettings settings;
    settings.law = sutherland ? "sutherland" : "normal";
    settings.prandtl = .73;
    settings.reynolds = 7.;
    settings.t_amb = 300.;
    settings.sutherland_s = 111.;
    viscosity::Aes aes;
    aes.prepare(part,nullptr,{},world,settings,gamma,mach,mesh.node_count,[](MeshIndex id,int) { return id; });
    aes.allocate_host(mesh.node_count);
    const auto view = aes.view();
    std::vector<double> field(mesh.node_count*width);
    for (MeshIndex node = 0; node < mesh.node_count; ++node) {
        field[node*width] = 1.;
        field[node*width+d+1] = 3.;
    }
    aes.evaluate(field.data());
    for (MeshIndex i = 0; i < mesh.node_count; ++i)
        for (unsigned k = 0; k < width; ++k)
            aero_test::require(view.rows[i].residual[k] == 0.,"uniform field");

    // Аффинные u и T имеют постоянные производные на любом из разбиений.
    // Независимо интегрируем аналитические напряжения и тепловой поток по ячейке.
    const double gradient[3][3] = {{.3,-.1,.4},{.7,-.2,.5},{-.6,.15,.11}};
    const double grad_p[] = {.13,-.09,.08};
    double mu = 0., mu_u[3]{}, divergence = 0.;
    for (unsigned k = 0; k < d; ++k) divergence += gradient[k][k];
    for (MeshIndex i = 0; i < mesh.node_count; ++i) {
        double* q = &field[i*width];
        const auto point = mesh.coordinate(i);
        double p = 1., kinetic = 0.;
        for (unsigned k = 0; k < d; ++k) p += grad_p[k]*point[k];
        const double t = gamma*mach*mach*p;
        const double viscosity = (sutherland ? std::pow(t,1.5)*(300.+111.)/(300.*t+111.) : std::sqrt(t))/7.;
        mu += viscosity;
        for (unsigned m = 0; m < d; ++m) {
            double u = .31;
            for (unsigned k = 0; k < d; ++k) u += gradient[m][k]*point[k];
            q[1+m] = u;
            mu_u[m] += viscosity*u;
            kinetic += .5*u*u;
        }
        q[d+1] = p/(gamma-1.) + kinetic;
    }
    aes.evaluate(field.data());
    for (unsigned component = 0; component < width; ++component) {
        double sum = 0.;
        for (MeshIndex i = 0; i < mesh.node_count; ++i) sum += view.rows[i].residual[component];
        near(sum,0.,"conservation");
    }
    for (unsigned a = 0; a < d; ++a) {
        double flux[5]{};
        for (MeshIndex i = 0; i < mesh.node_count; ++i)
            for (unsigned c = 0; c < width; ++c)
                flux[c] += mesh.coordinate(i)[a]*view.rows[i].residual[c];
        const double scale = volume/static_cast<double>(mesh.node_count);
        double energy = mu*gamma/((gamma-1.)*.73)*grad_p[a];
        for (unsigned m = 0; m < d; ++m) {
            const double stress = gradient[m][a]+gradient[a][m]-(a==m ? 2.*divergence/3. : 0.);
            near(flux[m+1],scale*mu*stress,"momentum flux");
            energy += mu_u[m]*stress;
        }
        near(flux[d+1],scale*energy,"energy flux");
    }
    // Неоднородная плотность исключает случайное совпадение производных по U и по примитивным полям.
    for (MeshIndex i = 0; i < mesh.node_count; ++i) field[i*width] += .017*static_cast<double>(i);
    const auto check_device = [&](int device) {
        check_jacobian(aes,part.mesh,field,device);
        if (d == 2 && mesh.node_count == 4) {
            aes.wall.front() = 1;
            check_jacobian(aes,part.mesh,field,device);
            aes.wall.front() = 0;
        }
    };
    check_device(-1);
#ifdef AERO_WITH_KOKKOS_GPU
    int devices = 0;
    const auto status = gpu::count(&devices);
    if (status != gpu::Status::NoDevice && status != gpu::Status::Unavailable) gpu::checked(status);
    for (int device = 0; device < devices; ++device) {
        gpu::DeviceScope scope(device);
        gpu::checked(scope.status);
        check_device(device);
    }
#endif
}
}

int main(int argc, char** argv) {
    aero::Processes world(argc,argv);
    try {
        const aero::viscosity::Law normal{1.4,4.,.01,.72,0.};
        near(normal.dynamic_viscosity(.25),.01,"reference viscosity");
        near(normal.dynamic_viscosity(1.),.02,"normal law");
        const Vertex a{0,0,0}, b{1,0,0}, c{1,1,0}, d{0,1,0}, e{0,0,1}, f{1,0,1}, g{1,1,1}, h{0,1,1};
        check_element(world,element(1,{a,b},{{0},{1}}),1.);
        check_element(world,element(2,{a,b,d},{{0,1},{1,2},{2,0}}),.5);
        const auto quad = element(2,{a,b,c,d},{{0,1},{1,2},{2,3},{3,0}});
        check_element(world,quad,1.,false);
        check_quadrilateral_diagonal(world,quad);
        check_element(world,element(3,{a,b,d,e},{{0,2,1},{0,1,3},{1,2,3},{2,0,3}}),1./6.);
        check_element(world,element(3,{a,b,c,d,e},{{3,2,1,0},{0,1,4},{1,2,4},{2,3,4},{3,0,4}}),1./3.);
        check_element(world,element(3,{a,b,d,e,f,h},{{2,1,0},{3,4,5},{0,1,4,3},{1,2,5,4},{2,0,3,5}}),.5);
        check_element(world,element(3,{a,b,c,d,e,f,g,h},
            {{3,2,1,0},{4,5,6,7},{0,1,5,4},{1,2,6,5},{2,3,7,6},{3,0,4,7}}),1.);
        std::cout << "AES fluxes, LF+AES Jacobian columns and conservation passed on all seven element types.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
