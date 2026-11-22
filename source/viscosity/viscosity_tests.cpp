#include "aes.h"
#include "../implicit/spatial_jacobian.h"
#include "../residual/residual_detail.h"
#include "../../tests/test_field.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_arrays.h"
#include "../parallel/device_work.h"
#endif
#include "../../tests/test_checks.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

namespace {
using namespace aero;
using Vertex = std::array<double,3>;
using Face = std::vector<MeshIndex>;

void near(double value, double expected, const char* reason) {
    aero_test::require(std::abs(value-expected) <= 2.e-12*(1.+std::abs(expected)),reason);
}

MeshPartition local_partition(Mesh mesh) {
    const auto nodes=mesh.view().node_count;
    MeshPartition part;
    part.mesh=std::move(mesh);
    part.node_owners.resize(nodes,0);
    part.nodes.resize(nodes);
    std::iota(part.nodes.begin(),part.nodes.end(),MeshIndex(0));
    return part;
}

// Одна ячейка с косыми осями даёт ненулевые смешанные производные базиса.
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
    return local_partition(Mesh(std::move(mesh)));
}

void check_jacobian(viscosity::Aes&,const Mesh&,const std::vector<double>&,int,bool,bool);

// Получаем те же указатели на постоянную геометрию, которые при обычной загрузке
// указывают на GPU. Проверка только читает их и не запрашивает рабочей памяти.
struct GeometryProbe {
    template<class T> T* upload(const std::vector<T>& values) { return const_cast<T*>(values.data()); }
    template<class T> T* allocate(std::size_t,const T* = nullptr) { return nullptr; }
};
void check_coordinate_pairs(const viscosity::Aes& aes,MeshIndex nodes,bool shared) {
    GeometryProbe probe;
    auto geometry=aes.upload(probe,nodes).tetra;
    aero_test::require((geometry.cells.shared!=nullptr)==shared,"coordinate and solution node numbering");
    geometry.cached_cells=0;
    const auto& cpu=aes.view();
    const viscosity::CoordinatePairTile direct{};
    for (MeshIndex e=0;e<cpu.cell_count;++e) {
        const auto& cell=cpu.cells[e];
        for (unsigned i=0;i<cell.count;++i) for (unsigned j=i+1;j<cell.count;++j) {
            const auto stress=direct.get(geometry,e,i,j);
            const auto& table=cpu.stencil.metric(cell,i,j);
            for (unsigned m=0;m<3;++m) for (unsigned n=0;n<3;++n)
                near(stress(m,n),table.stress(m,n),"mixed AES pair stress from coordinates");
        }
    }
}

// В короткой сетке из 12 узлов свободных строк невязки не хватает на тензоры
// даже одной ячейки. Четыре копии дают 48 реальных узлов и место для одной
// ячейки в плитке. В каждой копии сохраняются общие грани разных типов.
Mesh tiled_mixed_mesh(const Mesh& seed) {
    MeshData mesh;
    const auto& data=seed.data();
    mesh.dimension=data.dimension;
    mesh.patch_names=data.patch_names;
    const auto append_rows=[](MeshCsr& to,const MeshCsr& from,MeshIndex shift) {
        const auto old_end=to.indices.size();
        for (auto index:from.indices) to.indices.push_back(index+shift);
        for (std::size_t i=1;i<from.offsets.size();++i) to.offsets.push_back(old_end+from.offsets[i]);
    };
    for (unsigned copy=0;copy<4;++copy) {
        const auto nodes=mesh.x.size(),faces=mesh.face_nodes.offsets.size()-1;
        for (double x:data.x) mesh.x.push_back(x+4.*copy);
        mesh.y.insert(mesh.y.end(),data.y.begin(),data.y.end());
        mesh.z.insert(mesh.z.end(),data.z.begin(),data.z.end());
        append_rows(mesh.face_nodes,data.face_nodes,nodes);
        append_rows(mesh.cell_faces,data.cell_faces,faces);
        mesh.cell_face_orientation.insert(mesh.cell_face_orientation.end(),data.cell_face_orientation.begin(),
                                          data.cell_face_orientation.end());
        for (auto i:data.boundary_faces) mesh.boundary_faces.push_back(faces+i);
        mesh.boundary_patch.insert(mesh.boundary_patch.end(),data.boundary_patch.begin(),data.boundary_patch.end());
    }
    return Mesh(std::move(mesh));
}
// Три общие грани: пирамида примыкает к кубу и тетраэдру, призма — к кубу.
// Пирамида идёт первой, поэтому у следующих ячеек смещение уже отлично от 8*e.
void check_mixed_coordinates(const Processes& world,bool implicit,bool renumbered=false,bool tiled=false) {
    auto mesh=Mesh::read("qa/cases/mixed_3d/grid.mesh");
    const auto part=local_partition(tiled ? tiled_mixed_mesh(mesh) : std::move(mesh));
    const auto nodes=part.mesh.view().node_count;
    ViscositySettings settings;
    settings.law="normal";
    settings.prandtl=.72;
    settings.reynolds=9.;
    viscosity::Aes aes;
    aes.prepare_device_geometry();
#ifdef AERO_HAS_TURBULENCE
    turbulence::Settings ddes;
    ddes.enabled=ddes.ddes=true;
    ddes.les.scale=turbulence::Scale::Sla;
    aes.configure_les(ddes,settings.prandtl,true);
#endif
    aes.prepare(part,nullptr,{},world,settings,1.4,.75,nodes,[&](MeshIndex n,int){return renumbered ? nodes-1-n : n;});
    aes.allocate_host(nodes);
    // Нелинейное поле даёт различные напряжения на соседних тетраэдрах одной ячейки.
    std::vector<double> field(5*nodes);
    for (MeshIndex n=0;n<nodes;++n) {
        const double t=static_cast<double>(n)+.5;
        double* u=field.data()+5*n;
        u[0]=1.+.03*t;
        for (unsigned m=1;m<=3;++m) u[m]=.3*std::sin(t*m);
        u[4]=3.+.13*std::cos(t);
    }
#ifdef AERO_HAS_TURBULENCE
    // SLA использует направление завихренности. Сверим отдельно его длину и
    // норму градиента, которая входит в защитную функцию DDES.
    aes.evaluate(field.data(),nullptr,false,viscosity::Evaluation::DdesGasOnly);
    aes.evaluate(field.data(),nullptr,true,viscosity::Evaluation::FilterOnly);
    for (MeshIndex n=0;n<nodes;++n) aes.view().nodal_filter().gather<true>(n);
#endif
    aes.evaluate(field.data());
    const auto* reference=aes.view().rows;
    for (unsigned m=0;m<5;++m) {
        double sum=0.;
        for (MeshIndex n=0;n<nodes;++n) sum+=reference[n].residual[m];
        near(sum,0.,"conservation on shared mixed-element nodes");
    }
    aero_test::require(aes.use_coordinate_geometry(implicit),"mixed mesh must exercise coordinate AES");
    aes.finish_device_geometry();
    check_coordinate_pairs(aes,nodes,implicit && !renumbered);
    if (renumbered) return;
#ifdef AERO_WITH_KOKKOS_GPU
    int devices=0;
    const auto status=gpu::count(&devices);
    if (status!=gpu::Status::NoDevice && status!=gpu::Status::Unavailable) gpu::checked(status);
    for (int device=0;device<devices;++device) {
        gpu::DeviceScope scope(device);
        gpu::checked(scope.status);
        gpu::DeviceArrays arrays;
        arrays.device=device;
        const auto op=aes.upload(arrays,nodes);
        aero_test::require(op.tetra.cells && !op.tetra.all_hex,"mixed GPU geometry");
        if (tiled) {
            // Проверяем условия выбора компактного ядра независимо от значений результата.
            // При одном элементе на плитку второй варп группы дойдёт до обоих барьеров впустую.
            aero_test::require(op.rows && viscosity::CoordinatePairTile::capacity(op.row_count*sizeof(viscosity::Row))==1,
                               "one mixed AES cell fits in the reusable row buffer");
            aero_test::require(op.gradient_tile_count==op.cell_count && op.gradient_cache && op.cell_count==16,
                               "tiled mixed AES must build sixteen one-cell tiles");
        }
        auto* state=arrays.upload(field);
        const auto evaluate=[&](viscosity::Evaluation phase) {
            if (const auto* error=device_work::aero_device_viscosity(op,state,nullptr,phase))
                throw std::runtime_error(error);
        };
#ifdef AERO_HAS_TURBULENCE
        aero_test::require((op.tetra.cached_cells>0)==!implicit,"DDES tetrahedron volume cache");
        if (const auto* error=device_work::aero_device_prepare_tetra_volumes(op)) throw std::runtime_error(error);
        evaluate(viscosity::Evaluation::DdesGasOnly);
        evaluate(viscosity::Evaluation::FilterOnly);
        std::vector<turbulence::DdesPoint> model(nodes);
        gpu::checked(gpu::copy(model.data(),op.sa.ddes,nodes*sizeof(model.front()),gpu::Copy::ToHost));
        for (MeshIndex n=0;n<nodes;++n) {
            const auto& exact=aes.view().sa.ddes[n];
            aero_test::require(exact.grid_length>0. && exact.gradient_norm>0.,"nonzero mixed DDES reference");
            near(model[n].grid_length,exact.grid_length,"mixed DDES filter width");
            near(model[n].gradient_norm,exact.gradient_norm,"mixed DDES velocity gradient");
        }
#endif
        evaluate(viscosity::Evaluation::Full);
#ifdef AERO_HAS_TURBULENCE
        if (implicit) {
            aero_test::require(static_cast<const void*>(op.samples)==op.material,"DDES reuses cell material memory");
            // Пробуем другое поле. После расчёта его газовой невязки Ньютон
            // уточняет DDES с прежним полем SA. Фильтр может перезаписать
            // материал, но все пять компонент невязки и вязкий CFL уже сохранены.
            auto trial=field;
            for (MeshIndex i=0;i<nodes;++i) trial[5*i+1]*=2.;
            gpu::checked(gpu::copy(state,trial.data(),trial.size()*sizeof(double),gpu::Copy::ToDevice));
            evaluate(viscosity::Evaluation::DdesGasOnly);
            evaluate(viscosity::Evaluation::Full);
            std::vector<viscosity::Row> before(nodes),after(nodes);
            const auto bytes=nodes*sizeof(viscosity::Row);
            gpu::checked(gpu::copy(before.data(),op.rows,bytes,gpu::Copy::ToHost));
            evaluate(viscosity::Evaluation::FilterOnly);
            gpu::checked(gpu::copy(after.data(),op.rows,bytes,gpu::Copy::ToHost));
            aero_test::require(std::memcmp(before.data(),after.data(),bytes)==0,"DDES prediction preserves residual and CFL");
            // Отклоняем предиктор. По восстановленному полю повторяем фильтр и
            // сборку с нулём итераций SA. Ниже все строки сравниваются с CPU.
            gpu::checked(gpu::copy(state,field.data(),field.size()*sizeof(double),gpu::Copy::ToDevice));
            evaluate(viscosity::Evaluation::LesClosure);
            evaluate(viscosity::Evaluation::BeforeSaIterations);
            evaluate(viscosity::Evaluation::RowsOnly);
        }
#endif
        std::vector<viscosity::Row> rows(nodes);
        gpu::checked(gpu::copy(rows.data(),op.rows,rows.size()*sizeof(rows.front()),gpu::Copy::ToHost));
        double max_error=0.;
        for (MeshIndex n=0;n<nodes;++n) {
            near(rows[n].cfl_rate,reference[n].cfl_rate,"mixed AES viscous CFL");
            for (unsigned m=0;m<5;++m) {
                near(rows[n].residual[m],reference[n].residual[m],"mixed AES nodal residual");
                max_error=std::max(max_error,std::abs(rows[n].residual[m]-reference[n].residual[m]));
            }
        }
        std::cout << "Mixed AES GPU " << device << ", maximum CPU/GPU residual difference " << max_error << '\n';
        // Проверяем производную каждого столбца компактного Якобиана на той же сетке:
        // эталон содержит все пары полной невязки, включая диагонали призм и пирамид.
        if (implicit) {
            check_jacobian(aes,part.mesh,field,device,true,true);
            if (tiled) std::cout << "Tiled mixed AES GPU " << device << ": " << op.gradient_tile_count
                                 << " one-cell tiles, " << field.size() << " Jacobian columns passed.\n";
            // Центральные разности изменили рабочие массивы CPU. Восстановим эталон
            // перед сверкой со следующей картой, если в машине несколько устройств.
#ifdef AERO_HAS_TURBULENCE
            aes.evaluate(field.data(),nullptr,false,viscosity::Evaluation::DdesGasOnly);
#endif
            aes.evaluate(field.data());
        }
    }
#endif
}

// Проверяем каждый блок, в том числе смешанные производные напряжений. Вершины
// намеренно переставлены; опорный тетраэдр выбирается по координатам.
void check_affine_hex(double thickness) {
    using namespace viscosity;
    Element e{};
    e.count=8;
    const unsigned order[8]={5,3,6,0,1,7,2,4};
    for (unsigned i=0;i<8;++i) {
        const unsigned n=order[i];
        e.position[n]={{double(i&1)+.11*std::sin(i+1.),double((i>>1)&1)+.13*std::sin(3.*i),
                        thickness*(double(i>>2)+.08*std::cos(i))}};
        for (unsigned d=0;d<3;++d) e.neighbours[n]|=1u<<order[i^(1u<<d)];
    }
    const double gas_gamma=1.4,prandtl=.73;
    Cell cell{}; cell.count=8;
    MeanGradient mean{};
    std::vector<PairMetric> pairs;
    GradientBuilder builder;
    builder.prepare(e,3,gas_gamma,prandtl,cell,pairs,&mean);
    const auto reduced=make_affine_hex(e,cell,mean,pairs);
    for(unsigned i=0;i<8;++i) for(unsigned j=0;j<8;++j) if(i!=j) {
        const auto& full=pairs[cell.pair_index(i,j)];
        const auto compact=reduced.metric(i,j,full.heat);
        for(unsigned m=0;m<3;++m) for(unsigned d=0;d<3;++d)
            near(compact.stress(m, d),i<j ? full.stress(m, d) : full.stress(d, m),"affine AES pair");
    }
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
void check_jacobian(viscosity::Aes& aes, const Mesh& mesh, const std::vector<double>& field, int device,
                    bool shared_csgs = false, bool compact = false) {
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
                  offsets.data(),incidence.data(),{},{},options,{},false};
    implicit::Partition partition;
    partition.shared_csgs = shared_csgs;
    partition.compact_jacobian = compact;
    partition.host_aes = partition.active_aes = viscous;
    auto active = host;
#ifdef AERO_WITH_KOKKOS_GPU
    gpu::DeviceArrays arrays;
    if (device >= 0) {
        arrays.device = device;
        partition.active_aes = aes.upload(arrays,host.nodes);
        if (partition.active_aes.tetra.cached_cells)
            if (const auto* error=device_work::aero_device_prepare_tetra_volumes(partition.active_aes))
                throw std::runtime_error(error);
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
#ifdef AERO_HAS_TURBULENCE
        // При DDES декодирование газа вынесено в отдельную фазу до вязкой невязки.
        if (viscous.c_des>0.) aes.evaluate(u.data(),nullptr,false,viscosity::Evaluation::DdesGasOnly);
#endif
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
        // Из производной полной невязки получаем независимый эталон компактного
        // столбца. Сумму удалённых внедиагональных блоков переносим на диагональ.
        const auto owner = column/width;
        std::vector<bool> retained(host.nodes,!compact);
        retained[owner] = true;
        for (auto p=offsets[owner]; p<offsets[owner+1]; ++p) {
            const auto& edge = faces[incidence[p].face()];
            retained[edge.begin] = retained[edge.end] = true;
        }
        for (unsigned r = 0; r < width; ++r) {
            double sum = 0., norm = 0.;
            double omitted = 0.;
            for (size_t i=r; i<field.size(); i+=width)
                if (!retained[i/width]) omitted += (rp[i]-rm[i])/(2.*h);
            for (size_t i = r; i < field.size(); i += width) {
                const double expected = retained[i/width] ?
                    (rp[i]-rm[i])/(2.*h) + (i/width==owner ? omitted : 0.) : 0.;
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
    const FixedMatrix<double,3,3> gradient = {{.3,-.1,.4,.7,-.2,.5,-.6,.15,.11}};
    const double grad_p[] = {.13,-.09,.08};
    double mu = 0., mu_u[3]{}, divergence = 0.;
    for (unsigned k = 0; k < d; ++k) divergence += gradient(k, k);
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
            for (unsigned k = 0; k < d; ++k) u += gradient(m, k)*point[k];
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
            const double stress = gradient(m, a)+gradient(a, m)-(a==m ? 2.*divergence/3. : 0.);
            near(flux[m+1],scale*mu*stress,"momentum flux");
            energy += mu_u[m]*stress;
        }
        near(flux[d+1],scale*energy,"energy flux");
    }
    // Неоднородная плотность исключает случайное совпадение производных по U и по примитивным полям.
    for (MeshIndex i = 0; i < mesh.node_count; ++i) field[i*width] += .017*static_cast<double>(i);
    const auto check_device = [&](int device) {
        check_jacobian(aes,part.mesh,field,device);
        // Те же разностные производные проверяют адреса общей матрицы CSGS на GPU.
        if (device >= 0) check_jacobian(aes,part.mesh,field,device,true);
        check_jacobian(aes,part.mesh,field,device,device >= 0,true);
        if (d == 2 && mesh.node_count == 4) {
            aes.wall.front() = 1;
            check_jacobian(aes,part.mesh,field,device);
            aes.wall.front() = 0;
        }
    };
    check_device(-1);
#ifdef AERO_HAS_TURBULENCE
    // Производная турбулентной вязкости входит в газовый блок Якобиана при фиксированном
    // поле SA. Сверяем тот же набор столбцов с конечной разностью при Pr_t, отличном от Pr.
    aes.turbulence_field.distance.assign(mesh.node_count,0.);
    turbulence::Settings sa;
    sa.nut_nu_amb=.8;
    sa.prandtl=1.2;
    aes.prepare_turbulence(sa,true);
    check_device(-1);
#endif
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
        check_affine_hex(1.);
        check_affine_hex(.001);
        check_mixed_coordinates(world,false);
        check_mixed_coordinates(world,true);
        check_mixed_coordinates(world,true,true);
        check_mixed_coordinates(world,true,false,true);
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
