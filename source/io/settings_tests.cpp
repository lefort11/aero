#include "settings.h"
#include "settings_boundary.h"
#include "fields.h"
#include "scheduled_output.h"
#ifdef AERO_TEST_OUTPUT_DEVICE
#include "../explicit/explicit.h"
#endif
#include "../initial/initial.h"
#include "../bc/periodic.h"
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
using namespace aero;
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F call, const std::string& fragment) {
    try { call(); }
    catch (const std::exception& e) {
        require(std::string(e.what()).find(fragment) != std::string::npos, e.what());
        return;
    }
    throw std::runtime_error("expected error containing " + fragment);
}

struct CaseFiles {
    std::string dir;
    CaseFiles() {
        const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        for (unsigned n = 0; ; ++n) {
            dir = "build/settings-test-" + std::to_string(stamp) + "-" + std::to_string(n);
#ifdef _WIN32
            if (_mkdir(dir.c_str()) == 0) break;
#else
            if (mkdir(dir.c_str(), 0700) == 0) break;
#endif
            require(errno == EEXIST, "cannot create test directory");
        }
    }
    ~CaseFiles() {
        for (const char* name : {"main.txt","bc.txt","init.txt","cuda.txt","gas.txt","test.fields",
                                "output.txt","latest.fields","p_000001.fields","p_000002.fields","p_000003.fields"})
            std::remove((dir + "/" + name).c_str());
#ifdef _WIN32
        _rmdir(dir.c_str());
#else
        rmdir(dir.c_str());
#endif
    }
    void write(const std::string& name, const std::string& content) const {
        std::ofstream out((dir + "/" + name).c_str(), std::ios::binary);
        out << content;
        out.close();
        require(bool(out), "cannot write " + name);
    }
    RunSettings load() const { return read_run_settings({dir,"./results"}); }
};

void arguments() {
    const char* args[] = {"aero","--output=out dir","--input=./params"};
    const auto dirs = parse_run_arguments(3, args);
    require(dirs.input == "./params" && dirs.output == "out dir", "CLI directories");
    rejects([&]{parse_run_arguments(2,args);}, "--input");
    args[2] = "--output=x";
    rejects([&]{parse_run_arguments(3,args);}, "duplicate");
    args[2] = "--input=";
    rejects([&]{parse_run_arguments(3,args);}, "directory");
    args[2] = "--wrong=1";
    rejects([&]{parse_run_arguments(3,args);}, "unknown argument");
    const char* edit[]={"aero","--interactive","--input=in","--output=out","--interactive"};
    require(parse_run_arguments(4,edit).interactive,"interactive flag");
    rejects([&]{parse_run_arguments(5,edit);},"duplicate --interactive");
    const char* automatic[]={"aero","--recover","--input=in","--output=out"};
    const auto selected=parse_run_arguments(4,automatic);
    require(selected.recover && selected.recovery_slot==-1,"automatic recovery");
    const char* restart[]={"aero","--input=in","--output=out","--recover","0"};
    require(parse_run_arguments(5,restart).recovery_slot==0,"recovery slot 0");
    restart[3]="--recover=1";
    require(parse_run_arguments(4,restart).recovery_slot==1,"recovery slot 1");
    restart[4]="--recover";
    rejects([&]{parse_run_arguments(5,restart);},"duplicate --recover");
    for (const char* bad:{"--recover=","--recover=2","--recover=-1"}) {
        restart[3]=bad;
        rejects([&]{parse_run_arguments(4,restart);},"expects 0 or 1");
    }
    restart[3]="--recover"; restart[4]="2";
    rejects([&]{parse_run_arguments(5,restart);},"expects 0 or 1");
}

void templates_case() {
    const char* args[] = {"aero","--input=templates","--output=./results"};
    const auto run = read_run_settings(parse_run_arguments(3,args));
    require(run.output_zones.size()==2 && run.output_zones[0].every_steps==100 &&
            run.output_zones[1].every_time==0.05 && run.output_zones[1].multi,"template output zones");
    require(!run.use_cuda && run.devices.empty() && run.reconstruction == "ebr3" &&
            run.euler_solver == "roe" && run.time_scheme == "ssprk3" &&
            run.cfl == 0.5 && run.final_time == 2 && run.omp_threads == 8 &&
            run.flowreport_every == 100, "template settings");
    const Mesh mesh = Mesh::read(run.mesh_path);
    const auto boundary = read_case_boundaries(run.boundary_path, mesh.data());
    require(boundary.periodic_pairs.size() == 1, "template seam count");
    for (std::size_t i = 0; i < mesh.data().patch_names.size(); ++i) {
        const std::string& name = mesh.data().patch_names[i];
        const auto& rule = boundary.by_patch.at(i);
        if (name == "ymin") require(rule.type == BoundaryType::Wall, "template wall");
        else if (name == "ymax") require(rule.type == BoundaryType::Free &&
                rule.inflow.source == BoundarySource::Current && rule.inflow.mode == BoundaryMode::Pressure &&
                rule.outflow.source == BoundarySource::Current && rule.outflow.mode == BoundaryMode::NormalVelocity,
                "template free");
        else require(rule.type == BoundaryType::Periodic, "template periodic");
    }
    PeriodicDomain domain(mesh,boundary.periodic_pairs);
    const auto grid = domain.view();
    require(run.gamma == 1.4 && run.reconstruction_variables == "conservative","gas and reconstruction variables");
    const auto initial = read_initial_file(mesh,run.gamma,run.initial_path,&grid);
    require(grid.node_count < mesh.view().node_count && initial.size() == grid.node_count*4 &&
            initial[0] == 1 && std::abs(initial[1]-0.3) < 1e-12, "template initial state");
}

void output_zones(const CaseFiles& files) {
    const std::string latest="[latest]\nfile=latest.fields\nevery_steps=2\nvariables=E, ux, rho, uy\nformat=text\n";
    const std::string series="[history]\nfile=p.fields\nevery_time=0.1\nmulti=1\nvariables=p\nformat=lzma\n";
    const auto path=files.dir+"/output.txt";
    require(read_output_zones(path).empty(),"absent output.txt");
    for (const auto& invalid:std::vector<std::string>{latest+"every_time=.1",latest+"every_steps=3",latest+"multi=2",
            latest+latest,latest+"oops=0","file=f","[s]\nvariables=p\nevery_steps=1",
            "[s]\nfile=x\nevery_steps=0\nvariables=p","[s]\nfile=x\nevery_time=nan\nvariables=p",
            "[s]\nfile=x\nevery_steps=-1\nvariables=p","[s]\nfile=x\nevery_steps=2\nvariables=p,p",
            "[s]\nfile=x\nevery_steps=2\nvariables=q","[s]\nfile=../x\nevery_steps=2\nvariables=p",
            "[s]\nfile=x\nevery_steps=1\nvariables=p\nlayout=other",
            "[s]\nfile=x\nevery_steps=1\nvariables=p\nlayout=global\nformat=lzma",
            "[s]\nfile=recovery-1.bin\nevery_steps=1\nvariables=p\nlayout=global",
            series+"[alias]\nfile=p_000003.fields\nevery_steps=2\nvariables=E"}) {
        files.write("output.txt",invalid);
        rejects([&]{read_output_zones(path);},"output.txt");
    }
    files.write("output.txt",latest+series);
    const auto zones=files.load().output_zones;
    require(output_filename(zones[1],1000000)=="p_1000000.fields","wide step number");
    const auto mesh=Mesh::read("examples/tecplot/demo_2d.mesh");
    const auto bounds=read_case_boundaries("templates/bc.txt",mesh.data());
    PeriodicDomain domain(mesh,bounds.periodic_pairs);
    const auto grid=domain.view();
    std::vector<MeshIndex> map(grid.node_dof,grid.node_dof+grid.original_node_count);
    ScheduledOutput writer(zones,files.dir,mesh,1.4,map);
    std::vector<double> state;
    for (MeshIndex i=0;i<grid.node_count;++i) {
        const double rho=1+static_cast<double>(i);
        state.insert(state.end(),{rho,2*rho,-rho,2.5*rho+5});
    }
    require(!writer.due(0,0) && !writer.due(1,0.09) && writer.limit_dt(0,1)==0.1,"initial deadline");
    rejects([&]{writer.due(1,0.11);},"missed");
    writer.write(1,0.1,state);
    const auto first=FieldSnapshot::read(files.dir+"/p_000001.fields");
    first.check_mesh(mesh);
    require(first.step==1 && first.time==0.1 && first.fields.size()==1 && first.fields[0].name=="p" &&
            std::abs(first.fields[0].values[0]-2)<1e-12,"first pressure snapshot");
    // A pre-existing history file must survive; the first zone has already committed.
    files.write("p_000002.fields","existing");
    rejects([&]{writer.write(2,0.2,state);},"publish");
    require(std::remove((files.dir+"/p_000002.fields").c_str())==0,"release conflicting file");
    auto changed=state;
    changed[3]+=1;
    writer.write(2,0.2,changed);
    const auto current=FieldSnapshot::read(files.dir+"/latest.fields");
    current.check_mesh(mesh);
    require(current.step==2 && current.fields.size()==4 && current.fields[0].name=="E" &&
            current.fields[1].name=="ux" && current.fields[2].name=="rho" &&
            current.fields[3].name=="uy","simultaneous zones and variable order");
    for (std::size_t i=0;i<map.size();++i)
        require(current.fields[2].values[i]==state[4*map[i]] && current.fields[0].values[i]==state[4*map[i]+3] &&
                current.fields[1].values[i]==2 && current.fields[3].values[i]==-1,"periodic node field");
    writer.write(3,0.3,changed);
    require(!writer.due(3,0.3) && writer.due(4,0.31),"accepted step schedule");
    writer.write(4,0.31,changed);
    require(FieldSnapshot::read(files.dir+"/latest.fields").step==4 &&
            FieldSnapshot::read(files.dir+"/p_000001.fields").step==1,"overwrite and history");
    require(FieldSnapshot::read(files.dir+"/p_000003.fields").time==0.3,"rounded deadline");
    auto clock=zones.front();
    clock.every_steps=0;
    clock.every_time=0.3;
    auto fast=zones.back();
    fast.multi=false;
    fast.file="p_000001.fields";
    ScheduledOutput clocks({clock,fast},files.dir,mesh,1.4,map);
    require(clocks.limit_dt(0,0.7)==0.1,"nearest of two time clocks");
    clocks.write(1,0.1,state);
    clocks.write(2,0.2,state);
    const double accepted=0.2+clocks.limit_dt(0.2,0.7);
    clocks.write(3,accepted,state);
    require(FieldSnapshot::read(files.dir+"/latest.fields").time==accepted &&
            FieldSnapshot::read(files.dir+"/p_000001.fields").time==accepted &&
            clocks.limit_dt(accepted,0.01)==0.01,"coincident physical deadlines");
    auto invalid_zone=zones[1];
    invalid_zone.variables={"uz"};
    rejects([&]{ScheduledOutput bad({invalid_zone},files.dir,mesh,1.4);},"dimension");
    for (unsigned d:{1u,3u}) {
        const auto other=Mesh::read("examples/tecplot/demo_"+std::to_string(d)+"d.mesh");
        auto zone=zones.front();
        zone.variables={d==1 ? "ux" : "uz","p"};
        zone.format=FieldFormat::Binary;
        ScheduledOutput plain({zone},files.dir,other,1.4);
        std::vector<double> q;
        for (MeshIndex i=0;i<other.view().node_count;++i) {
            q.push_back(2);
            for (unsigned j=1;j<=d;++j) q.push_back(2.0*j);
            q.push_back(d==1 ? 6 : 19);
        }
        plain.write(2,0.09,q);
        const auto saved=FieldSnapshot::read(files.dir+"/latest.fields");
        saved.check_mesh(other);
        require(saved.fields[0].values.back()==d && std::abs(saved.fields[1].values.back()-2)<1e-12,
                "plain mesh output");
#ifdef AERO_TEST_OUTPUT_DEVICE
        const int devices=ExplicitIntegrator::cuda_device_count();
        require(devices>0,"CUDA output test requires a device");
        std::vector<double> volumes(q.size()/(d+2),1.0);
        for (int device=0;device<devices;++device) {
            ExplicitIntegrator solver(volumes.size(),d+2,volumes.data(),q.data(),ExplicitBackend::Cuda,device);
            ScheduledOutput gpu_writer({zone},files.dir,other,1.4);
            if (gpu_writer.due(2,0.09)) gpu_writer.write(2,0.09,solver.state());
            const auto gpu=FieldSnapshot::read(files.dir+"/latest.fields");
            gpu.check_mesh(other);
            require(gpu.fields[0].values==saved.fields[0].values && gpu.fields[1].values==saved.fields[1].values,
                    "output using the CUDA integrator snapshot");
        }
        std::cout<<"Verified "<<d<<"D disk output on "<<devices<<" CUDA devices.\n";
#endif
    }
}

void main_and_cuda(const CaseFiles& files) {
    const std::string head = "\xef\xbb\xbf# test\r\nmesh = \"path #1/grid.mesh\" # relative\n";
    files.write("gas.txt","gamma=1.67");
    const std::string common = "time_scheme=euler\ncfl=2e-1\neuler_solver=lf\nfinal_time=0\nreconstruction_variables=flux\n";
    const std::string base = head + common + "reconstruction=weno5\n";
    for (const char* scheme : {"first_order","ebr3","ebr5","weno3","weno5"}) {
        files.write("main.txt",head + common + "backend=cpu\nreconstruction=" + scheme);
        const auto run = files.load();
        require(run.reconstruction == scheme && run.euler_solver == "lf" && run.time_scheme == "euler" &&
                run.cfl == 0.2 && run.final_time == 0 && run.devices.empty() &&
                run.mesh_path == files.dir + "/path #1/grid.mesh", "CPU main.txt");
    }
    files.write("main.txt",base + "backend=cuda");
    rejects([&]{files.load();},"cuda.txt");
    files.write("cuda.txt","devices = 2, 0, 1");
    require(files.load().devices == std::vector<int>({2,0,1}), "device order");
    for (const char* list : {"1,1","-1","1,","0,,2","1.0","99999999999999","0 1"}) {
        files.write("cuda.txt",std::string("devices = ") + list);
        rejects([&]{files.load();},"cuda.txt:1");
    }
    files.write("main.txt",base + "backend=cpu");
    require(files.load().devices.empty(), "CPU ignores invalid cuda.txt");
    for (const char* extra : {"backend=cpu\nother=x","backend=cpu\ncfl=0.5","backend=gpu"}) {
        files.write("main.txt",base + extra);
        rejects([&]{files.load();},"main.txt");
    }
    files.write("main.txt",base);
    rejects([&]{files.load();},"missing backend");
    const std::string valid = base + "backend=cpu";
    files.write("main.txt",valid + "\nomp_threads=3\nflowreport_every=17\ntrace_ratio_min=.5\ntrace_ratio_max=1.5");
    const auto configured = files.load();
    require(configured.omp_threads == 3 && configured.flowreport_every == 17, "explicit report settings");
    require(configured.trace_ratio_min==.5 && configured.trace_ratio_max==1.5,"trace ratio bounds");
#ifdef _OPENMP
    require(omp_get_max_threads()==3, "configured OpenMP runtime");
#endif
    for (const char* entry : {"omp_threads=-1","omp_threads=1.5","omp_threads=2147483648",
                              "flowreport_every=2x","flowreport_every=-1","flowreport_every=18446744073709551616",
                              "trace_ratio_min=-.1","trace_ratio_min=1","trace_ratio_max=1","trace_ratio_max=nan"}) {
        files.write("main.txt",valid + "\n" + entry);
        rejects([&]{files.load();},std::string(entry).substr(0,std::string(entry).find('=')));
    }
    files.write("main.txt",valid);
    const auto defaults=files.load();
    require(defaults.trace_ratio_min==0 && defaults.trace_ratio_max==1e16,"default trace ratio bounds");
    require(defaults.gamma == 1.67 && !defaults.omp_threads && !defaults.flowreport_every,"case defaults");
    for (const auto& change : std::vector<std::pair<std::string,std::string> >{
             {"2e-1","0"},{"2e-1","nan"},{"final_time=0","final_time=-1"},
             {"weno5","eno5"},{"euler\n","rk4\n"},{"=lf","=hll"},{"=flux","=other"}}) {
        std::string invalid = valid;
        invalid.replace(invalid.find(change.first),change.first.size(),change.second);
        files.write("main.txt",invalid);
        rejects([&]{files.load();},"main.txt");
    }
    files.write("main.txt",valid);
    for (const char* value : {"1","0.9","nan","1e500","2x"}) {
        files.write("gas.txt",std::string("gamma=") + value);
        rejects([&]{files.load();},"gas.txt");
    }
    files.write("gas.txt","gamma=1.4");
}

void boundaries(const CaseFiles& files) {
    MeshData data;
    data.dimension = 2;
    data.patch_names = {"top #2","left","right","bottom #1"};
    const std::string physical = "[right]\nbc_type superout\n[left]\nbc_type superin initial p=2\n";
    const std::string seam = "[\"bottom #1\"]\nbc_type periodic\npair \"top #2\"\ntranslation 0 1 0\n";
    const std::string path = files.dir + "/bc.txt";
    files.write("bc.txt", physical + seam);
    const auto bc = read_case_boundaries(path,data);
    require(bc.by_patch[1].type == BoundaryType::SuperIn && bc.by_patch[2].type == BoundaryType::SuperOut &&
            bc.periodic_pairs[0].patch == "bottom #1" && bc.periodic_pairs[0].pair == "top #2" &&
            bc.periodic_pairs[0].translation[1] == 1, "patch order and labels");
    for (const std::string& invalid : std::vector<std::string>{
        physical, physical + seam + "[\"top #2\"]\nbc_type wall",
        physical + seam + "[wing]\nbc_type wall", physical + physical,
        "bc_type wall", "[left]\nbc_type free\ninflow base_p",
        "[right]\nbc_type periodic\npair right\ntranslation 0 1 0",
        "[right]\nbc_type periodic\npair nowhere\ntranslation 0 1 0"}) {
        files.write("bc.txt",invalid);
        rejects([&]{read_case_boundaries(path,data);},"bc.txt");
    }
}

void initial_snapshot(const CaseFiles& files) {
    files.write("main.txt","mesh=../../examples/tecplot/demo_1d.mesh\nreconstruction=first_order\n"
                "time_scheme=euler\ncfl=1\nbackend=cpu\neuler_solver=lf\nfinal_time=1\nreconstruction_variables=characteristic");
    files.write("init.txt","initial_type file\npath test.fields\nfield U");
    const auto run = files.load();
    const Mesh mesh = Mesh::read(run.mesh_path);
    FieldSnapshot snap(mesh);
    Field u;
    u.name = "U"; u.location = FieldLocation::Node; u.components = {"rho","rho_ux","E"};
    const auto n = static_cast<std::size_t>(mesh.view().node_count);
    u.values.assign(n,1);
    u.values.insert(u.values.end(),n,0.5);
    u.values.insert(u.values.end(),n,3);
    snap.fields.push_back(u);
    snap.write(files.dir + "/test.fields",FieldFormat::Lzma);
    const auto state = read_initial_file(mesh,run.gamma,run.initial_path);
    require(state.size() == 3*n && state[0] == 1 && state[1] == 0.5 && state[2] == 3,
            "snapshot relative to init.txt");
}
}

int main() {
    try {
#ifdef _OPENMP
        const int original_threads=omp_get_max_threads();
#endif
        arguments();
        templates_case();
        CaseFiles files;
        main_and_cuda(files);
#ifdef _OPENMP
        require(omp_get_max_threads()==original_threads, "restore OpenMP runtime default");
#endif
        boundaries(files);
        initial_snapshot(files);
        output_zones(files);
        std::cout << "Settings tests passed (CLI, templates, boundaries, initial snapshot and scheduled output).\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
