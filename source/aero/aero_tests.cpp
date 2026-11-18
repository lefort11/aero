#include "../mesh/mesh.h"
#include "../io/fields.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
void require(bool ok,const std::string& msg) { if (!ok) throw std::runtime_error(msg); }
std::string read(const std::string& path) {
    std::ifstream in(path.c_str());
    std::ostringstream text;
    text<<in.rdbuf();
    return text.str();
}
void write(const std::string& path,const std::string& text) {
    std::ofstream out(path.c_str()); out<<text; out.close();
    require(bool(out),"Cannot write "+path);
}
void remove_dir(const std::string& path) {
#ifdef _WIN32
    _rmdir(path.c_str());
#else
    rmdir(path.c_str());
#endif
}
std::string run_case(const std::string& program,const std::string& device,bool interactive,bool expect_fallback) {
    const auto stamp=std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::string dir="build/aero-test-"+std::to_string(stamp),out=dir+"/out",log=dir+"/run.log";
#ifdef _WIN32
    require(_mkdir(dir.c_str())==0,"Cannot create "+dir);
    const std::string command="call \""+program+"\"";
#else
    require(mkdir(dir.c_str(),0700)==0,"Cannot create "+dir);
    const std::string command="\""+program+"\"";
#endif
    const std::string backend=device.empty()?"cpu":"cuda";
    write(dir+"/main.txt","mesh=../../examples/tecplot/demo_1d.mesh\nreconstruction=first_order\n"
          "cfl=.35\nomp_threads=2\nflowreport_every=1\nfinal_time=.008\nbackend="+backend+
          (interactive?"\ntime_scheme=ssprk3\neuler_solver=roe\nreconstruction_variables=characteristic\n":
                       "\ntime_scheme=euler\neuler_solver=lf\nreconstruction_variables=flux\n"));
    write(dir+"/gas.txt","gamma=1.4\n");
    write(dir+"/init.txt","initial_type formula\nrho=1+0.1*x\nux=0\np=1+0.1*x\n");
    if (!device.empty()) write(dir+"/cuda.txt","devices="+device+"\n");
    write(dir+"/output.txt","[latest]\nfile=state.fields\nevery_steps=1\nvariables=rho,ux,p\n"
          "[clock]\nfile=history/p.fields\nevery_time=.004\nmulti=1\nvariables=p\nformat=lzma\n");
    const std::string launch=command+" --input="+dir+" --output="+out;
    const auto mesh=aero::Mesh::read("examples/tecplot/demo_1d.mesh");
    std::string rules;
    for (const auto& name:mesh.data().patch_names) rules+="["+name+"]\nbc_type wall\n";
    const std::string redirect=" > "+log+" 2>&1";
    if (interactive) {
#ifdef _WIN32
        std::FILE* pipe=_popen((launch+" --interactive"+redirect).c_str(),"w");
#else
        std::FILE* pipe=popen((launch+" --interactive"+redirect).c_str(),"w");
#endif
        require(pipe!=nullptr,"Cannot launch interactive aero");
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(20);
        while (read(log).find("press Enter")==std::string::npos && std::chrono::steady_clock::now()<deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const auto generated=read(dir+"/bc.txt");
        write(dir+"/bc.txt",rules);
        std::fputs("\n",pipe);
#ifdef _WIN32
        const int rc=_pclose(pipe);
#else
        const int rc=pclose(pipe);
#endif
        require(rc==0,read(log));
        for (const auto& name:mesh.data().patch_names)
            require(generated.find("["+name+"]")!=std::string::npos,"missing generated section");
    } else {
        require(std::system((launch+redirect).c_str())!=0,"missing bc must pause the batch calculation");
        require(read(log).find("Created ")!=std::string::npos && read(log).find("Starting time")==std::string::npos,
                "missing BC workflow");
        write(dir+"/bc.txt",rules);
        require(std::system((launch+redirect).c_str())==0,"aero failed; see "+log);
    }
    const auto trace=read(log);
    if (expect_fallback) require(trace.find("CUDA unavailable; running on CPU.")!=std::string::npos,"missing CPU fallback");
    std::string expected=device.empty() || expect_fallback ? "CPU" : "CUDA device "+device;
    for (const auto& key:std::vector<std::string>{"Active backend = "+expected,"Boundary label",
            "Building reconstruction stencils... done","Flow report    Step = 1","Final time = 0.008"})
        require(trace.find(key)!=std::string::npos,"missing "+key+" in "+log);
    require(read(dir+"/bc.txt")==rules,"existing boundary settings must be preserved");
    for (const auto& name:mesh.data().patch_names)
        require(trace.find(name+"\t1\n")!=std::string::npos,"boundary face count for "+name);
    const auto final=aero::FieldSnapshot::read(out+"/state.fields");
    final.check_mesh(mesh);
    require(final.time==0.008 && final.step>=2 && final.fields.size()==3,"final step and fields");
    bool evolved=false;
    for (double u:final.fields[1].values) if (std::abs(u)>1e-8) evolved=true;
    require(evolved,"the computed velocity must evolve");
    unsigned records=0;
    for (std::uint64_t step=1;step<=final.step;++step) {
        const auto digits=std::to_string(step);
        const auto path=out+"/history/p_"+std::string(6-digits.size(),'0')+digits+".fields";
        if (!std::ifstream(path.c_str())) continue;
        const auto snap=aero::FieldSnapshot::read(path);
        snap.check_mesh(mesh);
        ++records;
        require(snap.step==step && snap.time==records*0.004 && snap.fields.size()==1 &&
                snap.fields[0].name=="p","clock history");
        std::remove(path.c_str());
    }
    require(records==2,"two scheduled times");
    std::remove((out+"/state.fields").c_str());
    remove_dir(out+"/history"); remove_dir(out);
    for (const char* name:{"main.txt","gas.txt","bc.txt","init.txt","cuda.txt","output.txt","run.log"})
        std::remove((dir+"/"+name).c_str());
    remove_dir(dir);
    return expected;
}
}
int main(int argc,char** argv) {
    try {
        require(argc==2 || argc==3,"usage: aero_tests <aero executable> [CUDA ordinal | --expect-cpu-fallback]");
        const bool fallback=argc==3 && std::string(argv[2])=="--expect-cpu-fallback";
        const std::string device=fallback?"0":(argc==3?argv[2]:"");
        const auto backend=run_case(argv[1],device,false,fallback);
        run_case(argv[1],device,true,fallback);
        std::cout<<"Aero end-to-end tests passed ("<<backend<<", Euler, SSPRK3, interactive BC, reports and fields).\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
