// Build-time tool. Only the generated GPU image is embedded in the solver.
#include <nvrtc.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(nvrtcResult result) {
    if (result!=NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(result));
}
struct Program {
    nvrtcProgram handle=nullptr;
    ~Program() {if (handle) nvrtcDestroyProgram(&handle);}
};
}

int main(int argc,char** argv) {
    if (argc!=4) {
        std::cerr<<"Usage: compile_ptx INPUT SM_NUMBER OUTPUT.ptx\n";
        return 1;
    }
    try {
        const std::string path=argv[1];
        std::ifstream in(path.c_str(),std::ios::binary);
        if (!in) throw std::runtime_error("Cannot read "+path);
        const std::string source((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
        const auto slash=path.find_last_of("/\\");
        const std::string include="-I"+(slash==std::string::npos?".":path.substr(0,slash));
        const std::string arch=std::string("--gpu-architecture=compute_")+argv[2];
        const char* options[]={"--std=c++11","--device-as-default-execution-space",include.c_str(),arch.c_str()};
        Program program;
        check(nvrtcCreateProgram(&program.handle,source.c_str(),path.c_str(),0,nullptr,nullptr));
        const auto result=nvrtcCompileProgram(program.handle,sizeof(options)/sizeof(options[0]),options);
        std::size_t size=0;
        check(nvrtcGetProgramLogSize(program.handle,&size));
        if (size>1) {
            std::vector<char> log(size);
            check(nvrtcGetProgramLog(program.handle,log.data()));
            std::cerr<<log.data();
        }
        check(result);
        check(nvrtcGetPTXSize(program.handle,&size));
        std::vector<char> ptx(size);
        check(nvrtcGetPTX(program.handle,ptx.data()));
        std::ofstream out(argv[3],std::ios::binary);
        out.write(ptx.data(),static_cast<std::streamsize>(size-1));
        out.close();
        if (!out) throw std::runtime_error(std::string("Cannot write ")+argv[3]);
    } catch (const std::exception& e) {
        std::cerr<<argv[1]<<": "<<e.what()<<'\n';
        return 1;
    }
    return 0;
}
