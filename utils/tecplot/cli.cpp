#include "tecplot.h"
#include <limits>
#include <ostream>

namespace aero { namespace tecplot {
namespace {
void usage(std::ostream& out) {
    out<<"Usage: fields_to_tecplot MESH [FIELDS] OUTPUT.plt [--threads N]\n\n"
          "  MESH        Native AEROMESH v2: text, binary or LZMA.\n"
          "  FIELDS      Optional AEROFIELDS v1: text, binary or LZMA.\n"
          "  OUTPUT.plt  Binary Tecplot file with geometry and any supplied fields.\n"
          "  --threads N Positive thread count (default: OpenMP runtime settings).\n"
          "  -h, --help  Show this help.\n\n"
          "Mesh SHA-256 must match. Coordinates are exported in meters.\n"
          "Existing output files are never overwritten.\n"
          "OpenMP: "<<(openmp_available()?"enabled":"disabled (one thread)")<<"\n";
}
int positive(const std::string& value) {
    if(value.empty())throw ExportError("Missing thread count");
    int result=0;
    for(char c:value) {
        if(c<'0'||c>'9'||result>(std::numeric_limits<int>::max()-(c-'0'))/10)
            throw ExportError("Thread count must be a positive integer");
        result=result*10+c-'0';
    }
    if(!result)throw ExportError("Thread count must be positive");
    return result;
}
}
int run_cli(int argc,const char* const* argv,std::ostream& out,std::ostream& err) {
    try {
        if(argc==2&&(std::string(argv[1])=="--help"||std::string(argv[1])=="-h")) {usage(out);return 0;}
        int end_paths=1;
        while(end_paths<argc && std::string(argv[end_paths]).compare(0,2,"--")!=0) {
            if(!argv[end_paths][0]) {usage(err);return 1;}
            ++end_paths;
        }
        if(end_paths!=3 && end_paths!=4) {usage(err);return 1;}
        Options options;
        if(end_paths<argc) {
            if(std::string(argv[end_paths])!="--threads" || argc-end_paths!=2) {usage(err);return 1;}
            options.threads=positive(argv[end_paths+1]);
        }
        if(!openmp_available()&&options.threads>1)throw ExportError("This build has no OpenMP; use --threads 1");
        const Mesh mesh=Mesh::read(argv[1]);
        const char* path=argv[end_paths-1];
        if(end_paths==4) {
            const FieldSnapshot fields=FieldSnapshot::read(argv[2]);
            write(path,mesh,fields,options);
            out<<"Written: "<<path<<"; dimension="<<fields.dimension<<", cells="<<fields.counts[3]
               <<", fields="<<fields.fields.size()<<", step="<<fields.step<<"\n";
        } else {
            write(path,mesh,options);
            out<<"Written: "<<path<<"; dimension="<<mesh.view().dimension<<", cells="<<mesh.view().cell_count<<"\n";
        }
        return 0;
    } catch(const std::exception& error) {err<<"Tecplot export failed: "<<error.what()<<"\n";return 1;}
}
} }
