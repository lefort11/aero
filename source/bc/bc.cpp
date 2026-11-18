#include "bc.h"

#include <sstream>
#include <locale>
#include <string>

namespace aero {
namespace {
typedef std::vector<std::string> Tokens;

double number(const std::string& token) {
    std::istringstream stream(token);
    stream.imbue(std::locale::classic());
    double x;
    if (!(stream>>x) || !stream.eof() || !euler_detail::finite(x))
        throw std::invalid_argument("invalid finite number: "+token);
    return x;
}

BoundaryRule read_rule(const Tokens& t,bool inflow,unsigned d) {
    BoundaryRule r={};
    const std::string& name=t.at(1);
    if (name=="superout") {
        if (inflow || t.size()!=2) throw std::invalid_argument("superout requires outflow and no arguments");
        r.mode=BoundaryMode::SuperOut;
        return r;
    }
    if (name=="superin") {
        if (!inflow) throw std::invalid_argument("superin requires inflow");
        r.mode=BoundaryMode::SuperIn;
        const char* fields[]={"rho","p","ux","uy","uz"};
        for (size_t i=2;i<t.size();++i) {
            if (i==2 && t[i]=="initial") {
                r.source=BoundarySource::Initial;
                continue;
            }
            const size_t eq=t[i].find('=');
            const std::string key=t[i].substr(0,eq);
            unsigned k=0;
            while (k<2+d && key!=fields[k]) ++k;
            if (eq==std::string::npos || k==2+d || (r.mask&(1u<<k)))
                throw std::invalid_argument("invalid or repeated superin field: "+t[i]);
            r.primitive[k]=number(t[i].substr(eq+1));
            if (k<2 && r.primitive[k]<=0) throw std::invalid_argument("expected positive "+key);
            r.mask|=1u<<k;
        }
        return r;
    }
    if (name=="base_p") r.mode=BoundaryMode::Pressure;
    else if (name=="base_u") r.mode=BoundaryMode::NormalVelocity;
    else if (name=="base_rho") r.mode=BoundaryMode::Density;
    else throw std::invalid_argument("unknown boundary rule: "+name);
    if (t.size()>3) throw std::invalid_argument("too many arguments for "+name);
    if (t.size()==3) {
        if (t[2]=="initial") r.source=BoundarySource::Initial;
        else {
            r.source=BoundarySource::Constant;
            r.value=number(t[2]);
            if (r.mode!=BoundaryMode::NormalVelocity && r.value<=0)
                throw std::invalid_argument("expected positive value for "+name);
        }
    }
    return r;
}
}

BoundaryConfig read_boundary(std::istream& input,unsigned dimension,PeriodicPair& periodic) {
    if (dimension<1 || dimension>3) throw std::invalid_argument("invalid boundary dimension");
    BoundaryConfig config={};
    PeriodicPair spec={};
    bool translation=false;
    bool type=false, incoming=false, outgoing=false;
    std::string line;
    size_t row=0;
    while (std::getline(input,line)) {
        ++row;
        std::istringstream words(line.substr(0,line.find('#')));
        Tokens t;
        for (std::string word; words>>word;) t.push_back(word);
        if (t.empty()) continue;
        try {
            if (t.size()<2) throw std::invalid_argument("missing boundary setting");
            if (t[0]=="bc_type" && !type) {
                type=true;
                if (t[1]=="superin" || t[1]=="superout") {
                    const bool in=t[1]=="superin";
                    config.type=in ? BoundaryType::SuperIn : BoundaryType::SuperOut;
                    (in ? config.inflow : config.outflow)=read_rule(t,in,dimension);
                } else if (t.size()==2 && t[1]=="periodic") config.type=BoundaryType::Periodic;
                else if (t.size()==2 && t[1]=="wall") config.type=BoundaryType::Wall;
                else if (t.size()==2 && t[1]=="free") config.type=BoundaryType::Free;
                else throw std::invalid_argument("invalid bc_type");
            } else if (t[0]=="patch" && t.size()==2 && spec.patch.empty()) spec.patch=t[1];
            else if (t[0]=="pair" && t.size()==2 && spec.pair.empty()) spec.pair=t[1];
            else if (t[0]=="translation" && t.size()==4 && !translation) {
                for (unsigned k=0;k<3;++k) spec.translation[k]=number(t[k+1]);
                translation=true;
            } else if (t[0]=="inflow" && !incoming) {
                incoming=true;
                config.inflow=read_rule(t,true,dimension);
            } else if (t[0]=="outflow" && !outgoing) {
                outgoing=true;
                config.outflow=read_rule(t,false,dimension);
            } else throw std::invalid_argument("unknown or repeated keyword: "+t[0]);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument("boundary line "+std::to_string(row)+": "+e.what());
        }
    }
    if (input.bad() || !input.eof()) throw std::runtime_error("boundary settings read failed");
    if (!type || (config.type==BoundaryType::Free ? !(incoming && outgoing) : incoming || outgoing))
        throw std::invalid_argument("free requires inflow and outflow; other types take no branch lines");
    if (config.type==BoundaryType::Periodic) {
        if (!translation || spec.patch.empty() || spec.pair.empty() || spec.patch==spec.pair)
            throw std::invalid_argument("periodic requires different patch and pair names and translation x y z");
    } else if (translation || !spec.patch.empty() || !spec.pair.empty())
        throw std::invalid_argument("periodic keywords on a nonperiodic boundary");
    periodic=spec;
    return config;
}
BoundaryConfig read_boundary(std::istream& input,unsigned dimension) {
    PeriodicPair spec;
    BoundaryConfig result=read_boundary(input,dimension,spec);
    if (result.type==BoundaryType::Periodic)
        throw std::invalid_argument("read_boundary requires a PeriodicPair output for periodic settings");
    return result;
}
}
