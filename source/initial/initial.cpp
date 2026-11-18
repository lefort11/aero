#include "initial.h"
#include "../parallel/parallel_for.h"
#include "../bc/periodic.h"
#include "../io/expression.h"
#include "../io/fields.h"
#include "../eulersolver/euler.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>

namespace aero {
namespace {
std::string trim(const std::string& s) {
    const auto first=s.find_first_not_of(" \t\r\n");
    if (first==std::string::npos) return {};
    return s.substr(first,s.find_last_not_of(" \t\r\n")-first+1);
}
struct Setting { std::string value; std::size_t line; };
using Settings=std::map<std::string,Setting>;
std::invalid_argument error(std::size_t line,const std::string& msg) {
    return std::invalid_argument("initial line "+std::to_string(line)+": "+msg);
}
Settings read_settings(std::istream& input) {
    Settings out;
    std::string line;
    std::size_t row=0;
    while (std::getline(input,line)) {
        ++row;
        bool quoted=false;
        for (std::size_t j=0;j<line.size();++j) {
            if (line[j]=='"') quoted=!quoted;
            if (line[j]=='#' && !quoted) { line.resize(j); break; }
        }
        if (quoted) throw error(row,"unclosed quote");
        line=trim(line);
        if (line.empty()) continue;
        const auto cut=line.find_first_of(" \t=");
        if (cut==std::string::npos) throw error(row,"missing value");
        std::string key=line.substr(0,cut),value=trim(line.substr(cut));
        if (!value.empty() && value[0]=='=') value=trim(value.substr(1));
        if (key.empty() || value.empty()) throw error(row,"missing key or value");
        if (!out.emplace(key,Setting{value,row}).second) throw error(row,"duplicate "+key);
    }
    if (input.bad() || !input.eof()) throw std::runtime_error("initial settings read failed");
    return out;
}
Setting take(Settings& settings,const std::string& key) {
    const auto it=settings.find(key);
    if (it==settings.end()) throw std::invalid_argument("initial: missing "+key);
    const Setting s=it->second;
    settings.erase(it);
    return s;
}
std::vector<std::string> words(const Setting& s) {
    std::vector<std::string> result;
    std::size_t pos=0;
    while ((pos=s.value.find_first_not_of(" \t",pos))!=std::string::npos) {
        const bool quote=s.value[pos]=='"';
        if (quote) ++pos;
        const auto end=quote?s.value.find('"',pos):s.value.find_first_of(" \t",pos);
        if (quote && end==std::string::npos) throw error(s.line,"unclosed quote");
        const std::string word=s.value.substr(pos,end-pos);
        if (word.empty() || word.find('"')!=std::string::npos) throw error(s.line,"invalid name");
        result.push_back(word);
        if (end==std::string::npos) break;
        pos=end+(quote?1:0);
        if (quote && pos<s.value.size() && s.value[pos]!=' ' && s.value[pos]!='\t')
            throw error(s.line,"missing separator after quoted name");
    }
    return result;
}
std::string single(const Setting& s) {
    const auto tokens=words(s);
    if (tokens.size()!=1) throw error(s.line,"expected one name or path");
    return tokens[0];
}
std::string resolve(const std::string& dir,const std::string& path) {
    if (dir.empty() || path[0]=='/' || path[0]=='\\' || (path.size()>1 && path[1]==':')) return path;
    return dir+"/"+path;
}
void require_consumed(const Settings& s) {
    if (!s.empty()) throw error(s.begin()->second.line,"unexpected key "+s.begin()->first);
}
std::vector<std::string> primitives(unsigned d) {
    std::vector<std::string> names={"rho","ux"};
    if (d>=2) names.push_back("uy");
    if (d==3) names.push_back("uz");
    names.push_back("p");
    return names;
}
void convert(double* q,unsigned d,double gamma) {
    const double rho=q[0],p=q[d+1];
    if (!(rho>0 && p>0)) throw std::domain_error("expected positive rho and p");
    double energy=p/(gamma-1);
    for (unsigned k=1;k<=d;++k) {
        const double u=q[k];
        q[k]=rho*u;
        energy+=.5*q[k]*u;
    }
    q[d+1]=energy;
}
struct Column { const Field* field; std::size_t column; };
const Field& node_field(const FieldSnapshot& s,const std::string& name) {
    for (const Field& f:s.fields) if (f.location==FieldLocation::Node && f.name==name) return f;
    throw std::invalid_argument("initial: missing NODE field '"+name+"'");
}
Column component(const Field& f,const std::string& name) {
    const auto it=std::find(f.components.begin(),f.components.end(),name);
    if (it==f.components.end()) throw std::invalid_argument("initial: missing component '"+name+"' of '"+f.name+"'");
    return {&f,static_cast<std::size_t>(it-f.components.begin())};
}
}

void check_periodic_initial(const double* ref,const double* q,unsigned width,MeshIndex owner,MeshIndex node) {
    for (unsigned k=0;k<width;++k) {
        const double scale=std::max(std::fabs(q[k]),std::fabs(ref[k]));
        if (std::fabs(q[k]-ref[k])>1e-13+1e-10*scale)
            throw std::domain_error("initial: periodic mismatch between nodes "+std::to_string(owner)+" and "+std::to_string(node));
    }
}
std::vector<double> read_initial(const Mesh& mesh,double gamma,std::istream& input,
                                 const std::string& directory,const PeriodicView* seam,const FieldNodeSelection* selection) {
    if (!(gamma>1 && std::isfinite(gamma))) throw std::invalid_argument("initial: invalid gamma");
    const auto m=mesh.view();
    const unsigned d=m.dimension,width=d+2;
    const auto names=primitives(d);
    Settings settings=read_settings(input);
    const std::string type=single(take(settings,"initial_type"));
    std::vector<Expression> expressions;
    FieldSnapshot snapshot;
    std::vector<Column> columns;
    bool conservative=false;
    if (type=="formula") {
        for (const auto& name:names) {
            const auto s=take(settings,name);
            try { expressions.emplace_back(s.value); }
            catch(const std::invalid_argument& e) { throw error(s.line,name+": "+e.what()); }
        }
    } else if (type=="file") {
        const std::string path=resolve(directory,single(take(settings,"path")));
        if (selection) {
            if (selection->nodes.size()!=m.node_count) throw std::invalid_argument("initial: invalid local node count");
            snapshot.fields=FieldSnapshot::read_nodes(path,*selection);
        } else {
            snapshot=FieldSnapshot::read(path);
            snapshot.check_mesh(mesh);
        }
        conservative=settings.count("field")!=0;
        if (conservative) {
            const std::string name=single(take(settings,"field"));
            const Field& f=node_field(snapshot,name);
            if (f.components.size()!=width) throw std::invalid_argument("initial: incorrect conservative component count");
            columns.push_back(component(f,"rho"));
            for (unsigned k=1;k<=d;++k) columns.push_back(component(f,"rho_"+names[k]));
            columns.push_back(component(f,"E"));
        } else {
            for (const auto& name:names) {
                const Setting s=take(settings,name);
                const auto w=words(s);
                if (w.empty() || w.size()>2) throw error(s.line,"expected field name and optional component");
                const Field& f=node_field(snapshot,w[0]);
                if (w.size()==1 && f.components.size()!=1) throw error(s.line,"choose a component of "+w[0]);
                columns.push_back(component(f,w.size()==2?w[1]:f.components[0]));
            }
        }
    } else throw std::invalid_argument("initial: unknown initial_type "+type);
    require_consumed(settings);
    const MeshIndex count=seam?seam->node_count:m.node_count;
    if (seam && (seam->original_node_count!=m.node_count || !seam->node_dof || count>m.node_count))
        throw std::invalid_argument("initial: incompatible periodic view");
    std::vector<double> result;
    if (count>result.max_size()/width) throw std::length_error("initial state too large");
    result.resize(static_cast<std::size_t>(count)*width);
    std::vector<MeshIndex> owner(seam?count:0,m.node_count);
    auto evaluate=[&](MeshIndex n,double* q) {
        try {
            for (unsigned k=0;k<width;++k) {
                if (type=="file") q[k]=columns[k].field->values[columns[k].column*m.node_count+n];
                else {
                    try { q[k]=expressions[k].evaluate(m.x[n],m.y[n],m.z[n]); }
                    catch(const std::domain_error& e) { throw std::domain_error(names[k]+": "+e.what()); }
                }
            }
            if (!conservative) convert(q,d,gamma);
            EulerState physical;
            if (euler_state(d,q,gamma,&physical)!=EulerStatus::Success)
                throw std::domain_error("invalid conservative state");
        } catch(const std::domain_error& e) {
            throw std::domain_error("initial node "+std::to_string(n)+": "+e.what());
        }
    };
    if (seam) {
        for (MeshIndex n=0;n<m.node_count;++n) {
            const MeshIndex id=seam->node_dof[n];
            if (id>=count) throw std::invalid_argument("initial: periodic node index out of range");
            if (owner[id]==m.node_count) owner[id]=n;
        }
        if (std::find(owner.begin(),owner.end(),m.node_count)!=owner.end())
            throw std::invalid_argument("initial: incomplete periodic map");
    }
    parallel_for(count,[&](MeshIndex id) { evaluate(seam ? owner[id] : id,result.data()+width*id); });
    if (seam) {
        parallel_for(m.node_count,[&](MeshIndex n) {
            const MeshIndex id=seam->node_dof[n];
            if (owner[id]==n) return;
            double q[5]={};
            evaluate(n,q);
            const double* dst=result.data()+width*id;
            check_periodic_initial(dst,q,width,owner[id],n);
        });
    }
    return result;
}
std::vector<double> read_initial_file(const Mesh& mesh,double gamma,const std::string& path,const PeriodicView* seam) {
    std::ifstream in(path.c_str());
    if (!in) throw std::runtime_error("cannot open initial settings: "+path);
    const auto slash=path.find_last_of("/\\");
    return read_initial(mesh,gamma,in,slash==std::string::npos?".":path.substr(0,slash+1),seam);
}
}
