#include "../../source/io/submesh.h"
#include "../../source/io/scheduled_output.h"
#include "../../source/mesh/partition.h"
#include "../../source/parallel/processes.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <set>

namespace {
using namespace aero;
using Id=MeshIndex;
using Key=std::pair<Id,Id>;
using Ring=std::vector<Key>;

// Разбирает конечный числовой параметр секущей плоскости.
double number(const char* text) {
    std::size_t used=0;
    const double value=std::stod(text,&used);
    if (text[used] || !std::isfinite(value)) throw MeshError("Expected a finite plane parameter");
    return value;
}

// Строим связность обычной Aero-сетки меньшей размерности. Смежные элементы
// разделяют вершину (в 1D) или ребро (в 2D), поэтому Tecplot видит непрерывную зону.
class SliceBuilder {
    MeshView grid;
    Coordinate normal;
    std::vector<double> distance;
    MeshData data;
    std::map<Key,Id> vertices, faces;
    std::vector<unsigned> uses;
    std::vector<SlicePoint> mapping;

    // Вычисляет интерполяционный вес пересечения ребра с плоскостью.
    double weight(Key k) const {
        return k.first==k.second ? 0. : distance[k.first]/(distance[k.first]-distance[k.second]);
    }
    // Вычисляет координаты точки среза по концам ребра и весу.
    Coordinate position(Key k) const {
        const auto a=grid.coordinate(k.first), b=grid.coordinate(k.second);
        Coordinate p{};
        for (unsigned j=0;j<3;++j) p[j]=a[j]+weight(k)*(b[j]-a[j]);
        return p;
    }
    // Создаёт или находит уже добавленную вершину среза.
    Id vertex(Key k) {
        auto added=vertices.emplace(k,vertices.size());
        if (added.second) {
            const auto p=position(k);
            data.x.push_back(p[0]); data.y.push_back(p[1]); data.z.push_back(p[2]);
            mapping.push_back({k.first,k.second,weight(k)});
        }
        return added.first->second;
    }
    // Добавляет грань среза с сохранением общей нумерации вершин.
    void append_face(Id a,Id b,int orientation) {
        const auto key=std::minmax(a,b);
        auto added=faces.emplace(key,faces.size());
        if (added.second) {
            data.face_nodes.indices.push_back(key.first);
            if (a!=b) data.face_nodes.indices.push_back(key.second);
            data.face_nodes.offsets.push_back(data.face_nodes.indices.size());
            uses.push_back(0);
        }
        const Id f=added.first->second;
        ++uses[f];
        data.cell_faces.indices.push_back(f);
        data.cell_face_orientation.push_back(static_cast<std::int8_t>(orientation));
    }
    // Добавляет ячейку полученной подсетки.
    void append_cell(Ring ring) {
        if (data.dimension==1) {
            auto a=position(ring[0]), b=position(ring[1]);
            if (std::lexicographical_compare(b.begin(),b.end(),a.begin(),a.end())) std::swap(ring[0],ring[1]);
            const Id left=vertex(ring[0]), right=vertex(ring[1]);
            append_face(left,left,-1); append_face(right,right,1);
        } else {
            // Соседние многоугольники ориентируем по одной нормали плоскости.
            double area=0.;
            const auto origin=position(ring.front());
            for (std::size_t j=1;j+1<ring.size();++j)
                area+=dot3(normal,cross3(position(ring[j])-origin,position(ring[j+1])-origin));
            if (area==0.) throw MeshError("Degenerate plane intersection");
            if (area<0.) std::reverse(ring.begin(),ring.end());
            for (std::size_t j=0;j<ring.size();++j) {
                const Id a=vertex(ring[j]), b=vertex(ring[(j+1)%ring.size()]);
                append_face(a,b,a<b ? 1 : -1);
            }
        }
        data.cell_faces.offsets.push_back(data.cell_faces.indices.size());
    }
    // На одной грани отрезки пересечения лежат на прямой. Вершину касания
    // учитываем только при переходе через плоскость; лежащее в ней ребро сохраняем целиком.
    std::vector<std::pair<Key,Key>> segments(MeshIndexSpan nodes) const {
        std::vector<std::pair<Key,Key>> result;
        Ring hits;
        for (Id j=0;j<nodes.size;++j) {
            const Id a=nodes[j], b=nodes[(j+1)%nodes.size], prev=nodes[(j+nodes.size-1)%nodes.size];
            if (!distance[a] && !distance[b]) result.push_back({{a,a},{b,b}});
            if (!distance[a] && distance[prev] && distance[b] && (distance[prev]<0)!=(distance[b]<0))
                hits.emplace_back(a,a);
            if (distance[a] && distance[b] && (distance[a]<0)!=(distance[b]<0))
                hits.push_back(std::minmax(a,b));
        }
        if (hits.size()%2) throw MeshError("Ambiguous intersection along a face; move the plane or adjust tolerance");
        std::sort(hits.begin(),hits.end(),[&](Key a,Key b){
            const auto x=position(a), y=position(b);
            return std::lexicographical_compare(x.begin(),x.end(),y.begin(),y.end());
        });
        for (std::size_t k=0;k<hits.size();k+=2) result.emplace_back(hits[k],hits[k+1]);
        return result;
    }
    // Пересекает исходную ячейку плоскостью и упорядочивает контур сечения.
    void cut_cell(const MeshCell& cell) {
        bool above=false, below=false;
        for (Id n:cell.node_ids()) { above|=distance[n]>0.; below|=distance[n]<0.; }
        if (!above || !below) return; // Совпадающие грани уже обработаны по глобальному номеру.
        if (grid.dimension==2) {
            // В 2D cell.node_ids() упорядочен по номерам. Восстанавливаем контур по рёбрам.
            std::map<Id,Id> next;
            for (Id j=0;j<cell.face_count();++j) {
                auto nodes=cell.face(j).node_ids();
                const bool forward=cell.face_orientation(j)>0;
                next[nodes[forward?0:1]]=nodes[forward?1:0];
            }
            std::vector<Id> loop;
            Id n=next.begin()->first;
            do {loop.push_back(n); n=next.at(n);} while (n!=loop.front());
            for (auto seg:segments({loop.data(),loop.size()})) append_cell({seg.first,seg.second});
            return;
        }
        // Отрезки на гранях замыкаются в контуры внутри ячейки. Это сохраняет
        // исходное соседство на тетраэдрах, призмах, пирамидах и гексаэдрах.
        std::map<Key,std::set<Key>> links;
        for (Id f:cell.face_ids()) for (auto seg:segments(grid.face_nodes(f))) {
            links[seg.first].insert(seg.second); links[seg.second].insert(seg.first);
        }
        for (const auto& entry:links)
            if (entry.second.size()!=2) throw MeshError("Non-manifold plane intersection; adjust the plane or tolerance");
        while (!links.empty()) {
            Ring loop;
            Key current=links.begin()->first, previous=current;
            do {
                loop.push_back(current);
                const auto neighbors=links.at(current);
                const auto next=*neighbors.begin()==previous ? *neighbors.rbegin() : *neighbors.begin();
                links.erase(current); previous=current; current=next;
            } while (current!=loop.front());
            append_cell(loop);
        }
    }
public:
    // Вычисляет расстояния узлов до секущей плоскости и готовит построение среза.
    SliceBuilder(MeshView mesh,Coordinate n,double offset,double tolerance):grid(mesh),normal(n),distance(mesh.node_count) {
        data.dimension=grid.dimension-1;
        if (grid.dimension<2) throw MeshError("A plane slice requires a 2D or 3D mesh");
        for (Id i=0;i<mesh.node_count;++i) {
            const double d=dot3(normal,mesh.coordinate(i))-offset;
            distance[i]=std::abs(d)<=tolerance ? 0. : d;
        }
        if (std::all_of(distance.begin(),distance.end(),[](double d){return d==0.;}))
            throw MeshError("The entire mesh lies within the plane tolerance");
    }
    // Строит геометрию подсетки и связи интерполяции с исходной сеткой.
    Submesh build(bool cut) {
        // Каждую совпавшую грань берём один раз, включая внутренние грани двух ячеек.
        for (Id f=0;f<grid.face_count;++f) {
            const auto nodes=grid.face_nodes(f);
            if (!std::all_of(nodes.begin(),nodes.end(),[&](Id n){return distance[n]==0.;})) continue;
            Ring ring;
            for (Id n:nodes) ring.emplace_back(n,n);
            append_cell(ring);
        }
        if (cut) for (Id c=0;c<grid.cell_count;++c) cut_cell(grid.cell(c));
        if (mapping.empty()) throw MeshError("Plane selected no elements");
        for (Id f=0;f<uses.size();++f) if (uses[f]==1) {
            data.boundary_faces.push_back(f); data.boundary_patch.push_back(0);
        }
        if (!data.boundary_faces.empty()) data.patch_names.push_back("perimeter");
        Submesh result;
        result.points=std::move(mapping); result.mesh=Mesh(std::move(data)); result.parent_nodes=grid.node_count;
        return result;
    }
};
}

// Разбирает параметры плоскости и сохраняет сечение исходной сетки.
int main(int argc,char** argv) {
    using namespace aero;
    try {
        if (argc!=8 && argc!=9) {
            std::cout<<"Usage: mesh_slice INPUT.mesh|INPUT.mdec OUTPUT.submesh faces|cut NX NY NZ OFFSET [TOLERANCE]\n"
                       "Plane: NX*x + NY*y + NZ*z = OFFSET. Tolerance is a distance in mesh units.\n";
            return argc==2 && std::string(argv[1])=="--help" ? 0 : 1;
        }
        const std::string mode=argv[3], output=argv[2];
        if (output.size()<8 || output.substr(output.size()-8)!=".submesh") throw MeshError("Expected OUTPUT.submesh");
        if (mode!="faces" && mode!="cut") throw MeshError("Expected faces or cut");
        Coordinate n={{number(argv[4]),number(argv[5]),number(argv[6])}};
        const double scale=coordinate_length(n);
        if (!(scale>0.) || !std::isfinite(scale)) throw MeshError("Invalid plane normal");
        for (auto& v:n) v/=scale;
        const double offset=number(argv[7])/scale;
        Mesh mesh;
        std::string hash;
        if (mdec::Archive::matches(argv[1])) {
            Processes world(argc,argv);
            auto part=load_partition(argv[1],world);
            mesh=std::move(part.mesh); hash=part.mesh_sha256;
        } else mesh=Mesh::read(argv[1]);
        if (hash.empty()) hash=FieldSnapshot::mesh_fingerprint(mesh);
        const auto grid=mesh.view();
        double length=std::abs(offset);
        for (Id i=0;i<grid.node_count;++i)
            for (double v:grid.coordinate(i)) length=std::max(length,std::abs(v));
        const double tolerance=argc==9 ? number(argv[8]) : 64*std::numeric_limits<double>::epsilon()*length;
        if (tolerance<0.) throw MeshError("Negative plane tolerance");
        auto slice=SliceBuilder(grid,n,offset,tolerance).build(mode=="cut");
        slice.parent_hash=hash;
        StagedFieldFile file(output);
        std::ofstream out(file.temporary(),std::ios::binary);
        slice.write(out); out.close();
        if (!out) throw MeshError("Cannot write "+output);
        file.publish(true);
        std::cout<<"Written "<<output<<": "<<slice.points.size()<<" nodes, "<<slice.mesh.view().cell_count<<" elements\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<"mesh_slice: "<<e.what()<<'\n'; return 1; }
}
