#pragma once
#include "fields.h"
#include "../mesh/mesh.h"
#include <fstream>
#include <cmath>
#include <iomanip>

namespace aero {
// Для исходного узла first == second. На ребре значение равно (1-w)*first + w*second.
struct SlicePoint { MeshIndex first, second; double weight; };
// Геометрия подсетки и её связь с узлами исходной сетки для интерполяции.
struct Submesh {
    // Отпечаток исходной сетки для проверки совместимости.
    std::string parent_hash;
    // Количество узлов в исходной сетке.
    MeshIndex parent_nodes = 0;
    // Узлы среза и данные интерполяции.
    std::vector<SlicePoint> points;
    // Геометрия и топология самого среза.
    Mesh mesh;

    // Сохраняет подсетку, отпечаток родителя и точки интерполяции.
    void write(std::ostream& out) const {
        out << "AEROSLICE 1\n" << parent_hash << ' ' << parent_nodes << ' ' << points.size() << '\n'
            << std::setprecision(17);
        for (const auto& p : points) out << p.first << ' ' << p.second << ' ' << p.weight << '\n';
        mesh.write(out, MeshFormat::Binary);
    }
    // Читает подсетку и проверяет её связи с родительской сеткой.
    static Submesh read(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        Submesh s;
        std::string signature;
        MeshIndex count;
        if (!std::getline(in,signature) || signature!="AEROSLICE 1" ||
            !(in >> s.parent_hash >> s.parent_nodes >> count) || s.parent_hash.size()!=64 || !count)
            throw FieldError("Invalid submesh header: "+path);
        for (MeshIndex i=0;i<count;++i) {
            SlicePoint p;
            if (!(in >> p.first >> p.second >> p.weight) || p.first>=s.parent_nodes ||
                p.second>=s.parent_nodes || !(p.weight>=0 && p.weight<=1))
                throw FieldError("Invalid submesh node mapping: "+path);
            s.points.push_back(p);
        }
        in >> std::ws;
        s.mesh=Mesh::read(in);
        if (s.mesh.view().node_count!=count) throw FieldError("Submesh mapping size mismatch: "+path);
        return s;
    }
};
}
