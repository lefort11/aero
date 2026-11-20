#include "initial.h"
#include "../../qa/test_checks.h"
#include "../io/expression.h"
#include "../bc/periodic.h"
#include "../io/fields.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

using namespace aero;
namespace {
using aero_test::require;
void near(double a, double b) {
    require(std::fabs(a - b) < 1e-12, "wrong state or expression result");
}
template <class F> void rejects(F f, const std::string& message) {
    try {
        f();
    } catch (const std::exception& e) {
        if (std::string(e.what()).find(message) != std::string::npos)
            return;
        throw;
    }
    throw std::runtime_error("expected error: " + message);
}
Mesh line(double scale = 5) {
    MeshData d;
    d.dimension = 1;
    d.length_scale_to_meters = scale;
    d.x = {0, .25, 1};
    d.y = d.z = {0, 0, 0};
    d.face_nodes = MeshCsr({0, 1, 2, 3}, {0, 1, 2});
    d.cell_faces = MeshCsr({0, 2, 4}, {0, 1, 1, 2});
    d.cell_face_orientation = {-1, 1, -1, 1};
    d.patch_names = {"left", "right"};
    d.boundary_faces = {0, 2};
    d.boundary_patch = {0, 1};
    return Mesh(d);
}
Mesh simplex(unsigned dim) {
    MeshData d;
    d.dimension = dim;
    d.x = {0, 1, 0};
    d.y = {0, 0, 1};
    d.z = {0, 0, 0};
    if (dim == 2)
        d.face_nodes = MeshCsr({0, 2, 4, 6}, {0, 1, 1, 2, 2, 0});
    else {
        d.x.push_back(0);
        d.y.push_back(0);
        d.z.push_back(1);
        d.face_nodes = MeshCsr({0, 3, 6, 9, 12}, {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3});
    }
    for (MeshIndex i = 0; i <= dim; ++i) {
        d.cell_faces.indices.push_back(i);
        d.cell_face_orientation.push_back(1);
        d.boundary_faces.push_back(i);
        d.boundary_patch.push_back(0);
    }
    d.cell_faces.offsets.push_back(dim + 1);
    d.patch_names = {"wall"};
    return Mesh(d);
}
std::vector<double> load(const Mesh& m, const std::string& text, const PeriodicView* seam = nullptr) {
    std::istringstream s(text);
    return read_initial(m, 1.4, s, "build", seam);
}
// Проверяет вычисление начальных полей по координатам и диагностику ошибок в выражениях.
void formulas() {
    for (unsigned d : {1u, 2u, 3u}) {
        const auto mesh = d == 1 ? line() : simplex(d);
        std::istringstream input(d == 3 ? "initial_type freestream\nphi=90\ntheta=30" :
                                 d == 2 ? "initial_type freestream\nphi=90" : "initial_type freestream");
        const auto q = read_initial(mesh, 1.4, input, ".", nullptr, nullptr, {}, nullptr, .5);
        near(q[0], 1.);
        near(q[1], d == 1 ? 1. : 0.);
        if (d > 1) near(q[2], d == 3 ? std::sqrt(.75) : 1.);
        if (d == 3) near(q[3], .5);
        near(q[d+1], .5 + 4./(1.4*.4));
    }
    rejects([&] { load(line(), "initial_type freestream"); }, "mach");
    near(Expression("if(t>0 && x!=0, atan2(y,x)+ln(e)+hypot(3,4),0)").evaluate(1,0,0,2), 6);
    near(Expression("if(!t || 1/0,max(2**3,7)+min(3,4)%2,0)").evaluate(0,0,0), 9);
    near(Expression("-2^2+2^3^2+2^-2").evaluate(0, 0, 0), 508.25);
    near(Expression("if(x<=0,if(y!=1,cos(pi)+abs(-2),1/0),sqrt(-1))").evaluate(0, 0, 0), 1);
    near(Expression("if(x==1,exp(0)+sin(pi/2)+sqrt(4),0)").evaluate(1, 0, 0), 4);
    rejects([] { Expression e("2^^3"); }, "column");
    rejects([] { Expression e("missing(x)"); }, "unknown identifier");
    rejects([] { Expression e(std::string(300, '-') + "x"); }, "nesting");
    rejects([] { Expression("exp(10000)>0").evaluate(0, 0, 0); }, "non-finite");

    const Mesh m = line();
    const std::string sod = "initial_type formula\nrho=if(x<.5,1,.125)\nux=0\np=if(x<.5,1,.1)";
    auto u = load(m, sod);
    require(load(m, "\xef\xbb\xbf" + sod + " # comment") == u, "UTF-8 BOM and formula comment");
    near(u[3], 1);
    near(u[6], .125);
    near(u[8], .25);
    for (unsigned d : {2u, 3u}) {
        Mesh mesh = simplex(d);
        const std::string block = "initial_type formula\nrho=2\nux=1+x\nuy=-2\np=3\n";
        auto q = load(mesh, block + (d == 3 ? "uz=z" : ""));
        const unsigned base = d * (d + 2);
        near(q[base], 2);
        near(q[base + 1], 2);
        near(q[base + 2], -4);
        if (d == 3)
            near(q[base + 3], 2);
        near(q[base + d + 1], 12.5 + (d == 3 ? 1 : 0));
    }
    rejects([&] { load(m, sod + "\nuy=1"); }, "unexpected");
    rejects([&] { load(m, sod + "\np=2"); }, "duplicate");
    rejects([&] { load(m, "initial_type formula\nrho=1\nux=0\np=-1"); }, "node 0");
    rejects([&] { load(m, "initial_type formula\nrho=1\nux=1/x\np=1"); }, "ux: division");
    PeriodicDomain domain(m, {{"left", "right", {1, 0, 0}}});
    const auto seam = domain.view();
    u = load(m, "initial_type formula\nrho=2+sin(2*pi*x)\nux=0\np=1", &seam);
    require(u.size() == 6, "periodic compaction");
    near(u[0], 2);
    near(u[3], 3);
    rejects([&] { load(m, sod, &seam); }, "periodic mismatch");
}
struct Temp {
    std::string filename, path;
    explicit Temp(const std::string& ext) {
#ifdef _WIN32
        int process = _getpid();
#else
        int process = getpid();
#endif
        filename = "initial #" + std::to_string(process) + "_" +
                   std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ext;
        path = "build/" + filename;
    }
    ~Temp() { std::remove(path.c_str()); }
};
Field field(const std::string& name, std::vector<std::string> components, std::vector<double> values) {
    Field f;
    f.location = FieldLocation::Node;
    f.name = name;
    f.components = components;
    f.values = values;
    return f;
}
// Записывает контрольные поля, читает их как начальное состояние и сверяет компоненты.
void files() {
    const Mesh m = line();
    FieldSnapshot snap(m, 17, .2);
    snap.fields.push_back(field("U", {"E", "rho_ux", "rho"}, {3, 4, 6, 0, -1, 2, 1, 1, 2}));
    snap.fields.push_back(field("density", {"value"}, {1, 1, 2}));
    snap.fields.push_back(field("gas velocity", {"horizontal"}, {0, -1, 1}));
    snap.fields.push_back(field("pressure", {"value"}, {1.2, 1.4, 2}));
    Temp data(".fields"), cfg(".cfg");
    const std::string block = "initial_type file\npath \"" + data.filename + "\" # relative to settings\n";
    {
        std::ofstream s(cfg.path);
        s << block << "field U\n";
        require(bool(s), "cannot write test config");
    }
    const std::vector<double> expected = {1, 0, 3, 1, -1, 4, 2, 2, 6};
    for (auto format : {FieldFormat::Text, FieldFormat::Binary, FieldFormat::Lzma}) {
        snap.write(data.path, format);
        require(read_initial_file(m, 1.4, cfg.path) == expected, "snapshot component order");
        const auto u = load(m, block + "rho=density\nux=\"gas velocity\" horizontal\np=pressure");
        for (std::size_t i = 0; i < u.size(); ++i)
            near(u[i], expected[i]);
    }
    // Одна и та же привязка должна читать как начальные, так и эталонные поля.
    // Запрашиваем узлы в обратном порядке, как при выборке представителей.
    const auto reference = [&](const Mesh& mesh, const std::string& mapping, const LinearBackground& background) {
        std::istringstream input("path = \"" + data.filename + "\"\n" + mapping);
        return read_exact_fields(mesh, 1.4, input, "build", nullptr, background, {2, 0});
    };
    const std::string mapped = "rho=density\nux=\"gas velocity\" horizontal\np=pressure\n";
    auto values = reference(m, mapped, {});
    near(values[0][4], 2.);
    near(values[0][5], 6.);
    values = reference(m, "field=U", {});
    near(values[0][1], 1.);
    near(values[1][5], 3.);
    LinearBackground background;
    background.rho = 2.;
    background.velocity[0] = .25;
    values = reference(m, "field=U", background);
    near(values[0][1], .75);
    near(values[0][5], 6.);
    values = reference(m, mapped + "E=density", {});
    near(values[0][5], 2.);
    rejects([&] { reference(line(6), mapped, {}); }, "SHA-256");
    rejects([&] { reference(m, "rho=missing", {}); }, "exact.txt:");
    rejects([&] { load(line(6), block + "field U"); }, "SHA-256");
    rejects([&] { load(m, block + "field density"); }, "component count");
    Field cell = field("cell density", {"value"}, {1, 1});
    cell.location = FieldLocation::Cell;
    snap.fields.push_back(cell);
    snap.write(data.path);
    rejects([&] { load(m, block + "rho=\"cell density\"\nux=\"gas velocity\" horizontal\np=pressure"); },
            "missing NODE");
    snap.fields[0].values[0] = -1;
    snap.write(data.path);
    rejects([&] { load(m, block + "field U"); }, "node 0");
}
}
int main() {
    try {
        formulas();
        files();
        std::cout << "Initial condition formula, periodic and AEROFIELDS tests passed.\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
