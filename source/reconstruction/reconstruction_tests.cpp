#include "weno.h" // Заголовок должен компилироваться и при первом включении.
#include "../../tests/test_checks.h"
#include "reconstruction.h"
#include "mp6_hybrid.h"
#include "curved_stencil.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#ifdef AERO_WITH_KOKKOS_GPU
#include "../../tests/device_test_buffer.h"
#include "../parallel/device_memory.h"
#endif

namespace {
using namespace aero;
typedef std::vector<MeshIndex> Polygon;
std::vector<std::pair<std::string, std::function<void()>>> tests;
using aero_test::require;
// Проверяет совпадение чисел в пределах заданного допуска.
void near(double actual, double expected, double tolerance = 2.e-11) {
    require(std::abs(actual - expected) <= tolerance * (1 + std::abs(expected)),
            "Expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}
// Проверяет, что некорректный ввод завершается ожидаемой ошибкой.
template <class F> void rejects(F operation) {
    try {
        operation();
    } catch (const ReconstructionError&) {
        return;
    }
    throw std::runtime_error("Expected ReconstructionError");
}
// Добавляет именованную проверку в список тестов модуля.
void add(const std::string& name, std::function<void()> operation) {
    tests.push_back({name, operation});
}

// Проверяет ограничение слабой поперечной компоненты гибрида MP6.
template<bool weno> void check_weak_transverse_mp6() {
    const Mp5Settings settings{!weno, weno, weno, 4.};
    aero::FixedMatrix<double,5,5> slopes{};
    const double first_column[]={.4,.3,.7,-.1,1.};
    for (unsigned i=0;i<5;++i) slopes(i,0)=first_column[i];
    double base_l[5], base_r[5];
    mp6::reconstruct<weno>(slopes,5,settings,.6,base_l,base_r);
    for (double signal : {-1.e-12, 1.e-12}) {
        // У обеих сторон нулевой допустимый интервал для поперечной компоненты.
        // Для MP5 два дальних наклона по 12*signal обнуляют также линейный
        // полином (0.8*signal - 12*signal/15) при досрочном принятии.
        slopes(0, 3) = signal;
        slopes(3, 3) = slopes(4, 3) = weno ? 0. : 12.*signal;
        double left[5], right[5];
        mp6::reconstruct<weno>(slopes,5,settings,.6,left,right);
        near(left[3], 0., 1.e-27);
        near(right[3], 0., 1.e-27);
        require(std::abs(left[0]-base_l[0]) <= 2.*std::abs(signal) &&
                std::abs(right[0]-base_r[0]) <= 2.*std::abs(signal),
                "Weak transverse perturbation amplified in acoustic reconstruction");
    }
}

// Обход всех рёбер организует тест. Так сравниваем результат для всей сетки, вызывая рабочую функцию для каждой стороны.
struct TestResult {
    // Второй набор приращений по сторонам для сверки.
    std::vector<double> left_increment2, right_increment2;
};
// Проверяет размер тестового массива перед выделением памяти.
std::size_t checked_test_size(ReconstructionIndex count, ReconstructionIndex components) {
    const ReconstructionIndex limit =
        static_cast<ReconstructionIndex>((std::numeric_limits<std::size_t>::max)() / sizeof(double));
    if (!components || (count && components > limit / count))
        throw ReconstructionError("Invalid test field size");
    return static_cast<std::size_t>(count * components);
}
// Проверяет согласованность размеров входного поля и тестового шаблона.
void check_test_input(ReconstructionStencilView s, const std::vector<double>& values,
                      ReconstructionIndex components) {
    if (values.size() != checked_test_size(s.node_count, components))
        throw ReconstructionError("Invalid test field length");
    checked_test_size(2 * s.edge_count, components);
}
// Вычисляет все стороны через невладеющее представление тестового шаблона.
TestResult reconstruct_all_view(ReconstructionStencilView s, const std::vector<double>& values,
                                ReconstructionIndex components) {
    check_test_input(s, values, components);
    TestResult result;
    result.left_increment2.resize(checked_test_size(s.edge_count, components));
    result.right_increment2.resize(result.left_increment2.size());
    const ReconstructionFieldView neighbours = {values.data(), 1, s.node_count};
#ifdef _OPENMP
#pragma omp parallel for if (s.edge_count * components > 4096)
#endif
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(2 * s.edge_count); ++r) {
        const ReconstructionIndex row = static_cast<ReconstructionIndex>(r), edge = row / 2;
        const double* own = values.data() + s.row_owner[row];
        double* output = (row % 2 ? result.right_increment2.data() : result.left_increment2.data()) + edge;
        reconstruct_increment2(s, edge, row % 2 ? ReconstructionSide::Right : ReconstructionSide::Left, own,
                               components, neighbours, output, s.node_count, s.edge_count);
    }
    return result;
}
// Вычисляет все стороны реконструкции для тестового поля.
TestResult reconstruct_all_sides(const ReconstructionStencil& stencil, const std::vector<double>& values,
                                 ReconstructionIndex components) {
    return reconstruct_all_view(stencil.view(), values, components);
}
// Вычисляет выбранную сторону шаблона для сравнения реализаций.
template <class Reader>
double evaluate_test_side(ReconstructionStencilView s, ReconstructionIndex row, ReconstructionIndex component,
                          const Reader& reader) {
    std::vector<double> own(static_cast<std::size_t>(component + 1), 0), output(own.size());
    if (s.offsets[row] != s.offsets[row + 1])
        for (ReconstructionIndex c = 0; c <= component; ++c)
            own[static_cast<std::size_t>(c)] = reader(row, s.row_owner[row], c);
    const ReconstructionStatus status =
        reconstruct_increment2(s, row / 2, row % 2 ? ReconstructionSide::Right : ReconstructionSide::Left,
                               own.data(), component + 1, reader, output.data());
    if (status != ReconstructionStatus::Success)
        throw ReconstructionError("Invalid test side");
    return output[static_cast<std::size_t>(component)];
}
#ifdef AERO_WITH_KOKKOS_GPU
extern "C" {
// Создаёт GPU-состояние тестового реконструктора.
int test_reconstruction_device_create(ReconstructionStencilView, int, void**, char*, unsigned);
// Освобождает GPU-ресурсы тестового реконструктора.
void test_reconstruction_device_destroy(void*);
// Переносит тестовое поле на GPU и вычисляет реконструкцию.
int test_reconstruction_device_host(void*, const double*, ReconstructionIndex, double*, double*, char*,
                                  unsigned);
// Выполняет проверку реконструкции над полем, уже находящимся на GPU.
int test_reconstruction_device_device(void*, const double*, ReconstructionIndex, double*, double*, void*, char*,
                                    unsigned);
// Вычисляет одну выбранную сторону реконструкции на GPU.
int test_reconstruction_device_one(ReconstructionStencilView, ReconstructionIndex, unsigned, const double*,
                                 ReconstructionIndex, const double*, double*, void*);
}
#endif
// Связывает тестовые вызовы реконструкции с CPU- или GPU-контекстом.
class TestReconstructor {
    // Контекст реконструкции теста.
    ReconstructionContext context_;
    // Рабочий объект теста на устройстве.
    void* gpu_;

public:
    // Связывает тестовый шаблон с CPU- или GPU-исполнителем.
    explicit TestReconstructor(ReconstructionStencil stencil,
                               ReconstructionBackend backend = ReconstructionBackend::Auto, int device = 0)
        : context_(std::move(stencil), backend, device), gpu_(nullptr) {
#ifdef AERO_WITH_KOKKOS_GPU
        if (context_.backend() == ReconstructionBackend::Cuda) {
            char error[512] = {};
            if (test_reconstruction_device_create(context_.device_view(), device, &gpu_, error, sizeof(error)))
                throw ReconstructionError(error);
        }
#endif
    }
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~TestReconstructor() {
#ifdef AERO_WITH_KOKKOS_GPU
        test_reconstruction_device_destroy(gpu_);
#endif
    }
    // Сообщает, собран ли модуль с поддержкой CUDA.
    static bool cuda_compiled() { return ReconstructionContext::cuda_compiled(); }
    // Возвращает число доступных CUDA-устройств.
    static int cuda_device_count() { return ReconstructionContext::cuda_device_count(); }
    // Возвращает выбранный исполнитель вычислений.
    ReconstructionBackend backend() const { return context_.backend(); }
    // Восстанавливает тестовые следы на выбранном исполнителе.
    TestResult reconstruct(const std::vector<double>& values, ReconstructionIndex components) {
        const ReconstructionStencilView s = context_.host_view();
        check_test_input(s, values, components);
        if (backend() == ReconstructionBackend::Cpu)
            return reconstruct_all_view(s, values, components);
        TestResult result;
        result.left_increment2.resize(checked_test_size(s.edge_count, components));
        result.right_increment2.resize(result.left_increment2.size());
#ifdef AERO_WITH_KOKKOS_GPU
        char error[512] = {};
        if (test_reconstruction_device_host(gpu_, values.data(), components, result.left_increment2.data(),
                                          result.right_increment2.data(), error, sizeof(error)))
            throw ReconstructionError(error);
#endif
        return result;
    }
    // Вычисляет тестовые следы по данным, уже размещённым на GPU.
    void reconstruct_device(const double* values, ReconstructionIndex components, double* left, double* right,
                            void* stream = nullptr) {
        if (backend() != ReconstructionBackend::Cuda)
            throw ReconstructionError("Device driver requires CUDA");
        const ReconstructionStencilView s = context_.host_view();
        checked_test_size(s.node_count, components);
        checked_test_size(2 * s.edge_count, components);
        if (s.edge_count && (!values || !left || !right))
            throw ReconstructionError("Null test device field");
#ifdef AERO_WITH_KOKKOS_GPU
        char error[512] = {};
        if (test_reconstruction_device_device(gpu_, values, components, left, right, stream, error,
                                            sizeof(error)))
            throw ReconstructionError(error);
#else
        (void)stream;
#endif
    }
};

// Собирает тестовую сетку из узлов и ячеек с заданной топологией.
struct Builder {
    // Координаты и топология строящейся тестовой сетки.
    MeshData d;
    // Поиск повторяющихся граней по вершинам.
    std::map<Polygon, MeshIndex> faces;
    // Кратность использования граней.
    std::vector<unsigned> count;
    // Возвращает номер опорного узла строки реконструкции.
    MeshIndex node(double x, double y, double z = 0) {
        d.x.push_back(x);
        d.y.push_back(y);
        d.z.push_back(z);
        return static_cast<MeshIndex>(d.x.size() - 1);
    }
    // Добавляет ячейку в строящуюся тестовую сетку.
    void cell(const std::vector<Polygon>& shell) {
        for (const Polygon& p : shell) {
            Polygon key = p;
            std::sort(key.begin(), key.end());
            const auto found = faces.find(key);
            MeshIndex f;
            int sign = 1;
            if (found == faces.end()) {
                f = static_cast<MeshIndex>(count.size());
                faces[key] = f;
                count.push_back(0);
                d.face_nodes.indices.insert(d.face_nodes.indices.end(), p.begin(), p.end());
                d.face_nodes.offsets.push_back(static_cast<MeshIndex>(d.face_nodes.indices.size()));
            } else {
                f = found->second;
                sign = -1;
            }
            ++count[static_cast<std::size_t>(f)];
            d.cell_faces.indices.push_back(f);
            d.cell_face_orientation.push_back(static_cast<std::int8_t>(sign));
        }
        d.cell_faces.offsets.push_back(static_cast<MeshIndex>(d.cell_faces.indices.size()));
    }
    // Создаёт многоугольную ячейку тестовой сетки.
    void polygon(const Polygon& p) {
        std::vector<Polygon> shell;
        for (std::size_t k = 0; k < p.size(); ++k)
            shell.push_back({p[k], p[(k + 1) % p.size()]});
        cell(shell);
    }
    // Создаёт тестовую тетраэдральную ячейку.
    void tetra(MeshIndex a, MeshIndex b, MeshIndex c, MeshIndex e) {
        const double ax = d.x[b] - d.x[a], ay = d.y[b] - d.y[a], az = d.z[b] - d.z[a];
        const double bx = d.x[c] - d.x[a], by = d.y[c] - d.y[a], bz = d.z[c] - d.z[a];
        const double cx = d.x[e] - d.x[a], cy = d.y[e] - d.y[a], cz = d.z[e] - d.z[a];
        if (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx) < 0)
            std::swap(b, c);
        cell({{a, c, b}, {a, b, e}, {b, c, e}, {c, a, e}});
    }
    // Выбирает итоговое допустимое пересечение после обхода кандидатов.
    Mesh finish() {
        d.patch_names = {"boundary"};
        for (std::size_t f = 0; f < count.size(); ++f)
            if (count[f] == 1) {
                d.boundary_faces.push_back(static_cast<MeshIndex>(f));
                d.boundary_patch.push_back(0);
            }
        return Mesh(d);
    }
};
// Создаёт одномерную сетку для проверки алгоритма.
Mesh line(const std::vector<double>& x) {
    MeshData d;
    d.dimension = 1;
    d.x = x;
    d.y.assign(x.size(), 0);
    d.z = d.y;
    d.patch_names = {"left", "right"};
    for (std::size_t i = 0; i < x.size(); ++i) {
        d.face_nodes.indices.push_back(static_cast<MeshIndex>(i));
        d.face_nodes.offsets.push_back(static_cast<MeshIndex>(i + 1));
        if (i + 1 < x.size()) {
            d.cell_faces.indices.push_back(static_cast<MeshIndex>(i));
            d.cell_faces.indices.push_back(static_cast<MeshIndex>(i + 1));
            d.cell_faces.offsets.push_back(static_cast<MeshIndex>(2 * (i + 1)));
            d.cell_face_orientation.insert(d.cell_face_orientation.end(), {-1, 1});
        }
    }
    d.boundary_faces = {0, static_cast<MeshIndex>(x.size() - 1)};
    d.boundary_patch = {0, 1};
    return Mesh(d);
}
// Создаёт плоскую тестовую сетку для проверки шаблонов реконструкции.
Mesh plane(bool triangles, MeshIndex nx = 5, MeshIndex ny = 4) {
    Builder b;
    b.d.dimension = 2;
    for (MeshIndex j = 0; j < ny; ++j)
        for (MeshIndex i = 0; i < nx; ++i) {
            const double fi = static_cast<double>(i), fj = static_cast<double>(j);
            const double x = fi + 0.07 * fj * fj, y = fj + 0.04 * fi * fi;
            b.node(x, y);
        }
    for (MeshIndex j = 0; j + 1 < ny; ++j)
        for (MeshIndex i = 0; i + 1 < nx; ++i) {
            const MeshIndex a = nx * j + i, c = a + nx + 1;
            if (triangles) {
                b.polygon({a, a + 1, c});
                b.polygon({a, c, a + nx});
            } else
                b.polygon({a, a + 1, c, a + nx});
        }
    return b.finish();
}
// Создаёт объёмную тестовую сетку для проверки шаблонов реконструкции.
Mesh volume(bool tetrahedra, MeshIndex n = 4) {
    Builder b;
    for (MeshIndex k = 0; k < n; ++k)
        for (MeshIndex j = 0; j < n; ++j)
            for (MeshIndex i = 0; i < n; ++i)
                b.node(static_cast<double>(i) + 0.08 * static_cast<double>(j),
                       static_cast<double>(j) + 0.05 * static_cast<double>(k),
                       static_cast<double>(k) + 0.03 * static_cast<double>(i));
    for (MeshIndex k = 0; k + 1 < n; ++k)
        for (MeshIndex j = 0; j + 1 < n; ++j)
            for (MeshIndex i = 0; i + 1 < n; ++i) {
                const MeshIndex a = i + n * j + n * n * k, c = a + n + 1, e = a + n * n, g = e + n + 1;
                if (tetrahedra) {
                    b.tetra(a, a + 1, c, g);
                    b.tetra(a, c, a + n, g);
                    b.tetra(a, a + n, e + n, g);
                    b.tetra(a, e + n, e, g);
                    b.tetra(a, e, e + 1, g);
                    b.tetra(a, e + 1, a + 1, g);
                } else
                    b.cell({{a, a + n, c, a + 1},
                            {e, e + 1, g, e + n},
                            {a, a + 1, e + 1, e},
                            {a + 1, c, g, e + 1},
                            {c, a + n, e + n, g},
                            {a + n, a, e, e + n}});
            }
    return b.finish();
}
// Создаёт призматическую сетку для проверки геометрии и связности.
Mesh prism(bool concave, unsigned layers = 4) {
    Builder b;
    const std::vector<double> x =
        concave ? std::vector<double>{0, 2, 2, 1, 1, 0} : std::vector<double>{0, 2, 0};
    const std::vector<double> y =
        concave ? std::vector<double>{0, 0, 1, 1, 2, 2} : std::vector<double>{0, 0, 2};
    const MeshIndex n = static_cast<MeshIndex>(x.size());
    for (unsigned layer = 0; layer < layers; ++layer)
        for (std::size_t i = 0; i < x.size(); ++i)
            b.node(x[i], y[i], layer);
    for (MeshIndex layer = 0; layer + 1 < layers; ++layer) {
        Polygon top, bottom;
        for (MeshIndex i = 0; i < n; ++i) {
            top.push_back((layer + 1) * n + i);
            bottom.push_back(layer * n + n - 1 - i);
        }
        std::vector<Polygon> shell = {bottom, top};
        for (MeshIndex i = 0; i < n; ++i)
            shell.push_back(
                {layer * n + i, layer * n + (i + 1) % n, (layer + 1) * n + (i + 1) % n, (layer + 1) * n + i});
        b.cell(shell);
    }
    return b.finish();
}
// Вычисляет аналитические значения поля на узлах тестовой сетки.
std::vector<double> values(const Mesh& mesh, ReconstructionIndex components, bool affine) {
    const ReconstructionIndex n = mesh.view().node_count;
    std::vector<double> v(static_cast<std::size_t>(n * components));
    for (ReconstructionIndex c = 0; c < components; ++c)
        for (ReconstructionIndex i = 0; i < n; ++i) {
            const double x = mesh.data().x[i], y = mesh.data().y[i], z = mesh.data().z[i];
            v[static_cast<std::size_t>(c * n + i)] =
                affine ? 2 + static_cast<double>(c) * (0.7 * x - 0.3 * y + 1.2 * z)
                       : std::sin(x + 0.7 * y - 0.4 * z + static_cast<double>(c));
        }
    return v;
}
// На линейном поле известен точный прирост вдоль ребра. Сравниваем обе восстановленные стороны.
void affine_test(const Mesh& mesh, bool expect_full,
                 ReconstructionScheme scheme = ReconstructionScheme::Ebr3) {
    ReconstructionOptions options;
    options.scheme = scheme;
    const ReconstructionStencil s = prepare_reconstruction(mesh, options);
    const std::vector<double> input = values(mesh, 7, true);
    const TestResult r = reconstruct_all_sides(s, input, 7);
    const ReconstructionIndex n = mesh.view().node_count, e = mesh.view().edge_count;
    if (expect_full)
        require(s.incomplete_sides() < 2 * e, "No complete stencil sides");
    for (ReconstructionIndex c = 0; c < 7; ++c)
        for (ReconstructionIndex k = 0; k < e; ++k) {
            const double qi = input[c * n + mesh.edge_begin()[k]], qj = input[c * n + mesh.edge_end()[k]];
            near(r.left_increment2[c * e + k], qj - qi);
            near(r.right_increment2[c * e + k], qj - qi);
            near(qi + 0.5 * r.left_increment2[c * e + k], 0.5 * (qi + qj));
            near(qj - 0.5 * r.right_increment2[c * e + k], 0.5 * (qi + qj));
        }
}
// Сравнивает результаты двух реализаций или представлений с заданным допуском.
void same(const TestResult& a, const TestResult& b) {
    require(a.left_increment2.size() == b.left_increment2.size() &&
                a.right_increment2.size() == b.right_increment2.size(),
            "Result sizes differ");
    for (std::size_t i = 0; i < a.left_increment2.size(); ++i) {
        near(a.left_increment2[i], b.left_increment2[i]);
        near(a.right_increment2[i], b.right_increment2[i]);
    }
}

// Проверяет значения реконструкции возле концов одномерной сетки.
void endpoints(const Mesh& mesh, const std::vector<double>& input, ReconstructionIndex components,
               const TestResult& result) {
    const ReconstructionIndex n = mesh.view().node_count, e = mesh.view().edge_count;
    require(result.left_increment2.size() == e * components &&
                result.right_increment2.size() == e * components,
            "FirstOrder output size");
    for (ReconstructionIndex c = 0; c < components; ++c)
        for (ReconstructionIndex k = 0; k < e; ++k) {
            require(result.left_increment2[c * e + k] == 0 && result.right_increment2[c * e + k] == 0,
                    "FirstOrder increments must vanish");
            const double qi = input[c * n + mesh.edge_begin()[k]], qj = input[c * n + mesh.edge_end()[k]];
            require(qi + 0.5 * result.left_increment2[c * e + k] == qi, "FirstOrder left state changed");
            require(qj - 0.5 * result.right_increment2[c * e + k] == qj, "FirstOrder right state changed");
        }
}

// Проверяет сохранение нулевого поля при реконструкции.
void zeros(const TestResult& r) {
    for (double v : r.left_increment2)
        require(v == 0, "Nonzero left increment on constant field");
    for (double v : r.right_increment2)
        require(v == 0, "Nonzero right increment on constant field");
}
// Тестовый читатель поля, считающий число обращений к опорным значениям.
struct CountingReader {
    // Внешний счётчик обращений тестового чтения.
    unsigned* calls;
    // Считает обращения к полю и возвращает постоянное значение для проверки числа чтений.
    double operator()(ReconstructionIndex, ReconstructionIndex, ReconstructionIndex) const {
        ++*calls;
        return 7.0;
    }
};

// Читает значения локальных соседей для проверки нестандартной раскладки поля.
struct LocalNeighbours {
    // Ожидаемые номер строки и узел-владелец.
    ReconstructionIndex expected_row, owner;
    // Тестовые значения компонент по номерам узлов.
    std::map<ReconstructionIndex, std::vector<double>> values;
    // Проверяет локальность чтения опор и возвращает компоненту выбранного соседнего узла.
    double operator()(ReconstructionIndex row, ReconstructionIndex node,
                      ReconstructionIndex component) const {
        require(row == expected_row, "Reconstruction accessed a different edge side");
        require(node != owner, "Own vector must not be read through neighbour accessor");
        const auto found = values.find(node);
        require(found != values.end(), "Read outside the selected local stencil");
        return found->second.at(static_cast<std::size_t>(component));
    }
};

// Независимый эталон записан через восстановленные состояния на равномерной сетке.
// Независимая формула на равномерной линии служит эталоном для общей реализации на сетке.
double classical_weno5(const std::vector<double>& q, std::size_t i, double epsilon) {
    const double b0 = (13.0 / 12) * std::pow(q[i] - 2 * q[i + 1] + q[i + 2], 2) +
                      0.25 * std::pow(3 * q[i] - 4 * q[i + 1] + q[i + 2], 2);
    const double b1 =
        (13.0 / 12) * std::pow(q[i - 1] - 2 * q[i] + q[i + 1], 2) + 0.25 * std::pow(q[i - 1] - q[i + 1], 2);
    const double b2 = (13.0 / 12) * std::pow(q[i - 2] - 2 * q[i - 1] + q[i], 2) +
                      0.25 * std::pow(q[i - 2] - 4 * q[i - 1] + 3 * q[i], 2);
    const double a0 = 0.3 / std::pow(epsilon + b0, 2), a1 = 0.6 / std::pow(epsilon + b1, 2),
                 a2 = 0.1 / std::pow(epsilon + b2, 2);
    const double p0 = q[i] / 3 + 5 * q[i + 1] / 6 - q[i + 2] / 6;
    const double p1 = -q[i - 1] / 6 + 5 * q[i] / 6 + q[i + 1] / 3;
    const double p2 = q[i - 2] / 3 - 7 * q[i - 1] / 6 + 11 * q[i] / 6;
    return (a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2);
}
// Читает тестовые значения с привязкой к строке реконструкции.
struct RowReader {
    // Источник значений для чтения строки.
    const std::vector<double>* q;
    // Возвращает значение, зависящее от строки и компоненты, для проверки адресации шаблона.
    double operator()(ReconstructionIndex row, ReconstructionIndex node,
                      ReconstructionIndex component) const {
        return (*q)[static_cast<std::size_t>(node)] * static_cast<double>(row + component + 1) +
               static_cast<double>(row);
    }
};

#ifdef AERO_WITH_KOKKOS_GPU
using aero::gpu::checked;
// Сравнивает CPU и GPU при вычислении одной стороны шаблона.
void single_side_device_test(int device) {
    checked(aero::gpu::select(device));
    const Mesh m = line({0, 0.3, 1, 2, 4, 7});
    for (ReconstructionScheme scheme : std::vector<ReconstructionScheme>{
             ReconstructionScheme::FirstOrder, ReconstructionScheme::Ebr3, ReconstructionScheme::Ebr5,
             ReconstructionScheme::Weno3, ReconstructionScheme::Weno5}) {
        ReconstructionOptions o;
        o.scheme = scheme;
        const ReconstructionStencil s = prepare_reconstruction(m, o);
        ReconstructionContext context(s, ReconstructionBackend::Cuda, device);
        require(context.device() == device, "Wrong context device");
        for (unsigned side = 0; side < 2; ++side) {
            const ReconstructionIndex edge = 2, owner = s.data().row_owner[2 * edge + side];
            const std::vector<double> own = {4, -1, 2};
            auto neighbours = values(m, 3, false);
            // Значение узла-владельца в источнике нарочно непригодно: ядро должно взять переданный отдельно вектор этого узла.
            for (ReconstructionIndex c = 0; c < 3; ++c)
                neighbours[c * s.data().node_count + owner] = (std::numeric_limits<double>::quiet_NaN)();
            const ReconstructionFieldView reader = {neighbours.data(), 1, s.data().node_count};
            const auto expected = reconstruct_increment2(
                s, edge, side ? ReconstructionSide::Right : ReconstructionSide::Left, own, reader);
            std::vector<double> guarded(5, 12345);
            aero_test::DeviceBuffer<double> own_gpu(own.size()), neighbours_gpu(neighbours.size()),
                output_gpu(guarded.size());
            checked(
                aero::gpu::copy(own_gpu.data, own.data(), own.size() * sizeof(double), aero::gpu::Copy::ToDevice));
            checked(aero::gpu::copy(neighbours_gpu.data, neighbours.data(), neighbours.size() * sizeof(double),
                               aero::gpu::Copy::ToDevice));
            checked(aero::gpu::copy(output_gpu.data, guarded.data(), guarded.size() * sizeof(double),
                               aero::gpu::Copy::ToDevice));
            checked(static_cast<aero::gpu::Status>(
                test_reconstruction_device_one(context.device_view(), edge, side, own_gpu.data, 3,
                                             neighbours_gpu.data, output_gpu.data + 1, nullptr)));
            checked(aero::gpu::wait_all());
            checked(aero::gpu::copy(guarded.data(), output_gpu.data, guarded.size() * sizeof(double),
                               aero::gpu::Copy::ToHost));
            require(guarded.front() == 12345 && guarded.back() == 12345,
                    "CUDA wrote outside one side vector");
            for (std::size_t c = 0; c < 3; ++c)
                near(guarded[c + 1], expected[c]);
        }
    }
}
// Проверяет пакетную реконструкцию на доступных GPU.
void device_test(int device, ReconstructionScheme scheme = ReconstructionScheme::Ebr3, bool correction = true) {
    checked(aero::gpu::select(device));
    // Пятиузловым схемам нужны два слоя соседей. В середине этой сетки
    // сравниваем с CPU полный шаблон, а по краям того же поля — укороченный.
    const bool two_layers = reconstruction_scheme(scheme).layers == 2;
    const Mesh mesh = two_layers ? plane(true, 7, 7) : plane(true);
    ReconstructionOptions options;
    options.scheme = scheme;
    options.mpweno_ad_corr = correction;
    const ReconstructionStencil s = prepare_reconstruction(mesh, options);
    if (two_layers)
        require(s.incomplete_sides() < 2 * s.data().edge_count, "GPU fixture has no full stencils");
    TestReconstructor gpu(s, ReconstructionBackend::Cuda, device);
    const std::vector<double> v = values(mesh, 5, false);
    const TestResult expected = reconstruct_all_sides(s, v, 5);
    aero_test::DeviceBuffer<double> input(v.size()), left(expected.left_increment2.size()),
        right(expected.right_increment2.size());
    aero::gpu::Stream stream = nullptr;
    checked(aero::gpu::create_stream(&stream));
    try {
        checked(aero::gpu::copy(input.data, v.data(), v.size() * sizeof(double), aero::gpu::Copy::ToDevice));
        gpu.reconstruct_device(input.data, 5, left.data, right.data, stream);
        checked(aero::gpu::wait(stream));
        TestResult actual = expected;
        checked(aero::gpu::copy(actual.left_increment2.data(), left.data,
                           actual.left_increment2.size() * sizeof(double), aero::gpu::Copy::ToHost));
        checked(aero::gpu::copy(actual.right_increment2.data(), right.data,
                           actual.right_increment2.size() * sizeof(double), aero::gpu::Copy::ToHost));
        same(actual, expected);
        if (scheme == ReconstructionScheme::FirstOrder)
            endpoints(mesh, v, 5, actual);
    } catch (...) {
        aero::gpu::destroy_stream(stream);
        throw;
    }
    checked(aero::gpu::destroy_stream(stream));
}
#endif
}

namespace {
// Проверки доступа к одной стороне ребра и размещения данных реконструкции.
void register_side_access() {
    add("single side uses explicit owner vector and only local neighbours", [] {
        const Mesh m = line({0, 0.3, 1, 2, 4, 7});
        for (ReconstructionScheme scheme : std::vector<ReconstructionScheme>{
                 ReconstructionScheme::FirstOrder, ReconstructionScheme::Ebr3, ReconstructionScheme::Ebr5,
                 ReconstructionScheme::Weno3, ReconstructionScheme::Weno5}) {
            ReconstructionOptions o;
            o.scheme = scheme;
            const ReconstructionStencil s = prepare_reconstruction(m, o);
            for (ReconstructionIndex edge : std::vector<ReconstructionIndex>{0, 2, 4})
                for (unsigned side = 0; side < 2; ++side) {
                    const ReconstructionIndex row = 2 * edge + side, owner = s.data().row_owner[row];
                    const std::vector<double> own = {4, -1, 2};
                    auto field = values(m, 3, false);
                    LocalNeighbours reader;
                    reader.expected_row = row;
                    reader.owner = owner;
                    std::vector<ReconstructionIndex> ids;
                    for (ReconstructionIndex k = s.data().offsets[row]; k < s.data().offsets[row + 1]; ++k)
                        ids.push_back(s.data().nodes[k]);
                    if (!s.data().weno_order.empty())
                        for (ReconstructionIndex k = s.data().slope_offsets[4 * row];
                             k < s.data().slope_offsets[4 * row + 4]; ++k)
                            ids.push_back(s.data().slope_nodes[k]);
                    for (ReconstructionIndex n : ids)
                        if (n != owner) {
                            std::vector<double>& q = reader.values[n];
                            q.resize(3);
                            for (ReconstructionIndex c = 0; c < 3; ++c)
                                q[c] = field[c * s.data().node_count + n];
                        }
                    for (ReconstructionIndex c = 0; c < 3; ++c)
                        field[c * s.data().node_count + owner] = own[c];
                    const TestResult expected = reconstruct_all_sides(s, field, 3);
                    const ReconstructionSide direction =
                        side ? ReconstructionSide::Right : ReconstructionSide::Left;
                    std::vector<double> guarded(5, 12345);
                    require(reconstruct_increment2(s.view(), edge, direction, own.data(), 3, reader,
                                                   guarded.data() + 1) == ReconstructionStatus::Success,
                            "Single side failed");
                    require(guarded.front() == 12345 && guarded.back() == 12345,
                            "Wrote beyond the selected side vector");
                    const auto checked = reconstruct_increment2(s, edge, direction, own, reader);
                    const ReconstructionSideInput<LocalNeighbours> input = {s.view(), row, own.data(), 1,
                                                                            reader};
                    std::vector<double> direct(3);
                    if (scheme == ReconstructionScheme::Weno3 || scheme == ReconstructionScheme::Weno5)
                        reconstruct_weno_increment2(input, 3, direct.data());
                    else
                        reconstruct_ebr_increment2(input, 3, direct.data());
                    for (ReconstructionIndex c = 0; c < 3; ++c) {
                        near(guarded[c + 1],
                             (side ? expected.right_increment2
                                   : expected.left_increment2)[c * s.data().edge_count + edge]);
                        near(checked[c], guarded[c + 1]);
                        near(direct[c], guarded[c + 1]);
                    }
                }
        }
    });
    add("single-side invalid arguments leave output untouched", [] {
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2}));
        const ReconstructionFieldView reader = {nullptr, 1, 3};
        double q = 1, out = 77;
        require(reconstruct_increment2(s.view(), 2, ReconstructionSide::Left, &q, 1, reader, &out) ==
                    ReconstructionStatus::InvalidEdge,
                "Edge check");
        require(reconstruct_increment2(s.view(), 0, static_cast<ReconstructionSide>(7), &q, 1, reader,
                                       &out) == ReconstructionStatus::InvalidSide,
                "Side check");
        require(reconstruct_increment2(s.view(), 0, ReconstructionSide::Left, &q, 0, reader, &out) ==
                    ReconstructionStatus::InvalidLayout,
                "Empty vector check");
        require(reconstruct_increment2(s.view(), 0, ReconstructionSide::Left, &q, 1, reader, &out, 0) ==
                    ReconstructionStatus::InvalidLayout,
                "Stride check");
        require(reconstruct_increment2(s.view(), 0, ReconstructionSide::Left, &q, 3, reader, &out,
                                       (std::numeric_limits<ReconstructionIndex>::max)()) ==
                    ReconstructionStatus::InvalidLayout,
                "Stride overflow check");
        require(reconstruct_increment2(s.view(), 0, ReconstructionSide::Left, nullptr, 1, reader, &out) ==
                    ReconstructionStatus::NullPointer,
                "Owner pointer check");
        require(reconstruct_increment2(s.view(), 0, ReconstructionSide::Left, &q, 1, reader, nullptr) ==
                    ReconstructionStatus::NullPointer,
                "Output pointer check");
        require(out == 77, "Invalid call wrote output");
        rejects(
            [&] { reconstruct_increment2(s, 0, ReconstructionSide::Left, std::vector<double>(), reader); });
    });
    add("FirstOrder single side does not read fields", [] {
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::FirstOrder;
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1}), o);
        const double own[2] = {(std::numeric_limits<double>::quiet_NaN)(), 0};
        double out[2] = {9, 9};
        unsigned calls = 0;
        const CountingReader reader = {&calls};
        require(reconstruct_increment2(s.view(), 0, ReconstructionSide::Right, own, 2, reader, out) ==
                    ReconstructionStatus::Success,
                "FirstOrder single-side call");
        require(calls == 0 && out[0] == 0 && out[1] == 0, "FirstOrder read input or returned nonzero");
    });
    add("common context owns host geometry and rejects device view on CPU", [] {
        ReconstructionContext context(prepare_reconstruction(line({0, 1, 2})), ReconstructionBackend::Cpu);
        require(context.host_view().edge_count == 2 && context.device() == -1, "Host geometry lifetime");
        rejects([&] { context.device_view(); });
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::Weno5;
        rejects([&] { prepare_ebr(line({0, 1, 2}), o); });
    });
}

// Проверки весов WENO и выбора опорных точек на луче.
void register_weno_formulas() {
    add("MPWENO5 nonlinear weights preserve weak signals at either sign", [] {
        // Левая сторона этого шаблона использует WENO. Уменьшение амплитуды
        // должно уменьшать приращение во столько же раз при любом знаке поля.
        for (bool correction : {false, true}) {
            const double expected = correction ? .39057843304740597 : .38263893305467572;
            for (double a : {1.e-140, -1.e-12, 1.e-4, 1., 1.e100}) {
                double left, right;
                mpweno5::pair(-.1*a, .3*a, .4*a, .7*a, a, 1., correction, left, right);
                near(left/a, expected, 2.e-14);
                near(right/a, .48, 2.e-14);
            }
            // Любая нормированная сумма одинаковых кандидатов даёт сам кандидат,
            // включая случай двух совершенно гладких подшаблонов.
            for (double beta : {1.e-300, 1.e-12, 1.e200})
                near(mpweno5::weighted_value({0., 0., beta}, {.75,.75,.75}, correction), .75);
        }
    });
    add("MPWENO5 standalone reconstruction, independent left and right weights", [] {
        // У центрального ребра пять последовательных наклонов: −0.1, 0.3, 0.4, 0.7, 1.
        // Левая сторона выбирает нелинейные веса, правая — линейный полином.
        // Эталоны получены отдельной исходной формулой MPWENO5 при alpha = 1.
        const Mesh mesh = line({0, 1, 2, 3, 4, 5});
        const std::vector<double> state = {0., -.1, .2, .6, 1.3, 2.3};
        for (bool correction : {false, true}) {
            ReconstructionOptions options;
            options.scheme = ReconstructionScheme::MPWENO5;
            options.mp5_alpha = 1.;
            options.mpweno_ad_corr = correction;
            const auto stencil = prepare_reconstruction(mesh, options);
            const auto traces = reconstruct_all_sides(stencil, state, 1);
            const double nonlinear = correction ? .39057843304740597 : .38263893305467572;
            near(traces.left_increment2[2], nonlinear, 2.e-14);
            near(traces.right_increment2[2], .48, 2.e-14);

            // Разворачиваем само поле: теперь нелинейная сторона справа, меняется знак наклона.
            const auto reversed = reconstruct_all_sides(stencil, std::vector<double>(state.rbegin(), state.rend()), 1);
            near(reversed.left_increment2[2], -.48, 2.e-14);
            near(reversed.right_increment2[2], -nonlinear, 2.e-14);
        }
    });
    add("MPWENO5 preserves a constant and clips an isolated jump", [] {
        for (bool correction : {false, true}) {
            double left, right;
            mpweno5::pair(0., 0., 0., 0., 0., 1., correction, left, right);
            near(left, 0., 0.);
            near(right, 0., 0.);
            mpweno5::pair(1., 1., 1., 1., 1., 1., correction, left, right);
            near(left, 1.);
            near(right, 1.);
            mpweno5::pair(0., 0., 1., 0., 0., 1., correction, left, right);
            near(left, 0., 0.);
            near(right, 0., 0.);
        }
    });
    add("WENO3 hand-computed nonlinear weights and signs", [] {
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::Weno3;
        const ReconstructionStencil s = prepare_weno(line({0, 1, 2, 3}), o);
        const TestResult r = reconstruct_all_sides(s, {0, 1, 3, 7}, 1);
        near(r.left_increment2[1], 10.0 / 9);
        near(r.right_increment2[1], 68.0 / 33);
        require(std::abs(r.left_increment2[1] - 5.0 / 3) > 0.1,
                "Used optimal linear weights instead of WENO");
    });
    add("WENO5 independent classical state formula, both directions", [] {
        const std::vector<double> q = {2, -3, 7, 1, 9, -4};
        for (double h : std::vector<double>{0.03, 1, 7})
            for (double epsilon : std::vector<double>{1.e-40, 0.7}) {
                ReconstructionOptions o;
                o.scheme = ReconstructionScheme::Weno5;
                o.weno_epsilon = epsilon;
                const ReconstructionStencil s = prepare_weno(line({0, h, 2 * h, 3 * h, 4 * h, 5 * h}), o);
                const TestResult r = reconstruct_all_sides(s, q, 1);
                std::vector<double> reversed(q.rbegin(), q.rend());
                near(r.left_increment2[2], 2 * (classical_weno5(q, 2, epsilon * h * h) - q[2]));
                near(r.right_increment2[2], 2 * (q[3] - classical_weno5(reversed, 2, epsilon * h * h)));
                near(s.data().far_distance[4], 2 * h);
                near(s.data().outer_distance[4], h);
            }
    });
    add("WENO preparation rejects non-WENO schemes", [] {
        for (ReconstructionScheme scheme : std::vector<ReconstructionScheme>{
                 ReconstructionScheme::FirstOrder, ReconstructionScheme::Ebr3, ReconstructionScheme::Ebr5,
                 static_cast<ReconstructionScheme>(999)}) {
            ReconstructionOptions o;
            o.scheme = scheme;
            rejects([&] { prepare_weno(line({0, 1, 2}), o); });
        }
    });
    add("WENO5 virtual points use divided differences of second-layer values", [] {
        Builder b;
        b.d.dimension = 2;
        b.node(0, 0);
        b.node(1, 0);
        b.node(-1, -1);
        b.node(-1, 1);
        b.node(-3, -1);
        b.node(-2, 1);
        b.node(2, 0);
        b.polygon({0, 3, 2});
        b.polygon({2, 3, 5});
        b.polygon({2, 5, 4});
        b.polygon({0, 1, 3});
        b.polygon({0, 2, 1});
        b.polygon({1, 6, 3});
        b.polygon({1, 2, 6});
        const Mesh m = b.finish();
        ReconstructionIndex e = 0;
        while (!(m.edge_begin()[e] == 0 && m.edge_end()[e] == 1))
            ++e;
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::Weno5;
        const ReconstructionStencil s = prepare_reconstruction(m, o);
        require(s.data().side_scheme[2 * e] == ReconstructionSideScheme::Weno5,
                "No full virtual-point WENO5 stencil");
        near(s.data().outer_distance[2 * e], 1);
        near(s.data().far_distance[2 * e], 2.5);
        // При h = 1 обе интерполированные величины равны 8, наклоны D = (6, −6, 12, 0).
        const double a = 0.3 / (48 * 48), bweight = 0.6 / (156 * 156), c = 0.1 / (120 * 120);
        near(reconstruct_all_sides(s, {2, 8, 5, 11, 17, -1, 20}, 1).left_increment2[e],
             (4 * a + 2 * bweight - 10 * c) / (a + bweight + c));
        o.weno_epsilon = 1.e20;
        near(reconstruct_all_sides(prepare_reconstruction(m, o), {2, 8, 5, 11, 17, -1, 20}, 1)
                 .left_increment2[e],
             7.0 / 5);
    });
    add("WENO nonuniform metric factors match equivalent uniform slopes", [] {
        const Mesh m = line({-4, -2, 0, 1, 4, 8});
        const std::vector<double> q = {-4, 0, 2, 4, 13, 29};
        // Для ребра 0 → 1: D0 = 2, ближний/дальний слева = 1/2, справа = 3/4.
        // Тот же набор наклонов на равномерной сетке должен дать ту же реконструкцию.
        const Mesh uniform = line({-2, -1, 0, 1, 2, 3});
        const std::vector<double> equivalent = {-1, 1, 2, 4, 7, 11};
        for (ReconstructionScheme scheme :
             std::vector<ReconstructionScheme>{ReconstructionScheme::Weno3, ReconstructionScheme::Weno5}) {
            ReconstructionOptions o;
            o.scheme = scheme;
            o.weno_epsilon = 0.25;
            const TestResult actual = reconstruct_all_sides(prepare_reconstruction(m, o), q, 1),
                             expected =
                                 reconstruct_all_sides(prepare_reconstruction(uniform, o), equivalent, 1);
            near(actual.left_increment2[2], expected.left_increment2[2]);
            near(actual.right_increment2[2], expected.right_increment2[2]);
        }
    });
    add("WENO layer search selects farthest positive intersection", [] {
        Builder b;
        b.d.dimension = 2;
        b.node(-3, -1);
        b.node(-2, -1);
        b.node(-2, 1);
        b.node(0, 1);
        b.node(0, 0);
        b.node(1, 0);
        b.node(1, 2);
        b.node(-3, 2);
        b.polygon({0, 1, 2, 3, 4, 5, 6, 7});
        // Заполняем выемку ячейкой: луч идёт через внутреннюю грань на расстоянии 2
        // и достигает допустимой внешней опоры на расстоянии 3.
        b.node(0, -1);
        b.polygon({1, 8, 4, 3, 2});
        const Mesh m = b.finish();
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::Weno3;
        const ReconstructionStencil s = prepare_reconstruction(m, o);
        ReconstructionIndex e = 0;
        while (!(m.edge_begin()[e] == 4 && m.edge_end()[e] == 5))
            ++e;
        require(s.data().side_scheme[2 * e] == ReconstructionSideScheme::Weno3,
                "Missing WENO first-layer hit");
        near(s.data().outer_distance[2 * e], 3); // Пересечения на расстояниях 2 и 3.
        o.scheme = ReconstructionScheme::Ebr3;
        const auto linear = prepare_reconstruction(m, o);
        require(linear.data().offsets == s.data().offsets && linear.data().nodes == s.data().nodes &&
                    linear.data().weights == s.data().weights && linear.data().complete == s.data().complete &&
                    linear.data().outer_distance == s.data().outer_distance,
                "Different geometry or linear coefficients for EBR3 and WENO3");
    });
}

// Общие задачи для третьего и пятого порядков WENO на разных сетках.
void register_weno_schemes() {
    for (ReconstructionScheme scheme :
         std::vector<ReconstructionScheme>{ReconstructionScheme::Weno3, ReconstructionScheme::Weno5}) {
        const std::string name = scheme == ReconstructionScheme::Weno3 ? "WENO3" : "WENO5";
        add(name + " affine fields on nonuniform 1D, simplices and polytopes", [scheme] {
            for (const Mesh& m :
                 std::vector<Mesh>{line({0, 0.1, 0.7, 2, 4, 5, 9}), plane(true, 7, 7), plane(false, 7, 7),
                                   volume(true, 6), volume(false, 6), prism(false, 6), prism(true, 6)})
                affine_test(m, true, scheme);
        });
        add(name + " constants, offset invariance and reversal", [scheme] {
            ReconstructionOptions o;
            o.scheme = scheme;
            const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2, 3, 4, 5}), o);
            zeros(reconstruct_all_sides(s, std::vector<double>(6, 1.e200), 1));
            const std::vector<double> q = {2, -3, 7, 1, 9, -4};
            std::vector<double> shifted = q, rev(q.rbegin(), q.rend());
            for (double& v : shifted)
                v += 281474976710656.0;
            const TestResult r = reconstruct_all_sides(s, q, 1), back = reconstruct_all_sides(s, rev, 1);
            same(r, reconstruct_all_sides(s, shifted, 1));
            for (std::size_t e = 0; e < 5; ++e) {
                near(back.left_increment2[e], -r.right_increment2[4 - e]);
                near(back.right_increment2[e], -r.left_increment2[4 - e]);
            }
        });
        add(name + " WENO slope scaling at binary64 limits", [scheme] {
            const unsigned order = scheme == ReconstructionScheme::Weno5 ? 5 : 3;
            // При постоянном наклоне все кандидаты дают одну производную независимо от весов.
            for (double x : {(std::numeric_limits<double>::denorm_min)(), (std::numeric_limits<double>::min)(),
                             (std::numeric_limits<double>::max)() / 2}) {
                const double slopes[] = {x, x, x, x};
                near(weno_increment2(order, slopes, x) / x, 1);
            }
        });
        add(name + " jump suppression and extreme finite amplitudes", [scheme] {
            ReconstructionOptions o;
            o.scheme = scheme;
            const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2, 3, 4, 5}), o);
            const TestResult jump = reconstruct_all_sides(s, {0, 0, 0, 1, 1, 1}, 1);
            near(jump.left_increment2[2], 0, 1.e-12);
            near(jump.right_increment2[2], 0, 1.e-12);
            for (double amplitude : std::vector<double>{1.e-150, 1.e150}) {
                const TestResult r = reconstruct_all_sides(
                    s, {0, amplitude, 2 * amplitude, 3 * amplitude, 4 * amplitude, 5 * amplitude}, 1);
                for (double v : r.left_increment2)
                    near(v / amplitude, 1);
                for (double v : r.right_increment2)
                    near(v / amplitude, 1);
            }
        });
        add(name + " downgrade metadata, strict mode, empty mesh and epsilon validation", [scheme] {
            ReconstructionOptions o;
            o.scheme = scheme;
            const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2, 3, 4, 5}), o);
            require(s.data().side_scheme[0] == ReconstructionSideScheme::EdgeAverage,
                    "Missing boundary closure");
            require(s.data().side_scheme[2] == ReconstructionSideScheme::Weno3,
                    "Missing WENO3 boundary downgrade");
            for (std::size_t r = 0; r < 10; ++r) {
                const unsigned desired = scheme == ReconstructionScheme::Weno5 ? 5 : 3;
                require((s.data().complete[r] != 0) == (s.data().weno_order[r] == desired),
                        "Incorrect complete mask");
            }
            require(s.data().slope_offsets.size() == 41, "Four derivative rows per side required");
            o.require_complete_stencil = true;
            rejects([&] { prepare_reconstruction(line({0, 1, 2, 3, 4, 5}), o); });
            require(reconstruct_all_sides(prepare_reconstruction(Mesh(), o), {}, 1).left_increment2.empty(),
                    "Empty WENO output");
            for (double epsilon : std::vector<double>{0, -1, (std::numeric_limits<double>::infinity)(),
                                                      (std::numeric_limits<double>::quiet_NaN)()}) {
                o.weno_epsilon = epsilon;
                rejects([&] { prepare_reconstruction(Mesh(), o); });
            }
        });
        add(name + " large epsilon recovers optimal linear uniform stencil", [scheme] {
            const Mesh m = line({0, 1, 2, 3, 4, 5});
            const std::vector<double> q = {2, -3, 7, 1, 9, -4};
            ReconstructionOptions o;
            o.scheme = scheme;
            o.weno_epsilon = 1.e20;
            const TestResult nonlinear = reconstruct_all_sides(prepare_reconstruction(m, o), q, 1);
            o.scheme = scheme == ReconstructionScheme::Weno5 ? ReconstructionScheme::Ebr5
                                                             : ReconstructionScheme::Ebr3;
            same(nonlinear, reconstruct_all_sides(prepare_reconstruction(m, o), q, 1));
        });
        add(name + " smooth uniform-grid flux derivative convergence", [scheme] {
            ReconstructionOptions o;
            o.scheme = scheme;
            double previous = 0;
            for (unsigned n : std::vector<unsigned>{16, 32, 64}) {
                const double h = 1.0 / n;
                std::vector<double> x, q;
                for (unsigned i = 0; i < n + 7; ++i) {
                    x.push_back((static_cast<double>(i) - 3) * h);
                    q.push_back(std::exp(x.back()));
                }
                const ReconstructionStencil s = prepare_reconstruction(line(x), o);
                const TestResult r = reconstruct_all_sides(s, q, 1);
                double error = 0;
                for (unsigned i = 3; i < n + 3; ++i) {
                    require(s.data().complete[2 * i] && s.data().complete[2 * (i - 1)],
                            "Downgraded convergence stencil");
                    const double derivative =
                        (q[i] + r.left_increment2[i] / 2 - q[i - 1] - r.left_increment2[i - 1] / 2) / h;
                    error += std::pow(derivative - q[i], 2);
                }
                error = std::sqrt(error / n);
                if (previous > 0) {
                    const double ratio = previous / error;
                    require(scheme == ReconstructionScheme::Weno5 ? (ratio > 27 && ratio < 37)
                                                                  : (ratio > 7 && ratio < 9),
                            "Unexpected WENO convergence ratio: " + std::to_string(ratio));
                }
                previous = error;
            }
        });
        add(name + " row-dependent external reader applies before nonlinear weights", [scheme] {
            const Mesh m = line({0, 1, 2, 3, 4, 5});
            const std::vector<double> q = {2, -3, 7, 1, 9, -4};
            ReconstructionOptions o;
            o.scheme = scheme;
            o.weno_epsilon = 0.7;
            const ReconstructionStencil s = prepare_reconstruction(m, o);
            const RowReader reader = {&q};
            for (ReconstructionIndex row = 0; row < 10; ++row) {
                std::vector<double> transformed(6);
                for (ReconstructionIndex n = 0; n < 6; ++n)
                    transformed[static_cast<std::size_t>(n)] = reader(row, n, 2);
                const TestResult r = reconstruct_all_sides(s, transformed, 1);
                near(evaluate_test_side(s.view(), row, 2, reader),
                     (row % 2 ? r.right_increment2 : r.left_increment2)[static_cast<std::size_t>(row / 2)]);
            }
        });
    }
}

// Проверки коэффициентов EBR5, точности и понижения порядка.
void register_ebr5() {
    add("scheme selection defaults to EBR3 and rejects invalid values", [] {
        require(prepare_reconstruction(line({0, 1, 2})).data().scheme == ReconstructionScheme::Ebr3,
                "Default must remain EBR3");
        ReconstructionOptions o;
        o.scheme = static_cast<ReconstructionScheme>(999);
        rejects([&] { prepare_reconstruction(Mesh(), o); });
    });
    add("EBR5 uniform-grid coefficients and second-layer distances", [] {
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::Ebr5;
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2, 3, 4, 5}), o);
        require(s.data().scheme == ReconstructionScheme::Ebr5, "Wrong requested scheme");
        require(s.data().side_scheme[4] == ReconstructionSideScheme::Ebr5 &&
                    s.data().side_scheme[5] == ReconstructionSideScheme::Ebr5,
                "Incomplete central EBR5 stencil");
        const double left[] = {1.0 / 15, -13.0 / 30, -13.0 / 30, 9.0 / 10, -1.0 / 10, 0};
        const double right[] = {0, 1.0 / 10, -9.0 / 10, 13.0 / 30, 13.0 / 30, -1.0 / 15};
        for (unsigned k = 0; k < 6; ++k) {
            std::vector<double> impulse(6, 0);
            impulse[k] = 1;
            const TestResult r = reconstruct_all_sides(s, impulse, 1);
            near(r.left_increment2[2], left[k]);
            near(r.right_increment2[2], right[k]);
        }
        near(s.data().outer_distance[4], 1);
        near(s.data().far_distance[4], 2);
        near(s.data().outer_distance[5], 1);
        near(s.data().far_distance[5], 2);
    });
    add("EBR5 nonuniform 1D metric factors", [] {
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::Ebr5;
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 3, 6, 10, 15}), o);
        const TestResult r = reconstruct_all_sides(s, {2, -3, 7, 1, 9, -4}, 1);
        near(r.left_increment2[2], 11.0 / 10);
        near(r.right_increment2[2], -179.0 / 50);
        near(s.data().far_distance[4], 3);
        near(s.data().far_distance[5], 9);
    });
    add("EBR5 shares the WENO5 virtual points and optimal linear coefficients", [] {
        Builder b;
        b.d.dimension = 2;
        b.node(0, 0);
        b.node(1, 0);
        b.node(-1, -1);
        b.node(-1, 1);
        b.node(-3, -1);
        b.node(-2, 1);
        b.node(2, 0);
        b.polygon({0, 3, 2});
        b.polygon({2, 3, 5});
        b.polygon({2, 5, 4});
        b.polygon({0, 1, 3});
        b.polygon({0, 2, 1});
        b.polygon({1, 6, 3});
        b.polygon({1, 2, 6});
        const Mesh m = b.finish();
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::Ebr5;
        const ReconstructionStencil s = prepare_reconstruction(m, o);
        ReconstructionIndex e = 0;
        while (!(m.edge_begin()[e] == 0 && m.edge_end()[e] == 1))
            ++e;
        require(s.data().side_scheme[2 * e] == ReconstructionSideScheme::Ebr5,
                "Missing virtual-point EBR5 stencil");
        // В обоих пересечениях слева значение равно 8. Дальний наклон равен нулю.
        near(reconstruct_all_sides(s, {2, 8, 5, 11, 17, -1, 20}, 1).left_increment2[e], 7.0 / 5);
        o.scheme = ReconstructionScheme::Weno5;
        const auto nonlinear = prepare_reconstruction(m, o);
        require(s.data().offsets == nonlinear.data().offsets && s.data().nodes == nonlinear.data().nodes &&
                    s.data().weights == nonlinear.data().weights && s.data().complete == nonlinear.data().complete &&
                    s.data().outer_distance == nonlinear.data().outer_distance &&
                    s.data().far_distance == nonlinear.data().far_distance,
                "Different geometry or linear coefficients for EBR5 and WENO5");
        near(s.data().far_distance[2 * e], 2.5);
        require(s.data().side_scheme[2 * e + 1] == ReconstructionSideScheme::Ebr3,
                "Expected right-side downgrade only");
    });
    add("EBR5 affine fields in 1D/2D/3D", [] {
        for (const Mesh& m :
             std::vector<Mesh>{line({0, 0.1, 0.7, 2, 4, 5, 9}), plane(true, 7, 7), plane(false, 7, 7),
                               volume(true, 6), volume(false, 6), prism(false, 6), prism(true, 6)})
            affine_test(m, true, ReconstructionScheme::Ebr5);
    });
    add("EBR5 downgrade metadata and strict mode", [] {
        const Mesh m = line({0, 1, 2, 3, 4, 5});
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::Ebr5;
        const ReconstructionStencil s = prepare_reconstruction(m, o), third = prepare_reconstruction(m);
        const auto r = reconstruct_all_sides(s, {0, 1, 4, 9, 16, 25}, 1),
                   r3 = reconstruct_all_sides(third, {0, 1, 4, 9, 16, 25}, 1);
        unsigned incomplete = 0;
        for (ReconstructionIndex row = 0; row < 2 * s.data().edge_count; ++row) {
            const ReconstructionSideScheme actual = s.data().side_scheme[row];
            require((s.data().complete[row] != 0) == (actual == ReconstructionSideScheme::Ebr5),
                    "Inconsistent EBR5 metadata");
            if (actual != ReconstructionSideScheme::Ebr5) {
                ++incomplete;
                near(row % 2 ? r.right_increment2[row / 2] : r.left_increment2[row / 2],
                     row % 2 ? r3.right_increment2[row / 2] : r3.left_increment2[row / 2]);
            }
        }
        require(s.incomplete_sides() == incomplete, "Wrong downgrade count");
        require(s.data().side_scheme[0] == ReconstructionSideScheme::EdgeAverage,
                "Boundary closure metadata");
        require(s.data().side_scheme[2] == ReconstructionSideScheme::Ebr3, "Near-boundary EBR3 metadata");
        o.require_complete_stencil = true;
        rejects([&] { prepare_reconstruction(m, o); });
        const ReconstructionStencil empty = prepare_reconstruction(Mesh(), o);
        require(empty.data().side_scheme.empty() && empty.incomplete_sides() == 0, "Empty EBR5 stencil");
    });
    for (auto scheme : {ReconstructionScheme::Ebr5, ReconstructionScheme::Mp5})
    add(std::string(reconstruction_scheme(scheme).name) + " fifth-order uniform-grid flux derivative", [scheme] {
        ReconstructionOptions o;
        o.scheme = scheme;
        const double frequency = 2 * std::acos(-1.0);
        double previous = 0;
        for (unsigned n : std::vector<unsigned>{16, 32, 64}) {
            const double h = 1.0 / n;
            std::vector<double> x, u;
            for (unsigned i = 0; i < n + 7; ++i) {
                x.push_back((static_cast<double>(i) - 3) * h);
                u.push_back(std::sin(frequency * x.back()));
            }
            const ReconstructionStencil s = prepare_reconstruction(line(x), o);
            const TestResult r = reconstruct_all_sides(s, u, 1);
            double error = 0;
            for (unsigned i = 3; i < n + 3; ++i) {
                require(s.data().complete[2 * i] && s.data().complete[2 * (i - 1)],
                        "Convergence test used a downgrade");
                const double derivative =
                    (u[i] + 0.5 * r.left_increment2[i] - u[i - 1] - 0.5 * r.left_increment2[i - 1]) / h;
                const double residual = derivative - frequency * std::cos(frequency * x[i]);
                error += residual * residual;
            }
            error = std::sqrt(error / n);
            if (previous > 0)
                require(previous / error > 28 && previous / error < 35, "Expected fifth-order convergence");
            previous = error;
        }
    });
    add("EBR5 sign under reversal and constant-offset invariance", [] {
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::Ebr5;
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2, 3, 4, 5}), o);
        const std::vector<double> v = {2, -3, 7, 1, 9, -4};
        const TestResult r = reconstruct_all_sides(s, v, 1);
        const TestResult reversed = reconstruct_all_sides(prepare_reconstruction(line({5, 4, 3, 2, 1, 0}), o),
                                                          {-4, 9, 1, 7, -3, 2}, 1);
        for (unsigned e = 0; e < 5; ++e) {
            near(reversed.left_increment2[4 - e], -r.right_increment2[e]);
            near(reversed.right_increment2[4 - e], -r.left_increment2[e]);
        }
        std::vector<double> shifted = v;
        for (double& value : shifted)
            value += 281474976710656.0;
        same(r, reconstruct_all_sides(s, shifted, 1));
        zeros(reconstruct_all_sides(s, std::vector<double>(6, 1.e100), 1));
    });
    add("EBR5 rotation and scale invariance in 3D", [] {
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::Ebr5;
        const Mesh m = volume(true, 6);
        const ReconstructionStencil s = prepare_reconstruction(m, o);
        const auto input = values(m, 3, false);
        const TestResult expected = reconstruct_all_sides(s, input, 3);
        for (double scale : std::vector<double>{1.e-8, 1.e8}) {
            MeshData d = m.data();
            for (std::size_t i = 0; i < d.x.size(); ++i) {
                const double x = d.x[i], y = d.y[i], z = d.z[i];
                d.x[i] = scale * (3 + 0.6 * x - 0.8 * y);
                d.y[i] = scale * (-5 + 0.8 * x + 0.6 * y);
                d.z[i] = scale * (7 + z);
            }
            const ReconstructionStencil transformed = prepare_reconstruction(Mesh(d), o);
            require(transformed.data().side_scheme == s.data().side_scheme,
                    "EBR5 downgrade changed under rigid transform/scale");
            same(expected, reconstruct_all_sides(transformed, input, 3));
        }
    });
}

// Проверки значений первого порядка и сходимости на равномерной линии.
void register_first_order() {
    add("FirstOrder endpoints in 1D/2D/3D on CPU and Auto", [] {
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::FirstOrder;
        o.require_complete_stencil = true;
        for (const Mesh& m : std::vector<Mesh>{line({-2, 0, 0.1, 4}), plane(true), plane(false), volume(true),
                                               volume(false), prism(false), prism(true)}) {
            const ReconstructionStencil s = prepare_reconstruction(m, o);
            require(s.data().scheme == ReconstructionScheme::FirstOrder && s.incomplete_sides() == 0,
                    "FirstOrder has no incomplete sides");
            require(std::all_of(s.data().outer_distance.begin(), s.data().outer_distance.end(),
                                [](double d) { return d == 0; }),
                    "FirstOrder needs no outer points");
            TestReconstructor cpu(s, ReconstructionBackend::Cpu), automatic(s);
            for (ReconstructionIndex components : std::vector<ReconstructionIndex>{1, 7, 500, 2}) {
                const auto input = values(m, components, false);
                endpoints(m, input, components, reconstruct_all_sides(s, input, components));
                endpoints(m, input, components, cpu.reconstruct(input, components));
                endpoints(m, input, components, automatic.reconstruct(input, components));
            }
        }
    });
    add("FirstOrder preserves jumps at boundary and internal edges", [] {
        const Mesh m = line({0, 0.3, 1, 4});
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::FirstOrder;
        o.require_complete_stencil = true;
        const TestResult r = reconstruct_all_sides(prepare_reconstruction(m, o), {2, 2, -5, -5}, 1);
        endpoints(m, {2, 2, -5, -5}, 1, r);
        const Mesh single = line({0, 1});
        endpoints(single, {2, -5}, 1, reconstruct_all_sides(prepare_reconstruction(single, o), {2, -5}, 1));
    });
    add("FirstOrder does not require EBR geometric constructions", [] {
        MeshData d = volume(false).data();
        for (auto& sign : d.cell_face_orientation)
            sign = static_cast<std::int8_t>(-sign);
        const Mesh m(d);
        rejects([&] { prepare_reconstruction(m); });
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::FirstOrder;
        o.relative_tolerance = (std::numeric_limits<double>::quiet_NaN)();
        const auto input = values(m, 2, false);
        endpoints(m, input, 2, reconstruct_all_sides(prepare_reconstruction(m, o), input, 2));
    });
    add("FirstOrder empty mesh and strided components", [] {
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::FirstOrder;
        const ReconstructionStencil empty = prepare_reconstruction(Mesh(), o);
        require(empty.data().scheme == ReconstructionScheme::FirstOrder &&
                    empty.data().offsets == std::vector<ReconstructionIndex>{0},
                "Empty FirstOrder stencil");
        require(reconstruct_all_sides(empty, {}, 2).left_increment2.empty(), "Empty FirstOrder result");
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1}), o);
        const double input[] = {2, 7, -3, 11};
        const ReconstructionFieldView view = {input, 2, 1};
        near(evaluate_test_side(s.view(), 0, 0, view), 0);
        near(evaluate_test_side(s.view(), 1, 0, view), 0);
        near(evaluate_test_side(s.view(), 0, 1, view), 0);
        near(evaluate_test_side(s.view(), 1, 1, view), 0);
    });
    add("first-order upwind flux derivative convergence", [] {
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::FirstOrder;
        double previous = 0;
        for (unsigned n : std::vector<unsigned>{24, 48, 96}) {
            const double h = 1.0 / n;
            std::vector<double> x, u;
            for (unsigned i = 0; i < n + 3; ++i) {
                x.push_back((static_cast<double>(i) - 1) * h);
                u.push_back(std::sin(x.back()));
            }
            const TestResult r = reconstruct_all_sides(prepare_reconstruction(line(x), o), u, 1);
            double error = 0;
            for (unsigned i = 1; i <= n; ++i) {
                const double flux = u[i] + 0.5 * r.left_increment2[i],
                             previous_flux = u[i - 1] + 0.5 * r.left_increment2[i - 1];
                const double delta = (flux - previous_flux) / h - std::cos(x[i]);
                error += delta * delta;
            }
            error = std::sqrt(error / n);
            if (previous > 0)
                require(previous / error > 1.85 && previous / error < 2.15,
                        "Expected first-order convergence");
            previous = error;
        }
    });
}

// Проверки линейных полей, граничного замыкания и геометрических шаблонов EBR3.
// Сектор с тонкими радиальными слоями. По окружности поле Q = r постоянно,
// а продолжение ребра по прямой пересекает соседний слой, где Q меняется.
Mesh sector_mesh(bool prisms) {
    Builder b;
    b.d.dimension = prisms ? 3 : 2;
    const MeshIndex angles = 10, axial = prisms ? 5 : 1, layers = 5;
    const MeshIndex stride = angles * axial;
    for (MeshIndex k = 0; k < layers; ++k)
        for (MeshIndex j = 0; j < axial; ++j)
            for (MeshIndex i = 0; i < angles; ++i) {
                const double r = 1. + .02 * static_cast<double>(k);
                const double theta = .16 * static_cast<double>(i);
                b.node(r * std::cos(theta), r * std::sin(theta),
                       prisms ? .1 * static_cast<double>(j) + .02 * std::sin(2. * theta) : 0.);
            }
    for (MeshIndex k = 1; k < layers; ++k)
        for (MeshIndex j = 0; j < (prisms ? axial - 1 : 1); ++j)
            for (MeshIndex i = 0; i + 1 < angles; ++i) {
                const MeshIndex a = (k - 1) * stride + j * angles + i, b1 = a + 1;
                if (!prisms) {
                    b.polygon({a, a + stride, b1 + stride, b1});
                    continue;
                }
                // Треугольные основания лежат в слое, боковые рёбра направлены по радиусу.
                const auto prism_cell = [&](MeshIndex l, MeshIndex m, MeshIndex n) {
                    b.cell({{l, n, m},
                            {l + stride, m + stride, n + stride},
                            {l, m, m + stride, l + stride},
                            {m, n, n + stride, m + stride},
                            {n, l, l + stride, n + stride}});
                };
                prism_cell(a, b1, b1 + angles);
                prism_cell(a, b1 + angles, a + angles);
            }
    auto data = b.finish().data();
    data.patch_names.push_back("inner_wall");
    for (std::size_t i = 0; i < data.boundary_faces.size(); ++i) {
        const auto f = data.boundary_faces[i];
        bool wall = true;
        for (auto p = data.face_nodes.offsets[f]; p < data.face_nodes.offsets[f + 1]; ++p)
            wall &= data.face_nodes.indices[p] < stride;
        if (wall)
            data.boundary_patch[i] = 1;
    }
    return Mesh(std::move(data));
}

// Добавляет проверки криволинейных и призматических шаблонов.
void register_curved_stencils() {
    for (bool prisms : {false, true}) {
        add(prisms ? "PL cuts through triangular prism layers"
                   : "IJK and PL follow a curved quadrilateral row",
            [prisms] {
                const auto mesh = sector_mesh(prisms);
                const auto m = mesh.view();
                const MeshIndex stride = prisms ? 50 : 10;
                const auto depth = detail::PrismLayers(mesh, {"inner_wall"}).local_depths(m);
                std::vector<double> q(m.node_count);
                for (MeshIndex i = 0; i < m.node_count; ++i) {
                    require(depth[i] == static_cast<int>(i / stride), "Wrong wall layer");
                    q[i] = std::hypot(m.coordinate(i)[0], m.coordinate(i)[1]);
                }
                const MeshIndex a = 2 * stride + (prisms ? 24 : 4);
                MeshIndex edge = 0;
                while (edge < m.edge_count && (m.edge_begin[edge] != a || m.edge_end[edge] != a + 1))
                    ++edge;
                require(edge < m.edge_count, "Missing tangential test edge");
                ReconstructionOptions o;
                o.scheme = ReconstructionScheme::Ebr5;
                o.wall_labels = {"inner_wall"};
                const auto reference = reconstruct_all_sides(prepare_reconstruction(mesh, o), q, 1);
                require(std::abs(reference.left_increment2[edge]) > 1.e-4,
                        "Straight ray did not cross the layer");
                for (auto mode : {StencilGeometry::Ijk, StencilGeometry::Pl, StencilGeometry::Auto}) {
                    if (prisms && mode == StencilGeometry::Ijk)
                        continue;
                    o.geometry = mode;
                    if (prisms) {
                        detail::CurvedPoints support;
                        require(detail::CurvedStencil(mesh, o).points(edge, true, support), "Missing PL support");
                        require(support.near[0].weights.size() == 2 || support.near[1].weights.size() == 2,
                                "Plane should intersect a layer edge between its endpoints");
                    }
                    const auto stencil = prepare_reconstruction(mesh, o);
                    require(stencil.data().complete.at(2 * edge) && stencil.data().complete.at(2 * edge + 1),
                            "Missing fifth-order support");
                    const auto host = reconstruct_all_sides(stencil, q, 1);
                    near(host.left_increment2[edge], 0.);
                    near(host.right_increment2[edge], 0.);
                    if (TestReconstructor::cuda_device_count()) {
                        TestReconstructor gpu(stencil, ReconstructionBackend::Cuda);
                        same(gpu.reconstruct(q, 1), host);
                    }
                }
            });
    }
    add("IJK and unseeded PL fall back on triangles", [] {
        const auto mesh = plane(true);
        const auto q = values(mesh, 1, false);
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::Weno5;
        const auto straight = reconstruct_all_sides(prepare_reconstruction(mesh, o), q, 1);
        for (auto mode : {StencilGeometry::Ijk, StencilGeometry::Pl, StencilGeometry::Auto}) {
            o.geometry = mode;
            same(reconstruct_all_sides(prepare_reconstruction(mesh, o), q, 1), straight);
        }
    });
}

// Добавляет проверки реконструкции EBR третьего порядка.
void register_ebr3() {
    add("uniform 1D EBR3 coefficients, not a midpoint interpolant", [] {
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2, 3}));
        const TestResult r = reconstruct_all_sides(s, {0, 1, 4, 9}, 1);
        near(r.left_increment2[1], 7.0 / 3);
        near(r.right_increment2[1], 11.0 / 3);
        const TestResult cubic = reconstruct_all_sides(s, {0, 1, 8, 27}, 1);
        near(cubic.left_increment2[1], 5);
        near(cubic.right_increment2[1], 11);
        require(s.incomplete_sides() == 2, "Only external endpoint sides should be incomplete");
    });
    add("constant fields have exactly zero doubled increments", [] {
        for (const Mesh& m :
             std::vector<Mesh>{line({0, 0.1, 1, 4}), plane(true), volume(true), prism(true)}) {
            const ReconstructionStencil s = prepare_reconstruction(m);
            for (double constant : std::vector<double>{3.7, -17, 1.e100})
                zeros(reconstruct_all_sides(
                    s, std::vector<double>(static_cast<std::size_t>(m.view().node_count), constant), 1));
        }
    });
    add("doubled increments are invariant to a large constant offset", [] {
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2, 3}));
        const std::vector<double> input = {0, 1, 4, 9};
        std::vector<double> shifted = input;
        for (double& v : shifted)
            v += 281474976710656.0; // При 2^48 целочисленные разности ещё представляются точно.
        same(reconstruct_all_sides(s, input, 1), reconstruct_all_sides(s, shifted, 1));
    });
    add("FirstOrder zero rows do not access field data", [] {
        ReconstructionOptions o;
        o.scheme = ReconstructionScheme::FirstOrder;
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2}), o);
        require(s.data().nodes.empty() && s.data().weights.empty(),
                "FirstOrder should have empty sparse rows");
        unsigned calls = 0;
        for (ReconstructionIndex row = 0; row < 2 * s.data().edge_count; ++row)
            require(evaluate_test_side(s.view(), row, 0, CountingReader{&calls}) == 0,
                    "FirstOrder increment must be zero");
        require(calls == 0, "FirstOrder must not read field values");
    });
    add("nonuniform 1D divided differences", [] {
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 3, 6}));
        const TestResult r = reconstruct_all_sides(s, {0, 1, 9, 36}, 1);
        near(r.left_increment2[1], 6);
        near(r.right_increment2[1], 34.0 / 3);
        near(s.data().outer_distance[2], 1);
        near(s.data().outer_distance[3], 3);
    });
    add("boundary closure applies only to missing side", [] {
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2}));
        const TestResult r = reconstruct_all_sides(s, {0, 1, 4}, 1);
        near(r.left_increment2[0], 1);
        near(r.right_increment2[0], 5.0 / 3);
        near(r.left_increment2[1], 7.0 / 3);
        near(r.right_increment2[1], 3);
    });
    add("single interval has two boundary closures", [] { affine_test(line({-3, 2}), false); });
    add("affine field on nonuniform 1D", [] { affine_test(line({-4, -2.9, 0, 0.1, 3, 8}), true); });
    add("affine field on triangular 2D", [] { affine_test(plane(true), true); });
    add("affine field on quadrilateral 2D", [] { affine_test(plane(false), true); });
    add("affine field on tetrahedral 3D", [] { affine_test(volume(true), true); });
    add("affine field on hexahedral 3D", [] { affine_test(volume(false), true); });
    add("affine field on triangular prisms", [] { affine_test(prism(false), true); });
    add("concave polygon faces and polyhedral cells", [] { affine_test(prism(true), true); });
    add("concave 2D polygon", [] {
        Builder b;
        b.d.dimension = 2;
        b.node(0, 0);
        b.node(2, 0);
        b.node(2, 1);
        b.node(1, 1);
        b.node(1, 2);
        b.node(0, 2);
        b.polygon({0, 1, 2, 3, 4, 5});
        affine_test(b.finish(), false);
    });
    add("ray cannot jump outside a concave polygon", [] {
        Builder b;
        b.d.dimension = 2;
        b.node(-3, -1);
        b.node(-2, -1);
        b.node(-2, 1);
        b.node(0, 1);
        b.node(0, 0);
        b.node(1, 0);
        b.node(1, 2);
        b.node(-3, 2);
        b.polygon({0, 1, 2, 3, 4, 5, 6, 7});
        const Mesh m = b.finish();
        const ReconstructionStencil s = prepare_reconstruction(m);
        ReconstructionIndex e = 0;
        while (!(m.edge_begin()[e] == 4 && m.edge_end()[e] == 5))
            ++e;
        require(s.data().complete[2 * e] == 0, "Ray incorrectly jumped across the exterior gap");
    });
    add("all 3D interior-node sides have complete stencils", [] {
        for (bool tets : std::vector<bool>{false, true}) {
            const Mesh m = volume(tets);
            const ReconstructionStencil s = prepare_reconstruction(m);
            for (ReconstructionIndex e = 0; e < s.data().edge_count; ++e)
                for (unsigned side = 0; side < 2; ++side) {
                    const ReconstructionIndex node = side ? m.edge_end()[e] : m.edge_begin()[e];
                    const ReconstructionIndex i = node % 4, j = (node / 4) % 4, k = node / 16;
                    if (i > 0 && i < 3 && j > 0 && j < 3 && k > 0 && k < 3)
                        require(s.data().complete[2 * e + side] != 0, "Missing interior stencil");
                }
        }
    });
    add("third-order 1D upwind flux derivative on uniform grid", [] {
        double previous = 0;
        for (unsigned n : std::vector<unsigned>{24, 48, 96}) {
            const double h = 1.0 / n;
            std::vector<double> x, u;
            for (unsigned i = 0; i < n + 5; ++i) {
                x.push_back((static_cast<double>(i) - 2) * h);
                u.push_back(std::sin(x.back()));
            }
            const TestResult r = reconstruct_all_sides(prepare_reconstruction(line(x)), u, 1);
            double error = 0;
            for (unsigned i = 2; i < n + 2; ++i) {
                const double flux = u[i] + 0.5 * r.left_increment2[i],
                             previous_flux = u[i - 1] + 0.5 * r.left_increment2[i - 1];
                const double residual = (flux - previous_flux) / h - std::cos(x[i]);
                error += residual * residual;
            }
            error = std::sqrt(error / n);
            if (previous > 0)
                require(previous / error > 7.5 && previous / error < 8.5, "Unexpected convergence rate");
            previous = error;
        }
    });
    add("linear interpolation at a genuinely virtual 2D point", [] {
        Builder b;
        b.d.dimension = 2;
        b.node(0, 0);
        b.node(1, 0);
        b.node(-1, -1);
        b.node(-1, 2);
        b.polygon({0, 3, 2});
        b.polygon({0, 2, 1});
        const Mesh m = b.finish();
        const ReconstructionStencil s = prepare_reconstruction(m);
        ReconstructionIndex e = 0;
        while (!(m.edge_begin()[e] == 0 && m.edge_end()[e] == 1))
            ++e;
        const TestResult r = reconstruct_all_sides(s, {2, 8, 5, 11}, 1);
        // Интерполированное внешнее значение равно (2/3)·5 + (1/3)·11 = 7 при h = d = 1.
        near(r.left_increment2[e], 7.0 / 3);
        near(s.data().outer_distance[2 * e], 1);
    });
    add("rotation and scale invariance", [] {
        const Mesh original = volume(true);
        const ReconstructionStencil ref = prepare_reconstruction(original);
        const auto input = values(original, 3, false);
        const TestResult expected = reconstruct_all_sides(ref, input, 3);
        for (double scale : std::vector<double>{1.e-8, 1.e8}) {
            MeshData d = original.data();
            for (std::size_t i = 0; i < d.x.size(); ++i) {
                const double x = d.x[i], y = d.y[i], z = d.z[i];
                d.x[i] = scale * (3 + 0.6 * x - 0.8 * y);
                d.y[i] = scale * (-5 + 0.8 * x + 0.6 * y);
                d.z[i] = scale * (7 + z);
            }
            same(expected, reconstruct_all_sides(prepare_reconstruction(Mesh(d)), input, 3));
        }
    });
    add("barycentric interpolation at a virtual 3D point", [] {
        Builder b;
        b.node(0, 0, 0);
        b.node(1, 0, 0);
        b.node(-1, -1, -1);
        b.node(-1, 2, -1);
        b.node(-1, -1, 2);
        b.tetra(0, 2, 3, 4);
        b.tetra(0, 1, 2, 3);
        const Mesh m = b.finish();
        const ReconstructionStencil s = prepare_reconstruction(m);
        ReconstructionIndex e = 0;
        while (!(m.edge_begin()[e] == 0 && m.edge_end()[e] == 1))
            ++e;
        near(reconstruct_all_sides(s, {2, 8, 5, 11, 20}, 1).left_increment2[e], 2.0 / 3);
        near(s.data().outer_distance[2 * e], 1);
    });
    add("many components exercise the CPU parallel loop", [] {
        const Mesh m = plane(true);
        const ReconstructionStencil s = prepare_reconstruction(m);
        const auto input = values(m, 500, false);
        const TestResult actual = reconstruct_all_sides(s, input, 500);
        const ReconstructionIndex n = m.view().node_count, e = m.view().edge_count;
        for (ReconstructionIndex c = 0; c < 500; ++c) {
            const auto first = input.begin() + static_cast<std::ptrdiff_t>(c * n);
            const auto last = first + static_cast<std::ptrdiff_t>(n);
            const TestResult expected = reconstruct_all_sides(s, std::vector<double>(first, last), 1);
            for (ReconstructionIndex k = 0; k < e; ++k) {
                near(actual.left_increment2[c * e + k], expected.left_increment2[k]);
                near(actual.right_increment2[c * e + k], expected.right_increment2[k]);
            }
        }
    });
    add("node renumbering reverses left/right consistently", [] {
        MeshData d = line({0, 1, 3, 6}).data();
        std::reverse(d.x.begin(), d.x.end());
        const ReconstructionStencil s = prepare_reconstruction(Mesh(d));
        const TestResult r = reconstruct_all_sides(s, {36, 9, 1, 0}, 1);
        near(r.left_increment2[1], -34.0 / 3);
        near(r.right_increment2[1], -6);
    });
    add("custom component accessor and strided array", [] {
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2, 3}));
        const std::vector<double> aos = {0, 3, 1, 3, 4, 3, 9, 3};
        const ReconstructionFieldView v = {aos.data(), 2, 1};
        near(evaluate_test_side(s.view(), 2, 0, v), 7.0 / 3);
        near(evaluate_test_side(s.view(), 3, 1, v), 0);
    });
}

// Проверки времени жизни геометрии, ошибочных входов и выбора исполнителя.
void register_lifetime_and_errors() {
    add("ray just outside a stretched segment", [] {
        Builder b;
        b.d.dimension = 2;
        for (unsigned j = 0; j < 3; ++j) {
            b.node(-10., j*1.e-6);
            b.node(j == 2 ? 3.e-10 : 0., j*1.e-6);
            b.node(10., j*1.e-6);
        }
        for (MeshIndex i : {0u, 1u, 3u, 4u})
            b.polygon({i, i+1, i+4, i+3});
        const auto mesh = b.finish();
        const auto view = mesh.view();
        const detail::ReconstructionGeometry geometry(mesh, 1.e-10);
        for (MeshIndex e = 0; e < view.edge_count; ++e) {
            const auto a = view.edge_begin[e], z = view.edge_end[e];
            if (std::min(a,z) != 1 || std::max(a,z) != 4) continue;
            const unsigned side = z == 4 ? 1 : 0;
            // Луч x=0 попадает в левый отрезок. Правая грань начинается в x=3e-10.
            const auto right = b.faces.at({7,8}), left = b.faces.at({6,7});
            require(geometry.layer_hit(e,side,{right}).weights.empty(), "hit outside segment");
            const auto hit = geometry.layer_hit(e,side,{right,left});
            require(hit.weights.count(6) && hit.weights.count(7), "missed valid adjacent segment");
            near(hit.distance, 1.e-6);
            return;
        }
        throw std::runtime_error("Missing test edge");
    });
    add("geometry outlives original mesh", [] {
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2, 3}));
        near(reconstruct_all_sides(s, {0, 1, 4, 9}, 1).left_increment2[1], 7.0 / 3);
    });
    add("strict mode rejects missing stencil", [] {
        ReconstructionOptions o;
        o.require_complete_stencil = true;
        rejects([&] { prepare_reconstruction(line({0, 1, 2}), o); });
    });
    add("empty mesh and empty stencil", [] {
        const ReconstructionStencil s = prepare_reconstruction(Mesh());
        require(reconstruct_all_sides(s, {}, 3).left_increment2.empty(), "Expected empty output");
        require(s.data().offsets == std::vector<ReconstructionIndex>{0}, "Empty CSR");
    });
    add("bad component counts and input length", [] {
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1}));
        rejects([&] { reconstruct_all_sides(s, {0}, 1); });
        rejects([&] { reconstruct_all_sides(s, {}, 0); });
        rejects([&] { reconstruct_all_sides(s, {}, (std::numeric_limits<ReconstructionIndex>::max)()); });
    });
    add("zero length edge rejected", [] { rejects([] { prepare_reconstruction(line({0, 0, 1})); }); });
    add("affine reconstruction on warped faces", [] {
        MeshData d = volume(false).data();
        d.z[0] += 0.2;
        affine_test(Mesh(d), true);
    });
    add("inverted 3D cells rejected", [] {
        MeshData d = prism(false).data();
        for (auto& sign : d.cell_face_orientation)
            sign = static_cast<std::int8_t>(-sign);
        rejects([&] { prepare_reconstruction(Mesh(d)); });
    });
    add("invalid tolerances rejected", [] {
        for (double eps : std::vector<double>{0, -1, 0.5, (std::numeric_limits<double>::quiet_NaN)()}) {
            ReconstructionOptions o;
            o.relative_tolerance = eps;
            rejects([&] { prepare_reconstruction(line({0, 1}), o); });
        }
    });
    add("CPU executor and device-call rejection", [] {
        const Mesh m = plane(true);
        const ReconstructionStencil s = prepare_reconstruction(m);
        TestReconstructor cpu(s, ReconstructionBackend::Cpu);
        same(cpu.reconstruct(values(m, 4, false), 4), reconstruct_all_sides(s, values(m, 4, false), 4));
        rejects([&] { cpu.reconstruct_device(nullptr, 1, nullptr, nullptr); });
        rejects([&] { TestReconstructor bad(s, ReconstructionBackend::Auto, -1); });
    });
    add("automatic backend selection", [] {
        const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2}));
        TestReconstructor automatic(s);
        const bool available = TestReconstructor::cuda_device_count() > 0;
        require(automatic.backend() == (available ? ReconstructionBackend::Cuda : ReconstructionBackend::Cpu),
                "Wrong automatic backend");
        same(automatic.reconstruct({0, 1, 4}, 1), reconstruct_all_sides(s, {0, 1, 4}, 1));
        if (!available)
            rejects([&] { TestReconstructor unavailable(s, ReconstructionBackend::Cuda); });
    });
}
}

// Запускает проверки модуля и возвращает ненулевой код при ошибке.
int main(int argc, char** argv) {
    using namespace aero;
    const bool require_cuda = argc == 2 && std::string(argv[1]) == "--require-cuda";
    if (argc > 2 || (argc == 2 && !require_cuda)) {
        std::cerr << "Usage: reconstruction_tests [--require-cuda]\n";
        return 2;
    }
    register_curved_stencils();
    register_side_access();
    add("MP5 CFL, constant state, isolated jump and paired bounds", [] {
        near(mp5_alpha_from_cfl(.125), 4);
        near(mp5_alpha_from_cfl(.5), 1);
        near(mp5_alpha_from_cfl(1.25), 0);
        // Большой CFL уменьшает alpha только на явном шаге. При неявном
        // интегрировании он меняет длину шага, а ограничитель всегда получает 4.
        for (double cfl : {.125, .5, 1.25, 100.})
            near(mp5_alpha_for_time_scheme(cfl, true), 4.);
        near(mp5_alpha_for_time_scheme(.5, false), 1.);
        near(mp5::increment(0, 0, 1, 0, 4), 0);
        near(mp5::increment(0, 0, 0, 0, 1), 0);
        // Прямой эталон использует четыре аргумента minmod и явное пересечение интервалов.
        const auto dm = [](double a, double b) {
            const double x = 4*a-b, y = 4*b-a;
            return std::max(0., std::min({x,y,a,b})) + std::min(0., std::max({x,y,a,b}));
        };
        // Вычисляет независимое эталонное значение тестовой реконструкции.
        const auto reference = [&](double f, double l, double c, double r, double alpha) {
            const double v = -.1*r + .8*c + 11./30*l - f/15;
            const double bound = std::max(0., std::min(c,alpha*l)) + std::min(0., std::max(c,alpha*l));
            if (v*(v-bound) < 1.e-10) return v;
            const double md = c-dm(c-l,r-c), lc = l+8./3*dm(l-f,c-l);
            const double lo = std::max(std::min({0.,2*c,md}), std::min({0.,2*alpha*l,lc}));
            const double hi = std::min(std::max({0.,2*c,md}), std::max({0.,2*alpha*l,lc}));
            return std::min(hi,std::max(lo,v));
        };
        for (double alpha : {0., 1., 4.})
            for (int i = 1; i <= 500; ++i) {
                const double f = std::sin(i), l = std::cos(i*.29), c = std::sin(i*.51);
                const double r = std::sin(i*.75), g = std::cos(i*.91);
                double left, right;
                mp5::pair(f,l,c,r,g,alpha,left,right);
                near(left,reference(f,l,c,r,alpha));
                near(right,reference(g,r,c,l,alpha));
                near(mp5::increment(f,l,c,r,alpha),left);
            }
    });
    add("MP6 mean limiter is continuous for either sign and respects local bounds", [] {
        const double eps = mp6::mean_tolerance;
        const mp5::Bounds fixed = {0., 0.}, free = {-1., 1.};
        for (double sign : {-1., 1.}) {
            for (double x : {0., .25*eps, .75*eps, std::nextafter(eps, 0.),
                             eps, std::nextafter(eps, 2.*eps), 2.*eps}) {
                // Обе стороны малой волны закреплены в нуле. По мере роста |dS|
                // её вклад в общий вес ребра должен непрерывно убывать от 1 до 0.
                mp6::Correction small, large;
                const double weight = small.prepare(0., 0., 0., 0., true, true, sign*x, 1., fixed, fixed);
                const double expected = std::max(0., 1.-x/eps);
                near(weight, expected, 2.e-15);
                double left = 0., right = 0.;
                small.apply(mp6::MeanLimit::Shared, weight, left, right);
                near(left, 0., 0.);
                near(right, 0., 0.);

                // Вторая волна сама целиком допустима. Связанная компонента получает
                // общий вес ребра, а независимая сохраняет всю свою поправку.
                near(large.prepare(0., 0., 0., 0., false, false, .25, 1., free, free), 1.);
                for (unsigned width : {3u, 4u, 5u})
                    for (unsigned k = 0; k < width; ++k) {
                        left = right = 0.;
                        large.apply(mp6::face_mean_limit(k, width), weight, left, right);
                        const bool shear = (width >= 4 && k == 2) || (width == 5 && k == 3);
                        const double shift = .25 * (shear ? 1. : expected);
                        near(left, shift, 2.e-15);
                        near(right, shift, 2.e-15);
                    }
            }
        }
        // При частично допустимом сдвиге нормируем только запрещённую долю.
        near(mp6::mean_weight(.75*eps, 0., .25*eps), .5);
        near(mp6::mean_weight(-.75*eps, -.25*eps, 0.), .5);
        near(mp6::mean_weight(2.*eps, 0., .25*eps), .125);
    });
    add("MP5-MP6 weak transverse perturbation at an acoustic component", check_weak_transverse_mp6<false>);
    add("MPWENO5-MP6 weak transverse perturbation at an acoustic component", check_weak_transverse_mp6<true>);
    add("MP6 includes the early MP5 increment on its own side", [] {
        const Mp5Settings settings{true, false, false, 4.};
        // Центральный и оба ближних наклона равны нулю. Полный интервал каждой
        // стороны равен [0,0]. Малый дальний наклон даёт проход по допуску MP5,
        // а большой наклон с другой стороны заведомо требует ограничения до нуля.
        for (double sign : {-1., 1.}) for (bool left_passes : {false, true}) {
            const double small = sign * 1.5e-5, large = .3;
            const double fl = left_passes ? small : large, fr = left_passes ? large : small;
            const double accepted = -small / 15.;
            for (double h : {0., 1.e-14, .5, 1.}) {
                double left = 0., right = 0.;
                mp6::Correction actual;
                mp6::prepare_wave<false, false>(fl, 0., 0., 0., fr, settings, h, actual, left, right);
                near(left, left_passes ? accepted : 0., 1.e-20);
                near(right, left_passes ? 0. : accepted, 1.e-20);
                const double lm = std::max(left, 0.), lp = std::max(-left, 0.);
                const double rm = std::max(right, 0.), rp = std::max(-right, 0.);
                // Проверяем границы обеих связанных поправок. Если по ошибке
                // включить левое значение также справа, хотя бы одна проверка упадёт.
                near(actual.min1, -(1.5*lm + .5*rp), 1.e-20);
                near(actual.max1,   1.5*lp + .5*rm,  1.e-20);
                near(actual.min2, -(1.5*rm + .5*lp), 1.e-20);
                near(actual.max2,   1.5*rp + .5*lm,  1.e-20);
            }
        }
    });
    add("MPWENO5 hybrid uses its own accepted value on each side", [] {
        const Mp5Settings settings{false, true, true, 4.};
        // Полином MP5 проходит быструю проверку. На данном шаблоне обычная
        // MPWENO5 ограничит его до нуля. Нельзя расширять интервал полиномом MP5.
        double left = 0., right = 0.;
        mp6::Correction characteristic;
        mp6::prepare_wave<true, false>(-1.5e-5, 0., 0., 0., .3, settings, 1., characteristic, left, right);
        near(left, 0., 0.);
        near(right, 0., 0.);
        near(characteristic.min1, 0., 0.);
        near(characteristic.max1, 0., 0.);
        // После поворота базовые значения слева и справа задаёт вызывающий код.
        // Используем разные знаки и модули, чтобы обнаружить перестановку сторон.
        left = 2.e-6;
        right = -3.e-6;
        mp6::Correction rotated;
        mp6::prepare_wave<true, true>(-1.5e-5, 0., 0., 0., .3, settings, .5, rotated, left, right);
        near(rotated.min1, -4.5e-6, 2.e-20);
        near(rotated.max1, 0., 0.);
        near(rotated.min2, 0., 0.);
        near(rotated.max2, 5.5e-6, 2.e-20);
    });
    add("MP6 one-sided bound agrees with the complete interval", [] {
        // Если оба исходных значения уже внутри интервалов, достаточно границы
        // в направлении поправки. Полный расчёт здесь служит независимой проверкой
        // выбора стороны при разных знаках среднего сдвига и полуразности.
        for (unsigned i = 0; i < 180; ++i) {
            const double left = std::sin(.19*i), right = std::cos(.37*i);
            const mp5::Bounds bl = {left-.3, left+.7}, br = {right-.6, right+.2};
            const double center = .5*(left+right) + std::sin(.31*i)*(i%3 ? 2. : 1.e-8);
            const bool skip_l = i%2 == 0, skip_r = i%5 == 0;
            mp6::Correction reduced;
            const double w = reduced.prepare(left, right, .1, -.4, skip_l, skip_r, center, .75, bl, br);
            const double lo = std::max(std::min(0., reduced.ds), -mp6::Correction::upper_mean(
                                            -reduced.min1, -reduced.min2, -reduced.dd));
            const double hi = std::min(std::max(0., reduced.ds), mp6::Correction::upper_mean(
                                            reduced.max1, reduced.max2, reduced.dd));
            near(reduced.smin, lo, 0.);
            near(reduced.smax, hi, 0.);
            near(w, mp6::mean_weight(reduced.ds, lo, hi), 0.);
        }
    });
    add("MP6 coupled wave correction and MPWENO5 switch", [] {
        // Один несимметричный шаблон содержит гладкие и ограниченные волны.
        // Эталон вычислен прямой двухпроходной формулой при h=0.75 и alpha=1.
        const aero::FixedMatrix<double,5,5> slopes = {{.4,.3,.01,-.5,-.1, .3,.7,-.09,-.6,-.4,
                                     .7,-.3,.25,-.2,-.6, -.1,1.2,-.2,-.8,-.9, 1.,-.4,.6,.1,-.3}};
        const aero::FixedTensor<double,3,2,5> expected = {{.349166666666667,.504166666666667,-.041166666666667,-.559166666666667,-.066666666666667,
             .4625,.0575,.064166666666667,-.4325,-.16,
            .353218333224483,.495104166666667,-.044252981410357,-.562291666666667,-.054166666666667,
             .456569000065310,.0809375,.067235962206430,-.435625,-.1625,
            .356319700409143,.495104166666667,-.046496969964287,-.562291666666667,-.054166666666667,
             .454708179754514,.0809375,.068748185830593,-.435625,-.1625}};
        for (unsigned mode=0; mode<3; ++mode) {
            const Mp5Settings settings = {mode==0, mode!=0, mode==2, 1.};
            double left[5], right[5];
            if (settings.weno) mp6::reconstruct<true>(slopes,5,settings,.75,left,right);
            else mp6::reconstruct<false>(slopes,5,settings,.75,left,right);
            for (unsigned k=0; k<5; ++k) {
                near(left[k],expected(mode, 0, k),2.e-14);
                near(right[k],expected(mode, 1, k),2.e-14);
            }
        }
    });
    register_weno_formulas();
    register_weno_schemes();
    register_ebr5();
    register_first_order();
    register_ebr3();
    register_lifetime_and_errors();
    try {
        const int devices = TestReconstructor::cuda_device_count();
        if (require_cuda && devices == 0)
            throw ReconstructionError("GPU verification requested but no CUDA device is available");
        for (int device = 0; device < devices; ++device) {
#ifdef AERO_WITH_KOKKOS_GPU
            add("CUDA " + std::to_string(device) + " one side, explicit owner and guarded output",
                [device] { single_side_device_test(device); });
            add("CUDA " + std::to_string(device) + " MP5 full and shortened stencils",
                [device] { device_test(device, ReconstructionScheme::Mp5); });
            add("CUDA " + std::to_string(device) + " MPWENO5 full and shortened stencils, correction on/off",
                [device] {
                    for (bool correction : {false, true})
                        device_test(device, ReconstructionScheme::MPWENO5, correction);
                });
#endif
            for (ReconstructionScheme scheme : std::vector<ReconstructionScheme>{
                     ReconstructionScheme::Weno3, ReconstructionScheme::Weno5}) {
                const std::string name = scheme == ReconstructionScheme::Weno3 ? "WENO3" : "WENO5";
                add("CUDA " + std::to_string(device) + " " + name +
                        " dimensions, nonlinear fields and buffer reuse",
                    [device, scheme] {
                        ReconstructionOptions o;
                        o.scheme = scheme;
                        for (const Mesh& m : std::vector<Mesh>{
                                 line({0, 0.1, 0.7, 2, 4, 5, 9}), plane(true, 7, 7), plane(false, 7, 7),
                                 volume(true, 6), volume(false, 6), prism(true, 6)}) {
                            const ReconstructionStencil s = prepare_reconstruction(m, o);
                            require(s.incomplete_sides() < 2 * s.data().edge_count,
                                    "No full GPU WENO stencils");
                            TestReconstructor gpu(s, ReconstructionBackend::Cuda, device);
                            for (ReconstructionIndex components : std::vector<ReconstructionIndex>{1, 7, 2}) {
                                const auto q = values(m, components, false);
                                same(gpu.reconstruct(q, components), reconstruct_all_sides(s, q, components));
                                const auto affine = values(m, components, true);
                                same(gpu.reconstruct(affine, components),
                                     reconstruct_all_sides(s, affine, components));
                            }
                            zeros(gpu.reconstruct(
                                std::vector<double>(static_cast<std::size_t>(m.view().node_count), 1.e200),
                                1));
                            std::vector<double> jump(static_cast<std::size_t>(m.view().node_count));
                            for (std::size_t n = 0; n < jump.size(); ++n)
                                jump[n] = m.data().x[n] > 1 ? 1 : 0;
                            same(gpu.reconstruct(jump, 1), reconstruct_all_sides(s, jump, 1));
                        }
                        const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2, 3, 4, 5}), o);
                        TestReconstructor gpu(s, ReconstructionBackend::Cuda, device);
                        for (double a : std::vector<double>{1.e-150, 1.e150}) {
                            const std::vector<double> q = {0, a, 2 * a, 3 * a, 4 * a, 5 * a};
                            TestResult r = gpu.reconstruct(q, 1), expected = reconstruct_all_sides(s, q, 1);
                            for (double& v : r.left_increment2)
                                v /= a;
                            for (double& v : r.right_increment2)
                                v /= a;
                            for (double& v : expected.left_increment2)
                                v /= a;
                            for (double& v : expected.right_increment2)
                                v /= a;
                            same(r, expected);
                        }
                        TestReconstructor empty(prepare_reconstruction(Mesh(), o),
                                                ReconstructionBackend::Cuda, device);
                        require(empty.reconstruct({}, 1).left_increment2.empty(), "Empty WENO CUDA output");
                    });
#ifdef AERO_WITH_KOKKOS_GPU
                add("CUDA " + std::to_string(device) + " " + name + " device arrays and stream",
                    [device, scheme] { device_test(device, scheme); });
#endif
            }
            add("CUDA " + std::to_string(device) + " EBR5 1D/2D/3D and reusable buffers", [device] {
                ReconstructionOptions o;
                o.scheme = ReconstructionScheme::Ebr5;
                for (const Mesh& m :
                     std::vector<Mesh>{line({0, 0.1, 0.7, 2, 4, 5, 9}), plane(true, 7, 7), plane(false, 7, 7),
                                       volume(true, 6), volume(false, 6), prism(true, 6)}) {
                    const ReconstructionStencil s = prepare_reconstruction(m, o);
                    require(s.incomplete_sides() < 2 * s.data().edge_count, "No full GPU EBR5 stencils");
                    TestReconstructor gpu(s, ReconstructionBackend::Cuda, device);
                    for (ReconstructionIndex components : std::vector<ReconstructionIndex>{1, 7, 2}) {
                        const auto input = values(m, components, false);
                        same(gpu.reconstruct(input, components), reconstruct_all_sides(s, input, components));
                        const auto affine = values(m, components, true);
                        same(gpu.reconstruct(affine, components),
                             reconstruct_all_sides(s, affine, components));
                    }
                    zeros(gpu.reconstruct(
                        std::vector<double>(static_cast<std::size_t>(m.view().node_count), 1.e100), 1));
                }
                TestReconstructor empty(prepare_reconstruction(Mesh(), o), ReconstructionBackend::Cuda,
                                        device);
                require(empty.reconstruct({}, 1).left_increment2.empty(), "Empty EBR5 GPU output");
            });
            add("CUDA " + std::to_string(device) + " zero increments and constant-offset invariance",
                [device] {
                    const ReconstructionStencil s = prepare_reconstruction(line({0, 1, 2, 3}));
                    TestReconstructor gpu(s, ReconstructionBackend::Cuda, device);
                    zeros(gpu.reconstruct(std::vector<double>(4, 1.e100), 1));
                    const std::vector<double> input = {0, 1, 4, 9};
                    std::vector<double> shifted = input;
                    for (double& v : shifted)
                        v += 281474976710656.0;
                    same(gpu.reconstruct(shifted, 1), reconstruct_all_sides(s, input, 1));
                    ReconstructionOptions o;
                    o.scheme = ReconstructionScheme::FirstOrder;
                    TestReconstructor first(prepare_reconstruction(line({0, 1, 2, 3}), o),
                                            ReconstructionBackend::Cuda, device);
                    zeros(first.reconstruct(shifted, 1));
                });
            add("CUDA " + std::to_string(device) + " FirstOrder endpoints and buffer reuse", [device] {
                ReconstructionOptions o;
                o.scheme = ReconstructionScheme::FirstOrder;
                for (const Mesh& m : std::vector<Mesh>{line({0, 0.2, 1, 4}), plane(true), plane(false),
                                                       volume(true), volume(false), prism(true)}) {
                    TestReconstructor gpu(prepare_reconstruction(m, o), ReconstructionBackend::Cuda, device);
                    for (ReconstructionIndex components : std::vector<ReconstructionIndex>{1, 7, 2}) {
                        const auto input = values(m, components, false);
                        endpoints(m, input, components, gpu.reconstruct(input, components));
                    }
                }
                TestReconstructor empty(prepare_reconstruction(Mesh(), o), ReconstructionBackend::Cuda,
                                        device);
                require(empty.reconstruct({}, 1).left_increment2.empty(), "Empty FirstOrder GPU output");
            });
            add("CUDA " + std::to_string(device) + " 1D/2D/3D and buffer reuse", [device] {
                for (const Mesh& mesh : std::vector<Mesh>{line({0, 1, 3, 6}), plane(true), plane(false),
                                                          volume(true), volume(false), prism(true)}) {
                    const ReconstructionStencil s = prepare_reconstruction(mesh);
                    TestReconstructor gpu(s, ReconstructionBackend::Cuda, device);
                    for (ReconstructionIndex components : std::vector<ReconstructionIndex>{1, 7, 2, 9, 1}) {
                        const auto input = values(mesh, components, false);
                        same(gpu.reconstruct(input, components), reconstruct_all_sides(s, input, components));
                        const auto linear = values(mesh, components, true);
                        same(gpu.reconstruct(linear, components),
                             reconstruct_all_sides(s, linear, components));
                    }
                }
                TestReconstructor empty(ReconstructionStencil(), ReconstructionBackend::Cuda, device);
                require(empty.reconstruct({}, 1).left_increment2.empty(), "Empty GPU output");
            });
#ifdef AERO_WITH_KOKKOS_GPU
            add("CUDA " + std::to_string(device) + " device arrays and nondefault stream",
                [device] { device_test(device); });
            add("CUDA " + std::to_string(device) + " FirstOrder device arrays and stream",
                [device] { device_test(device, ReconstructionScheme::FirstOrder); });
            add("CUDA " + std::to_string(device) + " EBR5 device arrays and stream",
                [device] { device_test(device, ReconstructionScheme::Ebr5); });
#endif
        }
        unsigned passed = 0;
        for (const auto& test : tests) {
            try {
                test.second();
                ++passed;
                std::cout << "PASS " << test.first << '\n';
            } catch (const std::exception& error) {
                std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
                return 1;
            }
        }
        std::cout << passed << " reconstruction tests passed; CUDA devices tested: " << devices << '\n';
        if (!devices)
            std::cout << "CUDA runtime tests skipped (CPU checks completed).\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
