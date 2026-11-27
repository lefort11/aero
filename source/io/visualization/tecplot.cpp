#include "tecplot.h"
#include "binary_columns.h"
#include "../number_codec.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <ostream>
#include <streambuf>
#include <utility>
#include <vector>
#include "../scheduled_output.h"
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace aero {
namespace tecplot {
// Сообщает, собрана ли утилита с поддержкой OpenMP.
bool openmp_available() noexcept {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}
namespace {
typedef std::int32_t I32;
typedef std::int64_t I64;
typedef std::uint64_t U64;
static_assert(sizeof(float) == 4 && sizeof(double) == 8 && std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<double>::is_iec559,
              "PLT requires IEEE-754 binary32/binary64");
// Сообщает ошибку подготовки или записи Tecplot.
[[noreturn]] void fail(const std::string& message) {
    throw ExportError(message);
}
// Проверяет вместимость размера или индекса в ограниченный тип формата Tecplot.
I32 checked(U64 value, const char* what) {
    if (value > static_cast<U64>(std::numeric_limits<I32>::max()))
        fail(std::string(what) + " exceeds PLT v112 INT32 limit");
    return static_cast<I32>(value);
}
// Преобразует размещение исходного поля в размещение переменной Tecplot.
const char* location(FieldLocation loc) {
    switch (loc) {
    case FieldLocation::Node:
        return "node";
    case FieldLocation::Edge:
        return "edge";
    case FieldLocation::Face:
        return "face";
    case FieldLocation::Cell:
        return "cell";
    }
    fail("Invalid field location");
}
// Выбирает число потоков экспорта с учётом OpenMP и настроек.
int workers(const Options& options) {
    if (options.threads < 0)
        fail("Negative thread count");
#ifdef _OPENMP
    return options.threads ? options.threads : omp_get_max_threads();
#else
    if (options.threads > 1)
        fail("This build has no OpenMP; use --threads 1");
    return 1;
#endif
}
// Скалярная переменная Tecplot, её источник, диапазон и число значений.
struct Variable {
    // Выводимое имя переменной.
    std::string name;
    // Исходное поле для переменной.
    const Field* field;
    // Номер компоненты исходного поля.
    std::size_t component;
    // Адрес массива числовых значений.
    const double* values;
    // Число значений переменной.
    std::size_t count;
    // Диапазон значений для заголовка Tecplot.
    double minimum, maximum;
};
// Топология, нумерация и источники значений одной зоны Tecplot.
struct Zone {
    // Имя выходной зоны.
    std::string name;
    // Тип сущности сетки, к которой относятся данные.
    FieldLocation source;
    // Код типа зоны и числа узлов, ячеек и граней.
    I32 type, nodes, cells, faces;
    // Пустой node_ids означает обычную нумерацию. Иначе номер узла зоны переводится в номер узла сетки по таблице.
    std::vector<MeshIndex> node_ids;
    // Смещения связности и номера ячеек по сторонам граней.
    std::vector<I32> offsets, connectivity, left, right;
    // Диапазоны значений по переменным.
    std::vector<std::pair<double, double>> ranges;
    // Инициализирует пустую ячеечную зону до определения её топологии.
    Zone() : source(FieldLocation::Cell), type(0), nodes(0), cells(0), faces(0) {}
    // Проверяет, содержит ли зона данные выбранной переменной.
    bool active(const Variable& v) const {
        return !v.field || v.field->location == FieldLocation::Node || v.field->location == source;
    }
    // Определяет, записывается ли переменная по ячейкам текущей зоны.
    I32 cell_centered(const Variable& v) const {
        return type != 0 && v.field && v.field->location != FieldLocation::Node ? 1 : 0;
    }
    // Возвращает число значений переменной для текущей зоны.
    std::size_t count(const Variable& v) const {
        return static_cast<std::size_t>(cell_centered(v) ? cells : nodes);
    }
    // Поля ячеек индексируются напрямую. Узловым значениям иногда нужен переход по node_ids.
    bool contiguous(const Variable& v) const {
        return node_ids.empty() || (v.field && v.field->location != FieldLocation::Node);
    }
    // Получает значение переменной с учётом перенумерации узлов зоны.
    double value(const Variable& v, std::size_t i) const {
        return v.values[contiguous(v) ? i : static_cast<std::size_t>(node_ids[i])];
    }
};
// Владеет переменными и зонами экспорта, сохраняющими ссылки на исходные поля.
struct Plan {
    BinaryColumns* disk = nullptr; // Для общей зоны MPI значения читаются из завершённых столбцов файла.
    // Число потоков подготовки файла.
    int threads;
    // Контрольная сумма исходной сетки для метаданных.
    std::string mesh_sha256;
    // Переменные в порядке выходного файла.
    std::vector<Variable> variables;
    // Зоны подготовленного файла.
    std::vector<Zone> zones;
    // Копирование запрещено: указатели переменных ссылаются на данные исходной сетки и снимка.
    Plan(const Plan&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    Plan& operator=(const Plan&) = delete;
    // Подготавливает переменные, зоны и связи для записи Tecplot.
    Plan(const Mesh& mesh, const FieldSnapshot* snapshot, const Options& options, bool verify_mesh = true,
         BinaryColumns* columns = nullptr)
        : disk(columns), threads(workers(options)) {
        if (snapshot) {
            if (verify_mesh) snapshot->check_mesh(mesh);
            mesh_sha256 = snapshot->mesh_sha256;
        } else {
            mesh_sha256 = FieldSnapshot::mesh_fingerprint(mesh);
        }
        const MeshView view = mesh.view();
        const MeshData& d = mesh.data();
        if (!view.node_count || !view.cell_count)
            fail("Cannot export an empty mesh to Tecplot");
        const I32 nodes = checked(view.node_count, "Nodes"), cells = checked(view.cell_count, "Cells");
        const I32 faces = checked(view.face_count, "Faces");
        checked(view.edge_count, "Edges");
        checked(d.face_nodes.indices.size(), "Face-node count");
        add_coordinates(d, nodes);
        const auto requested = add_fields(snapshot);
        measure_variables(nodes);
        add_cell_zone(view, d, nodes, cells, faces);
        if (requested.edges)
            add_edge_zone(view, nodes);
        if (requested.faces)
            add_face_zone(view, d, nodes, faces);
        measure_zones(nodes);
    }

    // Стеночный снимок хранит исходный номер, три координаты и Cp в пяти узловых столбцах.
    explicit Plan(const FieldSnapshot& wall) : threads(workers(Options())), mesh_sha256(wall.mesh_sha256) {
        Zone zone;
        zone.name = "Wall";
        zone.source = FieldLocation::Node;
        zone.nodes = checked(wall.counts[0], "Wall points");
        // Упорядоченная зона Tecplot допускает только положительные размеры.

        for (unsigned axis = 0; axis < 3; ++axis) {
            const auto& field = wall.fields.at(axis + 1);
            variables.push_back({std::string(1, "XYZ"[axis]), nullptr, 0, field.values.data(),
                                 field.values.size(), 0, 0});
        }
        for (unsigned index : {0u, 4u}) {
            const auto& field = wall.fields.at(index);
            variables.push_back({field.name, &field, 0, field.values.data(), field.values.size(), 0, 0});
        }
        if (!zone.nodes) return; // Пустой раздел сохраняет заголовок и список переменных.
        measure_variables(zone.nodes);
        zones.push_back(std::move(zone));
        measure_zones(zones.front().nodes);
    }

private:
    // Сетка и снимок живут до конца экспорта. Tecplot читает координаты прямо
    // из исходных массивов, поэтому копия трёх полей размера всей сетки не нужна.
    void add_coordinates(const MeshData& d, I32 nodes) {
        const char* names[] = {"X", "Y", "Z"};
        const double* coordinates[] = {d.x.data(), d.y.data(), d.z.data()};
        for (unsigned axis = 0; axis < 3; ++axis) {
            Variable v = {names[axis], nullptr, 0, coordinates[axis], static_cast<std::size_t>(nodes), 0, 0};
            variables.push_back(v);
        }
    }

    // Признаки необходимости отдельных зон рёбер и граней.
    struct RequestedZones {
        // Запрошены зоны для рёбер и граней.
        bool edges = false, faces = false;
    };
    // Поле с несколькими компонентами даёт по скалярной переменной на компоненту.
    // Поля рёбер и граней требуют отдельных зон со своей связностью.
    RequestedZones add_fields(const FieldSnapshot* snapshot) {
        RequestedZones requested;
        if (snapshot)
            for (const Field& field : snapshot->fields) {
                requested.edges = requested.edges || field.location == FieldLocation::Edge;
                requested.faces = requested.faces || field.location == FieldLocation::Face;
                const std::size_t count = static_cast<std::size_t>(snapshot->entity_count(field.location));
                for (std::size_t component = 0; component < field.components.size(); ++component) {
                    checked(variables.size() + 1, "Variable count");
                    std::string name = "v" + std::to_string(variables.size() - 2) + ":" +
                                       location(field.location) + ":" + field.name + "." +
                                       field.components[component];
                    if (!field.units.empty())
                        name += " [" + field.units + "]";
                    if (name.size() > 128) {
                        std::size_t end = 125;
                        while ((static_cast<unsigned char>(name[end]) & 0xc0) == 0x80)
                            --end;
                        name.resize(end);
                        name += "...";
                    }
                    Variable v = {name,  &field, component, disk ? nullptr : field.values.data() + component * count,
                                  count, 0,      0};
                    variables.push_back(v);
                }
            }
        return requested;
    }
    // Минимум и максимум каждой переменной входят в заголовок данных PLT.
    void measure_variables(I32 nodes) {
        (void)nodes; // Используется в условии параллельного цикла OpenMP.
        // Каждая переменная записывается в собственный диапазон. Память выделена и ошибки проверены до параллельного участка.
        const I64 nvars = static_cast<I64>(variables.size());
        int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) reduction(| : invalid) if (nodes > 2048)
#endif
        for (I64 i = 0; i < nvars; ++i) {
            Variable& v = variables[static_cast<std::size_t>(i)];
            double lo = std::numeric_limits<double>::max(), hi = -lo;
            if (disk && v.field) continue; // Общий входной поток обслуживается после параллельного цикла.
            for (std::size_t j = 0; j < v.count; ++j) {
                const double value = v.values[j];
                if (!std::isfinite(value))
                    invalid = 1;
                else {
                    lo = std::min(lo, value);
                    hi = std::max(hi, value);
                }
            }
            v.minimum = lo;
            v.maximum = hi;
        }
        if (disk) for (auto& v : variables) {
            if (!v.field) continue;
            v.minimum = std::numeric_limits<double>::max();
            v.maximum = -v.minimum;
            disk->visit(*v.field, v.count, [&](const char* bytes, std::size_t count) {
                for (std::size_t j = 0; j < count; ++j) {
                    const double value = number_codec::read_real(bytes + 8*j);
                    if (!std::isfinite(value)) fail("Tecplot export requires finite field values");
                    v.minimum = std::min(v.minimum, value);
                    v.maximum = std::max(v.maximum, value);
                }
            });
        }
        if (invalid)
            fail("Tecplot export requires finite field values (NaN/Inf found)");
    }
    // Основная зона содержит ячейки сетки. Связность зависит от размерности.
    void add_cell_zone(const MeshView& view, const MeshData& d, I32 nodes, I32 cells, I32 faces) {
        Zone main;
        main.name = "Cells";
        main.nodes = nodes;
        main.cells = cells;
        main.type = view.dimension == 1 ? 1 : (view.dimension == 2 ? 6 : 7);
        if (view.dimension == 1) {
            main.connectivity.resize(static_cast<std::size_t>(cells) * 2);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (cells > 2048)
#endif
            for (I64 c = 0; c < cells; ++c) {
                const MeshCell cell = view.cell(static_cast<MeshIndex>(c));
                for (MeshIndex k = 0; k < 2; ++k)
                    main.connectivity[static_cast<std::size_t>(c) * 2 +
                                      (cell.face_orientation(k) < 0 ? 0 : 1)] =
                        static_cast<I32>(cell.face(k).node_id(0));
            }
        } else {
            main.faces = faces;
            main.connectivity.resize(d.face_nodes.indices.size());
            if (main.type == 7)
                main.offsets.resize(d.face_nodes.offsets.size());
            main.left.assign(static_cast<std::size_t>(faces), -1);
            main.right.assign(static_cast<std::size_t>(faces), -1);
            const I64 entries = static_cast<I64>(main.connectivity.size());
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (entries > 2048)
#endif
            for (I64 i = 0; i < entries; ++i)
                main.connectivity[static_cast<std::size_t>(i)] =
                    static_cast<I32>(d.face_nodes.indices[static_cast<std::size_t>(i)]);
            for (std::size_t i = 0; i < main.offsets.size(); ++i)
                main.offsets[i] = static_cast<I32>(d.face_nodes.offsets[i]);
            // После проверки сетки у каждой стороны грани ровно один процесс записи.
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (cells > 2048)
#endif
            for (I64 c = 0; c < cells; ++c) {
                const std::size_t cell = static_cast<std::size_t>(c);
                for (MeshIndex k = d.cell_faces.offsets[cell]; k < d.cell_faces.offsets[cell + 1]; ++k) {
                    const std::size_t entry = static_cast<std::size_t>(k),
                                      face = static_cast<std::size_t>(d.cell_faces.indices[entry]);
                    if (d.cell_face_orientation[entry] > 0)
                        main.left[face] = static_cast<I32>(c);
                    else
                        main.right[face] = static_cast<I32>(c);
                }
            }
        }
        zones.push_back(std::move(main));
    }
    // В зоне рёбер каждый отрезок соединяет исходные узлы begin и end.
    void add_edge_zone(const MeshView& view, I32 nodes) {
        Zone zone;
        zone.name = "Edges";
        zone.source = FieldLocation::Edge;
        zone.type = 1;
        zone.nodes = nodes;
        zone.cells = static_cast<I32>(view.edge_count);
        zone.connectivity.resize(static_cast<std::size_t>(zone.cells) * 2);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (zone.cells > 2048)
#endif
        for (I64 e = 0; e < zone.cells; ++e) {
            const std::size_t i = static_cast<std::size_t>(e);
            zone.connectivity[2 * i] = static_cast<I32>(view.edge_begin[i]);
            zone.connectivity[2 * i + 1] = static_cast<I32>(view.edge_end[i]);
        }
        zones.push_back(std::move(zone));
    }
    // В зоне граней топологическая размерность на единицу меньше размерности сетки.
    void add_face_zone(const MeshView& view, const MeshData& d, I32 nodes, I32 faces) {
        Zone zone;
        zone.name = "Faces";
        zone.source = FieldLocation::Face;
        zone.cells = faces;
        if (view.dimension == 1) {
            zone.type = 0;
            zone.nodes = faces;
            zone.cells = 0;
            zone.node_ids = d.face_nodes.indices;
        } else if (view.dimension == 2) {
            zone.type = 1;
            zone.nodes = nodes;
            zone.connectivity = zones[0].connectivity;
        } else {
            // Создаём независимые граничные многоугольники, чтобы связи по рёбрам поверхности были однозначными.
            const I32 entries = checked(d.face_nodes.indices.size(), "Surface nodes");
            checked(U64(entries) * 2, "Surface face-node count");
            zone.type = 6;
            zone.nodes = entries;
            zone.faces = entries;
            zone.node_ids = d.face_nodes.indices;
            zone.connectivity.resize(static_cast<std::size_t>(entries) * 2);
            zone.left.resize(static_cast<std::size_t>(entries));
            zone.right.assign(static_cast<std::size_t>(entries), -1);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (faces > 2048)
#endif
            for (I64 f = 0; f < faces; ++f) {
                const std::size_t face = static_cast<std::size_t>(f);
                const MeshIndex first = d.face_nodes.offsets[face], last = d.face_nodes.offsets[face + 1];
                for (MeshIndex k = first; k < last; ++k) {
                    const std::size_t i = static_cast<std::size_t>(k);
                    zone.connectivity[2 * i] = static_cast<I32>(k);
                    zone.connectivity[2 * i + 1] = static_cast<I32>(k + 1 == last ? first : k + 1);
                    zone.left[i] = static_cast<I32>(f);
                }
            }
        }
        zones.push_back(std::move(zone));
    }
    // Зона, использующая подмножество узлов, получает диапазон только по собственным значениям.
    void measure_zones(I32 nodes) {
        (void)nodes;
        const auto nvars = static_cast<I64>(variables.size());
        for (Zone& zone : zones) {
            zone.ranges.resize(variables.size());
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (nodes > 2048)
#endif
            for (I64 i = 0; i < nvars; ++i) {
                const std::size_t index = static_cast<std::size_t>(i);
                const Variable& v = variables[index];
                if (!zone.active(v))
                    continue;
                double lo = v.minimum, hi = v.maximum;
                if (!zone.node_ids.empty() && (!v.field || v.field->location == FieldLocation::Node)) {
                    lo = std::numeric_limits<double>::max();
                    hi = -lo;
                    for (std::size_t j = 0; j < zone.count(v); ++j) {
                        const double x = zone.value(v, j);
                        lo = std::min(lo, x);
                        hi = std::max(hi, x);
                    }
                }
                zone.ranges[index] = std::make_pair(lo, hi);
            }
        }
    }
};

// Буферизует двоичную запись слов, строк и массивов формата Tecplot.
class Writer {
public:
    // Подключает поток и выделяет буфер двоичного вывода Tecplot.
    explicit Writer(std::ostream& output) : output_(output), buffer_(65536), used_(0) {}
    // Записывает блок байтов через буфер выходного потока.
    void raw(const char* data, std::size_t size) {
        flush();
        output_.write(data, static_cast<std::streamsize>(size));
        check();
    }
    // Записывает 32-битное целое значение Tecplot.
    void i32(I32 value) { bits(static_cast<std::uint32_t>(value), 4); }
    // Записывает вещественный маркер одинарной точности.
    void f32(float value) {
        std::uint32_t v;
        std::memcpy(&v, &value, 4);
        bits(v, 4);
    }
    // Записывает значение переменной двойной точности.
    void f64(double value) {
        U64 v;
        std::memcpy(&v, &value, 8);
        bits(v, 8);
    }
    // Записывает строку в представлении кодов символов Tecplot.
    void string(const std::string& value) {
        for (unsigned char c : value)
            i32(c);
        i32(0);
    }
    // Записывает вспомогательную пару имени и значения метаданных.
    void aux(const std::string& key, const std::string& value) {
        i32(1);
        string(key);
        i32(0);
        string(value);
    }
    // Передаёт накопленные байты в выходной поток.
    void flush() {
        if (used_) {
            output_.write(buffer_.data(), static_cast<std::streamsize>(used_));
            used_ = 0;
        }
        check();
    }

private:
    // Проверяет, что выходной поток не сообщил об ошибке записи.
    void check() {
        if (!output_)
            fail("Tecplot output failed");
    }
    // Преобразует числовое значение в его двоичное представление.
    void bits(U64 value, unsigned count) {
        if (buffer_.size() - used_ < count)
            flush();
        for (unsigned i = 0; i < count; ++i)
            buffer_[used_++] = static_cast<char>((value >> (8 * i)) & 255);
    }
    // Поток записи двоичного PLT.
    std::ostream& output_;
    // Промежуточный буфер байтов.
    std::vector<char> buffer_;
    // Заполненная часть буфера.
    std::size_t used_;
};
// Описание зоны задаёт тип сетки, расположение переменных и размеры связности.
void write_zone_header(Writer& writer, const Plan& plan, const FieldSnapshot* snapshot, std::size_t i) {
    const Zone& z = plan.zones[i];
    writer.f32(299);
    writer.string(z.name);
    writer.i32(-1);
    writer.i32(snapshot ? static_cast<I32>(i) : -1);
    writer.f64(snapshot ? snapshot->time : 0.0);
    writer.i32(-1);
    writer.i32(z.type);
    writer.i32(1);
    for (const Variable& v : plan.variables)
        writer.i32(z.cell_centered(v));
    writer.i32(0);
    writer.i32(0);
    writer.i32(z.nodes);
    if (z.type == 0) {
        writer.i32(1);
        writer.i32(1);
    } else {
        if (z.type == 6 || z.type == 7) {
            writer.i32(z.faces);
            writer.i32(checked(z.connectivity.size(), "Face-node count"));
            writer.i32(0);
            writer.i32(0);
        }
        writer.i32(z.cells);
        writer.i32(0);
        writer.i32(0);
        writer.i32(0);
    }
    writer.aux("Aero.MeshSHA256", plan.mesh_sha256);
    if (snapshot)
        writer.aux("Aero.Step", std::to_string(snapshot->step));
    writer.aux("Aero.SourceLocation", location(z.source));
    writer.i32(0);
}

// Атрибуты связывают каждую переменную PLT с компонентой и единицами исходного поля.
void write_field_attributes(Writer& writer, const Plan& plan) {
    for (std::size_t i = 0; i < plan.variables.size(); ++i) {
        const Variable& v = plan.variables[i];
        if (!v.field)
            continue;
        const std::pair<std::string, std::string> metadata[] = {
            {"Aero.Field", v.field->name},
            {"Aero.Component", v.field->components[v.component]},
            {"Aero.Units", v.field->units},
            {"Aero.Location", location(v.field->location)}};
        for (const auto& item : metadata) {
            writer.f32(899);
            writer.i32(static_cast<I32>(i));
            writer.string(item.first);
            writer.i32(0);
            writer.string(item.second);
        }
    }
}

// Записываем по одному столбцу. Для зон со своей нумерацией собираем значения в рабочем блоке.
void write_variable_values(Writer& writer, const Zone& z, const Variable& v, std::vector<char>& bytes,
                           int threads) {
    (void)threads;
    const std::size_t count = z.count(v);
    for (std::size_t first = 0; first < count;) {
        const I64 batch =
            static_cast<I64>(std::min<std::size_t>(count - first, bytes.size() / sizeof(double)));
        if (number_codec::native_little_endian && z.contiguous(v)) {
            // В этой зоне порядок значений совпадает с порядком в снимке.
            writer.raw(reinterpret_cast<const char*>(v.values + first),
                       static_cast<std::size_t>(batch) * sizeof(double));
        } else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (batch > 2048)
#endif
            for (I64 j = 0; j < batch; ++j) {
                const std::size_t i = static_cast<std::size_t>(j);
                const double value = z.value(v, first + i);
                number_codec::store_real(bytes.data() + i * 8, value);
            }
            writer.raw(bytes.data(), static_cast<std::size_t>(batch) * 8);
        }
        first += static_cast<std::size_t>(batch);
    }
}

// Записывает значения переменных и топологию одной зоны.
void write_zone_data(Writer& writer, const Plan& plan, const Zone& z, std::vector<char>& bytes) {
    writer.f32(299);
    for (std::size_t i = 0; i < plan.variables.size(); ++i)
        writer.i32(2);
    writer.i32(1);
    for (const Variable& v : plan.variables)
        writer.i32(z.active(v) ? 0 : 1);
    writer.i32(0);
    writer.i32(-1);
    for (std::size_t i = 0; i < plan.variables.size(); ++i)
        if (z.active(plan.variables[i])) {
            writer.f64(z.ranges[i].first);
            writer.f64(z.ranges[i].second);
        }
    for (const auto& v : plan.variables)
        if (z.active(v)) {
            if (plan.disk && v.field) {
                // Двоичные AEROFIELDS и PLT используют одинаковый порядок байтов.
                plan.disk->visit(*v.field, v.count, [&](const char* data, std::size_t n) { writer.raw(data, 8*n); });
            } else write_variable_values(writer, z, v, bytes, plan.threads);
        }
    for (I32 value : z.offsets)
        writer.i32(value);
    for (I32 value : z.connectivity)
        writer.i32(value);
    for (I32 value : z.left)
        writer.i32(value);
    for (I32 value : z.right)
        writer.i32(value);
}

// Формат PLT требует сначала описать все зоны и переменные, затем последовательно записать их данные.
void write_plan(std::ostream& out, const Plan& plan, const FieldSnapshot* snapshot) {
    Writer writer(out);
    writer.raw("#!TDV112", 8);
    writer.i32(1);
    writer.i32(0);
    writer.string(snapshot ? "Aero step " + std::to_string(snapshot->step) : "Aero mesh");
    writer.i32(static_cast<I32>(plan.variables.size()));
    for (const Variable& v : plan.variables)
        writer.string(v.name);
    for (std::size_t i = 0; i < plan.zones.size(); ++i)
        write_zone_header(writer, plan, snapshot, i);
    write_field_attributes(writer, plan);
    writer.f32(357); // Маркер конца заголовков PLT v112.
    std::vector<char> bytes(65536 * sizeof(double));
    for (const Zone& z : plan.zones)
        write_zone_data(writer, plan, z, bytes);
    writer.flush();
}

// Записывает заголовок, описания зон и блоки данных файла Tecplot.
void write_file(const std::string& path, const Mesh& mesh, const FieldSnapshot* snapshot,
                const Options& options) {
    if (path.empty() || path.find('\0') != std::string::npos)
        fail("Invalid output path");
    const Plan plan(mesh, snapshot, options);
    // Сначала завершаем запись рядом с назначением. Только готовый результат заменяет прежний PLT.
    StagedFieldFile file(path);
    std::ofstream output(file.temporary().c_str(), std::ios::binary);
    write_plan(output, plan, snapshot);
    output.close();
    if (!output) fail("Cannot finish Tecplot output: " + path);
    file.publish(true);
}
} // namespace
// Расписание уже привязало поля к сетке. Повторный обход сетки ради SHA-256 здесь не требуется.
void write_snapshot(std::ostream& out, const Mesh* mesh, const FieldSnapshot& snapshot, BinaryColumns* columns) {
    if (mesh) write_plan(out, Plan(*mesh, &snapshot, Options(), false, columns), &snapshot);
    else write_plan(out, Plan(snapshot), &snapshot);
}
// Проверяет входные данные и экспортирует сетку или снимок в Tecplot.
void write(std::ostream& output, const Mesh& mesh, const FieldSnapshot& snapshot, const Options& options) {
    const Plan plan(mesh, &snapshot, options);
    write_plan(output, plan, &snapshot);
}
// Проверяет входные данные и экспортирует сетку или снимок в Tecplot.
void write(const std::string& path, const Mesh& mesh, const FieldSnapshot& snapshot, const Options& options) {
    write_file(path, mesh, &snapshot, options);
}
// Проверяет входные данные и экспортирует сетку или снимок в Tecplot.
void write(const std::string& path, const Mesh& mesh, const Options& options) {
    write_file(path, mesh, nullptr, options);
}
}
}
