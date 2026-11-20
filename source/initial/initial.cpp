#include "initial.h"
#include "../parallel/parallel_for.h"
#include "../bc/periodic.h"
#include "../io/expression.h"
#include "../io/flow_values.h"
#include "../io/settings_text.h"
#include "../io/fields.h"
#include "../eulersolver/euler.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>

namespace aero {
namespace {
using settings_detail::trim;
struct Setting {
    std::string value;
    std::string where; // Имя файла и номер строки для диагностики привязки полей.
};
struct Settings {
    const char* filename;
    std::map<std::string, Setting> entries;
};
std::invalid_argument error(const std::string& where, const std::string& msg) {
    return std::invalid_argument(where + ": " + msg);
}
Settings read_settings(std::istream& input, const char* filename) {
    Settings out{filename, {}};
    std::string line;
    std::size_t row = 0;
    while (std::getline(input, line)) {
        ++row;
        const auto where = std::string(filename) + ":" + std::to_string(row);
        line = settings_detail::clean(line, where, row);
        if (line.empty())
            continue;
        const auto cut = line.find_first_of(" \t=");
        if (cut == std::string::npos)
            throw error(where, "missing value");
        std::string key = line.substr(0, cut), value = trim(line.substr(cut));
        if (!value.empty() && value[0] == '=')
            value = trim(value.substr(1));
        if (key.empty() || value.empty())
            throw error(where, "missing key or value");
        if (!out.entries.emplace(key, Setting{value, where}).second)
            throw error(where, "duplicate " + key);
    }
    if (input.bad() || !input.eof())
        throw std::runtime_error(std::string(filename) + ": read failed");
    return out;
}
Setting take(Settings& settings, const std::string& key) {
    const auto it = settings.entries.find(key);
    if (it == settings.entries.end())
        throw std::invalid_argument(std::string(settings.filename) + ": missing " + key);
    const Setting s = it->second;
    settings.entries.erase(it);
    return s;
}
// Разбирает имена полей и путей, сохраняя пробелы внутри кавычек и номер исходной строки при ошибке.
std::vector<std::string> words(const Setting& s) {
    std::vector<std::string> result;
    std::size_t pos = 0;
    while ((pos = s.value.find_first_not_of(" \t", pos)) != std::string::npos) {
        const bool quote = s.value[pos] == '"';
        if (quote)
            ++pos;
        const auto end = quote ? s.value.find('"', pos) : s.value.find_first_of(" \t", pos);
        if (quote && end == std::string::npos)
            throw error(s.where, "unclosed quote");
        const std::string word = s.value.substr(pos, end - pos);
        if (word.empty() || word.find('"') != std::string::npos)
            throw error(s.where, "invalid name");
        result.push_back(word);
        if (end == std::string::npos)
            break;
        pos = end + (quote ? 1 : 0);
        if (quote && pos < s.value.size() && s.value[pos] != ' ' && s.value[pos] != '\t')
            throw error(s.where, "missing separator after quoted name");
    }
    return result;
}
std::string single(const Setting& s) {
    const auto tokens = words(s);
    if (tokens.size() != 1)
        throw error(s.where, "expected one name or path");
    return tokens[0];
}
std::string resolve(const std::string& dir, const std::string& path) {
    if (dir.empty() || path[0] == '/' || path[0] == '\\' || (path.size() > 1 && path[1] == ':'))
        return path;
    return dir + "/" + path;
}
void finish_settings(const Settings& settings, std::vector<std::string>* warnings) {
    for (const auto& item : settings.entries) {
        if (!warnings) throw error(item.second.where, "unexpected key " + item.first);
        warnings->push_back(item.second.where + ": unknown parameter " + item.first);
    }
}
std::vector<std::string> primitives(unsigned d) {
    std::vector<std::string> names = {"rho", "ux"};
    if (d >= 2)
        names.push_back("uy");
    if (d == 3)
        names.push_back("uz");
    names.push_back("p");
    return names;
}
// Однородный поток задаём в осях компонент импульса. Азимут отсчитывается от x к y,
// угол возвышения — от плоскости xy к z. Оба входных угла заданы в градусах.
std::vector<double> freestream(Settings& settings, unsigned d, double gamma, double mach) {
    if (!(mach > 0. && std::isfinite(mach)))
        throw std::invalid_argument("initial_type freestream requires gas.txt:mach > 0");
    const auto angle = [&](const char* name) {
        if (!settings.entries.count(name)) return 0.;
        const auto s = take(settings, name);
        return settings_detail::finite_real(s.value, s.where + ": " + name);
    };
    const double degrees = std::acos(-1.) / 180.;
    const double phi = std::remainder(angle("phi"), 360.) * degrees;
    const double elevation = angle("theta");
    if (std::abs(elevation) > 90. || (d < 3 && elevation != 0.))
        throw std::invalid_argument("init.txt: theta must be in [-90,90] and zero below 3D");
    const double theta = elevation * degrees;
    const double velocity[] = {std::cos(theta)*std::cos(phi), std::cos(theta)*std::sin(phi), std::sin(theta)};
    if (d == 1 && std::abs(velocity[1]) > 1.e-14)
        throw std::invalid_argument("init.txt: phi must be a multiple of 180 degrees in 1D");
    // При rho_inf = |u_inf| = 1 из a_inf^2 = gamma*p_inf/rho_inf получаем давление.
    std::vector<double> q(d + 2, 1.);
    std::copy_n(velocity, d, q.begin() + 1);
    q.back() = 1. / mach / gamma / mach;
    return q;
}

// Преобразует примитивные переменные одного узла в консервативные прямо на месте.
void primitive_to_conservative(double* q, unsigned d, double gamma) {
    const double rho = q[0], p = q[d + 1];
    if (!(rho > 0 && p > 0))
        throw std::domain_error("expected positive rho and p");
    // Начинаем с внутренней энергии. При замене каждой скорости импульсом добавляем её кинетический вклад.
    double energy = p / (gamma - 1);
    for (unsigned k = 1; k <= d; ++k) {
        const double u = q[k];
        q[k] = rho * u;
        energy += .5 * q[k] * u;
    }
    q[d + 1] = energy;
}
// Компоненты в снимке хранятся целыми столбцами: сначала все узлы первой компоненты, затем второй.
struct Column {
    const Field* field;
    std::size_t column;
};
const Field* find_node_field(const FieldSnapshot& snapshot, const std::string& name) {
    for (const Field& field : snapshot.fields)
        if (field.location == FieldLocation::Node && field.name == name) return &field;
    return nullptr;
}
const Field& node_field(const FieldSnapshot& snapshot, const std::string& name, const std::string& where) {
    const Field* field = find_node_field(snapshot, name);
    if (!field) throw error(where, "missing NODE field '" + name + "'");
    return *field;
}
Column component(const Field& f, const std::string& name, const std::string& where) {
    const auto it = std::find(f.components.begin(), f.components.end(), name);
    if (it == f.components.end())
        throw error(where, "missing component '" + name + "' of '" + f.name + "'");
    return {&f, static_cast<std::size_t>(it - f.components.begin())};
}
// В привязке «имя поля [компонента]» второй аргумент обязателен только для составных полей.
Column mapped_column(const FieldSnapshot& snapshot, const Setting& setting) {
    const auto names = words(setting);
    if (names.empty() || names.size() > 2) throw error(setting.where, "expected field name and optional component");
    const Field& f = node_field(snapshot, names.front(), setting.where);
    if (names.size() == 1 && f.components.size() != 1) throw error(setting.where, "choose a component of " + f.name);
    return component(f, names.size() == 2 ? names.back() : f.components.front(), setting.where);
}
// Читает только нужные узлы архива в MPI или сверяет всю сетку при обычном запуске.
FieldSnapshot read_snapshot(const Mesh& mesh, const std::string& path, const FieldNodeSelection* selection) {
    FieldSnapshot snapshot;
    if (selection) {
        if (selection->nodes.size() != mesh.view().node_count)
            throw std::invalid_argument("invalid local node count for field snapshot");
        snapshot.fields = FieldSnapshot::read_nodes(path, *selection);
    } else {
        snapshot = FieldSnapshot::read(path);
        snapshot.check_mesh(mesh);
    }

    return snapshot;
}

// Общий для init.txt и exact.txt читатель. Снимок владеет памятью столбцов до конца чтения.
struct NodalSnapshot {
    FieldSnapshot snapshot;      // Таблицы файла в нумерации локальной сетки.
    std::vector<Column> columns; // Плотность, скорости и давление либо консервативные компоненты.
    Column energy{nullptr, 0};   // Необязательная отдельная колонка E для сравнения с эталоном.
    bool conservative = false;  // Составное поле содержит импульсы и полную энергию.
    MeshIndex node_count = 0;    // Расстояние между компонентами одного поля в памяти.

    void load(Settings& settings, const Mesh& mesh, const std::string& directory,
              const FieldNodeSelection* selection, const std::vector<std::string>& names, bool exact) {
        const Setting path = take(settings, "path");
        snapshot = read_snapshot(mesh, resolve(directory, single(path)), selection);
        node_count = mesh.view().node_count;
        if (settings.entries.count("field")) {
            const Setting spec = take(settings, "field");
            const auto name = single(spec);
            const Field& f = node_field(snapshot, name, spec.where);
            if (f.components.size() != names.size()) throw error(spec.where, "incorrect conservative component count");
            columns.push_back(component(f, "rho", spec.where));
            for (std::size_t j = 1; j + 1 < names.size(); ++j)
                columns.push_back(component(f, "rho_" + names[j], spec.where));
            columns.push_back(component(f, "E", spec.where));
            conservative = true;
            return;
        }
        // В штатном снимке имена скалярных полей совпадают с именами величин в отчёте.
        for (const auto& name : names) {
            const Setting spec = exact && !settings.entries.count(name) ? Setting{name, path.where} : take(settings, name);
            columns.push_back(mapped_column(snapshot, spec));
        }
        if (exact && settings.entries.count("E"))
            energy = mapped_column(snapshot, take(settings, "E"));
        else if (exact && find_node_field(snapshot, "E"))
            energy = mapped_column(snapshot, Setting{"E", path.where});
    }

    double value(const Column& c, MeshIndex node) const {
        return c.field->values[c.column * node_count + node];
    }
    void read_node(MeshIndex node, double* q) const {
        for (std::size_t k = 0; k < columns.size(); ++k) q[k] = value(columns[k], node);
    }
};
}

// Из снимка сохраняем только представителей, которые участвуют в нормах данного процесса.
std::vector<std::array<double, 6>> read_exact_fields(const Mesh& mesh, double gamma, std::istream& input,
        const std::string& directory, const FieldNodeSelection* selection, const LinearBackground& background,
        const std::vector<MeshIndex>& nodes, std::vector<std::string>* warnings) {
    const unsigned d = mesh.view().dimension;
    Settings settings = read_settings(input, "exact.txt");
    NodalSnapshot file;
    file.load(settings, mesh, directory, selection, primitives(d), true);
    finish_settings(settings, warnings);
    std::vector<std::array<double, 6>> reference(nodes.size());
    parallel_for(nodes.size(), [&](MeshIndex i) {
        const MeshIndex node = nodes[i];
        if (node >= file.node_count) throw std::out_of_range("exact.txt: invalid local node");
        double q[5];
        file.read_node(node, q);
        if (file.conservative) {
            const double energy = q[d + 1];
            if (background.active()) background.to_primitive(d, q, gamma);
            reference[i] = output_values(d, q, gamma, node, background);
            reference[i][5] = energy;
        } else {
            reference[i] = primitive_output_values(d, q, gamma, background);
            // Если E записана, сравниваем именно сохранённые числа, без пересчёта из давления.
            if (file.energy.field) reference[i][5] = file.value(file.energy, node);
        }
    });
    return reference;
}

void check_periodic_initial(const double* ref, const double* q, unsigned width, MeshIndex owner,
                            MeshIndex node) {
    for (unsigned k = 0; k < width; ++k) {
        const double scale = std::max(std::fabs(q[k]), std::fabs(ref[k]));
        if (std::fabs(q[k] - ref[k]) > 1e-13 + 1e-10 * scale)
            throw std::domain_error("initial: periodic mismatch between nodes " + std::to_string(owner) +
                                    " and " + std::to_string(node));
    }
}
std::vector<double> read_initial(const Mesh& mesh, double gamma, std::istream& input,
                                 const std::string& directory, const PeriodicView* seam,
                                 const FieldNodeSelection* selection, const LinearBackground& background, std::vector<std::string>* warnings, double mach) {
    if (!(gamma > 1 && std::isfinite(gamma)))
        throw std::invalid_argument("initial: invalid gamma");
    const auto m = mesh.view();
    const unsigned d = m.dimension, width = d + 2;
    // Размерность определяет состав переменных: плотность, активные скорости и давление.
    const auto names = primitives(d);
    Settings settings = read_settings(input, "init.txt");
    const std::string type = single(take(settings, "initial_type"));
    std::vector<Expression> expressions;
    NodalSnapshot file;
    std::vector<double> uniform;
    if (type == "formula") {
        for (const auto& name : names) {
            const auto s = take(settings, name);
            try {
                expressions.emplace_back(s.value);
            } catch (const std::invalid_argument& e) {
                throw error(s.where, name + ": " + e.what());
            }
        }
    } else if (type == "freestream") {
        if (background.active()) throw std::invalid_argument("initial_type freestream requires Equations = Euler");
        uniform = freestream(settings, d, gamma, mach);
    } else if (type == "file") {
        file.load(settings, mesh, directory, selection, names, false);
    } else
        throw std::invalid_argument("initial: unknown initial_type " + type);
    finish_settings(settings, warnings);
    const MeshIndex count = seam ? seam->node_count : m.node_count;
    if (seam && (seam->original_node_count != m.node_count || !seam->node_dof || count > m.node_count))
        throw std::invalid_argument("initial: incompatible periodic view");
    std::vector<double> result;
    if (count > result.max_size() / width)
        throw std::length_error("initial state too large");
    result.resize(static_cast<std::size_t>(count) * width);
    std::vector<MeshIndex> owner(seam ? count : 0, m.node_count);
    const bool from_file = type == "file";
    // Один вызов заполняет смежные компоненты одного узла и проверяет получившиеся давление и плотность.
    auto evaluate = [&](MeshIndex n, double* q) {
        try {
            if (!uniform.empty()) {
                std::copy(uniform.begin(), uniform.end(), q);
            } else if (from_file) {
                file.read_node(n, q);
            } else {
                for (unsigned k = 0; k < width; ++k) {
                    try {
                        q[k] = expressions[k].evaluate(m.x[n], m.y[n], m.z[n]);
                    } catch (const std::domain_error& e) {
                        throw std::domain_error(names[k] + ": " + e.what());
                    }
                }
            }
            if (background.active()) {
                // Формулы и скалярные поля содержат пульсации. Составное поле
                // (rho', (rho*u)', E') переводим в рабочие переменные один раз.
                if (file.conservative) background.to_primitive(d,q,gamma);
                for (unsigned k=0; k<width; ++k)
                    if (!std::isfinite(q[k])) throw std::domain_error("nonfinite perturbation");
                return;
            }
            if (!file.conservative)
                primitive_to_conservative(q, d, gamma);
            EulerState physical;
            if (euler_state(d, q, gamma, &physical) != EulerStatus::Success)
                throw std::domain_error("invalid conservative state");
        } catch (const std::domain_error& e) {
            throw std::domain_error("initial node " + std::to_string(n) + ": " + e.what());
        }
    };
    // На периодической границе несколько геометрических узлов делят одну степень свободы.
    // Представителем берём первый узел в порядке сетки.
    if (seam) {
        for (MeshIndex n = 0; n < m.node_count; ++n) {
            const MeshIndex id = seam->node_dof[n];
            if (id >= count)
                throw std::invalid_argument("initial: periodic node index out of range");
            if (owner[id] == m.node_count)
                owner[id] = n;
        }
        if (std::find(owner.begin(), owner.end(), m.node_count) != owner.end())
            throw std::invalid_argument("initial: incomplete periodic map");
    }
    parallel_for(count, [&](MeshIndex id) { evaluate(seam ? owner[id] : id, result.data() + width * id); });
    // Проверяем каждую копию независимо: формулы или входной файл обязаны давать согласованные значения.
    if (seam) {
        parallel_for(m.node_count, [&](MeshIndex n) {
            const MeshIndex id = seam->node_dof[n];
            if (owner[id] == n)
                return;
            double q[5] = {};
            evaluate(n, q);
            const double* dst = result.data() + width * id;
            check_periodic_initial(dst, q, width, owner[id], n);
        });
    }
    return result;
}
std::vector<double> read_initial_file(const Mesh& mesh, double gamma, const std::string& path,
                                      const PeriodicView* seam, const LinearBackground& background, double mach) {
    std::ifstream in(path.c_str());
    if (!in)
        throw std::runtime_error("cannot open initial settings: " + path);
    const auto slash = path.find_last_of("/\\");
    return read_initial(mesh, gamma, in, slash == std::string::npos ? "." : path.substr(0, slash + 1), seam,
                        nullptr, background, nullptr, mach);
}
}
