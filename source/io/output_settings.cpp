#include "output_settings.h"
#include "../parallel/modules.h"
#include "settings_text.h"
#include "diagnostic_fields.h"
#include "number_codec.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <iomanip>
#include <locale>
#include <set>
#include <sstream>

namespace aero {
namespace {
using namespace settings_detail;
// Читает список форматов одной зоны и сообщает о пропущенном или повторном значении.
void read_visualization(OutputZone& zone, const std::string& value, const std::string& where) {
    std::size_t start = 0;
    for (;;) {
        const auto comma = value.find(',', start);
        const auto token = trim(value.substr(start, comma - start));
        if (token == "tecplot" && !zone.tecplot) zone.tecplot = true;
        else if (token == "vtk" && !zone.vtk) zone.vtk = true;
        else throw std::invalid_argument(where + ": unknown or duplicate visualization format: " + token);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
}
// Заполняет один параметр текущей зоны. Совместимость всех параметров проверяется после чтения.
void entry(OutputZone& z, const std::string& key, const std::string& value, const std::string& where, std::vector<std::string>* warnings) {
    if (key == "file")
        z.file = label(value, where);
    else if (key == "wall_label") z.wall_label=label(value,where);
    else if (key == "submesh") z.submesh=label(value,where);
    else if (key == "every_steps") {
        z.every_steps = number_codec::parse_decimal(value, [&](const char* reason) {
            throw std::invalid_argument(where + ": " + reason);
        });
        if (!z.every_steps) throw std::invalid_argument(where + ": expected a positive step count");
    } else if (key == "every_time") {
        z.every_time = finite_real(value, where);
        if (z.every_time <= 0.) throw std::invalid_argument(where + ": expected a positive time interval");
    } else if (key == "multi" && (value == "1" || value == "0"))
        z.multi = value == "1";
    else if (key == "layout" && (value == "local" || value == "global"))
        z.global = value == "global";
    else if (key == "format" && value == "text")
        z.format = FieldFormat::Text;
    else if (key == "format" && value == "binary")
        z.format = FieldFormat::Binary;
    else if (key == "format" && value == "lzma")
        z.format = FieldFormat::Lzma;
    else if (key == "visualization")
        read_visualization(z, value, where);
    else if (key == "variables") {
        std::size_t start = 0;
        for (;;) {
            const auto comma = value.find(',', start);
            z.variables.push_back(trim(value.substr(start, comma - start)));
            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
    } else if (warnings && !key.empty() && key != "format" && key != "multi" && key != "layout") {
        warnings->push_back(where + ": unknown parameter " + key);
    } else
        throw std::invalid_argument(where + ": unknown setting or invalid value: " + key + " = " + value);
}
// Определяет расширение выходного имени для выбора формата.
std::size_t extension(const std::string& name) {
    const auto dot = name.rfind('.');
    // Точку ищем только в имени файла. Точка внутри имени каталога не относится к расширению.
    const auto leaf = name.find_last_of("/\\") + 1;
    return dot != std::string::npos && dot > leaf ? dot : name.size();
}
// Проверяет, может ли серия нумерованных файлов создать файл с указанным именем.
bool in_series(const std::string& file, const std::string& series) {
    const auto ext = extension(series);
    const auto prefix = series.substr(0, ext) + "_", suffix = series.substr(ext);
    if (file.size() < prefix.size() + 6 + suffix.size() || file.compare(0, prefix.size(), prefix) != 0 ||
        file.compare(file.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;
    const auto digits = file.substr(prefix.size(), file.size() - prefix.size() - suffix.size());
    return digits.find_first_not_of("0123456789") == std::string::npos &&
           (digits.size() == 6 || digits.front() != '0');
}
}
// Строит ключ сравнения путей, общий для проверки конфликтов между зонами и контрольными точками.
std::string output_path_key(std::string file) {
    std::replace(file.begin(), file.end(), '\\', '/');
    if (file.empty() || file.find_first_of(":*?\"<>|") != std::string::npos)
        throw std::invalid_argument("invalid relative file name: " + file);
    for (unsigned char c : file)
        if (c < 32)
            throw std::invalid_argument("control character in file name");
    std::string normalized;
    std::size_t start = 0;
    for (;;) {
        const auto slash = file.find('/', start);
        const auto part = file.substr(start, slash - start);
        if (part.empty() || (part == "." && slash == std::string::npos) ||
            (part != "." && (part.back() == '.' || part.back() == ' ')))
            throw std::invalid_argument("expected a relative file name under --output: " + file);
        if (part != ".") {
            if (!normalized.empty())
                normalized += '/';
            normalized += part;
        }
        if (slash == std::string::npos)
            break;
        start = slash + 1;
    }
#ifdef _WIN32
    for (char& c : normalized)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
#endif
    return normalized;
}
// Меняет расширение только у последнего компонента пути.
std::string visualization_filename(const std::string& name, const char* suffix) {
    return name.substr(0, extension(name)) + suffix;
}
// При раздельном выводе вставляет номер шага перед расширением файла.
std::string output_filename(const OutputZone& zone, std::uint64_t step) {
    if (!zone.multi)
        return zone.file;
    std::ostringstream suffix;
    suffix.imbue(std::locale::classic());
    suffix << '_' << std::setfill('0') << std::setw(6) << step;
    auto name = zone.file;
    return name.insert(extension(name), suffix.str());
}
// Проверяет имена, поля и расписания зон вывода.
void validate_output_zones(const std::vector<OutputZone>& zones) {
    std::set<std::string> names;
    std::vector<std::pair<std::string, const OutputZone*>> paths;
    std::set<std::string> supported={"primary","fluctuations","dp/dt"};
    for(unsigned c=0;c<diagnostic::ColumnCount;++c) supported.insert(diagnostic::name(c));
#ifdef AERO_HAS_TURBULENCE
    supported.insert("turbulence");
#else
    supported.erase("nut"); supported.erase("nu_tilde");
#endif
    for (const auto& z : zones) {
        const std::string prefix = "[" + z.name + "] ";
        if (z.name.empty() || !names.insert(z.name).second)
            throw std::invalid_argument(prefix + "invalid or duplicate zone");
        if (!std::isfinite(z.every_time) || z.every_time < 0 || (z.every_steps > 0) == (z.every_time > 0))
            throw std::invalid_argument(prefix + "choose exactly one of every_steps and every_time");
        if (z.variables.empty())
            throw std::invalid_argument(prefix + "missing variables");
        std::set<std::string> used;
        for (const auto& var : z.variables)
            if (!supported.count(var) || !used.insert(var).second)
                throw std::invalid_argument(prefix + "unknown or duplicate variable: " + var);
        if(used.count("Cp") && (used.size()!=1 || z.wall_label.empty()))
            throw std::invalid_argument(prefix+"Cp needs a separate zone with wall_label");
        if(!used.count("Cp") && !z.wall_label.empty())
            throw std::invalid_argument(prefix+"wall_label requires variables = Cp");
        if (!z.submesh.empty() && !z.wall_label.empty())
            throw std::invalid_argument(prefix+"choose wall_label or submesh");
        // Проверяем, могут ли две зоны при каком-либо шаге записать в один и тот же файл.
        const auto snapshot = output_path_key(output_filename(z, 1));
        const auto first = snapshot.substr(0, snapshot.find('/'));
        if (first == "aero.log" || first == "control.txt")
            throw std::invalid_argument(prefix + "field output overlaps a solver file: " + z.file);
        if (z.global) {
            if (z.submesh.empty() && z.format != FieldFormat::Binary)
                throw std::invalid_argument(prefix + "layout = global requires format = binary");
            if (first == "recovery-0.bin" || first == "recovery-1.bin" || first == "recovery.lock")
                throw std::invalid_argument(prefix + "field output overlaps recovery data: " + z.file);
        }
        // Проверяем также пересечения .fields с визуализацией этой и соседних зон.
        const auto check_path = [&](const std::string& name) {
            const auto key = output_path_key(name);
            for (const auto& previous : paths) {
                const auto& other = *previous.second;
                const bool overlap = z.multi == other.multi ? key == previous.first :
                    (z.multi ? in_series(previous.first, key) : in_series(key, previous.first));
                if (overlap)
                    throw std::invalid_argument(prefix + "file names overlap with zone " + other.name);
            }
            paths.emplace_back(key, &z);
        };
        check_path(z.file);
        if (z.tecplot) check_path(visualization_filename(z.file, ".plt"));
        if (z.vtk) check_path(visualization_filename(z.file, ".vtk"));
    }
}
// Разбирает секции зон вывода и проверяет их параметры.
std::vector<OutputZone> read_output_zones(const std::string& path, std::vector<std::string>* warnings) {
    errno = 0;
    std::ifstream input(path.c_str());
    if (!input) {
        if (errno == ENOENT)
            return {};
        throw std::runtime_error("cannot open output settings: " + path);
    }
    std::vector<OutputZone> zones;
    std::set<std::string> keys;
    std::string line, where;
    std::size_t row = 0;
    while (settings_detail::next_line(input, path, row, line, where)) {
        if (line.front() == '[' && line.back() == ']') {
            zones.emplace_back();
            zones.back().name =
                settings_detail::label(settings_detail::trim(line.substr(1, line.size() - 2)), where);
            // Имена параметров должны быть уникальны внутри каждой отдельной секции.
            keys.clear();
            continue;
        }
        const auto eq = line.find('=');
        if (zones.empty() || eq == std::string::npos)
            throw std::invalid_argument(where + ": expected [zone] then key = value");
        const auto key = settings_detail::trim(line.substr(0, eq));
        if (!keys.insert(key).second)
            throw std::invalid_argument(where + ": duplicate " + key);
        entry(zones.back(), key, settings_detail::trim(line.substr(eq + 1)), where, warnings);
        // Нумерация .submesh общая для всех процессов, включая точки на стыке разделов.
        if (!zones.back().submesh.empty() && keys.count("layout") && !zones.back().global)
            throw std::invalid_argument(where + ": submesh requires layout = global (or omit layout)");
    }
    if (input.bad() || !input.eof())
        throw std::runtime_error("error reading output settings: " + path);
    for (auto& zone : zones) if (!zone.submesh.empty()) zone.global = true;
    try {
        validate_output_zones(zones);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(path + ": " + e.what());
    }
    return zones;
}
}
