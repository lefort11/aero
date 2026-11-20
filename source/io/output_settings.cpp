#include "output_settings.h"
#include "settings_text.h"
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
// Заполняет один параметр текущей зоны. Совместимость всех параметров проверяется после чтения.
void entry(OutputZone& z, const std::string& key, const std::string& value, const std::string& where, std::vector<std::string>* warnings) {
    if (key == "file")
        z.file = label(value, where);
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
void validate_output_zones(const std::vector<OutputZone>& zones) {
    std::set<std::string> names;
    std::vector<std::string> keys;
    const std::set<std::string> supported = {"rho", "ux", "uy", "uz", "p", "E"};
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
        // Проверяем, могут ли две зоны при каком-либо шаге записать в один и тот же файл.
        const auto key = output_path_key(z.file);
        if (z.global) {
            if (z.format != FieldFormat::Binary)
                throw std::invalid_argument(prefix + "layout = global requires format = binary");
            const auto path = output_path_key(output_filename(z, 1));
            const auto first = path.substr(0, path.find('/'));
            if (first == "recovery-0.bin" || first == "recovery-1.bin" || first == "recovery.lock")
                throw std::invalid_argument(prefix + "field output overlaps recovery data: " + z.file);
        }
        for (std::size_t i = 0; i < keys.size(); ++i) {
            const bool overlap = z.multi == zones[i].multi
                                     ? key == keys[i]
                                     : (z.multi ? in_series(keys[i], key) : in_series(key, keys[i]));
            if (overlap)
                throw std::invalid_argument(prefix + "file names overlap with zone " + zones[i].name);
        }
        keys.push_back(key);
    }
}
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
    std::string line;
    std::size_t row = 0;
    while (std::getline(input, line)) {
        const auto where = path + ":" + std::to_string(++row);
        line = settings_detail::clean(line, where, row);
        if (line.empty())
            continue;
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
    }
    if (input.bad() || !input.eof())
        throw std::runtime_error("error reading output settings: " + path);
    try {
        validate_output_zones(zones);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(path + ": " + e.what());
    }
    return zones;
}
}
