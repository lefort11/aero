#include "startup.h"
#include "../io/settings_text.h"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <system_error>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace aero {
namespace {
bool directory_exists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFMT) == S_IFDIR;
}
}
void create_output_directory(std::string path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    if (directory_exists(path))
        return;
    while (path.size() > 1 && path.back() == '/')
        path.pop_back();
    const auto slash = path.rfind('/');
    if (slash != std::string::npos && slash > 0)
        create_output_directory(path.substr(0, slash));
#ifdef _WIN32
    const int result = _mkdir(path.c_str());
#else
    const int result = mkdir(path.c_str(), 0777);
#endif
    if (result && !directory_exists(path))
        throw std::runtime_error("Cannot create output directory " + path);
}
void stage(const char* name) {
    std::cout << name << "... " << std::flush;
}
void done() {
    std::cout << "done\n";
}
std::string settings_text(const std::string& path) {
    auto in = settings_detail::open(path);
    std::ostringstream text;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (first && line.compare(0, 3, "\xef\xbb\xbf") == 0)
            line.erase(0, 3);
        first = false;
        line = settings_detail::trim(line);
        if (!line.empty() && line.front() != '#')
            text << line << '\n';
    }
    if (in.bad() || !in.eof())
        throw std::runtime_error("Cannot read " + path);
    return text.str();
}
void show_settings(const RunSettings& r) {
    std::cout << std::setprecision(15) << "Run settings\ninput = " << r.directories.input
              << "\noutput = " << r.directories.output << "\ninteractive = " << r.directories.interactive
              << "\nmesh = " << r.mesh_path << "\nbackend = " << (r.use_cuda ? "cuda" : "cpu")
              << "\nreconstruction = " << r.reconstruction
              << "\nreconstruction_variables = " << r.reconstruction_variables
              << "\neuler_solver = " << r.euler_solver << "\ntime_scheme = " << r.time_scheme
              << "\ntrace_ratio_min = " << r.trace_ratio_min << "\ntrace_ratio_max = " << r.trace_ratio_max
              << "\ngamma = " << r.gamma << "\ncfl = " << r.cfl << "\nfinal_time = " << r.final_time
              << "\nomp_threads = " << r.omp_threads << " (runtime maximum ";
#ifdef _OPENMP
    std::cout << omp_get_max_threads();
#else
    std::cout << 1;
#endif
    std::cout << ")\nflowreport_every = " << r.flowreport_every << '\n';
    if (r.use_cuda) {
        std::cout << "devices =";
        for (auto id : r.devices)
            std::cout << ' ' << id;
        std::cout << '\n';
    }
    std::cout << "Output zones = " << r.output_zones.size() << '\n';
    for (const auto& z : r.output_zones) {
        std::cout << "  [" << z.name << "] file = " << z.file;
        if (z.every_steps)
            std::cout << "  every_steps = " << z.every_steps;
        else
            std::cout << "  every_time = " << z.every_time;
        std::cout << "  layout = " << (z.global ? "global" : "local") << "  multi = " << z.multi
                  << "  format = "
                  << (z.format == FieldFormat::Text     ? "text"
                      : z.format == FieldFormat::Binary ? "binary"
                                                        : "lzma")
                  << "  variables =";
        for (const auto& v : z.variables)
            std::cout << ' ' << v;
        std::cout << '\n';
    }
}
void show_boundary_labels(const Mesh& mesh) {
    const auto& data = mesh.data();
    std::vector<std::size_t> counts(data.patch_names.size());
    for (auto patch : data.boundary_patch)
        ++counts.at(static_cast<std::size_t>(patch));
    std::cout << "Boundary label\tFaces\n";
    for (std::size_t i = 0; i < counts.size(); ++i)
        std::cout << data.patch_names[i] << '\t' << counts[i] << '\n';
}
bool prepare_boundary_file(const RunSettings& run, const std::vector<std::string>& names) {
    const std::string& path = run.boundary_path;
    std::FILE* file = nullptr;
    int error = 0;
    if (!std::ifstream(path.c_str())) {
#ifdef _MSC_VER
        error = fopen_s(&file, path.c_str(), "wx");
#else
        file = std::fopen(path.c_str(), "wx");
        if (!file)
            error = errno;
#endif
    }
    // В режиме исключительного создания только один процесс может записать заготовку.
    const bool created = file != nullptr;
    if (file) {
        std::string sections = "# Fill every boundary section. For a periodic pair, keep one section with "
                               "pair and translation.\n";
        for (const auto& name : names) {
            if (name.find_first_of("#\"") == std::string::npos && name == settings_detail::trim(name)) {
                sections += "\n[" + name + "]\n";
                continue;
            }
            sections += "\n[\"";
            for (char c : name) {
                sections += c;
                if (c == '"')
                    sections += c;
            }
            sections += "\"]\n";
        }
        const bool written = std::fwrite(sections.data(), 1, sections.size(), file) == sections.size();
        if (std::fclose(file) != 0 || !written)
            throw std::runtime_error("Cannot write " + path);
        std::cout << "Created " << path << " with empty boundary sections.\n";
    } else if (error && error != EEXIST)
        throw std::system_error(error, std::generic_category(), "Cannot open " + path);
    // В интерактивном запуске условия читаются снова после редактирования файла пользователем.
    if (run.directories.interactive) {
        std::cout << "Edit " << path << ", save it and press Enter to reload boundary conditions.\n"
                  << std::flush;
        std::string reply;
        if (!std::getline(std::cin, reply))
            throw std::runtime_error("End of input while waiting for Enter");
    } else if (created && !names.empty()) {
        std::cout << "Fill " << path << " and run aero again.\n";
        return false;
    }
    return true;
}
unsigned create_resume_directory(const std::string& prefix) {
    for (unsigned i = 1; i; i++) {
        // Номер резервируется созданием каталога. Если его занял предыдущий запуск, пробуем следующий.
        const std::string path = prefix + std::to_string(i);
#ifdef _WIN32
        const int status = _mkdir(path.c_str());
#else
        const int status = mkdir(path.c_str(), 0777);
#endif
        if (!status)
            return i;
        if (errno != EEXIST)
            throw std::runtime_error("Cannot create output directory " + path);
    }
    throw std::runtime_error("Too many recovery output directories");
}
namespace {
void prepare_zone_directory(const std::string& directory, const OutputZone& zone) {
    const auto slash = zone.file.find_last_of("/\\");
    if (slash != std::string::npos)
        create_output_directory(directory + "/" + zone.file.substr(0, slash));
}
}
void prepare_output_directories(const RunSettings& run) {
    create_output_directory(run.directories.output);
    if (!run.directories.geometry_only)
        for (const auto& z : run.output_zones)
            prepare_zone_directory(run.directories.output, z);
}
void prepare_zone_directories(const std::string& directory, const std::vector<OutputZone>& zones,
                              bool global) {
    create_output_directory(directory);
    for (const auto& z : zones)
        if (z.global == global)
            prepare_zone_directory(directory, z);
}
}
