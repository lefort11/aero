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
#include <fcntl.h>
#include <io.h>
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
              << "\nEquations = " << (r.background.active() ? "Linear" : "Euler")
              << "\nmesh = " << r.mesh_path << "\nbackend = " << (r.use_cuda ? "cuda" : "cpu")
              << "\nreconstruction = " << r.reconstruction
              << "\nstencil_geometry = " << r.stencil_geometry << "\nstencil_max_angle = " << r.stencil_max_angle
                << "\nreconstruction_variables = " << r.reconstruction_variables
              << "\neuler_solver = " << r.euler_solver << "\ntime_scheme = " << r.time_scheme
              << "\nupw_coeff = " << r.upw_coeff << "\ntrace_ratio_min = " << r.trace_ratio_min << "\ntrace_ratio_max = " << r.trace_ratio_max
              << "\ngamma = " << r.gamma << "\ncfl = " << r.cfl;
#ifdef AERO_HAS_VISCOSITY
    if (r.viscosity.law != "off") {
        const auto& v = r.viscosity;
        std::cout << "\nviscosity = " << v.law << " (AES)"
                  << "\nreynolds = " << v.reynolds << "\nprandtl = " << v.prandtl
                  << "\nt_amb = " << v.t_amb;
        if (v.law == "sutherland") std::cout << "\nsutherland_s = " << v.sutherland_s;
    }
#endif
#ifdef AERO_HAS_TURBULENCE
    if (r.turbulence.enabled) {
        const auto& t=r.turbulence;
        std::cout << "\nTurbulence = On (" << (t.ddes ? "DDES SA" : t.uses_sa() ? "RANS SA" : "LES Sigma") << ')';
        if (t.uses_sa()) std::cout << "\nnut_nu_amb = " << t.nut_nu_amb;
        if (t.les.scale!=turbulence::Scale::Off) std::cout << "\nsubgridscale = " << turbulence::scale_name(t.les.scale);
        if (t.ddes) std::cout << "\nc_des = " << t.c_des;
        if (t.uses_sigma()) std::cout << "\nc_sigma = " << t.les.c_sigma;
        if (t.les.scale==turbulence::Scale::Sla)
            std::cout << "\nswitch_on_vtm = " << (t.les.switch_on_vtm ? "on" : "off") << "\nF_KH_lim = " << t.les.f_kh_lim;
        std::cout << "\npr_t = " << t.prandtl;
    }
#endif
    if (r.mach > 0.) std::cout << "\nmach = " << r.mach;
    if (r.background.active()) {
        std::cout << "\nBackground: rho0 = " << r.background.rho << ", p0 = " << r.background.p;
        const char* names[] = {"ux0", "uy0", "uz0"};
        for (unsigned j = 0; j < r.background_velocities; ++j)
            std::cout << ", " << names[j] << " = " << r.background.velocity[j];
        std::cout << ", c0 = " << r.background.sound << "\nOutput fields are perturbations.";
    }
    if (r.t_fo > 0.) std::cout << "\nt_fo = " << r.t_fo;
    if (!r.steady()) std::cout << "\nfinal_time = " << r.final_time;
    std::cout << "\nomp_threads = " << r.omp_threads << " (runtime maximum ";
#ifdef _OPENMP
    std::cout << omp_get_max_threads();
#else
    std::cout << 1;
#endif
    std::cout << ")\nreport_every = " << r.report_every << '\n';
    if (r.prophecy_every_step) std::cout << "prophecy_every_step = " << r.prophecy_every_step << '\n';
    if (r.steady())
        std::cout << "local_time_step = " << (r.local_time_step ? "on" : "off") << "\nsteady_criterion = " << (r.steady_absolute ? "absolute" : "relative") << "\nsteady_tol = " << r.steady_tol << "\nsteady_max_steps = " << r.steady_max_steps << '\n';
    if (r.local_time_step) {
        std::cout << "local_time_step_max_ratio = ";
        if (r.local_time_step_max_ratio) std::cout << r.local_time_step_max_ratio;
        else std::cout << "off";
        std::cout << '\n';
    }
    if (r.uses_implicit_time())
        std::cout <<  "solver = BiCGSTAB\njacobian_flux = " << (r.implicit_settings.roe ? "Roe" : "LF")
                  << "\npreconditioner = " << (r.implicit_settings.csgs ? "CSGS" : "BJ") << "\nsolver_tol = "
                  << r.implicit_settings.linear_relative << "\nnewton_tol = " << r.implicit_settings.newton_relative
                  << "\nsimplified_jacobian = " << (r.implicit_settings.simplified_jacobian ? "on" : "off")
                  << "\nsimplified_newton = " << (r.implicit_settings.simplified_newton() ? "on" : "off")
                  << "\njacobian_update_every = " << r.implicit_settings.jacobian_update_every
                  << "\nnewton_max_iter = " << r.implicit_settings.max_iterations << '\n';
    if (r.reconstruction == "mp5" || r.reconstruction == "MPWENO5")
        std::cout << "mp6_hybrid = " << r.mp6_hybrid << '\n';
    if (r.reconstruction == "MPWENO5")
        std::cout << "mpweno_ad_corr = " << (r.mpweno_ad_corr ? "on" : "off") << '\n';
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
#ifdef _WIN32
        // В старой библиотеке времени выполнения MinGW режим fopen("wx") недоступен.
        // Флаг исключительного создания защищает уже существующий bc.txt и в этой сборке.
        const int fd = _open(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_TEXT, _S_IREAD | _S_IWRITE);
        if (fd >= 0) {
            file = _fdopen(fd, "w");
            if (!file) {
                error = errno;
                _close(fd);
            }
        } else {
            error = errno;
        }
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
