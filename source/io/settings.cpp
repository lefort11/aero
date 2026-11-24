#include "settings.h"
#include "diagnostic_fields.h"
#include "../reconstruction/scheme.h"
#include "settings_text.h"
#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cctype>
#include <initializer_list>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace aero {
namespace {
using namespace settings_detail;

// Разрешает путь настройки относительно каталога постановки.
std::string resolve(const std::string& directory, const std::string& path) {
    if (path.front() == '/' || path.front() == '\\' || (path.size() > 1 && path[1] == ':'))
        return path;
    const char last = directory.back();
    return directory + (last == '/' || last == '\\' ? "" : "/") + path;
}

// Текст параметра и место его определения для сообщений об ошибках.
struct Entry {
    std::string value, where;
};

// Таблица ещё не прочитанных ключей одного файла. take удаляет ключ, а finish находит опечатки и лишние поля.
class Entries {
public:
    // Читает файл настроек и применяет переопределения командной строки.
    Entries(const RunDirectories& dirs, const std::string& file, bool optional = false) : path_(resolve(dirs.input, file + ".txt")) {
        const auto& path = path_;
        std::ifstream input(path.c_str());
        const bool absent = !input && optional && errno == ENOENT;
        if (!input && !absent)
            throw std::runtime_error("cannot open settings file: " + path);
        std::string line;
        std::size_t row = 0;
        while (std::getline(input, line)) {
            const std::string where = path + ":" + std::to_string(++row);
            line = clean(line, where, row);
            if (line.empty())
                continue;
            const auto eq = line.find('=');
            if (eq == std::string::npos)
                throw std::invalid_argument(where + ": expected key = value");
            const std::string key = trim(line.substr(0, eq));
            const std::string value = trim(line.substr(eq + 1));
            if (key.empty() || value.empty())
                throw std::invalid_argument(where + ": empty key or value");
            if (!entries_.emplace(key, Entry{value, where + " (" + key + ")"}).second)
                throw std::invalid_argument(where + ": duplicate " + key);
        }
        if (!absent && (input.bad() || !input.eof()))
            throw std::runtime_error("error reading " + path);
        // Подменяем значения в памяти до общей проверки типов и допустимых диапазонов.
        const auto overrides = dirs.overrides.find(file);
        if (overrides != dirs.overrides.end())
            for (const auto& item : overrides->second)
                entries_[item.first] = {item.second, file + ":" + item.first};
    }

    // Извлекает параметр и удаляет его из таблицы непрочитанных ключей.
    Entry take(const std::string& key) {
        const auto it = entries_.find(key);
        if (it == entries_.end())
            throw std::invalid_argument(path_ + ": missing " + key);
        Entry entry = it->second;
        entries_.erase(it);
        return entry;
    }
    // Проверяет наличие ещё не использованного ключа настройки.
    bool contains(const std::string& key) const { return entries_.count(key) != 0; }
    // Читает необязательное целочисленное значение с проверкой диапазона.
    template <class T> T optional_count(const std::string& key) {
        if (!entries_.count(key))
            return 0;
        return count<T>(key);
    }
    // Разбирает неотрицательное целое значение настройки с проверкой диапазона типа.
    template <class T> T count(const std::string& key) {
        const Entry e = take(key);
        std::istringstream stream(e.value);
        stream.imbue(std::locale::classic());
        T count;
        if (e.value.find_first_not_of("0123456789") != std::string::npos || !(stream >> count) ||
            !stream.eof())
            throw std::invalid_argument(e.where + ": expected a nonnegative integer in range");
        return count;
    }

    // Сообщает о неиспользованных ключах как об ошибках или предупреждениях.
    void finish(std::vector<std::string>* warnings) const {
        for (const auto& item : entries_) {
            const auto message = item.second.where + ": unknown parameter";
            if (!warnings) throw std::invalid_argument(message);
            warnings->push_back(message);
        }
    }

private:
    std::string path_;
    std::map<std::string, Entry> entries_;
};

// Проверяет значение настройки по списку поддерживаемых вариантов.
std::string choice(const Entry& e, std::initializer_list<const char*> allowed) {
    if (std::find(allowed.begin(), allowed.end(), e.value) == allowed.end())
        throw std::invalid_argument(e.where + ": unknown value " + e.value);
    return e.value;
}

// Для сокращений BDF1, LF и BJ разрешаем также строчные буквы.
Entry lowercase(Entry e) {
    e.value = lower_value(e.value);
    return e;
}

// Используем те же имена режимов, которые солвер затем выведет при запуске.
ResidualLimit read_residual_limit(const Entry& entry) {
    const auto name = lower_value(entry.value);
    for (auto mode : {ResidualLimit::Off, ResidualLimit::Node, ResidualLimit::Stencil, ResidualLimit::Work})
        if (name == residual_limit_name(mode)) return mode;
    throw std::invalid_argument(entry.where + ": expected off, on, stencil or work");
}

// Разбирает конечное вещественное значение параметра.
double finite_number(const Entry& e) {
    return finite_real(e.value, e.where);
}

// Разбирает числовое значение настройки с диагностикой исходной строки.
double number(const Entry& e, bool zero_allowed) {
    const double value = finite_number(e);
    if (value < 0 || (!zero_allowed && value == 0))
        throw std::invalid_argument(
            e.where + (zero_allowed ? ": expected finite value >= 0" : ": expected finite value > 0"));
    return value;
}

// Проверяем все явно заданные величины. Для активного закона take также проверяет
// наличие обязательных ключей. Это позволяет менять закон, сохранив общий gas.txt.
#ifdef AERO_HAS_VISCOSITY
// Читает и согласует параметры закона вязкости и теплопроводности.
void read_viscosity_parameters(Entries& gas, ViscositySettings& settings) {
    const bool active = settings.law != "off";
    // Извлекает именованный параметр с диагностикой его исходной записи.
    const auto parameter = [&](const char* key, bool required, bool allow_zero, double& value) {
        if (required || gas.contains(key)) value = number(gas.take(key), allow_zero);
    };
    parameter("reynolds", active, false, settings.reynolds);
    parameter("prandtl", active, false, settings.prandtl);
    parameter("t_amb", active, false, settings.t_amb);
    parameter("sutherland_s", settings.law == "sutherland", true, settings.sutherland_s);
}

#endif

// Разбирает список положительных чисел из настройки.
std::vector<double> positive_list(const Entry& e) {
    std::vector<double> values;
    std::size_t start = 0;
    for (;;) {
        const auto comma = e.value.find(',', start);
        values.push_back(number({trim(e.value.substr(start, comma - start)), e.where}, false));
        if (comma == std::string::npos)
            return values;
        start = comma + 1;
    }
}

// Разбирает перечень номеров GPU для назначения процессам.
std::vector<int> device_list(const Entry& e) {
    std::vector<int> result;
    std::set<int> used;
    std::size_t start = 0;
    for (;;) {
        const auto comma = e.value.find(',', start);
        const auto token = trim(e.value.substr(start, comma - start));
        int id;
        std::istringstream stream(token);
        stream.imbue(std::locale::classic());
        if (token.find_first_not_of("0123456789") != std::string::npos || !(stream >> id) || !stream.eof())
            throw std::invalid_argument(e.where + ": invalid CUDA device ID '" + token + "'");
        if (!used.insert(id).second)
            throw std::invalid_argument(e.where + ": duplicate CUDA device " + token);
        result.push_back(id);
        if (comma == std::string::npos)
            return result;
        start = comma + 1;
    }
}

// Пустой список означает автоматический выбор. Явные веса и бюджеты обязаны быть положительными.
std::vector<double> optional_mpi_list(Entries& entries, const char* key) {
    if (!entries.contains(key))
        return {};
    const auto entry = entries.take(key);
    if (entry.value == "auto")
        return {};
    return positive_list(entry);
}

// Собирает настройки планировщика до чтения архива. Оценка памяти по умолчанию зависит от схемы.
void read_partition_settings(Entries& main, double default_memory_factor, RunSettings& run) {
    run.mpi_balance = main.contains("mpi_rank_weights") || main.contains("mpi_node_memory_mib") ||
                      main.contains("mpi_memory_factor");
    run.mpi_rank_weights = optional_mpi_list(main, "mpi_rank_weights");
    run.mpi_node_memory_mib = optional_mpi_list(main, "mpi_node_memory_mib");
    run.mpi_gpu_memory_mib = optional_mpi_list(main, "mpi_gpu_memory_mib");
    run.mpi_memory_factor = default_memory_factor;
    if (main.contains("mpi_memory_factor"))
        run.mpi_memory_factor = number(main.take("mpi_memory_factor"), false);
    if (main.contains("mpi_timing"))
        run.mpi_timing = choice(main.take("mpi_timing"), {"0", "1"}) == "1";
}
}

// Разбирает каталоги запуска и переопределения параметров командной строки.
RunDirectories parse_run_arguments(int argc, const char* const argv[]) {
    RunDirectories dirs;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto eq = arg.find('=');
        const std::string key = arg.substr(0, eq);
        if (key == "--recover") {
            if (dirs.recover)
                throw std::invalid_argument("duplicate --recover");
            dirs.recover = true;
            std::string slot;
            if (eq != std::string::npos)
                slot = arg.substr(eq + 1);
            else if (i + 1 < argc && std::string(argv[i + 1]).find('=') == std::string::npos && std::string(argv[i + 1]).compare(0, 2, "--") != 0)
                slot = argv[++i];
            else
                continue;
            if (slot != "0" && slot != "1")
                throw std::invalid_argument("--recover expects 0 or 1");
            dirs.recovery_slot = slot == "0" ? 0 : 1;
            continue;
        }
        if (arg == "--geometry-only") {
            if (dirs.geometry_only)
                throw std::invalid_argument("duplicate --geometry-only");
            dirs.geometry_only = true;
            continue;
        }
        if (arg == "--interactive") {
            if (dirs.interactive)
                throw std::invalid_argument("duplicate --interactive");
            dirs.interactive = true;
            continue;
        }
        const auto colon = key.find(':');
        if (colon != std::string::npos && eq != std::string::npos) {
            std::string file = key.substr(0, colon);
            if (file.size() >= 4 && file.substr(file.size() - 4) == ".txt")
                file.resize(file.size() - 4);
            if ((file != "main" && file != "gas" && file != "cuda" && file != "scheme" &&
                 file != "cfl" && file != "implicit" && file != "linear"
#ifdef AERO_HAS_TURBULENCE
                 && file != "turbulence"
#endif
                 ) || colon + 1 == key.size() || eq + 1 == arg.size())
                throw std::invalid_argument("invalid parameter override: " + arg);
            dirs.overrides[file][key.substr(colon + 1)] = arg.substr(eq + 1);
            continue;
        }
        if (key != "--input" && key != "--output")
            throw std::invalid_argument("unknown argument: " + arg);
        if (eq == std::string::npos || eq + 1 == arg.size())
            throw std::invalid_argument("expected " + key + "=<directory>");
        std::string& path = key == "--input" ? dirs.input : dirs.output;
        if (!path.empty())
            throw std::invalid_argument("duplicate " + key);
        path = arg.substr(eq + 1);
    }
    if (dirs.input.empty() && dirs.output.empty())
        dirs.input = dirs.output = ".";
    if (dirs.input.empty())
        throw std::invalid_argument("missing --input=<directory>");
    if (dirs.output.empty())
        throw std::invalid_argument("missing --output=<directory>");
    return dirs;
}

// Читает main.txt, gas.txt, необязательный output.txt и параметры CUDA из cuda.txt.
// Начальные и граничные условия разбираются после загрузки сетки.
std::vector<std::string> read_exact_formulas(const RunDirectories& dirs, const std::vector<std::string>& names, std::vector<std::string>* warnings) {
    Entries file(dirs, "exact");
    std::vector<std::string> result;
    for (const auto& name : names) result.push_back(file.take(name).value);
    file.finish(warnings);
    return result;
}

// Читает и согласует настройки постановки, модулей и командной строки.
RunSettings read_run_settings(const RunDirectories& directories) {
    if (directories.input.empty() || directories.output.empty())
        throw std::invalid_argument("input and output directories are required");
    RunSettings run{};
    run.directories = directories;
    Entries main(directories, "main");
    const auto mesh = main.take("mesh");
    run.mesh_path = resolve(directories.input, settings_detail::label(mesh.value, mesh.where));
    const auto reconstruction = main.take("reconstruction");
    const auto scheme = reconstruction_scheme(reconstruction.value, reconstruction.where);
    run.reconstruction = scheme.name;
    Entries scheme_settings(directories, "scheme", true);
    if (scheme_settings.contains("stencil_geometry"))
        run.stencil_geometry = choice(lowercase(scheme_settings.take("stencil_geometry")), {"straight","ijk","pl","auto"});
    if (scheme_settings.contains("stencil_max_angle"))
        run.stencil_max_angle = number(scheme_settings.take("stencil_max_angle"), false);
    if (!(run.stencil_max_angle > 0. && run.stencil_max_angle < 90.))
        throw std::invalid_argument("scheme.txt: stencil_max_angle must be between 0 and 90 degrees");
    run.mpweno_ad_corr = !scheme_settings.contains("mpweno_ad_corr") ||
        choice(scheme_settings.take("mpweno_ad_corr"), {"on", "off"}) == "on";
    run.mp6_hybrid = scheme_settings.contains("mp6_hybrid") ? number(scheme_settings.take("mp6_hybrid"), true) : 0;
    if (run.mp6_hybrid > 1)
        throw std::invalid_argument("scheme.txt: mp6_hybrid must be in [0, 1]");
    scheme_settings.finish(&run.input_warnings);
    run.reconstruction_variables =
        choice(main.take("reconstruction_variables"), {"characteristic", "flux"});
    const bool linear = main.contains("Equations") &&
        choice(lowercase(main.take("Equations")), {"euler","linear"}) == "linear";
    if (linear) {
        Entries base(directories, "linear");
        run.background.rho = number(base.take("rho0"), false);
        run.background.p = number(base.take("p0"), false);
        for (const char* key : {"ux0","uy0","uz0"}) {
            if (!base.contains(key)) break;
            run.background.velocity[run.background_velocities++] = finite_number(base.take(key));
        }
        base.finish(&run.input_warnings);
    }
    run.time_scheme = choice(lowercase(main.take("time_scheme")), {"euler", "ssprk3", "bdf1", "bdf2", "steady"});
    if (run.uses_implicit_time()) {
        Entries settings(directories, "implicit");
        choice(lowercase(settings.take("solver")), {"bicgstab"});
        auto& controls = run.implicit_settings;
        if (settings.contains("steady_multigrid")) {
            const auto entry = settings.take("steady_multigrid");
            controls.steady_multigrid = choice(lowercase(entry), {"on","off"}) == "on";
            if (controls.steady_multigrid && !run.steady())
                throw std::invalid_argument(entry.where + ": steady_multigrid requires time_scheme=steady");
#ifndef AERO_HAS_MULTIGRID
            if (controls.steady_multigrid)
                throw std::invalid_argument(entry.where + ": this build has no source/multigrid module");
#endif
        }
        controls.second_order = run.time_scheme == "bdf2";
        controls.roe = choice(lowercase(settings.take("jacobian_flux")), {"lf","roe"}) == "roe";
        controls.csgs = choice(lowercase(settings.take("preconditioner")), {"bj","csgs"}) == "csgs";
        // Разрешает относительный путь по каталогу исходной настройки.
        const auto relative = [&](const char* key) {
            const auto entry = settings.take(key);
            const double value = number(entry, false);
            if (value >= 1.) throw std::invalid_argument(entry.where + ": expected 0 < tolerance < 1");
            return value;
        };
        controls.simplified_jacobian = settings.contains("simplified_jacobian") &&
            choice(lowercase(settings.take("simplified_jacobian")), {"on","off"}) == "on";
        controls.compact_jacobian = settings.contains("compact_jacobian") &&
            choice(lowercase(settings.take("compact_jacobian")), {"on","off"}) == "on";
        controls.jacobian_update_every = settings.optional_count<unsigned>("jacobian_update_every");
        // В старых файлах переиспользование матрицы уже задавалось интервалом.
        // Явный переключатель имеет приоритет и задаёт тот же режим решателя.
        if (settings.contains("simplified_newton"))
            controls.set_simplified_newton(choice(lowercase(settings.take("simplified_newton")), {"on","off"}) == "on");
        if (settings.contains("local_time_step")) {
            const auto entry = settings.take("local_time_step");
            run.local_time_step = parse_local_time_step(entry.value, entry.where);
            if (run.local_time_step != LocalStepMode::Off && !run.steady())
                throw std::invalid_argument("implicit.txt: local_time_step requires time_scheme=steady");
        }
        if (settings.contains("local_time_step_residual_limit")) {
            run.local_time_step_residual_limit = read_residual_limit(settings.take("local_time_step_residual_limit"));
            if (run.local_time_step_residual_limit != ResidualLimit::Off && !run.steady())
                throw std::invalid_argument("local_time_step_residual_limit requires steady");
        }
        if (settings.contains("local_time_step_probe_node")) {
            run.local_time_step_probe_node = settings.count<std::uint64_t>("local_time_step_probe_node");
            if (!run.steady()) throw std::invalid_argument("local_time_step_probe_node requires steady");
        }
        if (settings.contains("local_time_step_max_ratio")) {
            const auto entry = settings.take("local_time_step_max_ratio");
            run.local_time_step_max_ratio = parse_local_step_ratio(entry.value, entry.where);
        }
        if (settings.contains("local_time_step_neighbor_ratio")) {
            const auto entry = settings.take("local_time_step_neighbor_ratio");
            run.local_time_step_neighbor_ratio = parse_local_step_ratio(entry.value, entry.where);
            if (run.local_time_step_neighbor_ratio && !run.steady())
                throw std::invalid_argument(entry.where + ": local_time_step_neighbor_ratio requires time_scheme=steady");
        }
        if (settings.contains("steady_criterion"))
            run.steady_absolute = choice(lowercase(settings.take("steady_criterion")), {"relative", "absolute"}) == "absolute";
        if (settings.contains("steady_tol"))
            run.steady_tol = run.steady_absolute ? number(settings.take("steady_tol"), false) : relative("steady_tol");
        if (settings.contains("steady_max_steps")) {
            run.steady_max_steps = settings.count<unsigned>("steady_max_steps");
            if (!run.steady_max_steps) throw std::invalid_argument("implicit.txt: steady_max_steps must be positive");
        }
        controls.linear_relative = relative("solver_tol");
        if (settings.contains("solver_max_iter")) {
            controls.linear_max_iterations = settings.count<unsigned>("solver_max_iter");
            if (!controls.linear_max_iterations)
                throw std::invalid_argument("implicit.txt: solver_max_iter must be positive");
        }
        controls.newton_relative = relative("newton_tol");
        controls.max_iterations = settings.count<unsigned>("newton_max_iter");
        if (!controls.max_iterations) throw std::invalid_argument("implicit.txt: newton_max_iter must be positive");
        settings.finish(&run.input_warnings);
    }
    run.upw_coeff = main.contains("upw_coeff") ? number(main.take("upw_coeff"), true) : 1;
    if (run.upw_coeff > 1)
        throw std::invalid_argument("main.txt: upw_coeff must be in [0, 1]");
    auto& stable = run.stabilization;
    if (main.contains("stabilization"))
        stable.mode = choice(main.take("stabilization"), {"off", "cfl", "low", "on"});
    if (main.contains("cfl_method"))
        run.cell_height_cfl = choice(main.take("cfl_method"), {"height", "spectral"}) == "height";
    if (!stable.active() || main.contains("cfl")) {
        const auto entry = main.take("cfl");
        run.cfl = finite_number(entry);
        // В main.txt проверяется обычный CFL; адаптивный диапазон задаётся ниже в cfl.txt.
        Stabilization{}.validate(run.cfl, entry.where);
    }
    if (stable.active()) {
        if (run.background.active()) throw std::invalid_argument("stabilization requires Equations = Euler");
        Entries control(directories, "cfl");
        stable.interval = control.count<std::uint64_t>("cfl_num_steps");
        stable.factor = finite_number(control.take("cfl_coeff"));
        run.cfl = finite_number(control.take("cfl_init"));
        stable.minimum = finite_number(control.take("cfl_min"));
        stable.maximum = finite_number(control.take("cfl_max"));
        stable.validate(run.cfl, "cfl.txt");
        control.finish(&run.input_warnings);
    }
    run.use_cuda = choice(main.take("backend"), {"cpu", "cuda"}) == "cuda";
    run.euler_solver = choice(main.take("euler_solver"), {"roe", "lf"});
    run.steps_fo = main.optional_count<std::uint64_t>("steps_fo");
    run.t_fo = main.contains("t_fo") ? number(main.take("t_fo"), true) : 0.;
    run.final_time = run.steady() && !main.contains("final_time") ? 0. : number(main.take("final_time"), true);
    run.trace_ratio_min = main.contains("trace_ratio_min") ? number(main.take("trace_ratio_min"), true) : 0;
    run.trace_ratio_max =
        main.contains("trace_ratio_max") ? number(main.take("trace_ratio_max"), false) : 1e16;
    if (run.trace_ratio_min >= 1 || run.trace_ratio_max <= 1)
        throw std::invalid_argument("main.txt: expected 0 <= trace_ratio_min < 1 < trace_ratio_max");
    read_partition_settings(main, scheme.default_memory_factor, run);
    run.omp_threads = main.optional_count<int>("omp_threads");
    if (main.contains("exact_solution")) {
        const auto mode = choice(main.take("exact_solution"), {"off", "formula", "fields", "on"});
        if (mode == "fields") run.exact_solution = ExactSolution::Fields;
        if (mode == "formula" || mode == "on") run.exact_solution = ExactSolution::Formula;
    }
    if (main.contains("control_every")) run.control_every = main.count<std::uint64_t>("control_every");
    run.prophecy_every_step = main.optional_count<std::uint64_t>("prophecy_every_step");
    run.report_every = main.optional_count<std::uint64_t>("report_every");
    run.recovery_every_steps = main.optional_count<std::uint64_t>("recovery_every_steps");
#ifdef AERO_HAS_VISCOSITY
    if (main.contains("viscosity"))
        run.viscosity.law = choice(lowercase(main.take("viscosity")), {"off", "normal", "sutherland"});
    if (linear && run.viscosity.law != "off") throw std::invalid_argument("viscosity requires Equations = Euler");
#endif
    if (main.contains("Turbulence")) {
        const bool enabled = choice(lowercase(main.take("Turbulence")), {"on","off"}) == "on";
        if (enabled) {
#ifdef AERO_HAS_TURBULENCE
            if (linear || run.viscosity.law == "off")
                throw std::invalid_argument("Turbulence = On requires Equations = Euler and viscosity = normal or sutherland");
            Entries model(directories, "turbulence");
            run.turbulence.enabled = true;
            const auto kind=choice(lowercase(model.take("model")), {"rans","les","ddes"});
            run.turbulence.ddes=kind=="ddes";
            if (kind=="les") choice(lowercase(model.take("les_eq")), {"sigma"});
            if (kind!="rans") {
                auto& les = run.turbulence.les;
                const auto scale = lowercase(model.take("subgridscale"));
                using turbulence::Scale;
                const auto name = scale.value=="delta_omega^tilde" ? "delta_omega_tilde" : scale.value;
                for (Scale candidate : {Scale::Max,Scale::Min,Scale::Vol,Scale::Omega,Scale::OmegaTilde,Scale::Sla})
                    if (name==turbulence::scale_name(candidate)) les.scale=candidate;
                if (les.scale==Scale::Off) throw std::invalid_argument(scale.where + ": unknown subgridscale " + scale.value);
                if (kind=="les" && model.contains("c_sigma")) les.c_sigma = number(model.take("c_sigma"),false);
                if (model.contains("switch_on_vtm"))
                    les.switch_on_vtm = choice(lowercase(model.take("switch_on_vtm")), {"on","off"}) == "on";
                if (model.contains("F_KH_lim")) {
                    les.f_kh_lim = number(model.take("F_KH_lim"),true);
                    if (les.f_kh_lim>1.) throw std::invalid_argument("turbulence.txt: F_KH_lim must be in [0,1]");
                }
            }
            if (kind!="les") {
                if (run.turbulence.ddes && model.contains("c_des"))
                    run.turbulence.c_des=number(model.take("c_des"),false);
                choice(lowercase(model.take("rans_eq")), {"sa"});
                if (model.contains("sa_high_order"))
                    run.turbulence.high_order = choice(lowercase(model.take("sa_high_order")), {"on","off"}) == "on";
                run.turbulence.nut_nu_amb = number(model.take("nut_nu_amb"), true);
            }
            if (model.contains("pr_t")) run.turbulence.prandtl = number(model.take("pr_t"), false);
            model.finish(&run.input_warnings);
#else
            throw std::invalid_argument("Turbulence = On requires the turbulence and viscosity modules");
#endif
        }
    }
    main.finish(&run.input_warnings);
    Entries gas(directories, "gas");
    const Entry gamma = gas.take("gamma");
    run.gamma = number(gamma, false);
    if (run.gamma <= 1)
        throw std::invalid_argument(gamma.where + ": expected gamma > 1");
    if (gas.contains("mach")) run.mach = number(gas.take("mach"), false);
    if (gas.contains("phi")) run.reference_phi = finite_number(gas.take("phi"));
    if (gas.contains("theta")) run.reference_theta = finite_number(gas.take("theta"));
#ifdef AERO_HAS_VISCOSITY
    read_viscosity_parameters(gas, run.viscosity);
    if (run.viscosity.law != "off" && !run.mach) throw std::invalid_argument("gas.txt: missing mach");
#endif
    gas.finish(&run.input_warnings);
    if (linear) {
        run.background.sound = std::sqrt(run.gamma*(run.background.p/run.background.rho));
        if (!(run.background.sound > 0.) || !std::isfinite(run.background.sound))
            throw std::invalid_argument("linear.txt: invalid background sound speed");
    }
    if (run.use_cuda) {
        Entries cuda(directories, "cuda");
        run.devices = device_list(cuda.take("devices"));
        cuda.finish(&run.input_warnings);
    }
    run.initial_path = resolve(directories.input, "init.txt");
    run.boundary_path = resolve(directories.input, "bc.txt");
    run.output_zones = read_output_zones(resolve(directories.input, "output.txt"), &run.input_warnings);
    for (auto& zone : run.output_zones)
        if (!zone.submesh.empty()) zone.submesh=resolve(directories.input,zone.submesh);
#ifdef AERO_HAS_TURBULENCE
    if (run.turbulence.uses_sigma()) {
        for (auto& zone:run.output_zones) for (auto& field:zone.variables) {
            if (field=="nu_tilde") throw std::invalid_argument("output.txt: nu_tilde requires RANS SA or DDES SA");
            if (field=="turbulence") field="nut";
        }
    }
    if (!run.turbulence.enabled) for (const auto& zone:run.output_zones)
        for (const auto& field:zone.variables) if(diagnostic::turbulent(field))
            throw std::invalid_argument("output.txt: "+field+" requires Turbulence = On");
#endif
#ifdef _OPENMP
    // Настройка OpenMP действует на вызывающую задачу. При повторном запуске с omp_threads=0
    // возвращаем исходные параметры, чтобы предыдущая постановка не определяла число потоков.
    static thread_local const int default_threads = omp_get_max_threads();
    static thread_local const int default_dynamic = omp_get_dynamic();
    omp_set_dynamic(run.omp_threads ? 0 : default_dynamic);
    omp_set_num_threads(run.omp_threads ? run.omp_threads : default_threads);
#endif
    return run;
}
}
