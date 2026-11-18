#include "settings.h"
#include "settings_text.h"
#include <algorithm>
#include <cmath>
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

std::string resolve(const std::string& directory, const std::string& path) {
    if (path.front() == '/' || path.front() == '\\' || (path.size() > 1 && path[1] == ':'))
        return path;
    const char last = directory.back();
    return directory + (last == '/' || last == '\\' ? "" : "/") + path;
}

struct Entry {
    std::string value, where;
};

class Entries {
public:
    explicit Entries(const std::string& path) : path_(path) {
        auto input = open(path);
        std::string line;
        std::size_t row = 0;
        while (std::getline(input, line)) {
            const std::string where = path + ":" + std::to_string(++row);
            line = clean(line, where, row);
            if (line.empty()) continue;
            const auto eq = line.find('=');
            if (eq == std::string::npos) throw std::invalid_argument(where + ": expected key = value");
            const std::string key = trim(line.substr(0, eq));
            const std::string value = trim(line.substr(eq + 1));
            if (key.empty() || value.empty()) throw std::invalid_argument(where + ": empty key or value");
            if (!entries_.emplace(key, Entry{value, where + " (" + key + ")"}).second)
                throw std::invalid_argument(where + ": duplicate " + key);
        }
        if (input.bad() || !input.eof()) throw std::runtime_error("error reading " + path);
    }

    Entry take(const std::string& key) {
        const auto it = entries_.find(key);
        if (it == entries_.end()) throw std::invalid_argument(path_ + ": missing " + key);
        Entry entry = it->second;
        entries_.erase(it);
        return entry;
    }
    bool contains(const std::string& key) const { return entries_.count(key)!=0; }
    template<class T> T optional_count(const std::string& key) {
        if (!entries_.count(key)) return 0;
        const Entry e=take(key);
        std::istringstream stream(e.value);
        stream.imbue(std::locale::classic());
        T count;
        if (e.value.find_first_not_of("0123456789")!=std::string::npos || !(stream>>count) || !stream.eof())
            throw std::invalid_argument(e.where+": expected a nonnegative integer in range");
        return count;
    }

    void finish() const {
        if (!entries_.empty()) throw std::invalid_argument(entries_.begin()->second.where + ": unknown key");
    }
private:
    std::string path_;
    std::map<std::string, Entry> entries_;
};

std::string choice(const Entry& e, std::initializer_list<const char*> allowed) {
    if (std::find(allowed.begin(), allowed.end(), e.value) == allowed.end())
        throw std::invalid_argument(e.where + ": unknown value " + e.value);
    return e.value;
}

double number(const Entry& e, bool zero_allowed) {
    std::istringstream stream(e.value);
    stream.imbue(std::locale::classic());
    double value;
    if (!(stream >> value) || !stream.eof() || !std::isfinite(value) ||
        value < 0 || (!zero_allowed && value == 0))
        throw std::invalid_argument(e.where + (zero_allowed ? ": expected finite value >= 0" : ": expected finite value > 0"));
    return value;
}

std::vector<double> positive_list(const Entry& e) {
    std::vector<double> values;
    std::size_t start=0;
    for (;;) {
        const auto comma=e.value.find(',',start);
        values.push_back(number({trim(e.value.substr(start,comma-start)),e.where},false));
        if (comma==std::string::npos) return values;
        start=comma+1;
    }
}

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
        if (!used.insert(id).second) throw std::invalid_argument(e.where + ": duplicate CUDA device " + token);
        result.push_back(id);
        if (comma == std::string::npos) return result;
        start = comma + 1;
    }
}
}

RunDirectories parse_run_arguments(int argc, const char* const argv[]) {
    RunDirectories dirs;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto eq = arg.find('=');
        const std::string key = arg.substr(0, eq);
        if (key=="--recover") {
            if (dirs.recover) throw std::invalid_argument("duplicate --recover");
            dirs.recover=true;
            std::string slot;
            if (eq!=std::string::npos) slot=arg.substr(eq+1);
            else if (i+1<argc && std::string(argv[i+1]).compare(0,2,"--")!=0) slot=argv[++i];
            else continue;
            if (slot!="0" && slot!="1") throw std::invalid_argument("--recover expects 0 or 1");
            dirs.recovery_slot=slot=="0"?0:1;
            continue;
        }
        if (arg=="--geometry-only") {
            if (dirs.geometry_only) throw std::invalid_argument("duplicate --geometry-only");
            dirs.geometry_only=true;
            continue;
        }
        if (arg=="--interactive") {
            if (dirs.interactive) throw std::invalid_argument("duplicate --interactive");
            dirs.interactive=true;
            continue;
        }
        if (key != "--input" && key != "--output")
            throw std::invalid_argument("unknown argument: " + arg);
        if (eq == std::string::npos || eq + 1 == arg.size())
            throw std::invalid_argument("expected " + key + "=<directory>");
        std::string& path = key == "--input" ? dirs.input : dirs.output;
        if (!path.empty()) throw std::invalid_argument("duplicate " + key);
        path = arg.substr(eq + 1);
    }
    if (dirs.input.empty() || dirs.output.empty())
        throw std::invalid_argument("usage: aero --input=<directory> --output=<directory> [--interactive] [--geometry-only] [--recover [0|1]]");
    return dirs;
}

RunSettings read_run_settings(const RunDirectories& directories) {
    if (directories.input.empty() || directories.output.empty())
        throw std::invalid_argument("input and output directories are required");
    RunSettings run{};
    run.directories = directories;
    Entries main(resolve(directories.input, "main.txt"));
    const auto mesh = main.take("mesh");
    run.mesh_path = resolve(directories.input, settings_detail::label(mesh.value, mesh.where));
    run.reconstruction = choice(main.take("reconstruction"), {"first_order","ebr3","ebr5","weno3","weno5"});
    run.reconstruction_variables = choice(main.take("reconstruction_variables"), {"conservative","characteristic","flux"});
    run.time_scheme = choice(main.take("time_scheme"), {"euler","ssprk3"});
    run.cfl = number(main.take("cfl"), false);
    run.use_cuda = choice(main.take("backend"), {"cpu","cuda"}) == "cuda";
    run.euler_solver = choice(main.take("euler_solver"), {"roe","lf"});
    run.final_time = number(main.take("final_time"), true);
    run.trace_ratio_min = main.contains("trace_ratio_min") ? number(main.take("trace_ratio_min"),true) : 0;
    run.trace_ratio_max = main.contains("trace_ratio_max") ? number(main.take("trace_ratio_max"),false) : 1e16;
    if (run.trace_ratio_min>=1 || run.trace_ratio_max<=1)
        throw std::invalid_argument("main.txt: expected 0 <= trace_ratio_min < 1 < trace_ratio_max");
    run.mpi_balance=main.contains("mpi_rank_weights") || main.contains("mpi_node_memory_mib") || main.contains("mpi_memory_factor");
    if (main.contains("mpi_rank_weights")) {
        const auto e=main.take("mpi_rank_weights");
        if (e.value!="auto") run.mpi_rank_weights=positive_list(e);
    }
    if (main.contains("mpi_node_memory_mib")) {
        const auto e=main.take("mpi_node_memory_mib");
        if (e.value!="auto") run.mpi_node_memory_mib=positive_list(e);
    }
    if (main.contains("mpi_gpu_memory_mib")) {
        const auto e=main.take("mpi_gpu_memory_mib");
        if (e.value!="auto") run.mpi_gpu_memory_mib=positive_list(e);
    }
    // Initial estimates for the temporary geometry and derivative rows at startup.
    if (run.reconstruction=="ebr3") run.mpi_memory_factor=12;
    else if (run.reconstruction=="ebr5" || run.reconstruction=="weno3") run.mpi_memory_factor=16;
    else if (run.reconstruction=="weno5") run.mpi_memory_factor=24;
    if (main.contains("mpi_memory_factor")) run.mpi_memory_factor=number(main.take("mpi_memory_factor"),false);
    if (main.contains("mpi_timing")) run.mpi_timing=choice(main.take("mpi_timing"),{"0","1"})=="1";
    run.omp_threads = main.optional_count<int>("omp_threads");
    run.flowreport_every = main.optional_count<std::uint64_t>("flowreport_every");
    run.recovery_every_steps = main.optional_count<std::uint64_t>("recovery_every_steps");
    main.finish();
    Entries gas(resolve(directories.input, "gas.txt"));
    const Entry gamma = gas.take("gamma");
    run.gamma = number(gamma, false);
    if (run.gamma <= 1) throw std::invalid_argument(gamma.where + ": expected gamma > 1");
    gas.finish();
    if (run.use_cuda) {
        Entries cuda(resolve(directories.input, "cuda.txt"));
        run.devices = device_list(cuda.take("devices"));
        cuda.finish();
    }
    run.initial_path = resolve(directories.input, "init.txt");
    run.boundary_path = resolve(directories.input, "bc.txt");
    run.output_zones = read_output_zones(resolve(directories.input, "output.txt"));
#ifdef _OPENMP
    // The runtime settings belong to the calling task. Restore its original
    // defaults when the same launcher loads another case with omp_threads=0.
    static thread_local const int default_threads=omp_get_max_threads();
    static thread_local const int default_dynamic=omp_get_dynamic();
    omp_set_dynamic(run.omp_threads ? 0 : default_dynamic);
    omp_set_num_threads(run.omp_threads ? run.omp_threads : default_threads);
#endif
    return run;
}
}
