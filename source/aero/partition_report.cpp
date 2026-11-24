#include "partition_report.h"
#include "../parallel/processes.h"
#include <array>
#include <fstream>
#include <limits>
#include <map>
#include <ostream>
#include <sstream>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace aero {
namespace {
using Count = std::uint64_t;
const Count missing = (std::numeric_limits<Count>::max)();
// Текущая и пиковая резидентная память процесса.
struct ResidentMemory {
    Count peak = missing, current = missing;
};
// Полная и доступная оперативная память вычислительного узла.
struct HostMemory {
    Count total = missing, available = missing;
};

#ifdef __linux__
// В procfs величины в КиБ подписаны «kB». В старых ядрах поле MemAvailable может отсутствовать.
void read_kib(const char* path, const char* key1, Count& value1, const char* key2, Count& value2) {
    std::ifstream in(path);
    std::string key, unit;
    while (in >> key) {
        if (key == key1 || key == key2) {
            Count value;
            if (in >> value >> unit && unit == "kB")
                (key == key1 ? value1 : value2) = value;
        }
        in.ignore((std::numeric_limits<std::streamsize>::max)(), '\n');
    }
}
#endif
// Запрашивает у ОС текущую и пиковую память процесса.
ResidentMemory resident_memory() {
    ResidentMemory memory;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counts{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counts, sizeof(counts))) {
        memory.peak = counts.PeakWorkingSetSize / 1024;
        memory.current = counts.WorkingSetSize / 1024;
    }
#elif defined(__linux__)
    read_kib("/proc/self/status", "VmHWM:", memory.peak, "VmRSS:", memory.current);
#endif
    return memory;
}
// Запрашивает у ОС полную и доступную физическую память.
HostMemory host_memory() {
    HostMemory memory;
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        memory.total = status.ullTotalPhys / 1024;
        memory.available = status.ullAvailPhys / 1024;
    }
#elif defined(__linux__)
    read_kib("/proc/meminfo", "MemTotal:", memory.total, "MemAvailable:", memory.available);
#endif
    return memory;
}
// Форматирует память в КиБ; неизвестное значение выводит как n/a.
std::string kib(Count value) {
    return value == missing ? "n/a" : std::to_string(value);
}

using Samples = std::array<Count, 3>;
// Группа процессов одного узла и суммарные измерения памяти.
struct NodeMemory {
    int leader, size = 1, local_rank = 0;
    Samples sums;
#ifdef AERO_WITH_MPI
    char host[MPI_MAX_PROCESSOR_NAME + 1]{};
#else
    char host[10] = "localhost";
#endif
};
// Группирует процессы по узлу и суммирует измерения памяти.
NodeMemory collect_node_memory(const Processes& world, const Samples& memory) {
    NodeMemory node;
    node.leader = world.rank;
    node.sums = memory;
#ifdef AERO_WITH_MPI
    int length = 0;
    MPI_Get_processor_name(node.host, &length);
    MPI_Comm group;
#if MPI_VERSION >= 3 || defined(MPI_COMM_TYPE_SHARED)
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world.rank, MPI_INFO_NULL, &group);
#else
    // В MPI-2 нет коммуникатора общей памяти. Группируем процессы по имени вычислительного узла.
    const auto names = world.gather_text(node.host);
    std::vector<int> leaders;
    world.local([&] {
        if (world.rank)
            return;
        std::map<std::string, int> first;
        for (int r = 0; r < world.size; ++r)
            leaders.push_back(first.emplace(names[r], r).first->second);
    });
    int color = 0;
    MPI_Scatter(leaders.data(), 1, MPI_INT, &color, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_split(MPI_COMM_WORLD, color, world.rank, &group);
#endif
    MPI_Comm_rank(group, &node.local_rank);
    MPI_Comm_size(group, &node.size);
    MPI_Bcast(&node.leader, 1, MPI_INT, 0, group);

    // Сумму по узлу можно вывести, только если значение прислал каждый его процесс.
    std::array<Count, 6> local{}, total{};
    for (std::size_t i = 0; i < memory.size(); ++i)
        if (memory[i] != missing) {
            local[i] = memory[i];
            local[i + memory.size()] = 1;
        }
    MPI_Reduce(local.data(), total.data(), 6, MPI_UINT64_T, MPI_SUM, 0, group);
    MPI_Comm_free(&group);
    if (node.leader == world.rank) {
        for (std::size_t i = 0; i < memory.size(); ++i)
            node.sums[i] = total[i + memory.size()] == static_cast<Count>(node.size) ? total[i] : missing;
    }
#endif
    return node;
}
}

// Пиковое потребление памяти по данным ОС за построение геометрии и шаблонов, КиБ.
std::uint64_t process_peak_kib() {
    return resident_memory().peak;
}

// Возвращает локальный ранг и доступные ресурсы узла.
NodeResources partition_node_resources(const Processes& world) {
    ResidentMemory process;
    world.local([&] { process = resident_memory(); });
    const auto node = collect_node_memory(world, {{missing, missing, process.current}});
    NodeResources result{node.leader, node.local_rank, node.sums[2], missing};
    world.local([&] {
        if (world.rank == node.leader)
            result.available_kib = host_memory().available;
    });
    return result;
}

// Печатает распределение сетки, объёмы обменов и потребление памяти.
void write_partition_report(const Processes& world, const PartitionWork& work, Count stencil_peak,
                            std::ostream& out) {
    ResidentMemory ready;
    world.local([&] { ready = resident_memory(); });
    const auto node = collect_node_memory(world, {{stencil_peak, ready.peak, ready.current}});
    std::string rank_line, node_line;
    world.local([&] {
        std::ostringstream rank;
        rank << world.rank << '\t' << node.leader << '\t' << work.leaves << '\t' << work.owned << '\t'
             << work.halo << '\t' << work.edges << '\t' << work.states.sent << '\t' << work.states.received
             << '\t' << work.fluxes.sent << '\t' << work.fluxes.received << '\t' << kib(stencil_peak) << '\t'
             << kib(ready.peak) << '\t' << kib(ready.current);
        rank_line = rank.str();
        if (world.rank != node.leader)
            return;
        const auto ram = host_memory();
        std::ostringstream host;
        host << node.leader << '\t' << node.host << '\t' << node.size;
        for (Count sum : node.sums)
            host << '\t' << kib(sum);
        host << '\t' << kib(ram.total) << '\t' << kib(ram.available);
        node_line = host.str();
    });
    const auto ranks = world.gather_text(std::move(rank_line));
    const auto nodes = world.gather_text(std::move(node_line));
    if (world.rank)
        return;
    out << "MPI partition report (bytes per stage, memory in KiB)\n"
        << "rank\tnode\tleaves\towned\thalo\tedges\tstate_send_B\tstate_recv_B\tflux_send_B\tflux_recv_B"
           "\tstencil_peak_KiB\tstartup_peak_KiB\tready_rss_KiB\n";
    for (const auto& line : ranks)
        out << line << '\n';
    out << "MPI node memory (sum of process RSS; peaks can occur at different times)\n"
        << "node\thost\tranks\tstencil_peaks_sum_KiB\tstartup_peaks_sum_KiB\tready_rss_sum_KiB\tram_total_"
           "KiB\tram_available_KiB\n";
    for (const auto& line : nodes)
        if (!line.empty())
            out << line << '\n';
}
}
