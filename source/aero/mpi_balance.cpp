#include "mpi_balance.h"
#include "../implicit/partition.h"
#include "../reconstruction/scheme.h"
#include "../explicit/stage_sequence.h"
#include "partition_report.h"
#include "../io/settings.h"
#include "../mesh/leaf_balance.h"
#include "../parallel/processes.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>

namespace aero {
namespace {
using Id = MeshIndex;
const long double mib = 1048576;
// Относительные скорости нормируются на самую быструю машину: выбор единицы не меняет разрезы.
std::vector<long double> rank_speeds(const std::vector<double>& weights, std::size_t ranks) {
    if (!weights.empty() && weights.size() != 1 && weights.size() != ranks)
        throw MeshError("mpi_rank_weights: expected one value or one weight per MPI rank");
    std::vector<long double> speeds(ranks, 1);
    if (!weights.empty()) {
        const auto maximum = *std::max_element(weights.begin(), weights.end());
        for (std::size_t r = 0; r < ranks; ++r) {
            speeds[r] =
                static_cast<long double>(weights.size() == 1 ? weights.front() : weights[r]) / maximum;
            if (speeds[r] == 0)
                throw MeshError("mpi_rank_weights: weight ratio is out of range");
        }
    }
    return speeds;
}

// Объединяем процессы одного узла: они делят общий бюджет RAM, даже если MPI выдал им несмежные номера.
std::vector<CpuNodeShare> node_memory_budgets(const RunSettings& run, const std::vector<Id>& resources,
                                              bool use_gpu, std::ostream& report) {
    const auto ranks = resources.size() / 3;
    std::map<int, std::vector<int>> members;
    for (std::size_t r = 0; r < ranks; ++r)
        members[static_cast<int>(resources[3 * r])].push_back(static_cast<int>(r));
    const auto& limits = run.mpi_node_memory_mib;
    if (!limits.empty() && limits.size() != 1 && limits.size() != members.size())
        throw MeshError("mpi_node_memory_mib: expected one limit or one limit per shared-memory node");
    report << (use_gpu ? "MPI GPU placement" : "MPI CPU placement")
           << " (estimated host mesh memory, factor = " << run.mpi_memory_factor << ")\n"
           << "node_leader\tinitial_rss_MiB\tmesh_budget_MiB\n";
    std::vector<CpuNodeShare> nodes;
    for (const auto& group : members) {
        const auto k = 3 * static_cast<std::size_t>(group.first);
        const Id rss = resources[k + 1], available = resources[k + 2];
        long double budget = std::numeric_limits<long double>::infinity();
        if (available != mdec::absent)
            budget = static_cast<long double>(available) * 1024;
        if (!limits.empty()) {
            if (rss == mdec::absent)
                throw MeshError("MPI partition: current RSS unavailable for node " +
                                std::to_string(group.first));
            const long double limit = limits.size() == 1 ? limits.front() : limits[nodes.size()];
            const long double remaining = limit * mib - static_cast<long double>(rss) * 1024;
            if (remaining < 0)
                throw MeshError(
                    "MPI partition: initial process RSS already exceeds mpi_node_memory_mib on node " +
                    std::to_string(group.first));
            budget = std::min(budget, remaining);
        }
        if (!std::isfinite(budget))
            throw MeshError("MPI partition: available RAM unavailable; set mpi_node_memory_mib");
        report << group.first << '\t';
        if (rss == mdec::absent)
            report << "n/a";
        else
            report << rss / 1024.0L;
        report << '\t' << budget / mib << '\n';
        nodes.push_back({group.second, budget / run.mpi_memory_factor});
    }

    return nodes;
}
// Распределяет листья на корневом ранге по ресурсам CPU и GPU.
void plan_on_root(const RunSettings& run, const std::vector<Id>& resources, LeafLayout& layout,
                  const std::vector<Id>& gpu, std::vector<GpuRankCapacity>* cards) {
    const auto ranks = resources.size() / 3;
    const auto& weights = run.mpi_rank_weights;
    const auto speeds = rank_speeds(weights, ranks);
    std::ostringstream report;
    const auto nodes = node_memory_budgets(run, resources, cards != nullptr, report);

    mdec::Archive archive(run.mesh_path);
    const bool fifth = reconstruction_scheme(run.reconstruction).layers == 2;
    const auto all = archive.prefix(archive.leaves, fifth);
    const Id cells = archive.counts[mdec::Cell];
    const Id cell_bytes = cells ? all.raw / cells + (all.raw % cells != 0) : 0;
    // Для BDF1 учитываем два поля расчёта, расширенный вход A*x и девять рабочих векторов.
    Id state_bytes = run.directories.geometry_only ? 0 : (archive.dimension + 2) * sizeof(double) *
        (3 + implicit::NewtonFieldCount + implicit::KrylovFieldCount);
    if (run.uses_implicit_time() && !run.directories.geometry_only) {
        if (run.background.active() || run.implicit_settings.jacobian_update_every) state_bytes += (archive.dimension+2)*sizeof(double);
        if (run.steady()) state_bytes += sizeof(double);
        if (run.implicit_settings.second_order) state_bytes += (archive.dimension+2)*sizeof(double);
        if (run.implicit_settings.csgs)
            state_bytes += (archive.dimension+2)*sizeof(double) + 3*sizeof(MeshIndex);
    }
    if (!run.uses_implicit_time()) {
        const auto stages = explicit_detail::stages_for(explicit_detail::scheme_from_name(run.time_scheme));
        state_bytes = (archive.dimension + 2) * sizeof(double) * (stages.count > 1 ? 3u : 2u);
    }
    // Храним недавно прочитанные границы листьев. Размер кэша зависит от числа процессов,
    // поэтому планирование не требует загрузки всего каталога архива в память.
    std::map<Id, LeafCost> cache;
    // Возвращает накопленные ресурсы архива до указанной границы листьев.
    const auto prefix = [&](Id end) -> LeafCost {
        const auto found = cache.find(end);
        if (found != cache.end())
            return found->second;
        const auto p = archive.prefix(end, fifth);
        const LeafCost cost = {partition_bytes(p.raw, p.nodes, 32),
                               partition_bytes(partition_bytes(p.raw, p.nodes, state_bytes), p.halo, cell_bytes)};
        if (cache.size() >= 128 + 4 * ranks)
            cache.clear();
        cache.emplace(end, cost);
        return cost;
    };
    std::cout << report.str() << std::flush;
    if (cards) {
        cards->resize(ranks);
        for (std::size_t r = 0; r < ranks; ++r) {
            const auto row = &gpu[6 * r];
            const Id free = row[0], budget = row[1], first = row[2], end = row[3], required = row[4];
            auto& card = (*cards)[r];
            if (!required)
                card.work_capacity = static_cast<long double>(budget);
            else if (first < end) {
                const Id proxy = prefix(end).work - prefix(first).work;
                card.work_capacity = static_cast<long double>(proxy) * budget / required;
            }
            if (row[5]) {
                if (first == end)
                    throw MeshError("MPI GPU memory: rank " + std::to_string(r) + " cannot reserve " +
                                    std::to_string(required) + " bytes even for an empty partition");
                card.rejected.emplace_back(first, end);
            }
            std::cout << "MPI GPU budget: rank " << r << " free_bytes " << free << " budget_bytes " << budget
                      << '\n';
        }
        layout = balance_gpu_leaves(archive.leaves, prefix, nodes, speeds, *cards);
    } else
        layout = balance_leaves(archive.leaves, prefix, nodes, speeds);
    report.str("");
    report.clear();
    report << "mpi_rank\tweight\tfirst_leaf\tend_leaf\testimated_mesh_MiB\n";
    for (std::size_t i = 0; i < ranks; ++i) {
        const int rank = layout.ranks[i];
        const Id bytes = prefix(layout.cuts[i + 1]).bytes - prefix(layout.cuts[i]).bytes;
        const double weight =
            weights.empty() ? 1 : weights[weights.size() == 1 ? 0 : static_cast<std::size_t>(rank)];
        report << rank << '\t' << weight << '\t' << layout.cuts[i] << '\t' << layout.cuts[i + 1] << '\t'
               << bytes * static_cast<long double>(run.mpi_memory_factor) / mib << '\n';
    }
    std::cout << report.str() << std::flush;
}
}

// Подбирает непрерывные диапазоны листьев с учётом скоростей процессов и доступной памяти.
LeafLayout plan_partition(const RunSettings& run, const Processes& world, GpuPlacement* gpu) {
    LeafLayout plan;
    if (world.size == 1)
        return plan;
    int enabled = (run.mpi_balance || gpu) ? 1 : 0;
    world.broadcast(&enabled, 1, 0);
    if (!enabled)
        return plan;
    const auto local = partition_node_resources(world);
    const Id values[] = {static_cast<Id>(local.leader), local.resident_kib, local.available_kib};
    std::vector<Id> resources;
    world.local([&] {
        if (!world.rank)
            resources.resize(3 * static_cast<std::size_t>(world.size));
    });
    world.gather(values, 3, resources.data(), 0);
    std::vector<Id> device_resources;
    if (gpu) {
        const auto& fit = gpu->local;
        const Id data[] = {fit.free_bytes, fit.budget,   fit.first,
                           fit.end,        fit.required, fit.failed ? Id(1) : Id(0)};
        world.local([&] {
            if (!world.rank)
                device_resources.resize(6 * static_cast<std::size_t>(world.size));
        });
        world.gather(data, 6, device_resources.data(), 0);
    }
    world.local([&] {
        if (!world.rank)
            plan_on_root(run, resources, plan, device_resources, gpu ? &gpu->cards : nullptr);
        else {
            plan.cuts.resize(static_cast<std::size_t>(world.size) + 1);
            plan.ranks.resize(world.size);
        }
    });
    world.broadcast(plan.cuts.data() + 1, world.size, 0);
    world.broadcast(plan.ranks.data(), world.size, 0);
    return plan;
}
}
