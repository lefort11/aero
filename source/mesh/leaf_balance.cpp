#include "leaf_balance.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace aero {
namespace {
using Id = MeshIndex;
template <class Fits> Id last_fit(Id first, Id last, const Fits& fits) {
    if (fits(last))
        return last;
    while (last - first > 1) {
        const Id middle = first + (last - first) / 2;
        if (fits(middle))
            first = middle;
        else
            last = middle;
    }
    return first;
}

Id nearest_cut(Id low, Id high, Id base, long double target, const LeafPrefix& prefix) {
    Id cut = last_fit(low, high, [&](Id end) { return prefix(end).work - base <= target; });
    if (cut < high && target - (prefix(cut).work - base) > (prefix(cut + 1).work - base) - target)
        ++cut;
    return cut;
}

void divide_node(LeafLayout& plan, Id first, Id last, const CpuNodeShare& node, const LeafPrefix& prefix,
                 const std::vector<long double>& speeds) {
    // Если листьев меньше, чем процессов, выдаём доступные участки самым быстрым процессам.
    auto active = node.ranks;
    if (last - first < active.size()) {
        std::stable_sort(active.begin(), active.end(), [&](int a, int b) { return speeds[a] > speeds[b]; });
        active.resize(static_cast<std::size_t>(last - first));
    }
    std::sort(active.begin(), active.end());
    const auto base = prefix(first).work, work = prefix(last).work - base;
    long double total = 0, used = 0;
    for (int r : active)
        total += speeds[r];
    std::size_t k = 0;
    Id cut = first;
    for (int r : node.ranks) {
        plan.ranks.push_back(r);
        if (k < active.size() && active[k] == r) {
            used += speeds[r];
            ++k;
            if (k == active.size())
                cut = last;
            else {
                const Id low = cut + 1, high = last - (active.size() - k);
                const long double target = work * (used / total);
                cut = nearest_cut(low, high, base, target, prefix);
            }
        }
        plan.cuts.push_back(cut);
    }
}
}

LeafLayout balance_leaves(Id leaves, const LeafPrefix& prefix, const std::vector<CpuNodeShare>& nodes,
                          const std::vector<long double>& speeds) {
    const std::size_t count = nodes.size();
    std::vector<long double> speed(count);
    for (std::size_t i = 0; i < count; ++i)
        for (int rank : nodes[i].ranks)
            speed[i] += speeds[rank];
    // При пробном времени узлу назначаем самый длинный префикс, который помещается в память и укладывается в срок.
    // Более короткий префикс только увеличил бы работу оставшихся узлов.
    std::vector<Id> ends(count + 1);
    const auto fits = [&](std::size_t i, const LeafCost& a, const LeafCost& b, long double duration) {
        return b.bytes - a.bytes <= nodes[i].mesh_bytes &&
               static_cast<long double>(b.work - a.work) / speed[i] <= duration;
    };
    const auto place = [&](long double duration) {
        for (std::size_t i = 0; i < count; ++i) {
            const auto base = prefix(ends[i]);
            ends[i + 1] =
                last_fit(ends[i], leaves, [&](Id end) { return fits(i, base, prefix(end), duration); });
        }
        return ends.back() == leaves;
    };
    if (!place(std::numeric_limits<long double>::infinity()))
        throw MeshError("MPI partition: node memory budgets cannot fit the estimated mesh in contiguous "
                        "whole leaves; first unplaced leaf " +
                        std::to_string(ends.back()));
    if (count > 1 && leaves) {
        long double high = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const Id work = prefix(ends[i + 1]).work - prefix(ends[i]).work;
            high = std::max(high, static_cast<long double>(work) / speed[i]);
        }
        if (!std::isfinite(high))
            throw MeshError("MPI partition: CPU weight ratio is out of range");
        long double low = 0;
        while (high - low > high * 1e-9L) {
            const long double middle = low + (high - low) / 2;
            if (place(middle))
                high = middle;
            else
                low = middle;
        }
        // Резервируем достаточно места для следующих узлов. В допустимом интервале берём разрез, ближайший к пропорциональному,
        // чтобы при равенстве оценок работа досталась и последним узлам.
        std::vector<Id> suffix(count + 1, leaves);
        std::vector<long double> remaining_speed(count + 1, 0);
        for (std::size_t i = count; i-- > 0;) {
            const Id end = suffix[i + 1];
            const auto finish = prefix(end);
            suffix[i] = end - last_fit(0, end, [&](Id length) {
                            return fits(i, prefix(end - length), finish, high);
                        });
            remaining_speed[i] = speed[i] + remaining_speed[i + 1];
        }
        const Id all = prefix(leaves).work;
        for (std::size_t i = 0; i < count; ++i) {
            const Id begin = ends[i];
            const auto base = prefix(begin);
            const Id last = last_fit(begin, leaves, [&](Id end) { return fits(i, base, prefix(end), high); });
            const long double target = (all - base.work) * (speed[i] / remaining_speed[i]);
            ends[i + 1] = nearest_cut(std::max(begin, suffix[i + 1]), last, base.work, target, prefix);
        }
    }
    LeafLayout plan;
    plan.cuts.push_back(0);
    for (std::size_t i = 0; i < count; ++i)
        divide_node(plan, ends[i], ends[i + 1], nodes[i], prefix, speeds);
    return plan;
}

LeafLayout balance_gpu_leaves(Id leaves, const LeafPrefix& prefix, const std::vector<CpuNodeShare>& nodes,
                              const std::vector<long double>& speeds,
                              const std::vector<GpuRankCapacity>& cards) {
    auto plan = balance_leaves(leaves, prefix, nodes, speeds);
    const auto allowed = [&](int rank, Id first, Id end, bool estimate) {
        const auto& card = cards[rank];
        if (estimate && prefix(end).work - prefix(first).work > card.work_capacity)
            return false;
        for (const auto& range : card.rejected)
            if (first <= range.first && end >= range.second)
                return false;
        return true;
    };
    bool fits = true;
    for (std::size_t i = 0; i < plan.ranks.size(); ++i)
        if (!allowed(plan.ranks[i], plan.cuts[i], plan.cuts[i + 1], true))
            fits = false;
    if (fits)
        return plan;

    // При заданном пределе времени покрываем максимально длинный допустимый префикс.
    // У процессов одного узла общая оперативная память, у каждого устройства свой предел памяти.
    const auto place = [&](long double duration, bool estimate) {
        std::size_t slot = 0;
        Id first = 0;
        for (const auto& node : nodes) {
            const Id base = prefix(first).bytes;
            for (int rank : node.ranks) {
                const Id work = prefix(first).work;
                const Id end = last_fit(first, leaves, [&](Id cut) {
                    const auto cost = prefix(cut);
                    return cost.bytes - base <= node.mesh_bytes &&
                           (cost.work - work) / speeds[rank] <= duration &&
                           allowed(rank, first, cut, estimate);
                });
                plan.cuts[++slot] = end;
                first = end;
            }
        }
        return first == leaves;
    };
    const auto infinity = std::numeric_limits<long double>::infinity();
    bool estimate = place(infinity, true);
    // Стоимость по архиву приблизительна. Одной завышенной оценки недостаточно для отбрасывания разбиения.
    if (!estimate && !place(infinity, false))
        throw MeshError("MPI GPU memory: cannot place whole leaves within the measured GPU and node budgets; "
                        "first unplaced leaf " +
                        std::to_string(plan.cuts.back()) + ". Use a finer .mdec or more device memory.");
    long double high = 0, low = 0;
    for (std::size_t i = 0; i < plan.ranks.size(); ++i) {
        const Id work = prefix(plan.cuts[i + 1]).work - prefix(plan.cuts[i]).work;
        high = std::max(high, work / speeds[plan.ranks[i]]);
    }
    if (!std::isfinite(high))
        throw MeshError("MPI GPU weight ratio is out of range");
    while (high - low > high * 1e-9L) {
        const auto middle = low + (high - low) / 2;
        if (place(middle, estimate))
            high = middle;
        else
            low = middle;
    }
    place(high, estimate);
    return plan;
}
}
