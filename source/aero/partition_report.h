#ifndef AERO_PARTITION_REPORT_H
#define AERO_PARTITION_REPORT_H
#include <cstdint>
#include <iosfwd>

namespace aero {
class Processes;
// Число отправленных и полученных байтов обмена.
struct ExchangeBytes {
    std::uint64_t sent, received;
};
// Размеры подобласти и объёмы обмена полями и потоками.
struct PartitionWork {
    std::uint64_t leaves, owned, halo, edges;
    ExchangeBytes states, fluxes;
};

// Пиковое потребление памяти по данным ОС за построение геометрии и шаблонов, КиБ.
std::uint64_t process_peak_kib();
// Лидер группы, ранг на узле и память для планирования подобластей.
struct NodeResources {
    int leader, local_rank;
    std::uint64_t resident_kib,
        available_kib; // Значения процесса-лидера. UINT64_MAX означает отсутствие измерения.
};
// Возвращает локальный ранг и доступные ресурсы узла.
NodeResources partition_node_resources(const Processes& world);
// Печатает распределение сетки, объёмы обменов и потребление памяти.
void write_partition_report(const Processes& world, const PartitionWork& work, std::uint64_t stencil_peak,
                            std::ostream& out);
}
#endif
