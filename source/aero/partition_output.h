#ifndef AERO_PARTITION_OUTPUT_H
#define AERO_PARTITION_OUTPUT_H
#include "../io/diagnostic_fields.h"
#include "../io/submesh.h"
#include "indexed_io.h"
#include "../bc/bc.h"
#include <map>
#include "mpi_periodic.h"
#include "../io/scheduled_output.h"

namespace aero {
struct RunSettings;

// Одно соответствие между номерами сетки и строками файлов для полей и восстановления.
struct PartitionFileLayout {
    FieldNodeSelection selection; // Исходные номера узлов, отпечаток сетки и её глобальные размеры.
    std::vector<OwnedNode> rows;  // Строки, за которые отвечает этот процесс в общем файле.
    std::string rank_directory;   // Имя подкаталога процесса относительно каталога результатов.
    bool global_fields = false;   // Есть хотя бы одна зона с коллективной записью в общий файл.
    bool local_fields = false;    // Есть хотя бы одна зона с записью из локальной части сетки.

    // Согласует размеры сетки, проверяет пути зон и находит узлы для выборочного ввода-вывода.
    PartitionFileLayout(const RunSettings& run, const MeshPartition& part, const PeriodicPartition* seam,
                        const Processes& world);
};

// Пишет все зоны по общему расписанию и выбирает способ записи по настройке зоны.
class PartitionOutput {
public:
    OutputSchedule schedule; // Номера шагов и моменты записи, общие для всех процессов.

    // Готовит каталоги и передаёт локальному выводу нумерацию расчётных узлов.
    PartitionOutput(const RunSettings& run, const MeshPartition& part, const PartitionFileLayout& layout,
                    const Processes& world, const std::string& directory, std::vector<MeshIndex>& node_dof,
                    MeshIndex state_nodes, const BoundaryPatches& patches);

    // Записывает назначенные зоны из одного уже доступного на CPU состояния.
    void write(std::uint64_t step, double time, const std::vector<double>& state, bool final_field = false,const diagnostic::Fields& scalars={});

private:
    struct SliceOutput {
        FieldSnapshot metadata;             // Геометрия и отпечаток именно подсетки.
        std::vector<SlicePoint> points;     // Номера концов в компактном списке источников.
        std::vector<OwnedNode> sources;     // На каждом ранге только принадлежащие ему строки.
        std::size_t source_count;
    };
    std::map<std::string,SliceOutput> slices_;
    void write_slice(const OutputZone&,const std::vector<unsigned>&,std::uint64_t,double,
                     const std::vector<double>&,const diagnostic::Fields&);
    struct WallNode {
        MeshIndex id, slot; // Исходный номер узла и его номер в массиве расчётных значений.
        Coordinate point;
    };
    std::map<std::string,std::vector<WallNode>> walls_;
    double reference_pressure_=0., dynamic_pressure_=0.;
    void write_wall(const OutputZone&, std::uint64_t, double, const std::vector<double>&);
    const Processes& world_;            // Участники коллективной записи полей.
    const PartitionFileLayout& layout_; // Нумерация узлов принадлежит постановке расчёта.
    std::string directory_;             // Корень результатов текущего запуска или продолжения.
    LinearBackground background_; // Величины фона для вычисления пульсации энергии.
    double gamma_;                      // Показатель адиабаты для вычисления полей газа.
    LocalFieldOutput local_;            // Метаданные сетки и нумерация локального вывода.
};
}
#endif
