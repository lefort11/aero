#include "zone_files.h"
#include "binary_columns.h"
#include "tecplot.h"
#include "vtk.h"
#include "../scheduled_output.h"
#include <algorithm>
#include <fstream>
#include <memory>

namespace aero {
namespace {
// Подготавливает один дополнительный файл в каталоге назначения. Исключение оставляет старый результат на месте.
std::unique_ptr<StagedFieldFile> stage_visualization(const std::string& path, bool plt,
                                                     const Mesh* mesh, const FieldSnapshot& snapshot,
                                                     BinaryColumns* columns) {
    std::unique_ptr<StagedFieldFile> file(new StagedFieldFile(visualization_filename(path, plt ? ".plt" : ".vtk")));
    std::ofstream stream(file->temporary(), std::ios::binary);
    if (plt) tecplot::write_snapshot(stream, mesh, snapshot, columns);
    else vtk::write(stream, mesh, &snapshot, columns);
    stream.close();
    if (!stream) throw FieldError("Cannot finish visualization for " + path);
    return file;
}
}

// Сначала завершаем все потоки, затем публикуем файлы, основной снимок последним.
// При конфликте имён в серии убираем только файлы, созданные текущей попыткой.
void publish_zone_files(StagedFieldFile& native, const std::string& path, const OutputZone& zone,
                        const Mesh* mesh, const FieldSnapshot& snapshot,
                        const std::vector<std::uint64_t>* offsets) {
    if (!mesh && zone.wall_label.empty()) throw FieldError("Missing mesh for visualization: " + path);
    std::unique_ptr<BinaryColumns> columns;
    if (offsets) columns.reset(new BinaryColumns(native.temporary(), snapshot, *offsets));
    auto plt = zone.tecplot ? stage_visualization(path, true, mesh, snapshot, columns.get()) : nullptr;
    auto vtk_file = zone.vtk ? stage_visualization(path, false, mesh, snapshot, columns.get()) : nullptr;
    columns.reset(); // Windows требует закрыть поток перед переименованием файла.
    try {
        if (plt) plt->publish(!zone.multi);
        if (vtk_file) vtk_file->publish(!zone.multi);
        native.publish(!zone.multi);
    } catch (...) {
        if (zone.multi) {
            if (plt) plt->withdraw();
            if (vtk_file) vtk_file->withdraw();
        }
        throw;
    }
}

// Передаёт один набор значений всем запрошенным форматам без повторного вычисления полей.
void write_zone_files(const std::string& directory, const OutputZone& zone,
                      const Mesh* mesh, const FieldSnapshot& snapshot) {
    auto name = output_filename(zone, snapshot.step);
    std::replace(name.begin(), name.end(), '\\', '/');
    const auto path = directory + "/" + name;
    StagedFieldFile native(path);
    snapshot.write(native.temporary(), zone.format);
    if (zone.visualize()) publish_zone_files(native, path, zone, mesh, snapshot);
    else native.publish(!zone.multi);
}
}
