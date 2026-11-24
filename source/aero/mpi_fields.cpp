#include "mpi_fields.h"
#include "../io/scheduled_output.h"
#include <algorithm>
#include <memory>

namespace aero {
// Записывает общий двоичный снимок без сбора поля на процессе 0.
void write_global_fields(const Processes& world, const std::string& directory, const OutputZone& zone,
                         const std::vector<unsigned>& columns, const FieldNodeSelection& mesh,
                         const std::vector<OwnedNode>& nodes, double gamma, std::uint64_t step, double time,
                         const std::vector<double>& state, const LinearBackground& background,const diagnostic::Fields& scalars) {
    std::unique_ptr<StagedFieldFile> staged;
    std::vector<std::uint64_t> offsets;
    std::string temporary;
    world.local([&] {
        // Нулевой процесс создаёт метаданные и резервирует место для каждого столбца.
        offsets.resize(columns.size());
        if (world.rank)
            return;
        auto name = output_filename(zone, step);
        std::replace(name.begin(), name.end(), '\\', '/');
        staged.reset(new StagedFieldFile(directory + "/" + name));
        temporary = staged->temporary();
        FieldSnapshot metadata;
        metadata.dimension = mesh.dimension;
        metadata.mesh_sha256 = mesh.mesh_sha256;
        metadata.counts = mesh.counts;
        metadata.step = step;
        metadata.time = time;
        for (const auto& var : zone.variables)
            metadata.fields.push_back(output_field(var));
        offsets = metadata.write_binary_layout(temporary);
    });
    // Всем владельцам нужны одинаковые путь и начальные смещения столбцов.
    world.broadcast(temporary, 0);
    world.broadcast(offsets.data(), static_cast<int>(offsets.size()), 0);
    const FieldRows rows{state.data(), mesh.dimension, gamma, background, scalars, columns};
    // Упаковывает узловую запись в формат распределённого файлового обмена.
    const auto encode = [&](const OwnedNode& n, char* dst, std::size_t pitch) {
        // Один раз восстанавливаем физические величины узла, затем раскладываем запрошенные столбцы.
        rows.visit(n.slot, n.id, [&](std::size_t k, double value) { indexed_io::put_real(dst + k * pitch, value); });
    };
    // Исходный номер узла определяет позицию в файле независимо от разбиения сетки.
    indexed_io::transfer(world, temporary, indexed_io::Access::Write, 8, nodes, offsets, std::ref(encode));
    world.local([&] {
        if (!world.rank)
            staged->publish(!zone.multi);
    });
}
}
