#pragma once
#include "../output_settings.h"

namespace aero {
class StagedFieldFile;
// Записывает снимок зоны и выбранные представления, используя одни и те же столбцы.
// Для Cp аргумент mesh равен nullptr, а координаты находятся в самом снимке.
void write_zone_files(const std::string& directory, const OutputZone& zone,
                      const Mesh* mesh, const FieldSnapshot& snapshot);
// Завершает уже записанный (в том числе коллективно) основной файл зоны.
void publish_zone_files(StagedFieldFile& native, const std::string& path, const OutputZone& zone,
                        const Mesh* mesh, const FieldSnapshot& snapshot,
                        const std::vector<std::uint64_t>* offsets = nullptr);
}
