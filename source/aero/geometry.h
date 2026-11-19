#ifndef AERO_GEOMETRY_H
#define AERO_GEOMETRY_H
#include "../mesh/partition.h"
namespace aero {
struct CvView;
struct RunSettings;
class Processes;
// Строит контрольные объёмы и записывает геометрию каждого процесса.
void export_case_geometry(const RunSettings& run, const Processes& world, bool archive);
// Записывает объёмы и площади с исходными номерами узлов и граней.
void write_geometry(const std::string& output, const Mesh& mesh, const CvView& cv,
                    const MeshPartition* partition, int rank, int ranks);
}
#endif
