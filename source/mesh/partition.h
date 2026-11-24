#ifndef AERO_MESH_PARTITION_H
#define AERO_MESH_PARTITION_H
#include "mdec.h"
#include <memory>
namespace aero {
class Processes;
// Диапазоны листьев идут в пространственном порядке. Номера процессов одного узла в MPI_COMM_WORLD могут чередоваться.
struct LeafLayout {
    // Диапазон листьев позиции i равен [cuts[i], cuts[i+1]).
    std::vector<MeshIndex> cuts;
    // Для каждой позиции разбиения хранится настоящий номер MPI-процесса.
    std::vector<int> ranks;
    // По номеру листа находит его интервал и возвращает ранг владельца.
    int owner(MeshIndex leaf) const;
};
// Локальная нумерация следует исходным номерам. Для каждого собственного узла храним все примыкающие ячейки.
// Публикуем только рёбра, у которых процесс владеет концом с меньшим номером.
struct MeshPartition {
    Mesh mesh;
    std::string mesh_sha256;
    std::array<MeshIndex, 3> global_counts{}; // Число узлов, граней и ячеек всей сетки.
    // Таблицы перехода от местного номера к исходному для каждого типа элемента.
    std::vector<MeshIndex> nodes, faces, cells;
    std::vector<MeshIndex>
        boundary_order; // Номер во входной таблице внешних граней; у искусственного разреза отсутствует.
    // MPI-процесс, отвечающий за состояние каждого локального узла.
    std::vector<int> node_owners;
    // Число исходных меток. Искусственная граница разреза может получить дополнительную метку.
    MeshIndex physical_patches = 0;
    // Полуинтервал собственных листьев в каталоге.
    MeshIndex first_leaf = 0, end_leaf = 0;
};
// Распакованные записи ядра раздела нужны до окончания построения строк реконструкции.
class PartitionInput {
public:
    // Открывает архив сетки и выбирает листья текущего MPI-процесса.
    PartitionInput(const std::string& path, const Processes& world, const LeafLayout& layout);
    // Читает собственные узлы и все ячейки, необходимые для их контрольных объёмов.
    MeshPartition core(const Processes& world);
    using NodeOwners = std::vector<std::pair<MeshIndex, int>>;
    // Расширяет входные данные геометрией, необходимой шаблону реконструкции.
    void prepare_stencil(const std::set<MeshIndex>& periodic_nodes, bool fifth, NodeOwners* near_vertices);
    // Добавляет ячейки вокруг запрошенных вершин, в том числе у периодических копий.
    void add_stars(const std::set<MeshIndex>& vertices);
    // Дополняет выбранный слой гранями для поиска пересечений луча WENO.
    void complete_layer_faces();
    // Собирает номера узлов указанных граничных участков.
    NodeOwners boundary_nodes(std::set<MeshIndex>* incomplete_stars);
    // Возвращает вспомогательную сетку со всеми запрошенными опорными ячейками.
    MeshPartition stencil();

private:
    // Источник записей листьев и индексов соседства на время подготовки раздела.
    mdec::Archive archive_;
    LeafLayout layout_;
    int slot_;
    // Уже прочитанные записи по паре «лист, смещение». Соседние ячейки могут ссылаться на одну запись.
    mdec::Records records_;
    mdec::Addresses owned_, cells_;
    // Возвращает строку граней указанной ячейки.
    mdec::Addresses cell_faces(const mdec::Addresses& cells);
    // Возвращает строку узлов указанной грани.
    mdec::Addresses face_nodes(const mdec::Addresses& faces);
    // Строит локальную сетку из выбранных глобальных сущностей.
    MeshPartition indexed_mesh(const mdec::Addresses& cells);
};
// Загружает собственную геометрию и требуемые halo-связи подобласти.
MeshPartition load_partition(const std::string& path, const Processes& world,
                             std::unique_ptr<PartitionInput>* stencil_input = nullptr,
                             const LeafLayout& layout = {});
}
#endif
