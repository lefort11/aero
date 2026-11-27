#include "partition.h"
#include "mdec.h"
#include "../parallel/processes.h"
#include <algorithm>
#include <memory>
#include <numeric>
#include <set>

namespace aero {
// По номеру листа находит его интервал и возвращает ранг владельца.
int LeafLayout::owner(MeshIndex leaf) const {
    mdec::check(leaf < cuts.back(), "reference to nonexistent leaf");
    return ranks[static_cast<std::size_t>(std::upper_bound(cuts.begin(), cuts.end(), leaf) - cuts.begin() -
                                          1)];
}
namespace {
using namespace mdec;

// Исходные номера узлов должны ровно один раз покрывать [0, count). Каждый процесс проверяет свой класс остатков.
// Проверка не требует собирать глобальную карту номеров.
void verify_owners(const Addresses& owned, const Records& records, Id count, const Processes& world) {
    Processes::Messages send;
    const Id peers = static_cast<unsigned>(world.size);
    world.local([&] {
        send.resize(static_cast<std::size_t>(peers));
        for (const auto& a : owned) {
            const Id id = records.at(a).id();
            append_word(send[id % peers], id);
        }
    });
    const auto incoming = world.exchange(send);
    world.local([&] {
        const Id rank = static_cast<unsigned>(world.rank);
        const Id expected = count / peers + (rank < count % peers ? 1u : 0u);
        Id actual = 0;
        for (const auto& msg : incoming)
            actual += msg.size() / 8;
        check(actual == expected, "missing or duplicate global node owner");
        std::vector<bool> seen(static_cast<std::size_t>(expected));
        for (const auto& msg : incoming)
            for (std::size_t pos = 0; pos < msg.size(); pos += 8) {
                const Id id = read_word(msg, pos);
                check(id < count && id % peers == rank && !seen[id / peers], "duplicate global node owner");
                seen[id / peers] = true;
            }
    });
}

// Разрешаем ссылки пакетами. Для одного запрашивающего процесса запись пересылается один раз,
// даже если несколько соседних ячеек ссылаются на неё.
void fetch(const Addresses& needed, Kind kind, Records& records, const LeafLayout& layout,
           const Processes& world) {
    Processes::Messages requests;
    world.local([&] {
        requests.resize(static_cast<std::size_t>(world.size));
        for (const auto& a : needed) {
            const auto it = records.find(a);
            if (it != records.end()) {
                check(it->second.kind() == kind, "wrong entity type in link");
                continue;
            }
            const int p = layout.owner(a.first);
            check(p != world.rank, "dangling local reference");
            append_word(requests[p], a.first);
            append_word(requests[p], a.second);
        }
    });
    const auto incoming = world.exchange(requests);
    Processes::Messages replies;
    world.local([&] {
        replies.resize(incoming.size());
        for (std::size_t p = 0; p < incoming.size(); ++p) {
            for (std::size_t k = 0; k < incoming[p].size(); k += 16) {
                const Address a = {read_word(incoming[p], k), read_word(incoming[p], k + 8)};
                const auto it = records.find(a);
                check(layout.owner(a.first) == world.rank && it != records.end() && it->second.kind() == kind,
                      "missing or incompatible remote record");
                append_word(replies[p], it->second.bytes.size());
                replies[p].append(it->second.bytes);
            }
        }
    });
    const auto answers = world.exchange(replies);
    world.local([&] {
        for (std::size_t p = 0; p < answers.size(); ++p) {
            std::size_t pos = 0;
            for (std::size_t k = 0; k < requests[p].size(); k += 16) {
                const Address a = {read_word(requests[p], k), read_word(requests[p], k + 8)};
                const Id n = read_word(answers[p], pos);
                pos += 8;
                check(n <= answers[p].size() - pos, "short halo response");
                records.emplace(a, Record{answers[p].substr(pos, static_cast<std::size_t>(n))});
                pos += static_cast<std::size_t>(n);
            }
            check(pos == answers[p].size(), "unexpected halo response");
        }
    });
}
// Читает связи сущностей, требуемые для сборки локальной подобласти.
Addresses links(const Addresses& source, const Records& records, bool secondary) {
    Addresses result;
    for (const auto& a : source) {
        const auto& r = records.at(a);
        const Id count = secondary ? r.secondary_count() : r.primary_count();
        for (Id i = 0; i < count; ++i)
            result.insert(r.link(i, secondary));
    }
    return result;
}
// Находит запись сущности по её исходному глобальному номеру.
std::vector<Address> by_id(const Addresses& addresses, const Records& records) {
    std::vector<std::pair<Id, Address>> keys;
    keys.reserve(addresses.size());
    for (const auto& a : addresses)
        keys.emplace_back(records.at(a).id(), a);
    std::sort(keys.begin(), keys.end());
    std::vector<Address> sorted;
    sorted.reserve(keys.size());
    Id previous = absent;
    for (const auto& key : keys) {
        check(previous != key.first, "duplicate global ID");
        sorted.push_back(key.second);
        previous = key.first;
    }
    return sorted;
}
// Собирает локальную сетку из записей архива и таблиц перенумерации.
MeshPartition assemble(const Archive& archive, const Records& records, const LeafLayout& layout, int slot,
                       const Addresses& nodes, const Addresses& faces, const Addresses& cells) {
    MeshData d;
    d.dimension = archive.dimension;
    d.patch_names = archive.patches;
    MeshPartition result;
    result.mesh_sha256 = archive.mesh_sha256;
    result.global_counts = archive.counts;
    result.first_leaf = layout.cuts[slot];
    result.end_leaf = layout.cuts[slot + 1];
    result.physical_patches = d.patch_names.size();
    std::map<Address, Id> node_number, face_number;
    for (const Address& a : by_id(nodes, records)) {
        const auto& r = records.at(a);
        node_number.emplace(a, result.nodes.size());
        result.nodes.push_back(r.id());
        result.node_owners.push_back(layout.owner(a.first));
        d.x.push_back(r.coordinate(0));
        d.y.push_back(r.coordinate(1));
        d.z.push_back(r.coordinate(2));
    }
    const auto sorted_faces = by_id(faces, records);
    for (const auto& a : sorted_faces) {
        const auto& r = records.at(a);
        face_number.emplace(a, result.faces.size());
        result.faces.push_back(r.id());
        for (Id i = 0; i < r.primary_count(); ++i)
            d.face_nodes.indices.push_back(node_number.at(r.link(i)));
        d.face_nodes.offsets.push_back(d.face_nodes.indices.size());
    }
    std::vector<unsigned> incidence(result.faces.size());
    for (const Address& a : by_id(cells, records)) {
        const auto& r = records.at(a);
        result.cells.push_back(r.id());
        for (Id i = 0; i < r.primary_count(); ++i) {
            const auto f = face_number.at(r.link(i));
            ++incidence[f];
            d.cell_faces.indices.push_back(f);
            d.cell_face_orientation.push_back(r.sign(i));
        }
        d.cell_faces.offsets.push_back(d.cell_faces.indices.size());
    }
    // При чтении всей области сохраняем исходный порядок границы. Грани разреза замыкают вспомогательную сетку;
    // в физическую поверхность контрольного объёма входят только настоящие граничные грани.
    std::vector<std::pair<Id, Id>> boundary;
    for (Id f = 0; f < sorted_faces.size(); ++f) {
        const auto& r = records.at(sorted_faces[f]);
        check(incidence[f] >= 1 && incidence[f] <= r.secondary_count(), "inconsistent face incidence");
        if (incidence[f] == 1)
            boundary.emplace_back(r.boundary_order(), f);
    }
    std::sort(boundary.begin(), boundary.end());
    for (auto entry : boundary) {
        const auto& r = records.at(sorted_faces[entry.second]);
        Id patch = r.boundary_patch();
        if (entry.first == absent) {
            patch = result.physical_patches;
            if (d.patch_names.size() == patch) {
                std::string name = "partition-cut";
                while (std::find(d.patch_names.begin(), d.patch_names.end(), name) != d.patch_names.end())
                    name += "_";
                d.patch_names.push_back(name);
            }
        }
        d.boundary_faces.push_back(entry.second);
        result.boundary_order.push_back(entry.first);
        d.boundary_patch.push_back(patch);
    }
    result.mesh = Mesh(std::move(d));
    return result;
}
}
// Открывает архив сетки и выбирает листья текущего MPI-процесса.
PartitionInput::PartitionInput(const std::string& path, const Processes& world, const LeafLayout& layout)
    : archive_(path), layout_(layout), slot_(world.rank) {
    if (layout_.cuts.empty()) {
        layout_.cuts = archive_.partition(world.size);
        layout_.ranks.resize(world.size);
        std::iota(layout_.ranks.begin(), layout_.ranks.end(), 0);
    } else {
        const auto it = std::find(layout_.ranks.begin(), layout_.ranks.end(), world.rank);
        check(it != layout_.ranks.end() && layout_.cuts.back() == archive_.leaves, "inconsistent MPI layout");
        slot_ = static_cast<int>(it - layout_.ranks.begin());
    }
    for (Id i = layout_.cuts[slot_]; i < layout_.cuts[slot_ + 1]; ++i)
        archive_.read_leaf(i, records_);
    for (const auto& item : records_)
        if (item.second.kind() == Node)
            owned_.insert(item.first);
}

// Возвращает строку граней указанной ячейки.
Addresses PartitionInput::cell_faces(const Addresses& cells) {
    archive_.read_records(cells, Cell, records_);
    auto faces = links(cells, records_, false);
    archive_.read_records(faces, Face, records_);
    return faces;
}
// Возвращает строку узлов указанной грани.
Addresses PartitionInput::face_nodes(const Addresses& faces) {
    auto nodes = links(faces, records_, false);
    archive_.read_records(nodes, Node, records_);
    return nodes;
}
// Строит локальную сетку из выбранных глобальных сущностей.
MeshPartition PartitionInput::indexed_mesh(const Addresses& cells) {
    const auto faces = cell_faces(cells);
    auto nodes = face_nodes(faces);
    nodes.insert(owned_.begin(), owned_.end());
    return assemble(archive_, records_, layout_, slot_, nodes, faces, cells);
}

// Расширяет входные данные геометрией, необходимой шаблону реконструкции.
void PartitionInput::prepare_stencil(const std::set<Id>& periodic_nodes, bool fifth,
                                     NodeOwners* near_vertices) {
    if (fifth)
        check(
            archive_.far_halo(),
            "fifth-order MPI reconstruction requires the far halo index: regenerate the .mdec with meshdec");
    check(archive_.indexed_halo(),
          "MPI reconstruction requires the halo index: regenerate the .mdec with meshdec");
    for (Id i = layout_.cuts[slot_]; i < layout_.cuts[slot_ + 1]; ++i)
        archive_.leaf_cells(i, true, cells_);
    Addresses seam_cells;
    for (Id n : periodic_nodes)
        archive_.cell_star(n, seam_cells);
    Addresses remote_cells;
    for (const auto& a : seam_cells)
        if (cells_.insert(a).second && fifth)
            remote_cells.insert(a);
    if (!fifth)
        return;
    if (near_vertices) {
        // Ближняя опора внутреннего ребра может попасть на периодический шов. Копии узлов дают соседей второго слоя.
        for (const auto& a : face_nodes(cell_faces(cells_)))
            near_vertices->emplace_back(records_.at(a).id(), layout_.owner(a.first));
        // Дополнительный поиск соседей нужен только за обычными слоями листа. Дальний каталог уже охватывает все обычные ближние ячейки.
        for (const auto& a : face_nodes(cell_faces(remote_cells)))
            archive_.cell_star(records_.at(a).id(), cells_);
    }
    for (Id i = layout_.cuts[slot_]; i < layout_.cuts[slot_ + 1]; ++i)
        archive_.far_cells(i, cells_);
}
// Добавляет ячейки вокруг запрошенных вершин, в том числе у периодических копий.
void PartitionInput::add_stars(const std::set<Id>& vertices) {
    for (Id n : vertices)
        archive_.cell_star(n, cells_);
}

// WENO проверяет каждую грань, чьи вершины вошли в слой. Такая грань может лежать за пределами исходных ячеек слоя.
// Берём индекс «узел–грань» из meshdec и сохраняем обе прилегающие ячейки для определения ориентации пересечения.
void PartitionInput::complete_layer_faces() {
    const auto nodes = face_nodes(cell_faces(cells_));
    const auto incident = links(nodes, records_, false);
    archive_.read_records(incident, Face, records_);
    for (const auto& a : incident) {
        const auto& f = records_.at(a);
        Id i = 0;
        while (i < f.primary_count() && nodes.count(f.link(i)))
            ++i;
        if (i == f.primary_count())
            for (Id k = 0; k < f.secondary_count(); ++k)
                cells_.insert(f.link(k, true));
    }
}

// Собирает номера узлов указанных граничных участков.
PartitionInput::NodeOwners PartitionInput::boundary_nodes(std::set<Id>* incomplete_stars) {
    const auto nodes = face_nodes(cell_faces(cells_));
    archive_.read_records(links(nodes, records_, false), Face, records_);
    NodeOwners boundary;
    for (const auto& a : nodes) {
        const auto& n = records_.at(a);
        bool physical = false, cut = false;
        for (Id i = 0; i < n.primary_count(); ++i) {
            const auto& f = records_.at(n.link(i));
            if (f.secondary_count() == 1)
                physical = true;
            if (incomplete_stars && !cut)
                for (Id j = 0; j < f.secondary_count(); ++j)
                    if (!cells_.count(f.link(j, true)))
                        cut = true;
        }
        if (physical)
            boundary.emplace_back(n.id(), layout_.owner(a.first));
        if (cut)
            incomplete_stars->insert(n.id());
    }
    return boundary;
}

// Возвращает вспомогательную сетку со всеми запрошенными опорными ячейками.
MeshPartition PartitionInput::stencil() {
    return indexed_mesh(cells_);
}

// Читает собственные узлы и все ячейки, необходимые для их контрольных объёмов.
MeshPartition PartitionInput::core(const Processes& world) {
    verify_owners(owned_, records_, archive_.counts[Node], world);
    MeshPartition result;
    if (archive_.indexed_halo()) {
        world.local([&] {
            for (Id i = layout_.cuts[slot_]; i < layout_.cuts[slot_ + 1]; ++i)
                archive_.leaf_cells(i, false, cells_);
            result = indexed_mesh(cells_);
        });
        return result;
    }
    // У архива версии 1 есть только непосредственные связи. При его чтении используем первоначальный порядок подготовки.
    Addresses incident_faces, faces, nodes;
    world.local([&] { incident_faces = links(owned_, records_, false); });
    fetch(incident_faces, Face, records_, layout_, world);
    world.local([&] { cells_ = links(incident_faces, records_, true); });
    fetch(cells_, Cell, records_, layout_, world);
    world.local([&] { faces = links(cells_, records_, false); });
    fetch(faces, Face, records_, layout_, world);
    world.local([&] {
        nodes = links(faces, records_, false);
        nodes.insert(owned_.begin(), owned_.end());
    });
    fetch(nodes, Node, records_, layout_, world);
    world.local([&] { result = assemble(archive_, records_, layout_, slot_, nodes, faces, cells_); });
    return result;
}

namespace {
// Таблица одного типа записей листа. Пары (смещение, исходный номер) уже отсортированы read_leaf.
using LeafIds = std::vector<std::pair<Id, Id>>;
// Индексируем только узлы и грани: именно на них ссылаются строки связности.
using GeometryIndex = std::vector<std::array<LeafIds, 2>>;

// Преобразует архивный адрес в исходный номер и проверяет тип связанной записи.
Id original_id(const GeometryIndex& index, Kind kind, const Address& address) {
    const auto& ids = index.at(address.first)[kind];
    const auto found = std::lower_bound(ids.begin(), ids.end(), std::make_pair(address.second, Id(0)));
    check(found != ids.end() && found->first == address.second, "invalid global geometry link");
    return found->second;
}

// Превращает длины строк в смещения; рабочий массив длин переиспользуется для готовой CSR.
void allocate_rows(MeshCsr& csr) {
    Id size = 0;
    for (auto& count : csr.offsets) {
        check(count <= csr.indices.max_size() - size, "global connectivity too large");
        size += count;
        count = size;
    }
    csr.indices.resize(size);
}

// Обходит архив по листьям. Распакованные записи освобождаются перед чтением следующего листа.
template<class Visitor> void visit_geometry(Archive& archive, const Visitor& visit) {
    for (Id leaf = 0; leaf < archive.leaves; ++leaf) {
        Records records;
        archive.read_leaf(leaf, records);
        for (const auto& item : records) visit(item.first, item.second);
    }
}

// В первый проход заполняем координаты и размеры строк в исходной нумерации.
// Во второй проходим по тем же листьям, когда уже известны номера удалённых концов связей.
MeshData archive_mesh_data(Archive& archive) {
    MeshData data;
    data.dimension = archive.dimension;
    data.patch_names = archive.patches;
    data.x.resize(archive.counts[Node]);
    data.y.resize(data.x.size());
    data.z.resize(data.x.size());
    data.boundary_faces.resize(archive.boundaries, absent);
    data.boundary_patch.resize(archive.boundaries);
    data.face_nodes.offsets.resize(1 + archive.counts[Face]);
    data.cell_faces.offsets.resize(1 + archive.counts[Cell]);
    GeometryIndex index(archive.leaves);
    std::array<std::vector<bool>, 3> seen;
    for (auto kind : {Node, Face, Cell}) seen[kind].resize(archive.counts[kind]);
    visit_geometry(archive, [&](const Address& address, const Record& record) {
        const auto kind = record.kind(), id = record.id();
        check(!seen[kind][id], "duplicate global geometry ID");
        seen[kind][id] = true;
        if (kind != Cell) index[address.first][kind].emplace_back(address.second, id);
        if (kind == Node) {
            data.x[id] = record.coordinate(0);
            data.y[id] = record.coordinate(1);
            data.z[id] = record.coordinate(2);
        } else if (kind == Face) {
            data.face_nodes.offsets[id + 1] = record.primary_count();
            const Id order = record.boundary_order();
            if (order != absent) {
                check(data.boundary_faces[order] == absent, "duplicate global boundary ID");
                data.boundary_faces[order] = id;
                data.boundary_patch[order] = record.boundary_patch();
            }
        } else data.cell_faces.offsets[id + 1] = record.primary_count();
    });
    for (const auto& ids : seen)
        check(std::find(ids.begin(), ids.end(), false) == ids.end(), "missing global geometry ID");
    allocate_rows(data.face_nodes);
    allocate_rows(data.cell_faces);
    data.cell_face_orientation.resize(data.cell_faces.indices.size());
    visit_geometry(archive, [&](const Address&, const Record& record) {
        if (record.kind() == Node) return;
        const bool cell = record.kind() == Cell;
        auto& row = cell ? data.cell_faces : data.face_nodes;
        const auto start = row.offsets[record.id()];
        for (Id j = 0; j < record.primary_count(); ++j) {
            row.indices[start + j] = original_id(index, cell ? Face : Node, record.link(j));
            if (cell) data.cell_face_orientation[start + j] = record.sign(j);
        }
    });
    return data;
}
}
// Читает полную геометрию для общей визуализации. Вспомогательный индекс освобождается
// до построения производных рёбер и связности Mesh, снижая пик памяти ранга 0.
Mesh read_archive_geometry(const std::string& path) {
    Archive archive(path);
    return Mesh(archive_mesh_data(archive));
}
// Загружает собственную геометрию и требуемые halo-связи подобласти.
MeshPartition load_partition(const std::string& path, const Processes& world,
                             std::unique_ptr<PartitionInput>* stencil_input, const LeafLayout& layout) {
    std::unique_ptr<PartitionInput> input;
    world.local([&] { input.reset(new PartitionInput(path, world, layout)); });
    auto partition = input->core(world);
    if (stencil_input)
        *stencil_input = std::move(input);
    return partition;
}
}
