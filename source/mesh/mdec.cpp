#include "mdec.h"
#include "mesh_reader.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace aero {
namespace mdec {
void check(bool condition, const char* message) {
    if (!condition)
        throw MeshError(std::string("mdec: ") + message);
}
namespace {
Id add(Id a, Id b) {
    check(b <= absent - a, "offset overflow");
    return a + b;
}
Id mul(Id a, Id b) {
    check(!a || b <= absent / a, "size overflow");
    return a * b;
}
Id read(std::istream& in) {
    char bytes[8];
    check(static_cast<bool>(in.read(bytes, 8)), "truncated directory");
    return number_codec::read_word(bytes);
}
std::string block(std::istream& in, Id n) {
    std::string s;
    check(n <= s.max_size() && n <= static_cast<Id>(std::numeric_limits<std::streamsize>::max()),
          "block too large");
    s.resize(static_cast<std::size_t>(n));
    if (n)
        check(static_cast<bool>(in.read(&s[0], static_cast<std::streamsize>(n))), "truncated input");
    return s;
}
std::string decode(const std::string& data, Id raw) {
    std::istringstream in(data);
    check(block(in, 8) == "AEROLZMA", "bad compressed leaf signature");
    io::LzmaInputBuffer codec(in, raw);
    std::istream decoded(&codec);
    decoded.exceptions(std::ios::badbit);
    std::string result;
    std::array<char, 32768> temp;
    while (decoded.read(temp.data(), temp.size()) || decoded.gcount()) {
        result.append(temp.data(), static_cast<std::size_t>(decoded.gcount()));
    }
    check(result.size() == raw, "incorrect decoded size");
    return result;
}
// По типу записи и числу ссылок определяет её размер с проверкой переполнения.
Id record_size(const std::string& data, std::size_t start, Id kind) {
    const auto field = [&](std::size_t k) {
        check(k <= data.size() - start, "short record header");
        return read_word(data, start + k);
    };
    if (kind == Node)
        return add(48, mul(16, field(40)));
    if (kind == Face)
        return add(48, mul(16, add(field(32), field(40))));
    check(kind == Cell, "unknown record kind");
    return add(24, mul(17, field(16)));
}
}
Id read_word(const std::string& data, std::size_t offset) {
    check(offset <= data.size() && data.size() - offset >= 8, "truncated record");
    return number_codec::read_word(data.data() + offset);
}
// Добавляет 64-битную величину в сообщение в том же формате, что и запись на диске.
void append_word(std::string& data, Id value) {
    char bytes[sizeof(value)];
    number_codec::store_word(bytes, value);
    data.append(bytes, sizeof(bytes));
}
double Record::coordinate(unsigned axis) const {
    return number_codec::real_from_bits(word(16 + 8 * axis));
}
Id Record::primary_count() const {
    switch (kind()) {
    case Node:
        return word(40);
    case Face:
        return word(32);
    default:
        return word(16);
    }
}
Id Record::secondary_count() const {
    return kind() == Face ? word(40) : 0;
}
Address Record::link(Id i, bool secondary) const {
    const Id k = kind(), pos = k == Cell ? 24 + 17 * i : 48 + 16 * (i + (secondary ? primary_count() : 0));
    return {word(static_cast<std::size_t>(pos)), word(static_cast<std::size_t>(pos + 8))};
}
std::int8_t Record::sign(Id i) const {
    const auto value = static_cast<unsigned char>(bytes.at(static_cast<std::size_t>(40 + 17 * i)));
    check(value == 1 || value == 255, "invalid face orientation");
    return value == 1 ? 1 : -1;
}
bool Archive::matches(const std::string& path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    char sig[8];
    return f.read(sig, 8) && std::memcmp(sig, archive_format::signature, 8) == 0;
}
// Открывает архив и проверяет его каталог перед обращением к адресам отдельных листьев.
Archive::Archive(const std::string& path) : file_(path.c_str(), std::ios::binary) {
    check(file_.is_open(), "cannot open archive");
    file_.seekg(0, std::ios::end);
    const auto end = file_.tellg();
    check(end >= static_cast<std::streamoff>(archive_format::header_bytes), "truncated archive header");
    length_ = static_cast<Id>(end);
    file_.seekg(0);
    check(block(file_, 8) == archive_format::signature, "unsupported archive");
    version_ = read(file_);
    check(version_ >= 1 && version_ <= archive_format::current_version, "unsupported archive version");
    const Id flags = read(file_), dim = read(file_);
    check(flags <= 1 && dim >= 1 && dim <= 3, "invalid flags or dimension");
    compressed_ = flags != 0;
    dimension = static_cast<unsigned>(dim);
    if (version_ < 5)
        read(file_); // Пропускаем старое поле единиц длины. Координаты записаны в узловых записях.
    for (auto& n : counts)
        n = read(file_);
    boundaries = read(file_);
    const Id names = read(file_);
    leaves = read(file_);
    entries_ = read(file_);
    root_ = read(file_);
    const Id header = archive_format::header_size(version_);
    check(read(file_) == header, "invalid patch table offset");
    index_ = read(file_);
    tree_ = read(file_);
    payload_ = read(file_);
    check(index_ >= header && names <= (index_ - header) / 8 &&
              boundaries <= counts[Face],
          "invalid patch table");
    check(entries_ == (leaves ? mul(2, leaves) - 1 : 0) && root_ == (leaves ? entries_ - 1 : absent),
          "invalid tree header");
    Id end_tree = add(tree_, mul(archive_format::tree_entry_bytes, entries_));
    check(tree_ == add(index_, mul(archive_format::leaf_entry_bytes, leaves)) && end_tree <= payload_ &&
              payload_ <= length_,
          "invalid directory offsets");
    read_patch_names(names);
    if (indexed_halo())
        end_tree = read_halo_directory(end_tree);
    // Перед данными листьев версия 4 хранит хеш исходной сетки для restart и ввода полей.
    if (version_ >= 4) {
        file_.seekg(static_cast<std::streamoff>(end_tree));
        mesh_sha256.resize(64);
        file_.read(&mesh_sha256[0], 64);
        check(file_ && mesh_sha256.find_first_not_of("0123456789abcdef") == std::string::npos,
              "invalid mesh SHA-256");
        end_tree = add(end_tree, 64);
    }
    check(end_tree == payload_, "invalid halo directory extent");
}

// Таблица имён занимает точный диапазон между заголовком и началом индекса листьев.
void Archive::read_patch_names(Id names) {
    Id pos = archive_format::header_size(version_);
    for (Id i = 0; i < names; ++i) {
        check(index_ - pos >= 8, "truncated patch table");
        const Id n = read(file_);
        check(n <= 4096 && n <= index_ - pos - 8, "invalid patch name size");
        patches.push_back(block(file_, n));
        check(mesh_input::valid_name(patches.back()), "invalid patch name");
        pos += 8 + n;
    }
    check(pos == index_, "invalid patch table length");
}

// Числа и смещения описывают строки CSR. Сами списки ячеек читаем позже только по запросу своего ранга.
Id Archive::read_halo_directory(Id end_tree) {
    file_.seekg(static_cast<std::streamoff>(end_tree));
    const Id n = read(file_), h = read(file_);
    end_tree = add(end_tree, 16);
    const auto rows = [&](Id count, Id entries) {
        Rows r = {end_tree, add(end_tree, mul(8, add(count, 1))), count, entries};
        end_tree = add(r.refs, mul(16, entries));
        return r;
    };
    stars_ = rows(counts[Node], n);
    halo_ = rows(mul(2, leaves), h);
    if (far_halo()) {
        check(end_tree <= payload_ && payload_ - end_tree >= 8, "truncated far halo directory");
        file_.seekg(static_cast<std::streamoff>(end_tree));
        const Id far_count = read(file_);
        end_tree = add(end_tree, 8);
        far_ = rows(leaves, far_count);
    }
    return end_tree;
}

Archive::Branch Archive::branch(Id i) {
    check(i < entries_, "invalid tree pointer");
    file_.seekg(static_cast<std::streamoff>(tree_ + archive_format::tree_entry_bytes * i));
    Branch row;
    for (Id& v : row)
        v = read(file_);
    return row;
}
Archive::Branch Archive::root() {
    const auto row = branch(root_);
    check(row[0] == 0 && row[1] == leaves && row[5] == counts[Node] && row[6] == counts[Face] &&
              row[7] == counts[Cell],
          "incorrect root totals");
    return row;
}
void Archive::children(const Branch& parent, Branch& left, Branch& right) {
    left = branch(parent[2]);
    right = branch(parent[3]);
    check(left[0] == parent[0] && left[1] > 0 && right[1] > 0 && add(left[1], right[1]) == parent[1] &&
              add(left[0], left[1]) == right[0],
          "invalid child ranges");
    for (unsigned k = 4; k < 8; ++k)
        check(add(left[k], right[k]) == parent[k], "incorrect branch totals");
}
Id Archive::row_offset(const Rows& rows, Id row) {
    check(row <= rows.count, "invalid halo prefix");
    file_.seekg(static_cast<std::streamoff>(rows.offsets + 8 * row));
    const Id value = read(file_);
    check(value <= rows.entries && (row || !value) && (row != rows.count || value == rows.entries),
          "invalid halo prefix");
    return value;
}
Archive::Prefix Archive::prefix(Id end_leaf, bool fifth) {
    check(end_leaf <= leaves, "invalid leaf prefix");
    Prefix result;
    if (!end_leaf)
        return result;
    auto row = root();
    while (end_leaf != row[0] + row[1]) {
        Branch left, right;
        children(row, left, right);
        if (end_leaf <= right[0])
            row = left;
        else {
            result.raw = add(result.raw, left[4]);
            result.nodes = add(result.nodes, left[5]);
            row = right;
        }
    }
    result.raw = add(result.raw, row[4]);
    result.nodes = add(result.nodes, row[5]);
    if (indexed_halo())
        result.halo = row_offset(halo_, 2 * end_leaf);
    if (fifth && far_halo())
        result.halo = add(result.halo, row_offset(far_, end_leaf));
    return result;
}

std::vector<Id> Archive::partition(int processes) {
    const auto weight = [](const std::array<Id, 8>& row) {
        return static_cast<long double>(row[4]) + 32.0L * row[5];
    };
    std::vector<Id> cuts(static_cast<std::size_t>(processes) + 1, leaves);
    cuts[0] = 0;
    if (!leaves)
        return cuts;
    const auto top = root();
    const Id active = std::min<Id>(leaves, static_cast<unsigned>(processes));
    for (Id r = 1; r < active; ++r) {
        long double target = weight(top) * r / active;
        auto row = top;
        while (row[1] > 1) {
            Branch left, right;
            children(row, left, right);
            if (target <= weight(left))
                row = left;
            else {
                target -= weight(left);
                row = right;
            }
        }
        const Id nearest = row[0] + (target >= weight(row) / 2 ? 1u : 0u);
        cuts[r] = std::max(cuts[r - 1] + 1, std::min(nearest, leaves - (active - r)));
    }
    return cuts;
}
void Archive::read_row(const Rows& rows, Id row, Addresses& result) {
    check(indexed_halo() && row < rows.count, "invalid halo row");
    file_.seekg(static_cast<std::streamoff>(rows.offsets + 8 * row));
    const Id first = read(file_), end = read(file_);
    check(first <= end && end <= rows.entries && (row || !first) &&
              (row + 1 < rows.count || end == rows.entries),
          "invalid halo row extent");
    if (first == end)
        return;
    file_.seekg(static_cast<std::streamoff>(rows.refs + 16 * first));
    for (Id i = first; i < end; ++i) {
        const Id leaf = read(file_), offset = read(file_);
        check(leaf < leaves, "halo reference to nonexistent leaf");
        result.emplace(leaf, offset);
    }
}
void Archive::cell_star(Id node, Addresses& cells) {
    read_row(stars_, node, cells);
}
void Archive::far_cells(Id leaf, Addresses& cells) {
    read_row(far_, leaf, cells);
}
void Archive::leaf_cells(Id leaf, bool outer, Addresses& cells) {
    read_row(halo_, 2 * leaf + (outer ? 1 : 0), cells);
}

// Получает недостающие записи из архива по их адресам и добавляет в локальный набор.
void Archive::read_records(const Addresses& needed, Kind kind, Records& records) {
    Id loaded = absent;
    Records leaf;
    for (const auto& a : needed) {
        auto existing = records.find(a);
        if (existing == records.end()) {
            if (loaded != a.first) {
                leaf.clear();
                read_leaf(a.first, leaf);
                loaded = a.first;
            }
            auto it = leaf.find(a);
            check(it != leaf.end(), "dangling halo record address");
            existing = records.emplace(a, std::move(it->second)).first;
        }
        check(existing->second.kind() == kind, "wrong entity type in halo");
    }
}

void Archive::read_leaf(Id leaf, Records& records) {
    check(leaf < leaves, "invalid leaf number");
    file_.seekg(static_cast<std::streamoff>(index_ + archive_format::leaf_entry_bytes * leaf));
    const Id pos = read(file_), packed = read(file_), raw = read(file_);
    std::array<Id, 3> remaining;
    for (Id& n : remaining)
        n = read(file_);
    const Id end = leaf + 1 == leaves ? length_ : read(file_);
    check(pos >= payload_ && end <= length_ && pos < end && packed == end - pos &&
              (leaf || pos == payload_) && (compressed_ || raw == packed),
          "invalid leaf extent");
    file_.seekg(static_cast<std::streamoff>(pos));
    auto data = block(file_, packed);
    if (compressed_)
        data = decode(data, raw);
    for (std::size_t k = 0; k < data.size();) {
        const Id kind = read_word(data, k), length = record_size(data, k, kind);
        check(length <= data.size() - k && remaining[kind] > 0, "invalid record extent or leaf counts");
        Record record{data.substr(k, static_cast<std::size_t>(length))};
        check(record.id() < counts[kind], "global entity ID out of range");
        if (kind == Node) {
            for (unsigned a = 0; a < 3; ++a)
                check(std::isfinite(record.coordinate(a)), "invalid coordinate");
            for (Id j = 0; j < record.primary_count(); ++j)
                check(record.link(j).first < leaves, "reference to nonexistent leaf");
        } else if (kind == Face) {
            const Id boundary = record.boundary_order(), patch = record.boundary_patch();
            check((boundary == absent && patch == absent && record.secondary_count() == 2) ||
                      (boundary < boundaries && patch < patches.size() && record.secondary_count() == 1),
                  "invalid physical face");
        }
        --remaining[kind];
        records.emplace(Address{leaf, k}, std::move(record));
        k += static_cast<std::size_t>(length);
    }
    check(remaining == std::array<Id, 3>{{0, 0, 0}}, "incomplete leaf");
}
}
}
