#include "indexed_io.h"
#include "../io/number_codec.h"
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>

namespace aero {
namespace indexed_io {
// Записывает двоичное представление вещественного числа без потери точности.
void put_real(char* dst, double value) {
    number_codec::store_real(dst, value);
}
namespace {
#ifdef AERO_WITH_MPI
// Преобразует код ошибки MPI-IO в исключение с названием операции.
void check(int code, const char* operation) {
    if (code == MPI_SUCCESS)
        return;
    char reason[MPI_MAX_ERROR_STRING];
    int size = 0;
    MPI_Error_string(code, reason, &size);
    throw std::runtime_error(std::string("indexed I/O: ") + operation + ": " + std::string(reason, size));
}

// Ограничиваем отдельно объём передаваемых байтов и расстояние в файле между первым и последним байтом пакета.
std::size_t batch_size(const std::vector<OwnedNode>& nodes, std::size_t first, unsigned stride,
                       std::size_t fields) {
    const auto count = std::min<std::size_t>(256 * 1024 / stride / fields, nodes.size() - first);
    if (!count)
        return 0;
    const auto begin = nodes.begin() + first;
    const auto span = static_cast<std::uint64_t>(INT_MAX) / stride;
    if (begin[count - 1].id - begin->id < span)
        return count;
    const auto end = std::lower_bound(begin, begin + count, begin->id + span,
                                      [](const OwnedNode& n, std::uint64_t limit) { return n.id < limit; });
    return static_cast<std::size_t>(end - begin);
}

// Представление MPI-файла начинается с первой выбранной записи и описывает участки между пропусками.
// Для непрерывного и пустого пакета достаточно обычного байтового представления.
struct Batch {
    std::vector<char> bytes;

    MPI_Offset base = 0;
    MPI_Datatype type = MPI_BYTE;
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~Batch() {
        if (type != MPI_BYTE && type != MPI_DATATYPE_NULL)
            MPI_Type_free(&type);
    }

    // Готовит файловое смещение и MPI-тип для пакета выбранных узловых записей.
    void prepare(const std::vector<OwnedNode>& nodes, std::size_t first, std::size_t count, unsigned stride,
                 std::size_t fields) {
        if (!count)
            return;
        const auto id = nodes[first].id;
        base = static_cast<MPI_Offset>(id * stride);
        bytes.resize(count * stride * fields);
        if (nodes[first + count - 1].id - id == count - 1)
            return;
        std::vector<int> lengths;
        std::vector<MPI_Aint> starts;
        for (std::size_t j = 0; j < count; ++j) {
            const auto current = nodes[first + j].id;
            if (!j || current != nodes[first + j - 1].id + 1) {
                starts.push_back(static_cast<MPI_Aint>((current - id) * stride));
                lengths.push_back(0);
            }
            lengths.back() += static_cast<int>(stride);
        }
        MPI_Datatype indexed = MPI_DATATYPE_NULL;
        check(MPI_Type_create_hindexed(static_cast<int>(starts.size()), lengths.data(), starts.data(),
                                       MPI_BYTE, &indexed),
              "create record view");
        type = indexed;
        check(MPI_Type_commit(&type), "commit record view");
    }
};

// Читает или записывает принадлежащие рангу строки через коллективный MPI-IO.
void transfer_mpi(const Processes& world, const std::string& file, bool writing, unsigned stride,
                  const std::vector<OwnedNode>& nodes, const std::vector<std::uint64_t>& offsets,
                  const Row& row) {
    // Все процессы должны сделать одинаковое число коллективных вызовов. Закончивший раньше передаёт ноль байтов.
    std::uint64_t rounds = 0;
    for (std::size_t i = 0; i < nodes.size(); i += batch_size(nodes, i, stride, offsets.size()))
        ++rounds;
    world.local([&] {
        check(MPI_Allreduce(MPI_IN_PLACE, &rounds, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD),
              "count record batches");
    });
    MPI_File handle = MPI_FILE_NULL;
    world.local([&] {
        check(MPI_File_open(MPI_COMM_WORLD, const_cast<char*>(file.c_str()),
                            writing ? MPI_MODE_RDWR : MPI_MODE_RDONLY, MPI_INFO_NULL, &handle),
              "open shared file");
    });
    std::exception_ptr failure;
    try {
        world.local(
            [&] { check(MPI_File_set_errhandler(handle, MPI_ERRORS_RETURN), "set file error handler"); });
        std::size_t first = 0;
        for (std::uint64_t round = 0; round < rounds; ++round) {
            Batch batch;
            const auto count = batch_size(nodes, first, stride, offsets.size());
            const auto pitch = count * stride;
            world.local([&] {
                batch.prepare(nodes, first, count, stride, offsets.size());
                if (writing)
                    for (std::size_t i = 0; i < count; ++i)
                        row(nodes[first + i], batch.bytes.data() + i * stride, pitch);
            });
            for (std::size_t k = 0; k < offsets.size(); ++k) {
                world.local([&] {
                    const auto base = static_cast<MPI_Offset>(offsets[k]) + batch.base;
                    check(MPI_File_set_view(handle, base, MPI_BYTE, batch.type, const_cast<char*>("native"),
                                            MPI_INFO_NULL),
                          "set record view");
                });
                world.local([&] {
                    const int n = static_cast<int>(pitch);
                    char dummy = 0;
                    char* data = n ? batch.bytes.data() + k * pitch : &dummy;
                    MPI_Status status;
                    const int code = writing ? MPI_File_write_all(handle, data, n, MPI_BYTE, &status)
                                             : MPI_File_read_all(handle, data, n, MPI_BYTE, &status);
                    check(code, writing ? "write records" : "read records");
                    int transferred = 0;
                    check(MPI_Get_count(&status, MPI_BYTE, &transferred), "check record count");
                    if (transferred != n)
                        throw std::runtime_error("indexed I/O: incomplete record transfer");
                });
            }
            if (!writing)
                world.local([&] {
                    for (std::size_t i = 0; i < count; ++i)
                        row(nodes[first + i], batch.bytes.data() + i * stride, pitch);
                });
            first += count;
        }
        if (writing)
            world.local([&] { check(MPI_File_sync(handle), "flush records"); });
    } catch (...) {
        // К выходу из пакета каждый процесс уже получил сведения об общей ошибке.
        failure = std::current_exception();
    }
    world.local([&] { check(MPI_File_close(&handle), "close shared file"); });
    if (failure)
        std::rethrow_exception(failure);
}
#endif
}

// Переносит выбранные узловые записи между полем и файлом по заданным смещениям.
void transfer(const Processes& world, const std::string& file, Access access, unsigned stride,
              const std::vector<OwnedNode>& nodes, const std::vector<std::uint64_t>& offsets,
              const Row& row) {
    if (!stride || offsets.empty() || offsets.size() > 256 * 1024 / stride)
        throw std::invalid_argument("Invalid indexed record size");
    const bool writing = access == Access::Write;
#ifdef AERO_WITH_MPI
    if (world.size > 1) {
        transfer_mpi(world, file, writing, stride, nodes, offsets, row);
        return;
    }
#endif
    world.local([&] {
        const auto mode = writing ? (std::ios::in | std::ios::out) : std::ios::in;
        std::fstream stream(file, std::ios::binary | mode);
        if (!stream)
            throw std::runtime_error("indexed I/O: cannot open " + file);
        std::vector<char> bytes(stride * offsets.size());
        auto next = std::numeric_limits<std::uint64_t>::max();
        for (const auto& node : nodes) {
            if (writing)
                row(node, bytes.data(), stride);
            for (std::size_t k = 0; k < offsets.size(); ++k) {
                if (offsets.size() > 1 || node.id != next) {
                    const auto pos = static_cast<std::streamoff>(offsets[k] + node.id * stride);
                    if (writing)
                        stream.seekp(pos);
                    else
                        stream.seekg(pos);
                }
                auto* data = bytes.data() + k * stride;
                if (writing)
                    stream.write(data, stride);
                else
                    stream.read(data, stride);
                if (!stream)
                    throw std::runtime_error("indexed I/O: incomplete record transfer in " + file);
            }
            if (!writing)
                row(node, bytes.data(), stride);
            next = node.id + 1;
        }
        stream.close();
        if (!stream)
            throw std::runtime_error("indexed I/O: cannot close " + file);
    });
}
}
}
