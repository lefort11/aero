#include "recovery.h"
#include "startup.h"
#include "../io/number_codec.h"
#include "../io/lzma_stream.h"
#include "../io/sha256.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sys/stat.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace aero {
namespace {
using Id = std::uint64_t;
const unsigned header_size = 120;
using Packet = std::array<char, header_size>;
void require(bool ok, const std::string& message) {
    if (!ok)
        throw std::runtime_error("recovery: " + message);
}
using indexed_io::put_real;
using number_codec::read_real;
using number_codec::read_word;
using number_codec::store_word;
// Версия 2 добавляет контрольную сумму и значение предыдущего dt после заголовка.
unsigned records_start(bool history) { return header_size + (history ? 16u : 0u); }
unsigned record_size(unsigned width, bool history) {
    return 24 + (history ? 24 : 16) * width;
}
std::streamoff offset(Id id, unsigned width, bool history) {
    const auto stride = record_size(width,history);
    require(id <= (static_cast<Id>(std::numeric_limits<std::streamoff>::max()) - records_start(history)) / stride,
            "file too large");
    return static_cast<std::streamoff>(records_start(history) + id * stride);
}
void seal(char* bytes, unsigned size) {
    store_word(bytes + size - 8, io::crc32(bytes, size - 8));
}
bool intact(const char* bytes, unsigned size) {
    return read_word(bytes + size - 8) == io::crc32(bytes, size - 8);
}
// Строка узла содержит два номера, текущее и начальное состояния, затем контрольную сумму.
// Оба состояния записываем в порядке [ρ, ρu, E], чтобы после перезапуска восстановить и данные для границ.
constexpr unsigned header_dimension = 8, header_nodes = 16, header_gamma = 24, header_step = 32,
                   header_time = 40, header_hash = 48;
constexpr unsigned id_bytes = 8, state_start = 2 * id_bytes;

// Смещение компоненты в первом или втором векторе. Поля файла располагаются подряд без выравнивания.
std::size_t component_offset(unsigned component) {
    return state_start + sizeof(double) * component;
}

void decode_node(const OwnedNode& n, const char* row, unsigned width, std::vector<double>& state,
                 std::vector<double>& initial, implicit::BdfHistory& history) {
    if (!intact(row, record_size(width,history.dt>0.)) || read_word(row) != n.id)
        throw std::runtime_error("recovery: damaged record at node " + std::to_string(n.id));
    if (read_word(row + id_bytes) != n.representative)
        throw std::runtime_error("recovery: incompatible periodic mapping at node " + std::to_string(n.id));
    // Копии на шве проходят проверку, а значение степени свободы задаёт её представитель.
    if (n.id != n.representative)
        return;
    const auto begin = n.slot * width;
    for (unsigned k = 0; k < width; ++k) {
        state[begin + k] = read_real(row + component_offset(k));
        initial[begin + k] = read_real(row + component_offset(width + k));
        if (history.dt>0.) history.previous[begin+k] = read_real(row+component_offset(2*width+k));
    }
}
void encode_node(const OwnedNode& n, char* row, unsigned width, const double* state, const double* initial, const double* previous) {
    store_word(row, n.id);
    store_word(row + id_bytes, n.representative);
    for (unsigned k = 0; k < width; ++k) {
        put_real(row + component_offset(k), state[k]);
        put_real(row + component_offset(width + k), initial[k]);
        if (previous) put_real(row + component_offset(2*width+k),previous[k]);
    }
    seal(row, record_size(width,previous!=nullptr));
}

Packet header(Id count, unsigned width, double gamma, const std::string& hash, Id step, double time, bool history) {
    Packet bytes{};
    std::memcpy(bytes.data(), history ? "AERORCV2" : "AERORCV1", 8);
    store_word(&bytes[header_dimension], width - 2);
    store_word(&bytes[header_nodes], count);
    put_real(&bytes[header_gamma], gamma);
    store_word(&bytes[header_step], step);
    put_real(&bytes[header_time], time);
    std::memcpy(&bytes[header_hash], hash.data(), 64);
    seal(bytes.data(), header_size);
    return bytes;
}
Packet read_header(std::ifstream& in, Id count, unsigned width, double gamma, const std::string& hash, double& saved_dt) {
    Packet bytes{};
    in.read(bytes.data(), header_size);
    const bool history = !std::memcmp(bytes.data(),"AERORCV2",8);
    require(in && (history || !std::memcmp(bytes.data(), "AERORCV1", 8)) && intact(bytes.data(), header_size),
            "incomplete or damaged header");
    require(read_word(&bytes[header_dimension]) == width - 2 && read_word(&bytes[header_nodes]) == count &&
                read_real(&bytes[header_gamma]) == gamma &&
                !std::memcmp(&bytes[header_hash], hash.data(), 64),
            "incompatible mesh or gas");
    require(read_word(&bytes[header_step]) && std::isfinite(read_real(&bytes[header_time])) &&
                read_real(&bytes[header_time]) > 0,
            "invalid step or time");
    saved_dt = 0.;
    if (history) {
        char extra[16];
        in.read(extra,sizeof(extra));
        require(in && intact(extra,sizeof(extra)),"damaged BDF2 time step");
        saved_dt=read_real(extra);
        require(saved_dt>0. && std::isfinite(saved_dt),"invalid BDF2 time step");
    }
    in.seekg(0, std::ios::end);
    require(in.tellg() == offset(count, width, history), "incomplete or damaged file length");
    return bytes;
}
bool exists(const std::string& path) {
    struct stat info;
    if (!stat(path.c_str(), &info))
        return true;
    require(errno == ENOENT, "cannot inspect " + path);
    return false;
}
void sync_file(const std::string& path) {
#ifdef _WIN32
    HANDLE f = CreateFileA(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           OPEN_EXISTING, 0, nullptr);
    require(f != INVALID_HANDLE_VALUE, "cannot open for flush: " + path);
    const bool ok = FlushFileBuffers(f) != 0;
    CloseHandle(f);
#else
    const int f = open(path.c_str(), O_RDWR);
    require(f >= 0, "cannot open for flush: " + path);
    const bool ok = fsync(f) == 0;
    close(f);
#endif
    require(ok, "cannot flush " + path);
}
// Старый архив не содержит хеша исходной сетки. Для его контрольных точек используем хеш байтов самого архива.
// Только нулевой процесс читает его целиком, один раз при запуске.
std::string archive_hash(const std::string& path, const Processes& world) {
    int digits[64] = {};
    world.local([&] {
        if (world.rank)
            return;
        std::ifstream in(path, std::ios::binary);
        require(in.is_open(), "cannot open " + path);
        io_detail::Sha256Buffer hash;
        char block[65536];
        while (in.read(block, sizeof block) || in.gcount())
            hash.sputn(block, in.gcount());
        require(in.eof() && !in.bad(), "cannot read " + path);
        const auto digest = hash.hex_digest();
        std::copy(digest.begin(), digest.end(), digits);
    });
    world.broadcast(digits, 64, 0);
    std::string hash(64, 0);
    for (unsigned i = 0; i < hash.size(); ++i)
        hash[i] = static_cast<char>(digits[i]);
    return hash;
}
}
// ОС освобождает блокировку записи при завершении владельца, в том числе аварийном.
struct Recovery::Lock {
#ifdef _WIN32
    HANDLE handle;
    explicit Lock(const std::string& path) {
        handle =
            CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        require(handle != INVALID_HANDLE_VALUE, "cannot lock " + path);
    }
    ~Lock() { CloseHandle(handle); }
#else
    int fd;
    explicit Lock(const std::string& path) {
        fd = open(path.c_str(), O_RDWR | O_CREAT, 0666);
        require(fd >= 0, "cannot open " + path);
        if (flock(fd, LOCK_EX | LOCK_NB)) {
            close(fd);
            throw std::runtime_error("recovery: cannot lock " + path);
        }
    }
    ~Lock() { close(fd); }
#endif
};
Recovery::~Recovery() = default;
Recovery::Recovery(const RunSettings& run, const Processes& world, Id count, unsigned dimension,
                   std::string mesh_sha256, const std::vector<OwnedNode>& nodes)
    : run_(run), world_(world), count_(count), width_(dimension + 2), hash_(std::move(mesh_sha256)),
      fields_(run.directories.output), nodes_(nodes) {
    world_.local([&] {
        if (world_.rank)
            return;
        if (world_.size == 1 && !run_.directories.recover)
            for (const auto& zone : run_.output_zones) {
                const auto key = output_path_key(output_filename(zone, 1));
                const auto first = key.substr(0, key.find('/'));
                require(first != "recovery-0.bin" && first != "recovery-1.bin" && first != "recovery.lock",
                        "field output overlaps recovery data: " + zone.file);
            }
        create_output_directory(run_.directories.output);
        lock_.reset(new Lock(run_.directories.output + "/recovery.lock"));
        if (!run_.directories.recover)
            require(!exists(path(0)) && !exists(path(1)),
                    "output already contains recovery data; use --recover or a new output directory");
    });
    if (hash_.empty())
        hash_ = archive_hash(run.mesh_path, world);
    require(hash_.size() == 64, "invalid mesh fingerprint");
}
std::string Recovery::path(int slot) const {
    return run_.directories.output + "/recovery-" + std::to_string(slot) + ".bin";
}
bool Recovery::due(Id accepted, double t, bool final_field) const {
    return accepted > 0 && run_.recovery_every_steps && (final_field || accepted % run_.recovery_every_steps == 0 || (!run_.steady() && t == run_.final_time));
}
void Recovery::restore(std::vector<double>& state, std::vector<double>& initial) {
    const int requested = run_.directories.recovery_slot;
    Id steps[2] = {};
    double times[2] = {}, saved_dt[2] = {};
    world_.local([&] {
        if (world_.rank)
            return;
        for (int s = 0; s < 2; ++s) {
            if (requested >= 0 && s != requested)
                continue;
            if (!exists(path(s)))
                continue;
            try {
                std::ifstream in(path(s), std::ios::binary);
                const auto h = read_header(in, count_, width_, run_.gamma, hash_, saved_dt[s]);
                steps[s] = read_word(&h[header_step]);
                times[s] = read_real(&h[header_time]);
            } catch (const std::exception& e) {
                std::cerr << path(s) << ": " << e.what() << '\n';
            }
        }
    });
    world_.broadcast(steps, 2, 0);
    world_.broadcast(saved_dt, 2, 0);
    const int first = steps[1] > steps[0] ? 1 : 0;
    for (int s : {first, 1 - first}) {
        if (!steps[s])
            continue;
        try {
            history.dt=saved_dt[s];
            world_.local([&] {
                history.previous.resize(history.dt>0. ? state.size() : 0);
            });
            const unsigned stride = record_size(width_,history.dt>0.);
            const auto decode = [&](const OwnedNode& n, char* row, std::size_t) {
                decode_node(n, row, width_, state, initial, history);
            };
            indexed_io::transfer(world_, path(s), indexed_io::Access::Read, stride, nodes_, {records_start(history.dt>0.)},
                                 decode);
            // Перед выдачей восстановленного состояния проверяем, что заголовок всё ещё указывает на выбранный завершённый шаг.
            world_.local([&] {
                if (world_.rank)
                    return;
                std::ifstream in(path(s), std::ios::binary);
                const auto h = read_header(in, count_, width_, run_.gamma, hash_, saved_dt[s]);
                require(saved_dt[s]==history.dt && read_word(&h[header_step]) == steps[s] && read_real(&h[header_time]) == times[s],
                        "checkpoint changed during read");
            });
        } catch (const std::exception& e) {
            if (!world_.rank)
                std::cerr << path(s) << ": " << e.what() << '\n';
            continue;
        }
        world_.broadcast(&times[s], 1, 0);
        time = times[s];
        last_ = s;
        step = steps[s];
        require(run_.steady() || time <= run_.final_time, "final_time precedes the saved time");
        if (!world_.rank)
            std::cout << "Recovered step " << step << " at time " << time << " from " << path(s) << '\n';
        if (!run_.output_zones.empty()) {
            const std::string prefix = run_.directories.output + "/resumed-" + std::to_string(step) + "-";
            Id number = 0;
            world_.local([&] {
                if (!world_.rank)
                    number = create_resume_directory(prefix);
            });
            world_.broadcast(&number, 1, 0);
            fields_ = prefix + std::to_string(number);
        }
        return;
    }
    throw std::runtime_error("recovery: no valid checkpoint in " +
                             (requested >= 0 ? path(requested) : run_.directories.output));
}
void Recovery::save(Id accepted, double t, const std::vector<double>& state,
                    const std::vector<double>& initial) {
    const bool has_history = history.dt>0.;
    const int slot = last_ == 0 ? 1 : 0;
    const std::string file = path(slot);
    // У незавершённой записи нет сигнатуры. Сначала фиксируем это на диске, затем записываем новые строки.
    world_.local([&] {
        if (world_.rank)
            return;
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        Packet blank{};
        out.write(blank.data(), header_size);
        out.seekp(offset(count_, width_, has_history) - 1);
        out.put(0);
        out.close();
        require(bool(out), "cannot prepare " + file);
        sync_file(file);
    });
    const unsigned stride = record_size(width_,has_history);
    const auto encode = [&](const OwnedNode& n, char* row, std::size_t) {
        const auto begin = n.slot * width_;
        encode_node(n, row, width_, state.data() + begin, initial.data() + begin,
                    has_history ? history.previous.data()+begin : nullptr);
    };
    indexed_io::transfer(world_, file, indexed_io::Access::Write, stride, nodes_, {records_start(has_history)}, encode);
    // MPI_File_sync уже сбросил параллельную запись. Перед фиксацией заголовка сбрасываем также обычный файловый поток.
    world_.local([&] {
        if (world_.rank)
            return;
        if (world_.size == 1)
            sync_file(file);
        std::fstream out(file, std::ios::in | std::ios::out | std::ios::binary);
        const auto bytes = header(count_, width_, run_.gamma, hash_, accepted, t, has_history);
        out.write(bytes.data(), header_size);
        if (has_history) {
            char extra[16];
            put_real(extra,history.dt);
            seal(extra,sizeof(extra));
            out.write(extra,sizeof(extra));
        }
        out.close();
        require(bool(out), "cannot commit " + file);
        sync_file(file);
    });
    last_ = slot;
}
}
