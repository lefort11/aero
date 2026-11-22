#include "recovery.h"
#include "../io/sha256.h"
#include <iomanip>
#include <sstream>
#include <locale>
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
unsigned record_size(unsigned width, bool history, bool scalar, bool samples) {
    return 24 + (history ? 24 : 16) * width + (scalar ? (history ? 16 : 8) : 0) + (samples ? 8*diagnostic::Moments::words : 0);
}
std::streamoff offset(Id id, unsigned width, bool history, bool scalar, bool samples) {
    const auto stride = record_size(width,history,scalar,samples);
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

void decode_node(const OwnedNode& n, const char* row, unsigned width, unsigned stride, bool saved_history,
                 std::vector<double>& state, std::vector<double>* initial, implicit::BdfHistory& history,
                 ScalarHistory* scalar, diagnostic::Moments* samples) {
    // CRC всегда охватывает всю запись, даже если для нового запуска берём только последний слой.
    if (!intact(row, stride) || read_word(row) != n.id)
        throw std::runtime_error("recovery: damaged record at node " + std::to_string(n.id));
    if (read_word(row + id_bytes) != n.representative)
        throw std::runtime_error("recovery: incompatible periodic mapping at node " + std::to_string(n.id));
    // Копии на шве проходят проверку, а значение степени свободы задаёт её представитель.
    if (n.id != n.representative)
        return;
    if(scalar) {
        const unsigned first=(saved_history ? 3u : 2u)*width;
        scalar->current[n.slot]=read_real(row+component_offset(first));
        if(history.dt>0.) scalar->previous[n.slot]=read_real(row+component_offset(first+1));
    }
    if (samples) {
        const unsigned first=(history.dt>0.?3u:2u)*width+(scalar?(history.dt>0.?2u:1u):0u);
        for (unsigned j=0;j<diagnostic::Moments::words;++j)
            samples[n.slot].set(j,read_real(row+component_offset(first+j)));
    }
    const auto begin = n.slot * width;
    for (unsigned k = 0; k < width; ++k) {
        state[begin + k] = read_real(row + component_offset(k));
        if (initial) (*initial)[begin + k] = read_real(row + component_offset(width + k));
        if (history.dt>0.) history.previous[begin+k] = read_real(row+component_offset(2*width+k));
    }
}
void encode_node(const OwnedNode& n, char* row, unsigned width, const double* state, const double* initial, const double* previous, const ScalarHistory* scalar, const diagnostic::Moments* samples) {
    store_word(row, n.id);
    store_word(row + id_bytes, n.representative);
    for (unsigned k = 0; k < width; ++k) {
        put_real(row + component_offset(k), state[k]);
        put_real(row + component_offset(width + k), initial[k]);
        if (previous) put_real(row + component_offset(2*width+k),previous[k]);
    }
    if(scalar) {
        const unsigned first=(previous?3u:2u)*width;
        put_real(row+component_offset(first),scalar->current[n.slot]);
        if(previous) put_real(row+component_offset(first+1),scalar->previous[n.slot]);
    }
    if (samples) {
        const unsigned first=(previous?3u:2u)*width+(scalar?(previous?2u:1u):0u);
        for (unsigned j=0;j<diagnostic::Moments::words;++j)
            put_real(row+component_offset(first+j),samples[n.slot].get(j));
    }
    seal(row, record_size(width,previous!=nullptr,scalar!=nullptr,samples!=nullptr));
}

Packet header(Id count, unsigned width, double gamma, const std::string& hash, Id step, double time, bool history, bool adaptive, bool scalar, bool samples) {
    Packet bytes{};
    std::memcpy(bytes.data(),scalar ? "AEROSAV" : "AERORCV",7);
    bytes[7]=static_cast<char>('1'+(history?1:0)+(adaptive?2:0)+(samples?4:0));
    store_word(&bytes[header_dimension], width - 2);
    store_word(&bytes[header_nodes], count);
    put_real(&bytes[header_gamma], gamma);
    store_word(&bytes[header_step], step);
    put_real(&bytes[header_time], time);
    std::memcpy(&bytes[header_hash], hash.data(), 64);
    seal(bytes.data(), header_size);
    return bytes;
}
Packet read_header(std::ifstream& in, Id count, unsigned width, double gamma, const std::string& hash, double& saved_dt, std::string& stabilization, bool scalar) {
    Packet bytes{};
    in.read(bytes.data(), header_size);
    require(in && intact(bytes.data(),header_size) && bytes[7]>='1' && bytes[7]<='8',
            "incomplete or damaged header");
    require(!std::memcmp(bytes.data(),scalar ? "AEROSAV" : "AERORCV",7),
            "incompatible recovery equation set");
    const unsigned flags=static_cast<unsigned>(bytes[7]-'1');
    const bool history=(flags&1)!=0, adaptive=(flags&2)!=0, samples=(flags&4)!=0;
    require(read_word(&bytes[header_dimension]) == width - 2 && read_word(&bytes[header_nodes]) == count &&
                read_real(&bytes[header_gamma]) == gamma &&
                !std::memcmp(&bytes[header_hash], hash.data(), 64),
            "incompatible mesh or gas (including the linear background)");
    // Стационарное начальное поле или final_time=0 дают корректную точку (шаг 0, t=0).
    const double time = read_real(&bytes[header_time]);
    require(std::isfinite(time) && (read_word(&bytes[header_step]) ? time > 0. : time == 0.),
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
    const auto records_end = offset(count, width, history, scalar, samples);
    const std::streamoff tail = in.tellg() - records_end;
    stabilization.clear();
    if (adaptive) {
        // Секция защищена собственной CRC. Первые версии контрольных точек
        // заканчивались сразу за последним узлом и читаются как прежде.
        require(tail > 8 && tail <= INT_MAX, "invalid stabilization section size");
        in.seekg(records_end);
        stabilization.resize(static_cast<std::size_t>(tail - 8));
        in.read(&stabilization[0], tail - 8);
        char crc[8];
        in.read(crc, 8);
        require(in && read_word(crc) == io::crc32(stabilization.data(), stabilization.size()),
                "damaged stabilization section");
    } else require(tail == 0, "incomplete or damaged file length");
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
#ifdef AERO_HAS_TURBULENCE
    if(run.turbulence.uses_sa()) scalar.reset(new ScalarHistory);
#endif
    if (hash_.empty())
        hash_ = archive_hash(run.mesh_path, world);
    require(hash_.size() == 64, "invalid mesh fingerprint");
    if (run.background.active()) {
        // Контрольная точка хранит пульсации и привязана к их фону. Обычные
        // контрольные точки Эйлера сохраняют исходный идентификатор сетки.
        io_detail::Sha256Buffer buffer;
        std::ostream key(&buffer);
        key.imbue(std::locale::classic());
        key << hash_ << " Linear " << std::setprecision(17) << run.background.rho << ' ' << run.background.p;
        for (double v : run.background.velocity) key << ' ' << v;
        hash_ = buffer.hex_digest();
    }
}
std::string Recovery::path(int slot) const {
    return run_.directories.output + "/recovery-" + std::to_string(slot) + ".bin";
}
bool Recovery::due(Id accepted) const {
    return accepted > 0 && run_.recovery_every_steps && accepted % run_.recovery_every_steps == 0;
}
void Recovery::restore(std::vector<double>& state, std::vector<double>& initial, const std::string& initial_file) {
    const bool new_run = !initial_file.empty();
    const int requested = new_run ? 0 : run_.directories.recovery_slot;
    // Для начального поля читаем ровно указанный файл. Для продолжения сохраняем выбор
    // новейшего исправного слота и возможность отката к предыдущей контрольной точке.
    const std::string files[] = {new_run ? initial_file : path(0), path(1)};
    std::string adaptive[2];
    Id steps[2] = {};
    int valid[2] = {}; // У целого заголовка номер шага может быть нулевым.
    double times[2] = {}, saved_dt[2] = {};
    world_.local([&] {
        if (world_.rank)
            return;
        for (int s = 0; s < 2; ++s) {
            if (requested >= 0 && s != requested)
                continue;
            if (!exists(files[s]))
                continue;
            try {
                std::ifstream in(files[s], std::ios::binary);
                const auto h = read_header(in, count_, width_, run_.gamma, hash_, saved_dt[s], adaptive[s], scalar!=nullptr);
                steps[s] = read_word(&h[header_step]);
                times[s] = read_real(&h[header_time]);
                valid[s] = h[7]>='5'?2:1;
            } catch (const std::exception& e) {
                std::cerr << files[s] << ": " << e.what() << '\n';
            }
        }
    });
    world_.broadcast(steps, 2, 0);
    world_.broadcast(valid, 2, 0);
    world_.broadcast(saved_dt, 2, 0);
    const int first = steps[1] > steps[0] ? 1 : 0;
    for (int s : {first, 1 - first}) {
        if (!valid[s])
            continue;
        try {
            if (!world_.rank)
                std::cout << "Reading recovery file " << files[s] << "..." << std::endl;
            const bool saved_history = saved_dt[s] > 0.;
            const bool saved_samples = valid[s] == 2;
            // В новом расчёте газ записываем прямо в начальное поле. Старые средние,
            // исходное поле RANS и предыдущий слой BDF пропускаем без выделения памяти.
            auto& gas = new_run ? initial : state;
            has_samples = !new_run && saved_samples;
            history.dt = new_run ? 0. : saved_dt[s];
            world_.local([&] {
                samples.resize(has_samples ? gas.size()/width_ : 0);
                history.previous.resize(history.dt > 0. ? gas.size() : 0);
                if(scalar) {
                    scalar->current.resize(gas.size()/width_);
                    scalar->previous.resize(history.previous.size()/width_);
                }
            });
            const unsigned stride = record_size(width_,saved_history,scalar!=nullptr,saved_samples);
            const auto decode = [&](const OwnedNode& n, char* row, std::size_t) {
                decode_node(n, row, width_, stride, saved_history, gas, new_run ? nullptr : &initial,
                            history, scalar.get(), has_samples ? samples.data() : nullptr);
            };
            indexed_io::transfer(world_, files[s], indexed_io::Access::Read, stride, nodes_, {records_start(saved_history)},
                                 decode);
            // Перед выдачей восстановленного состояния проверяем, что заголовок всё ещё указывает на выбранный завершённый шаг.
            world_.local([&] {
                if (world_.rank)
                    return;
                std::ifstream in(files[s], std::ios::binary);
                double confirmed_dt = 0.;
                const auto h = read_header(in, count_, width_, run_.gamma, hash_, confirmed_dt, adaptive[s], scalar!=nullptr);
                require(confirmed_dt == saved_dt[s] && (h[7] >= '5') == saved_samples &&
                        read_word(&h[header_step]) == steps[s] && read_real(&h[header_time]) == times[s],
                        "checkpoint changed during read");
            });
        } catch (const std::exception& e) {
            if (!world_.rank)
                std::cerr << files[s] << ": " << e.what() << '\n';
            continue;
        }
        if (new_run) {
            if (!world_.rank)
                std::cout << "Initialized from checkpoint step " << steps[s] << " at time " << times[s]
                          << ". Starting a new run at step 0, time 0." << std::endl;
            return;
        }
        stabilization_state.swap(adaptive[s]);
        world_.broadcast(stabilization_state, 0);
        // Новые снимки хранят начальную норму в проверяемом контрольной суммой разделе.
        if (stabilization_state.compare(0, 6, "steady") == 0) {
            const auto end = stabilization_state.find('\n');
            std::istringstream in(stabilization_state.substr(6, end - 6));
            in.imbue(std::locale::classic());
            require(end != std::string::npos && (in >> steady_reference) && std::isfinite(steady_reference)
                    && steady_reference >= 0., "invalid initial steady residual");
            stabilization_state.erase(0, end + 1);
        }
        world_.broadcast(&times[s], 1, 0);
        time = times[s];
        last_ = s;
        step = steps[s];
        require(run_.steady() || time <= run_.final_time, "final_time precedes the saved time");
        if (!world_.rank)
            std::cout << "Recovered step " << step << " at time " << time << " from " << files[s] << std::endl;
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
                             (requested >= 0 ? files[requested] : run_.directories.output));
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
        out.seekp(offset(count_, width_, has_history, scalar!=nullptr,has_samples) - 1);
        out.put(0);
        out.close();
        require(bool(out), "cannot prepare " + file);
        sync_file(file);
    });
    const unsigned stride = record_size(width_,has_history,scalar!=nullptr,has_samples);
    const auto encode = [&](const OwnedNode& n, char* row, std::size_t) {
        const auto begin = n.slot * width_;
        encode_node(n, row, width_, state.data() + begin, initial.data() + begin,
                    has_history ? history.previous.data()+begin : nullptr, scalar.get(),has_samples?samples.data():nullptr);
    };
    indexed_io::transfer(world_, file, indexed_io::Access::Write, stride, nodes_, {records_start(has_history)}, encode);
    // MPI_File_sync уже сбросил параллельную запись. Перед фиксацией заголовка сбрасываем также обычный файловый поток.
    world_.local([&] {
        if (world_.rank)
            return;
        if (world_.size == 1)
            sync_file(file);
        std::fstream out(file, std::ios::in | std::ios::out | std::ios::binary);
        std::string metadata = stabilization_state;
        if (steady_reference >= 0.) {
            std::ostringstream line;
            line.imbue(std::locale::classic());
            line << "steady " << std::setprecision(17) << steady_reference << '\n';
            metadata.insert(0, line.str());
        }
        const auto bytes = header(count_, width_, run_.gamma, hash_, accepted, t, has_history, !metadata.empty(), scalar!=nullptr,has_samples);
        out.write(bytes.data(), header_size);
        if (has_history) {
            char extra[16];
            put_real(extra,history.dt);
            seal(extra,sizeof(extra));
            out.write(extra,sizeof(extra));
        }
        if (!metadata.empty()) {
            require(metadata.size() < INT_MAX - 8u, "stabilization section too large");
            out.seekp(offset(count_, width_, has_history, scalar!=nullptr,has_samples));
            out.write(metadata.data(), static_cast<std::streamsize>(metadata.size()));
            char crc[8];
            store_word(crc, io::crc32(metadata.data(), metadata.size()));
            out.write(crc, 8);
        }
        out.close();
        require(bool(out), "cannot commit " + file);
        sync_file(file);
    });
    last_ = slot;
}
}
