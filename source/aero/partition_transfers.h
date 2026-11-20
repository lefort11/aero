#ifndef AERO_PARTITION_TRANSFERS_H
#define AERO_PARTITION_TRANSFERS_H
#include "partition_kernels.h"
#include "partition_report.h"
#include "../parallel/processes.h"
#include "../parallel/wall_timer.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "partition_gpu.h"
#endif

namespace aero {
class PartitionGpu;
// Буферы и порядок записей для многократного обмена с одними и теми же соседями.
class PartitionTransfers {
public:
    std::vector<std::vector<MeshIndex>> exports; // Локальные индексы для отправки каждому соседу.
    std::vector<std::vector<MeshIndex>> imports; // Локальные индексы для принятых от соседа записей.
    // Одна запись занимает stride чисел double.
    PartitionTransfers(int size, unsigned stride)
        : exports(size), imports(size), buffers_(size), stride_(stride) {}
    // Выделяет одну непрерывную область памяти и нарезает её на буферы соседей.
    void allocate(bool cuda) {
        std::size_t total = 0;
        const auto size_buffer = [&](DoubleExchange::Buffer& b, std::size_t items) {
            if (items > (host_.max_size() - total) / stride_) {
                throw std::length_error("MPI buffer too large");
            }
            b.count = items * stride_;
            total += b.count;
        };
        for (std::size_t p = 0; p < exports.size(); ++p) {
            if (exports[p].empty() && imports[p].empty()) {
                continue;
            }
            peers_.push_back(static_cast<int>(p));
            size_buffer(buffers_.send[p], exports[p].size());
            size_buffer(buffers_.receive[p], imports[p].size());
            sends_ = sends_ || !exports[p].empty();
            receives_ = receives_ || !imports[p].empty();
        }
        if (!total) {
            return;
        }
        double* next = allocate_storage(total, cuda);
        for (int p : peers_) {
            buffers_.send[p].data = next;
            next += buffers_.send[p].count;
            buffers_.receive[p].data = next;
            next += buffers_.receive[p].count;
        }
    }
#ifdef AERO_WITH_KOKKOS_GPU
    // Готовит индексы упаковки и распаковки на карте для каждого активного соседа.
    void prepare_gpu(PartitionGpu& gpu) {
        device_peers_.clear();
        for (int p : peers_) {
            device_peers_.push_back(gpu.prepare(exports[p], imports[p], stride_));
        }
    }
#endif
    // Считает объём данных для одного вызова обмена.
    ExchangeBytes bytes() const {
        ExchangeBytes bytes{};
        for (int p : peers_) {
            bytes.sent += buffers_.send[p].count * sizeof(double);
            bytes.received += buffers_.receive[p].count * sizeof(double);
        }
        return bytes;
    }
    // У одного списка соседей могут быть разные типы записей: поток и два блока
    // его производных. Нумерацию согласуем один раз, размеры буферов задаём раздельно.
    PartitionTransfers(const PartitionTransfers& layout, unsigned stride)
        : PartitionTransfers(static_cast<int>(layout.exports.size()), stride) {
        exports = layout.exports;
        imports = layout.imports;
    }

private:
    // MPI всегда получает адрес в памяти CPU. Для CUDA закрепляем её и запрашиваем адрес на карте.
    double* allocate_storage(std::size_t total, bool cuda) {
#ifdef AERO_WITH_KOKKOS_GPU
        if (cuda) {
            const auto status = pinned_.allocate(total);
            if (status != aero::gpu::Status::Success)
                throw std::runtime_error(std::string("MPI pinned buffer: ") + aero::gpu::message(status));
            return pinned_.host();
        }
#else
        (void)cuda;
#endif
        host_.resize(total);
        return host_.data();
    }

public:
    // Упаковывает записи, дожидается обмена MPI и раскладывает полученные значения.
    void transfer(const Processes& world, partition_detail::RecordField field,
                  PartitionGpu* gpu = nullptr, double* seconds = nullptr) {
        WallTimer timer(seconds);
        const auto visit = [&](bool pack) {
            for (std::size_t j = 0; j < peers_.size(); ++j) {
                const int p = peers_[j];
                auto& records = pack ? buffers_.send[p] : buffers_.receive[p];
#ifdef AERO_WITH_KOKKOS_GPU
                if (gpu) {
                    double* alias = pinned_.device() ? pinned_.device() + (records.data - pinned_.host()) : nullptr;
                    gpu->exchange(device_peers_[j], records, alias, field, pack);
                    continue;
                }
#endif
                const auto& ids = pack ? exports[p] : imports[p];
                for (std::size_t k = 0; k < ids.size(); ++k)
                    field.copy_record(ids[k], k, pack ? nullptr : records.data, pack ? records.data : nullptr);
            }
        };
        // До отправки все процессы должны узнать о возможной ошибке копирования с GPU.
        if (gpu) {
            world.local([&] {
                visit(true);
#ifdef AERO_WITH_KOKKOS_GPU
                // MPI может читать буфер только после завершения копирования на CPU.
                if (sends_)
                    gpu->synchronize();
#endif
            });
        } else {
            visit(true);
        }
        buffers_.transfer(peers_);
        if (gpu) {
            world.local([&] {
                visit(false);
#ifdef AERO_WITH_KOKKOS_GPU
                // У процесса без отправок нет другой точки ожидания перед повторным использованием буфера.
                if (receives_ && (seconds || !sends_))
                    gpu->synchronize();
#endif
            });
        } else {
            visit(false);
        }
    }
private:
    std::vector<double> host_; // Обычная память для обменов CPU.
#ifdef AERO_WITH_KOKKOS_GPU
    PinnedBuffer<double> pinned_; // Закреплённое хранилище записей всех соседей.
    std::vector<PartitionGpu::Peer> device_peers_; // Индексы соседей в памяти GPU.
#endif
    DoubleExchange buffers_; // Срезы буферов передачи и приёма.
    std::vector<int> peers_; // Только соседи с непустым обменом.
    bool sends_ = false;     // Есть хотя бы одна отправка с этого процесса.
    bool receives_ = false;  // Есть хотя бы один приём этим процессом.
    unsigned stride_;        // Число double на один узел или поток.
};
}
#endif
