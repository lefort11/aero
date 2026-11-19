#ifndef AERO_PARTITION_TRANSFERS_H
#define AERO_PARTITION_TRANSFERS_H
#include "partition_kernels.h"
#include "partition_report.h"
#include "../parallel/processes.h"
#include "../parallel/wall_timer.h"
#ifdef AERO_WITH_CUDA
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
#ifdef AERO_WITH_CUDA
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
    // Обменивает значения консервативных переменных на узлах границы подобласти.
    void exchange_states(const Processes& world, unsigned width, double* state, PartitionGpu* gpu = nullptr,
                         double* seconds = nullptr) {
        transfer(world, width, state, nullptr, false, gpu, seconds);
    }

    // Передаёт поток от владельца ребра соседу, который обновляет другой конец ребра.
    void exchange_fluxes(const Processes& world, unsigned width, EulerFluxResult* flux, PartitionGpu* gpu,
                         double* seconds) {
        transfer(world, width, nullptr, flux, true, gpu, seconds);
    }

private:
    // MPI всегда получает адрес в памяти CPU. Для CUDA закрепляем её и запрашиваем адрес на карте.
    double* allocate_storage(std::size_t total, bool cuda) {
#ifdef AERO_WITH_CUDA
        if (cuda) {
            void* data = nullptr;
            const auto status = cudaMallocHost(&data, total * sizeof(double));
            if (status != cudaSuccess) {
                throw std::runtime_error(std::string("MPI pinned buffer: ") + cudaGetErrorString(status));
            }
            pinned_.reset(static_cast<double*>(data));
            void* alias = nullptr;
            const auto mapping = cudaHostGetDevicePointer(&alias, data, 0);
            if (mapping == cudaSuccess) {
                mapped_ = static_cast<double*>(alias);
            } else if (mapping == cudaErrorInvalidValue || mapping == cudaErrorNotSupported) {
                // Без адреса карты обмен использует обычное копирование через закреплённую память.
                cudaGetLastError();
            } else {
                throw std::runtime_error(cudaGetErrorString(mapping));
            }
            return pinned_.get();
        }
#else
        (void)cuda;
#endif
        host_.resize(total);
        return host_.data();
    }

    // Упаковывает записи, дожидается обмена MPI и раскладывает полученные значения.
    void transfer(const Processes& world, unsigned width, double* state, EulerFluxResult* flux, bool edges,
                  PartitionGpu* gpu = nullptr, double* seconds = nullptr) {
        WallTimer timer(seconds);
        const auto visit = [&](bool pack) {
            for (std::size_t j = 0; j < peers_.size(); ++j) {
                const int p = peers_[j];
                auto& records = pack ? buffers_.send[p] : buffers_.receive[p];
#ifdef AERO_WITH_CUDA
                if (gpu) {
                    double* alias = mapped_ ? mapped_ + (records.data - pinned_.get()) : nullptr;
                    gpu->exchange(device_peers_[j], records, alias, edges, pack);
                    continue;
                }
#endif
                const auto& ids = pack ? exports[p] : imports[p];
                for (std::size_t k = 0; k < ids.size(); ++k)
                    partition_detail::transfer_record(ids[k], width, state, flux, records.data + k * stride_,
                                                      edges, pack);
            }
        };
        // До отправки все процессы должны узнать о возможной ошибке копирования с GPU.
        if (gpu) {
            world.local([&] {
                visit(true);
#ifdef AERO_WITH_CUDA
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
#ifdef AERO_WITH_CUDA
                // У процесса без отправок нет другой точки ожидания перед повторным использованием буфера.
                if (receives_ && (seconds || !sends_))
                    gpu->synchronize();
#endif
            });
        } else {
            visit(false);
        }
    }
    std::vector<double> host_; // Обычная память для обменов CPU.
#ifdef AERO_WITH_CUDA
    struct FreePinned {
        // Дожидается последних операций карты перед возвратом закреплённой памяти.
        void operator()(double* p) const {
            cudaStreamSynchronize(nullptr);
            cudaFreeHost(p);
        }
    };
    std::unique_ptr<double, FreePinned> pinned_;   // Закреплённая память CPU для передач CUDA.
    double* mapped_ = nullptr;                     // Адрес того же буфера на карте, если доступен.
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
