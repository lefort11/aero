#ifndef AERO_PROCESSES_H
#define AERO_PROCESSES_H
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef AERO_WITH_MPI
#include <mpi.h>
#endif

namespace aero {
// Внутри процесса к MPI обращается один поток. OpenMP используется в локальных вычислениях.
class Processes {
public:
    Processes(int& argc, char**& argv) {
#ifdef AERO_WITH_MPI
        int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
            throw std::runtime_error("Cannot initialize MPI");
        if (provided < MPI_THREAD_FUNNELED) {
            MPI_Finalize();
            throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
        (void)argc;
        (void)argv;
#endif
    }
    ~Processes() {
#ifdef AERO_WITH_MPI
        MPI_Finalize();
#endif
    }
    Processes(const Processes&) = delete;
    Processes& operator=(const Processes&) = delete;

    // До следующего обмена все процессы заканчивают локальную операцию и согласуют ошибку, если она возникла.
    // Процесс с пустым разделом участвует в каждой фазе.
    template <class Work> void local(const Work& work) const {
        if (size == 1) {
            work();
            return;
        }
        std::string error;
        bool failed = false;
        try {
            work();
        } catch (const std::exception& e) {
            failed = true;
            error = e.what();
        }
#ifdef AERO_WITH_MPI
        // Выбираем наименьший номер процесса с ошибкой. Он передаст одну и ту же причину всем рангам.
        int mine = failed ? rank : size, first = size;
        MPI_Allreduce(&mine, &first, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (first == size)
            return;
        int length = static_cast<int>(std::min<std::size_t>(error.size(), INT_MAX));
        MPI_Bcast(&length, 1, MPI_INT, first, MPI_COMM_WORLD);
        error.resize(static_cast<std::size_t>(length));
        if (length)
            MPI_Bcast(&error[0], length, MPI_CHAR, first, MPI_COMM_WORLD);
        throw std::runtime_error("rank " + std::to_string(first) + ": " + error);
#else
        if (failed)
            throw std::runtime_error(error);
#endif
    }
    using Messages = std::vector<std::string>;
    // Пересылает по одному сообщению каждому процессу. В ответе индекс означает номер отправителя.
    Messages exchange(const Messages& outgoing) const;
    // Собирает короткие диагностические сообщения в порядке номеров процессов на процессе 0.
    Messages gather_text(std::string message) const;
    // Согласует одно наименьшее значение на всех процессах, например допустимый шаг CFL.
    double minimum(double value) const;
    // Глобальная норма либо скалярное произведение распределённых векторов BiCGSTAB.
    double reduce_double(double value, bool maximum) const;
    std::uint64_t first_id(std::uint64_t value) const;
    std::uint64_t sum(std::uint64_t value) const;
    void broadcast(std::string& value, int root) const;
    void broadcast(double* values, int count, int root) const;
    void broadcast(std::uint64_t* values, int count, int root) const;
    void broadcast(int* values, int count, int root) const;
    void gather(const std::uint64_t* values, int count, std::uint64_t* collected, int root) const;
    void allgather(const double* values, int count, double* collected) const;
    // Номер данного процесса и число процессов в MPI_COMM_WORLD; при обычном запуске это 0 и 1.
    int rank = 0, size = 1;
};

// При подготовке обмена состояниями и потоками фиксируем размеры и порядок соседей.
// На шагах по времени пересылаются только значения.
class DoubleExchange {
public:
    struct Buffer {
        double* data = nullptr;
        std::size_t count = 0;
    };
    std::vector<Buffer> send, receive;
    explicit DoubleExchange(int ranks) : send(ranks), receive(ranks) {
#ifdef AERO_WITH_MPI
        requests_.reserve(2 * static_cast<std::size_t>(ranks));
#endif
    }
    // Обменивает заранее связанные числовые буферы с указанными соседями.
    void transfer(const std::vector<int>& peers);

private:
#ifdef AERO_WITH_MPI
    std::vector<MPI_Request> requests_;
#endif
};
}
#endif
