#include "meshdec.h"
#include <climits>
#include <iostream>
#ifdef AERO_MESHDEC_MPI
#include <mpi.h>
#endif

namespace {
using namespace aero::meshdec;
// Управляет MPI-процессами утилиты разбиения и согласованием ошибок.
class Workers {
public:
    // Инициализирует MPI-процессы утилиты разбиения сетки.
    Workers(int& argc, char**& argv) {
#ifdef AERO_MESHDEC_MPI
        if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
            throw std::runtime_error("MPI initialization failed");
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
        (void)argc;
        (void)argv;
#endif
    }
    // Завершает созданную объектом среду MPI.
    ~Workers() {
#ifdef AERO_MESHDEC_MPI
        MPI_Finalize();
#endif
    }
    // Распространяет данные или статус работы между процессами утилиты.
    void share(std::string& text, int root = 0) const {
#ifdef AERO_MESHDEC_MPI
        int bytes = rank == root ? static_cast<int>(text.size()) : 0;
        MPI_Bcast(&bytes, 1, MPI_INT, root, MPI_COMM_WORLD);
        if (rank != root)
            text.resize(static_cast<unsigned>(bytes));
        if (bytes)
            MPI_Bcast(&text[0], bytes, MPI_CHAR, root, MPI_COMM_WORLD);
#else
        (void)text;
        (void)root;
#endif
    }
    // Выполняет этап с согласованием ошибок между MPI-процессами.
    template <class Work> void phase(const Work& work) const {
        std::string message;
        try {
            work();
        } catch (const std::exception& e) {
            message = e.what();
        }
        int failed = message.empty() ? size : rank;
#ifdef AERO_MESHDEC_MPI
        int first = size;
        MPI_Allreduce(&failed, &first, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        failed = first;
#endif
        if (failed < size) {
            share(message, failed);
            throw std::runtime_error("rank " + std::to_string(failed) + ": " + message);
        }
    }
    int rank = 0, size = 1;
};
// Разбирает положительное целое значение параметра командной строки.
Id positive(const std::string& text) {
    Id result = 0;
    for (char c : text) {
        require(c >= '0' && c <= '9', "Expected positive integer: " + text);
        result = plus(times(result, 10), static_cast<unsigned>(c - '0'));
    }
    require(result != 0, "Expected a positive integer");
    return result;
}
// Показывает настройки будущего разбиения и временной памяти декомпозера.
void help() {
    std::cout << "meshdec INPUT.mesh OUTPUT.mdec [--target-ranks=N] [--leaf-bytes=N] [--memory-mib=N] "
                 "[--temp-dir=DIR] [--compress]\n"
              << "--target-ranks hints at the future solver size (about four leaf groups per rank).\n"
              << "Defaults: 4 MiB per leaf, 64 MiB sort buffer per process, scratch beside output.\n"
              << "The output must be new. MPI processes require a shared scratch and output directory.\n";
}
// Разбирает параметры памяти, разбиения и путей утилиты meshdec.
Options parse(int argc, char** argv) {
    Options o;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--compress")
            o.compress = true;
        else if (arg.compare(0, 15, "--target-ranks=") == 0)
            o.target_ranks = positive(arg.substr(15));
        else if (arg.compare(0, 13, "--leaf-bytes=") == 0)
            o.leaf_bytes = positive(arg.substr(13));
        else if (arg.compare(0, 13, "--memory-mib=") == 0)
            o.memory = times(positive(arg.substr(13)), 1024 * 1024);
        else if (arg.compare(0, 11, "--temp-dir=") == 0) {
            o.temp = arg.substr(11);
            require(!o.temp.empty(), "Empty scratch directory");
        } else if (arg.compare(0, 2, "--") == 0)
            throw std::invalid_argument("Unknown option: " + arg);
        else
            files.push_back(arg);
    }
    require(files.size() == 2, "Expected input mesh and output archive (see --help)");
    o.source = files[0];
    o.target = files[1];
    if (o.temp.empty())
        o.temp = directory(o.target);
    return o;
}
// Выполняет этапы чтения, упорядочения, разбиения и упаковки сетки.
int run(const Workers& w, int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        if (!w.rank)
            help();
        return 0;
    }
    const Options o = parse(argc, argv);
    Scratch scratch, destination;
    std::string dir, staged;
    w.phase([&] {
        if (w.rank)
            return;
        std::ifstream existing(o.target.c_str(), std::ios::binary);
        require(!existing.is_open(), "Output already exists: " + o.target);
        scratch.create(o.temp);
        destination.create(directory(o.target));
        dir = scratch.path;
        staged = destination.path + "/archive";
        unpack(o, dir);
        index_topology(dir, o.memory);
    });
    // Все процессы должны открывать один каталог временных таблиц и один подготовленный выходной файл.
    w.share(dir);
    w.share(staged);
    // Границы фаз синхронизируют ошибки до того, как следующая операция начнёт читать промежуточные файлы.
    w.phase([&] { order_entities(dir, o.memory, w.rank, w.size); });
    w.phase([&] {
        if (!w.rank && w.size > 1)
            choose_splitters(dir, o.memory, w.size);
    });
    w.phase([&] { sort_range(dir, o.memory, w.rank, w.size); });
    w.phase([&] {
        if (!w.rank)
            plan_leaves(dir, o, w.size);
    });
    w.phase([&] { pack_leaves(dir, o.compress, w.rank, w.size); });
    Id leaves = 0;
    w.phase([&] {
        if (!w.rank)
            leaves = write_catalog(dir, staged, o.compress, w.size);
    });
    w.phase([&] { write_part(dir, staged, w.rank); });
    w.phase([&] {
        if (!w.rank)
            publish(staged, o.target);
    });
    if (!w.rank) {
        std::cout << o.target << "  leaves = " << leaves << "  preprocessing processes = " << w.size;
        if (o.target_ranks)
            std::cout << "  future solver ranks (hint) = " << o.target_ranks;
        std::cout << "\n";
    }
    return 0;
}
}
// Запускает построение архива разбиения и согласует ошибки рабочих процессов.
int main(int argc, char** argv) {
    try {
        Workers workers(argc, argv);
        try {
            return run(workers, argc, argv);
        } catch (const std::exception& e) {
            if (!workers.rank)
                std::cerr << "meshdec: " << e.what() << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "meshdec: " << e.what() << "\n";
        return 1;
    }
}
