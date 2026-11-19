#ifndef AERO_IO_SETTINGS_H
#define AERO_IO_SETTINGS_H
#include "output_settings.h"

#include <string>
#include <cstdint>
#include <vector>

namespace aero {
struct RunDirectories {
    std::string input, output;  // Каталоги постановки и результатов запуска.
    bool interactive;           // Разрешить паузу для редактирования граничных условий.
    bool geometry_only = false; // Записать геометрию контрольных объёмов.
    bool recover = false;       // Возобновить расчёт с контрольной точки.
    int recovery_slot = -1;     // Без номера выбираем новейший пригодный слот.
    RunDirectories(std::string in = "", std::string out = "", bool edit = false)
        : input(in), output(out), interactive(edit) {}
};

// Названия схем проверяются при чтении настроек независимо от подключённых модулей солвера.
struct RunSettings {
    RunDirectories directories; // Режим запуска и каталоги из командной строки.
    std::string mesh_path, initial_path,
        boundary_path;                    // Разрешённые пути к сетке, начальному полю и границам.
    std::string reconstruction;           // Схема построения следов слева и справа от грани.
    std::string reconstruction_variables; // В каких переменных восстанавливаются следы.
    std::string time_scheme;              // Эйлер или SSPRK3.
    std::string euler_solver;             // Численный поток Roe или LF.
    double cfl;                           // Число Куранта для шага по времени.
    double final_time;                    // Физическое время окончания расчёта.
    double gamma;                         // Отношение теплоёмкостей идеального газа.
    double trace_ratio_min,
        trace_ratio_max;                  // Границы отношения восстановленной плотности и давления к узловым.
    bool use_cuda;                        // Вычислять потоки и стадии на GPU.
    std::vector<int> devices;             // Разрешённые локальные номера GPU.
    int omp_threads;                      // Ноль сохраняет стандартный выбор OpenMP.
    std::vector<double> mpi_rank_weights; // Относительная скорость каждого MPI-процесса.
    std::vector<double> mpi_node_memory_mib, mpi_gpu_memory_mib; // Бюджеты памяти узлов и видеокарт в МиБ.
    bool mpi_balance = false;     // Учитывать заданные параметры баланса при распределении листьев.
    bool mpi_timing = false;      // Собирать время операций по процессам.
    double mpi_memory_factor = 4; // Множитель оценки памяти по данным архива.
    std::uint64_t flowreport_every, recovery_every_steps = 0; // Интервалы отчёта и контрольной точки в шагах.
    std::vector<OutputZone> output_zones; // Какие поля, куда и с какой частотой записывать.
};

RunDirectories parse_run_arguments(int argc, const char* const argv[]);
// Читает main.txt, gas.txt, необязательный output.txt и параметры CUDA из cuda.txt.
// Начальные и граничные условия разбираются после загрузки сетки.
RunSettings read_run_settings(const RunDirectories& directories);
}
#endif
