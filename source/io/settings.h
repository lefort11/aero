#ifndef AERO_IO_SETTINGS_H
#define AERO_IO_SETTINGS_H
#include "output_settings.h"
#include "../timestep/stabilization.h"
#include "../implicit/bdf1_step.h"
#include "../eulersolver/linear_euler.h"

#include <map>
#include <string>
#include <cstdint>
#include <vector>

namespace aero {
struct RunDirectories {
    std::string input, output;  // Каталоги постановки и результатов запуска.
    using Parameters = std::map<std::string, std::string>;
    std::map<std::string, Parameters> overrides; // Значения по имени файла и ключу для этого запуска.
    bool interactive;           // Разрешить паузу для редактирования граничных условий.
    bool geometry_only = false; // Записать геометрию контрольных объёмов.
    bool recover = false;       // Возобновить расчёт с контрольной точки.
    int recovery_slot = -1;     // Без номера выбираем новейший пригодный слот.
    RunDirectories(std::string in = "", std::string out = "", bool edit = false)
        : input(in), output(out), interactive(edit) {}
};

// Названия схем проверяются при чтении настроек независимо от подключённых модулей солвера.
struct RunSettings {
    mutable std::vector<std::string> input_warnings; // Диагностика пополняется при чтении файлов, зависящих от размерности сетки.
    LinearBackground background;
    unsigned background_velocities = 0; // Число заданных компонент, проверяем по размерности сетки.
    RunDirectories directories; // Режим запуска и каталоги из командной строки.
    std::string mesh_path, initial_path,
        boundary_path;                    // Разрешённые пути к сетке, начальному полю и границам.
    std::string reconstruction;           // Схема построения следов слева и справа от грани.
    std::string reconstruction_variables; // В каких переменных восстанавливаются следы.
    std::string time_scheme;              // Эйлер, SSPRK3, BDF1, BDF2 или выход на стационар.
    implicit::NewtonSettings implicit_settings; // Допуски и предел итераций из implicit.txt для неявных схем.
    bool steady_absolute = false;         // Сравнивать норму напрямую или относительно начального поля.
    double steady_tol = 1.e-6;            // Порог в единицах выбранного критерия.
    unsigned steady_max_steps = 10000;    // Предел числа псевдошагов с начала этого запуска.
    bool steady() const { return time_scheme == "steady"; }
    bool uses_implicit_time() const { return steady() || time_scheme == "bdf1" || time_scheme == "bdf2"; }
    std::string euler_solver;             // Численный поток Roe или LF.
    bool mpweno_ad_corr; // Антидиссипативная поправка к весам MPWENO5 из scheme.txt.
    double mp6_hybrid; // Вес связанной центральной коррекции двух следов.
    double upw_coeff; // Доля диссипации LF/Roe и доля EBR при смешивании с центральной реконструкцией.
    Stabilization stabilization; // Управление откатом шага и локальным порядком.
    double cfl;                           // Число Куранта для шага по времени.
    std::uint64_t control_every = 16; // Интервал проверки control.txt в принятых шагах. Ноль отключает управление.
    double t_fo = 0.; // До этого времени (псевдовремени для steady) весь расчёт идёт с первым порядком по пространству.
    bool first_order_start(double time) const { return time < t_fo && reconstruction != "first_order"; }
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
    bool exact_solution = false; // Сравнивать решение с формулами exact.txt.
    bool mpi_timing = false;      // Собирать время операций по процессам.
    double mpi_memory_factor = 4; // Множитель оценки памяти по данным архива.
    std::uint64_t report_every, recovery_every_steps = 0; // Интервалы отчёта и контрольной точки в шагах.
    std::vector<OutputZone> output_zones; // Какие поля, куда и с какой частотой записывать.
};

RunDirectories parse_run_arguments(int argc, const char* const argv[]);
// Читает main.txt, gas.txt, необязательный output.txt и параметры CUDA из cuda.txt.
// Начальные и граничные условия разбираются после загрузки сетки.
std::vector<std::string> read_exact_formulas(const RunDirectories&, const std::vector<std::string>& names, std::vector<std::string>* warnings = nullptr);
RunSettings read_run_settings(const RunDirectories& directories);
}
#endif
