#ifndef AERO_IO_SCHEDULED_OUTPUT_H
#define AERO_IO_SCHEDULED_OUTPUT_H
#include "diagnostic_fields.h"
#include "output_settings.h"
#include "flow_values.h"
#include <functional>

namespace aero {
// Публикует готовый снимок одной операцией. При ошибке сохраняется предыдущий выходной файл.
class StagedFieldFile {
public:
    // Создаёт временный путь для последующей публикации законченного снимка.
    explicit StagedFieldFile(const std::string& path);
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~StagedFieldFile();
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    StagedFieldFile(const StagedFieldFile&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    StagedFieldFile& operator=(const StagedFieldFile&) = delete;
    // Возвращает путь временного файла незавершённого снимка.
    const std::string& temporary() const { return temporary_; }
    // Переименовывает законченный временный файл в итоговый путь.
    void publish(bool replace);

private:
    std::string path_, temporary_;
};

// Выбирает физическую или диагностическую величину для столбца вывода.
Field output_field(const std::string& name);

// Все зоны используют общую последовательность принятых шагов и моментов записи.
class OutputSchedule {
public:
    // Создаёт расписание зон и выбирает их столбцы по размерности задачи.
    OutputSchedule(const std::vector<OutputZone>& zones, unsigned dimension);
    // Переносит расписание за восстановленные шаг и время, исключая повтор старых снимков.
    void resume(std::uint64_t step, double time);
    // Ограничивает шаг ближайшим сроком записи активной зоны.
    double limit_dt(double time, double candidate) const;
    // Проверяет, наступил ли шаг или физическое время записи зоны.
    bool due(std::uint64_t step, double time) const;
    // Проверяет, понадобится ли столбец хотя бы одной зоне именно в этом снимке.
    bool column_due(unsigned column, std::uint64_t step, double time, bool final_field) const;
    using Writer = std::function<void(const OutputZone&, const std::vector<unsigned>& columns)>;
    // Проверяет наличие активных зон вывода.
    bool has_zones() const { return !zones_.empty(); }
    // Записывает зоны, срок которых наступил на принятом шаге.
    void write_due(std::uint64_t step, double time, const Writer& writer, bool final_field = false);

private:
    // Расписание и выбранные столбцы одной зоны с отметкой последней записи.
    struct Zone {
        OutputZone config;
        std::vector<unsigned> columns;
        std::uint64_t next_tick = 1, last_step = 0;
        // Вычисляет ближайшее физическое время записи зоны по целочисленному номеру срока.
        double deadline() const;
        // Проверяет, наступил ли шаг или физическое время записи зоны.
        bool due(std::uint64_t step, double time) const;
    };
    std::vector<Zone> zones_;
};

// Записывает выбранные поля локальной сетки в файлы зон.
class LocalFieldOutput {
public:
    // Инициализирует объект стандартными значениями его полей.
    LocalFieldOutput() = default;
    // Подготавливает локальную запись полей с учётом нумерации узлов.
    LocalFieldOutput(const std::string& directory, const Mesh& mesh, double gamma,
                     std::vector<std::uint64_t> node_dof, std::size_t state_nodes, const LinearBackground& background = {});
    // Формирует выбранные поля и записывает снимок зоны в нужном формате.
    void write(const OutputZone& zone, const std::vector<unsigned>& columns, std::uint64_t step, double time,
               const std::vector<double>& state,const diagnostic::Fields& scalars={}) const;

private:
    std::string directory_;
    FieldSnapshot metadata_;
    std::vector<std::uint64_t> node_dof_;
    std::size_t nodes_ = 0;
    double gamma_ = 0;
    LinearBackground background_;
};

// Связывает расписание зон с записью снимков поля.
class ScheduledOutput : public OutputSchedule {
public:
    // Связывает расписание зон с геометрией и полями для записи снимков.
    ScheduledOutput(const std::vector<OutputZone>& zones, const std::string& directory, const Mesh& mesh,
                    double gamma, std::vector<std::uint64_t> node_dof = {}, std::size_t state_nodes = 0);
    // Формирует выбранные поля и записывает снимок зоны в нужном формате.
    void write(std::uint64_t step, double time, const std::vector<double>& state);

private:
    LocalFieldOutput local_;
};
}
#endif
