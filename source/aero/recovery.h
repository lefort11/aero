#ifndef AERO_RECOVERY_H
#define AERO_RECOVERY_H
#include "../io/settings.h"
#include "indexed_io.h"
#include "../io/diagnostic_fields.h"
#include <memory>

namespace aero {
// Строки следуют исходным номерам узлов. Каждый процесс записывает свои узлы вместе с их периодическими копиями.
// При восстановлении значения соседей заполняются обменом.
struct ScalarHistory {
    std::vector<double> current, previous; // Поле SA и предыдущий слой BDF2 в расчётной нумерации.
};
class Recovery {
public:
    Recovery(const RunSettings& run, const Processes& world, std::uint64_t count, unsigned dimension,
             std::string mesh_sha256, const std::vector<OwnedNode>& nodes);
    ~Recovery();
    // Сначала пробует указанный или новейший слот. При успешной проверке устанавливает время и номер шага.
    void restore(std::vector<double>& state, std::vector<double>& initial);
    // Записывает данные владельцев в другой слот и завершает запись проверенным заголовком.
    void save(std::uint64_t step, double time, const std::vector<double>& state,
              const std::vector<double>& initial);
    // Только промежуточные точки подчиняются расписанию. Итоговую пишет интегратор.
    bool due(std::uint64_t step) const;
    const std::string& fields_directory() const { return fields_; }
    std::uint64_t step = 0;
    double time = 0;
    std::string stabilization_state; // CFL, счётчик успешных шагов и глобальные номера отмеченных узлов.
    bool has_samples = false; // Признак формата одинаков и на процессе с пустой подобластью.
    std::vector<diagnostic::Moments> samples; // Средние и последние значения для продолжения с тем же dt.
    std::unique_ptr<ScalarHistory> scalar; // Выделяется только для расчёта с уравнением SA.
    implicit::BdfHistory history; // Предыдущий слой и фактический шаг, если они есть в файле BDF2.

private:
    struct Lock;
    // Блокировка каталога исключает одновременную запись контрольных точек разными запусками.
    std::unique_ptr<Lock> lock_;
    const RunSettings& run_;
    const Processes& world_;
    // Число исходных узлов. Каждому соответствует одна запись независимо от разбиения.
    std::uint64_t count_;
    // Число консервативных компонент в текущем и начальном состояниях узла.
    unsigned width_;
    // Хеш сетки привязывает файл к геометрии, fields_ задаёт каталог полей текущего продолжения.
    std::string hash_, fields_;
    const std::vector<OwnedNode>& nodes_;
    // Номер последнего принятого слота. Следующая запись идёт в противоположный.
    int last_ = -1;
    std::string path(int slot) const;
};
}
#endif
