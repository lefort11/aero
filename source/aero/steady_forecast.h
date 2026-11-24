#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace aero {
// Оценивает число принятых псевдошагов до допуска. При остановке снижения
// невязки или недостаточном интервале наблюдения возвращает отрицательное число.
class SteadyResidualTrend {
    // Двенадцать последних средних log(R). На первом уровне блок равен шагу,
    // на каждом следующем — паре блоков предыдущего. Длинный расчёт не требует
    // хранения всех шагов, а короткая история остаётся доступной при смене темпа.
    struct Window {
        // Кольцевое окно наблюдений для оценки тренда.
        std::array<double,12> values{};
        // Позиция следующей записи и число заполненных позиций.
        unsigned next = 0, count = 0;
        // Ожидается вторая величина пары.
        bool half_pair = false;
        // Сохранённая первая величина незавершённой пары.
        double first = 0.;

        // Добавляет значение в кольцевое окно и передаёт среднее пары следующему масштабу.
        bool append(double& value) {
            values[next] = value;
            next = (next+1)%12;
            if (count < 12) ++count;
            half_pair = !half_pair;
            if (half_pair) {
                first = value;
                return false;
            }
            value = .5*(first+value);
            return true;
        }

        // В полном окне next указывает на самую старую запись. Медиана трёх
        // соседних блоков подавляет одиночный выброс в оценке наклона.
        double median_at(unsigned start) const {
            const double a = values[(next+start)%12], b = values[(next+start+1)%12];
            const double c = values[(next+start+2)%12];
            return std::max(std::min(a,b),std::min(std::max(a,b),c));
        }

        // Оценивает шаги до заданного снижения логарифма невязки по оконному тренду.
        double steps_to(double distance, double recent, std::uint64_t width, std::uint64_t remainder) const {
            const double a = median_at(0), b = median_at(3), c = median_at(6), d = median_at(9);
            const double span = 3.*static_cast<double>(width); // Расстояние между центрами соседних троек.
            double slow = std::min(a-b,std::min(b-c,c-d))/span;
            double fast = std::max(a-b,std::max(b-c,c-d))/span;
            if (width > 1) {
                // Последний полный крупный блок может отставать от текущего шага.
                // Сверяем его темп с последними тремя принятыми невязками.
                const double tail = 1.5*(static_cast<double>(width)-1.)+static_cast<double>(remainder);
                const double latest = (d-recent)/tail;
                slow = std::min(slow,latest);
                fast = std::max(fast,latest);
            }
            // Три участка должны снижаться с сопоставимым темпом. Разрешаем
            // экстраполяцию максимум на два уже наблюдавшихся логарифмических спада.
            const double drop = a-d;
            if (!(slow > 0.) || fast > 2.5*slow || drop < .01 || distance > 2.*drop) return -1.;
            return distance*(3.*span)/drop;
        }
    };

    std::array<Window,32> windows_{}; // Блоки от одного шага до 2^31 шагов.
    // Число поступивших оценок невязки.
    std::uint64_t samples_ = 0;

  public:
    // Добавляет логарифм положительной невязки в окна прогноза; недопустимые данные сбрасывают историю.
    void add(double residual) {
        if (!(residual > 0.) || !std::isfinite(residual)) {
            *this = SteadyResidualTrend{};
            return;
        }
        ++samples_;
        double value = std::log(residual);
        for (auto& window : windows_)
            if (!window.append(value)) break;
    }

    // Прогнозирует оставшиеся шаги; -1 означает недостаток пригодных данных.
    double remaining_steps(double tolerance) const {
        const auto& short_window = windows_.front();
        if (short_window.count < 12 || !(tolerance > 0.)) return -1.;
        const double recent = short_window.median_at(9);
        if (recent >= short_window.median_at(0)) return -1.;
        const double distance = recent-std::log(tolerance);
        if (!(distance > 0.)) return -1.;
        // Выбираем самое свежее окно, которое уже видело достаточно большое
        // снижение. Длинное окно также обязано подтвердить темп на последних шагах.
        std::uint64_t width = 1;
        for (const auto& window : windows_) {
            if (window.count < 12) break;
            const double steps = window.steps_to(distance,recent,width,samples_%width);
            if (steps > 0.) return std::max(1., steps-1.); // Центр последних трёх отсчётов на шаг позади.
            width *= 2;
        }
        return -1.;
    }
};
}
