#pragma once
#include "krylov_scratch.h"
#include "../timestep/node_step.h"
#include "../parallel/kernel_math.h"
#include <cmath>

namespace aero {
namespace implicit {
// Полином по принятым слоям служит только начальным приближением нелинейного решения.
struct TimePrediction {
    const MeshIndex* owned; // Соответствие компактных строк полному локальному полю MPI.
    unsigned width;         // Число компонент газа на узел.
    unsigned count;         // Число предыдущих слоёв, участвующих в полиноме.
    double* state;          // Текущий слой U_n, на место которого записывается проба.
    const double* fields[KrylovFieldCount]; // Предыдущие слои от нового к старому.
    double weights[KrylovFieldCount];       // Веса Лагранжа в момент нового слоя.
    const double* rate; // Начальный интегральный баланс узла при экстраполяции правой части в steady RANS.
    double* save_rate;  // Место для начального баланса текущего принятого слоя.
    const double* volumes; // Объёмы узлов для перевода баланса в скорость изменения.
    NodeSteps steps;       // Зафиксированные на текущей попытке узловые псевдошаги.
    double dt;             // Доля локального шага для формулы неявного Эйлера.
    double* intervals = nullptr; // Последние два узловых dt в свободном векторе BiCGSTAB.

    // Связывает компактные строки поля с представлением узловых псевдошагов.
    TimePrediction(const MeshIndex* ids, unsigned components, double* values, const double* volume = nullptr,
                   NodeSteps local = {}, double fraction = 0.)
        : owned(ids), width(components), count(0), state(values), fields{}, weights{}, rate(nullptr),
          save_rate(nullptr), volumes(volume), steps(local), dt(fraction) {}

    // Суммирует разности относительно U_n, сохраняя постоянное поле точно.
    AERO_HOST_DEVICE void operator()(MeshIndex row) const {
        const auto node=owned ? owned[row] : row;
        double* values = state + node*width;
        const double step = rate ? steps.at(node,dt) : 0.;
        const double multiplier = rate ? step/volumes[node] : 1.;
        double local_weights[2]{};
        const double* coefficients = weights;
        unsigned used = count;
        if (intervals) {
            // Меняющийся диапазон LTS даёт разную историю шага даже у соседних
            // узлов. Интерполируем балансы в локальном псевдовремени каждого узла.
            double* past = intervals + row*width;
            const double previous = count ? past[0] : 0.;
            if (count) {
                const double ratio = step/previous;
                local_weights[0] = -ratio;
                used = ratio <= 2. ? 1 : 0;
                if (count>1 && used && past[1]>0.) {
                    const double second = ratio*(step+previous)/(previous+past[1]) * (previous/past[1]);
                    if (second <= 2.) {
                        local_weights[0] = -ratio*(step+previous+past[1])/past[1];
                        local_weights[1] = second;
                        used=2;
                    }
                }
            }
            past[1]=previous;
            past[0]=step;
            coefficients=local_weights;
        }
        for (unsigned k=0; k<width; ++k) {
            const auto i=row*width+k;
            const double origin = rate ? rate[i] : values[k];
            double delta = 0.;
            for (unsigned j=0; j<used; ++j) delta += coefficients[j]*(fields[j][i]-origin);
            if (rate) {
                values[k] += multiplier * (origin + delta);
                if (save_rate) save_rate[i]=origin;
            } else values[k] = origin + delta;
        }
    }
};

// Кольцо прошлых слоёв занимает уже выделенные собственные векторы BiCGSTAB.
// Память, совмещённая с потоками или вязким оператором, сюда не передаётся.
class PredictionHistory {
public:
    // При создании интегратора связывает историю с доступной рабочей памятью.
    explicit PredictionHistory(KrylovScratch space) : space_(space), available_(space.count) {}
    // До записи первого слоя оставляет одинаковое число буферов на всех рангах.
    void limit_capacity(unsigned available) { available_=space_.count=std::min(available_,available); }
    // В режиме LTS отводим свободный вектор под интервалы узлов. На тесной
    // раскладке единственный вектор остаётся для истории самого баланса.
    double* local_intervals(bool local) {
        const bool reserve = local && available_>=2;
        space_.count = available_ - unsigned(reserve);
        return reserve ? space_.field[space_.count] : nullptr;
    }
    // После использования векторов линейным солвером все прежние слои утрачены.
    void clear() { count_=0; next_=0; }
    // Выбирает непрерывную последовательность последних слоёв для нового шага dt.
    void prepare(double dt, unsigned order, TimePrediction& trial) const {
        trial.count=0;
        double times[KrylovFieldCount]{};
        double past=0.;
        const unsigned limit=std::min(order,count_);
        for (unsigned j=0; j<limit; ++j) {
            const unsigned slot=(next_+space_.count-1-j)%space_.count;
            past -= intervals_[slot]/dt;
            if (!std::isfinite(past) || (j ? past>=times[j-1] : past>=0.)) break;
            times[j]=past;
            trial.fields[j]=space_.field[slot];
            // Пересчитываем веса для кандидата степени j+1. Координаты U_n и цели равны 0 и 1.
            double weights[KrylovFieldCount]{};
            double amplification=0.;
            for (unsigned i=0; i<=j; ++i) {
                double w=1./times[i];
                for (unsigned k=0; k<=j; ++k)
                    if (k!=i) w *= (1.-times[k])/(times[i]-times[k]);
                weights[i]=w;
                amplification+=std::abs(w);
            }
            // При большом скачке dt оставляем устойчивую степень с предыдущей попытки.
            if (!(amplification <= 64.)) break;
            std::copy_n(weights,j+1,trial.weights);
            trial.count=j+1;
        }
    }
    // Запоминает длину принятого шага и отдаёт место для копии поля его начала.
    double* append(double dt) {
        if (!space_.count) return nullptr;
        const unsigned slot=next_;
        intervals_[slot]=dt;
        next_=(slot+1)%space_.count;
        count_=std::min(count_+1,space_.count);
        return space_.field[slot];
    }
private:
    KrylovScratch space_; // Адреса независимых компактных полей на CPU или GPU.
    unsigned available_; // Общее согласованное число свободных векторов.
    double intervals_[KrylovFieldCount]{}; // Расстояние по времени до следующего принятого слоя.
    unsigned count_=0; // Число пригодных последовательных слоёв.
    unsigned next_=0;  // Позиция следующей записи в кольце.
};
}
}
