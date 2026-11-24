#pragma once
#include "aggregates.h"
#include "transfer.h"
#include "../implicit/partition.h"
#include "../implicit/workspace.h"
#include <functional>

namespace aero {
namespace implicit { class BdfLinearSystem; }
namespace multigrid {
// Грубая поправка к уже принятому мелкосеточному шагу steady.
// Постоянна только группировка узлов. Правая часть и действие матрицы всегда
// соответствуют текущей схеме и якобиану, использованному на этом шаге.
class CoarseStep {
public:
    // Готовит операторы переноса и рабочие векторы грубосеточной коррекции.
    CoarseStep(const residual_detail::Geometry& host, const residual_detail::Geometry& active,
               MeshIndex rows, int device, const implicit::Partition& partition);

    // Возвращает принятую долю поправки. После отказа поле и потоки восстановлены.
    double correct(implicit::BdfLinearSystem& matrix, double* state, const double* volumes,
                   double dt, double* negative_residual, double* product, double accepted_norm,
                   const std::function<double()>& trial_norm, const std::function<void()>& restore_flux,
                   const double* start = nullptr);
    // Сообщает, использовалась ли история направлений в последней коррекции.
    bool used_history() const { return used_history_; }
private:
    enum { Directions = 4 }; // Небольшая задача GCR для несимметричного оператора грубой сетки.
    const implicit::Partition& partition_; // Обмен на стыке и редукции предоставлены основной СЛАУ.
    Aggregates graph_;       // Построен один раз по связности локальной сетки.
    int device_;             // Номер карты либо -1 для CPU.
    MeshIndex size_;         // Число компонент на грубом уровне этого процесса.
    bool linear_gas_;        // Для пульсаций отрицательная плотность возмущения допустима.
    bool used_history_ = false; // Какую поправку приняли: по изменению поля или по блокам агрегатов.
    Transfer transfer_{};    // Адреса индексов и полей на выбранном исполнителе.
    implicit::Workspace workspace_;
    std::vector<double> host_; // Владелец рабочих векторов при расчёте на CPU.
    double* residual_;       // Непогашенная часть подписанной грубой правой части.
    double* solution_;       // Сумма принятых направлений GCR по агрегатам.
    double* directions_;     // Последовательные векторы поиска в одной линейной области памяти.
    double* images_;         // Соответствующие произведения PᵀJP на векторы поиска.

    // Выполняет ограничение или продление вектора на CPU либо GPU.
    void transfer(Transfer::Operation op, double* coarse);
    // Выполняет выбранное поэлементное обновление рабочих векторов коррекции.
    void update(VectorUpdate::Operation op, double* target, const double* source = nullptr, double factor = 0.);
    // Вычисляет глобальное скалярное произведение для подбора грубой поправки.
    double dot(const double* x, const double* y) const;
    // Решает малую систему для коэффициентов направления коррекции.
    bool solve(implicit::BdfLinearSystem& matrix);
    // Проверяет пробное направление по полной невязке и принимает только пригодную коррекцию.
    double try_direction(Transfer::Operation trial, double fraction, double accepted_norm,
                         const std::function<double()>& trial_norm);
    // Применяет заданную долю найденного направления к пробному полю.
    void set_fraction(Transfer::Operation trial,double fraction);
    // Оценивает долю коррекции с учётом предыдущего успешного направления.
    double history_fraction(implicit::BdfLinearSystem& matrix);
    // Один проход существующего BJ/CSGS по дефекту после грубой поправки.
    void smooth(implicit::BdfLinearSystem& matrix);
};
}
}
