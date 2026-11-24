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
    // Считает постоянные поля и только ту часть временных направлений, которой не хватило места BiCGSTAB.
    static MeshIndex storage_bytes(const Aggregates& graph,MeshIndex rows,unsigned width,const implicit::KrylovScratch& scratch,
                                   bool product_persistent=false);
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
    std::shared_ptr<const Aggregates> graph_; // Тот же граф используется при оценке памяти и при расчёте.
    int device_;             // Номер карты либо -1 для CPU.
    MeshIndex size_;         // Число компонент на грубом уровне этого процесса.
    bool linear_gas_;        // Для пульсаций отрицательная плотность возмущения допустима.
    bool used_history_ = false; // Какую поправку приняли: по изменению поля или по блокам агрегатов.
    Transfer transfer_{};    // Адреса индексов и полей на выбранном исполнителе.
    // Владеет отдельными рабочими выделениями грубого решателя.
    implicit::Workspace workspace_;
    std::vector<double> host_; // Владелец рабочих векторов при расчёте на CPU.
    double* residual_;       // Непогашенная часть подписанной грубой правой части.
    double* solution_;       // Сумма принятых направлений GCR по агрегатам.
    double* borrowed_product_ = nullptr; // Свободный хвост Krylov для временного J*v; correction хранит сглаживатель.
    double* directions_[Directions]{}; // При поиске направления можно занять свободные векторы BiCGSTAB.
    double* images_[Directions]{};     // Их содержимое больше не нужно при проверке пробного поля.

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
