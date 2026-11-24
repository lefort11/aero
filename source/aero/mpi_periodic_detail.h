#ifndef AERO_MPI_PERIODIC_DETAIL_H
#define AERO_MPI_PERIODIC_DETAIL_H
#include "mpi_periodic.h"
#include "../mesh/mdec.h"
#include <array>
#include <cmath>
#include <cstring>
#include <set>

namespace aero {
namespace periodic_mpi {
using Id = MeshIndex;
using Key = std::pair<Id, Id>; // Для узла пара (n,n), для ребра — упорядоченные исходные номера концов.
// Глобальный ключ сущности и ранг, отвечающий за её данные.
struct Ref {
    // Идентификатор для сопоставления периодических записей.
    Key key;
    // Ранг процесса, откуда получена запись.
    int rank;
};
// Сравнивает ключи сущностей для упорядочения и поиска.
inline bool operator<(const Ref& a, const Ref& b) {
    return a.key < b.key;
}
// Создаёт глобальную ссылку на узел и его владельца.
inline Ref vertex(const MeshPartition& p, Id i) {
    return {{p.nodes[i], p.nodes[i]}, p.node_owners[i]};
}
// Формирует канонический ключ ребра из упорядоченной пары ссылок.
inline Ref edge(Ref a, Ref b) {
    if (b < a)
        std::swap(a, b);
    return {{a.key.first, b.key.first}, a.rank};
}
// Получает три координаты точки в общем представлении Coordinate.
inline Coordinate point(CvVector a) {
    return {{a.x, a.y, a.z}};
}
// Переводит координатный массив в вектор контрольного объёма.
inline CvVector vector(Coordinate a) {
    return {a[0], a[1], a[2]};
}

// Дописывает поля значения в сообщение периодического обмена.
inline void put(std::string& s, Id n) {
    mdec::append_word(s, n);
}
// Дописывает поля значения в сообщение периодического обмена.
inline void put(std::string& s, Key k) {
    put(s, k.first);
    put(s, k.second);
}
// Дописывает поля значения в сообщение периодического обмена.
inline void put(std::string& s, Ref r) {
    put(s, r.key);
    put(s, static_cast<Id>(r.rank));
}
// Записывает двоичное представление вещественного числа без потери точности.
inline void put_real(std::string& s, double x) {
    Id n;
    std::memcpy(&n, &x, sizeof(n));
    put(s, n);
}
// Дописывает поля значения в сообщение периодического обмена.
inline void put(std::string& s, Coordinate v) {
    for (double x : v)
        put_real(s, x);
}
// Последовательно декодирует поля сообщения с текущей байтовой позиции.
struct Reader {
    // Сериализованное сообщение, доступное на время разбора.
    const std::string& data;
    // Байтовая позиция следующего чтения.
    std::size_t offset;
    // Читает восьмибайтовый идентификатор и сдвигает позицию сообщения.
    Id id() {
        const Id n = mdec::read_word(data, offset);
        offset += 8;
        return n;
    }
    // Читает пару идентификаторов ключа сущности.
    Key key() { return {id(), id()}; }
    // Читает ключ сущности и ранг её владельца.
    Ref ref() { return {key(), static_cast<int>(id())}; }
    // Восстанавливает double из сохранённых в сообщении битов.
    double real() {
        const Id n = id();
        double x;
        std::memcpy(&x, &n, sizeof(x));
        return x;
    }
    // Получает три координаты точки в общем представлении Coordinate.
    Coordinate point() { return {{real(), real(), real()}}; }
};
// При подготовке используем 64-битный формат архива с младшим байтом первым.
// Согласуем ошибки между процессами до начала следующего обмена.
using Messages = Processes::Messages;
// Готовит сообщения, выполняет обмен и согласованно обрабатывает ответы.
template <class Send, class Receive> void exchange(const Processes& w, Send send, Receive receive) {
    Messages out;
    w.local([&] {
        out.resize(w.size);
        send(out);
    });
    const auto in = w.exchange(out);
    w.local([&] {
        for (int p = 0; p < w.size; ++p) {
            Reader r = {in[p], 0};
            while (r.offset < r.data.size())
                receive(r, p);
        }
    });
}
using Labels = std::map<Key, Ref>;
// Возвращает периодического представителя или исходную ссылку.
inline Ref representative(const Labels& labels, Ref ref) {
    const auto it = labels.find(ref.key);
    return it == labels.end() ? ref : it->second;
}
// Сопоставляет периодические участки разных рангов и строит общие метки узлов.
Labels match_periodic(const MeshPartition&, CvView, const std::vector<PeriodicPair>&, const Processes&);
}
}
#endif
