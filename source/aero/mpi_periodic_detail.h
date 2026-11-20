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
struct Ref {
    Key key;
    int rank;
};
inline bool operator<(const Ref& a, const Ref& b) {
    return a.key < b.key;
}
inline Ref vertex(const MeshPartition& p, Id i) {
    return {{p.nodes[i], p.nodes[i]}, p.node_owners[i]};
}
inline Ref edge(Ref a, Ref b) {
    if (b < a)
        std::swap(a, b);
    return {{a.key.first, b.key.first}, a.rank};
}
inline Coordinate point(CvVector a) {
    return {{a.x, a.y, a.z}};
}
inline CvVector vector(Coordinate a) {
    return {a[0], a[1], a[2]};
}

inline void put(std::string& s, Id n) {
    mdec::append_word(s, n);
}
inline void put(std::string& s, Key k) {
    put(s, k.first);
    put(s, k.second);
}
inline void put(std::string& s, Ref r) {
    put(s, r.key);
    put(s, static_cast<Id>(r.rank));
}
inline void put_real(std::string& s, double x) {
    Id n;
    std::memcpy(&n, &x, sizeof(n));
    put(s, n);
}
inline void put(std::string& s, Coordinate v) {
    for (double x : v)
        put_real(s, x);
}
struct Reader {
    const std::string& data;
    std::size_t offset;
    Id id() {
        const Id n = mdec::read_word(data, offset);
        offset += 8;
        return n;
    }
    Key key() { return {id(), id()}; }
    Ref ref() { return {key(), static_cast<int>(id())}; }
    double real() {
        const Id n = id();
        double x;
        std::memcpy(&x, &n, sizeof(x));
        return x;
    }
    Coordinate point() { return {{real(), real(), real()}}; }
};
// При подготовке используем 64-битный формат архива с младшим байтом первым.
// Согласуем ошибки между процессами до начала следующего обмена.
using Messages = Processes::Messages;
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
inline Ref representative(const Labels& labels, Ref ref) {
    const auto it = labels.find(ref.key);
    return it == labels.end() ? ref : it->second;
}
Labels match_periodic(const MeshPartition&, CvView, const std::vector<PeriodicPair>&, const Processes&);
}
}
#endif
