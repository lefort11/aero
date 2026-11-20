#include "processes.h"
#include <limits>

namespace aero {
double Processes::minimum(double value) const {
    if (size == 1)
        return value;
#ifdef AERO_WITH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
    return value;
}
double Processes::reduce_double(double value, bool maximum) const {
#ifdef AERO_WITH_MPI
    if (size > 1) MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, maximum ? MPI_MAX : MPI_SUM, MPI_COMM_WORLD);
#else
    (void)maximum;
#endif
    return value;
}
std::uint64_t Processes::first_id(std::uint64_t value) const {
#ifdef AERO_WITH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
#endif
    return value;
}
std::uint64_t Processes::sum(std::uint64_t value) const {
#ifdef AERO_WITH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
    return value;
}
void Processes::broadcast(std::string& value, int root) const {
#ifdef AERO_WITH_MPI
    std::uint64_t length = value.size();
    broadcast(&length, 1, root);
    local([&] {
        if (length > INT_MAX)
            throw std::length_error("MPI string too long");
        value.resize(static_cast<std::size_t>(length));
    });
    if (length)
        MPI_Bcast(&value[0], static_cast<int>(length), MPI_CHAR, root, MPI_COMM_WORLD);
#else
    (void)value;
    (void)root;
#endif
}
void Processes::broadcast(double* values, int count, int root) const {
#ifdef AERO_WITH_MPI
    MPI_Bcast(values, count, MPI_DOUBLE, root, MPI_COMM_WORLD);
#else
    (void)values;
    (void)count;
    (void)root;
#endif
}
void Processes::broadcast(std::uint64_t* values, int count, int root) const {
#ifdef AERO_WITH_MPI
    MPI_Bcast(values, count, MPI_UINT64_T, root, MPI_COMM_WORLD);
#else
    (void)values;
    (void)count;
    (void)root;
#endif
}
void Processes::broadcast(int* values, int count, int root) const {
#ifdef AERO_WITH_MPI
    MPI_Bcast(values, count, MPI_INT, root, MPI_COMM_WORLD);
#else
    (void)values;
    (void)count;
    (void)root;
#endif
}
void Processes::gather(const std::uint64_t* values, int count, std::uint64_t* collected, int root) const {
#ifdef AERO_WITH_MPI
    MPI_Gather(values, count, MPI_UINT64_T, collected, count, MPI_UINT64_T, root, MPI_COMM_WORLD);
#else
    (void)root;
    std::copy_n(values, count, collected);
#endif
}
void Processes::allgather(const double* values, int count, double* collected) const {
#ifdef AERO_WITH_MPI
    MPI_Allgather(values, count, MPI_DOUBLE, collected, count, MPI_DOUBLE, MPI_COMM_WORLD);
#else
    std::copy_n(values, count, collected);
#endif
}
Processes::Messages Processes::gather_text(std::string message) const {
#ifdef AERO_WITH_MPI
    int width = 0;
    local([&] {
        if (message.size() >= INT_MAX)
            throw std::length_error("MPI report too large");
        width = static_cast<int>(message.size() + 1);
    });
    MPI_Allreduce(MPI_IN_PLACE, &width, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    std::vector<char> data;
    Messages reports;
    local([&] {
        message.resize(width, '\0');
        if (!rank) {
            if (static_cast<std::size_t>(size) > data.max_size() / width)
                throw std::length_error("MPI reports too large");
            data.resize(static_cast<std::size_t>(size) * width);
        }
    });
    MPI_Gather(&message[0], width, MPI_CHAR, data.data(), width, MPI_CHAR, 0, MPI_COMM_WORLD);
    local([&] {
        if (!rank)
            for (int p = 0; p < size; ++p)
                reports.emplace_back(data.data() + static_cast<std::size_t>(p) * width);
    });
    return reports;
#else
    return {message};
#endif
}
// Размеры и адреса уже известны обеим сторонам, поэтому значения можно пересылать сразу.
void DoubleExchange::transfer(const std::vector<int>& peers) {
#ifdef AERO_WITH_MPI
    for (std::size_t offset = 0;; offset += INT_MAX) {
        requests_.clear();
        for (int p : peers) {
            MPI_Request req;
            if (offset < receive[p].count) {
                const int n = static_cast<int>(std::min<std::size_t>(INT_MAX, receive[p].count - offset));
                MPI_Irecv(receive[p].data + offset, n, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &req);
                requests_.push_back(req);
            }
            if (offset < send[p].count) {
                const int n = static_cast<int>(std::min<std::size_t>(INT_MAX, send[p].count - offset));
                MPI_Isend(send[p].data + offset, n, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &req);
                requests_.push_back(req);
            }
        }
        if (requests_.empty())
            break;
        MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE);
    }
#else
    for (int p : peers) {
        if (receive[p].count != send[p].count)
            throw std::logic_error("Mismatched local exchange buffers");
        std::copy_n(send[p].data, send[p].count, receive[p].data);
    }
#endif
}

Processes::Messages Processes::exchange(const Messages& outgoing) const {
    if (size == 1)
        return outgoing;
#ifndef AERO_WITH_MPI
    return outgoing;
#else
    using Count = unsigned long long;
    const auto peers = static_cast<std::size_t>(size);
    std::vector<Count> sent, received;
    Messages incoming;
    std::vector<MPI_Request> requests;
    local([&] {
        if (outgoing.size() != peers || peers > INT_MAX / 2)
            throw std::runtime_error("Invalid MPI peer count");
        sent.resize(peers);
        received.resize(peers);
        incoming.resize(peers);
        requests.reserve(2 * peers);
        for (std::size_t p = 0; p < peers; ++p)
            sent[p] = outgoing[p].size();
    });
    // Для сообщений переменной длины сначала сообщаем каждому процессу, сколько байтов он получит.
    MPI_Alltoall(sent.data(), 1, MPI_UNSIGNED_LONG_LONG, received.data(), 1, MPI_UNSIGNED_LONG_LONG,
                 MPI_COMM_WORLD);
    local([&] {
        for (std::size_t p = 0; p < peers; ++p) {
            if (received[p] > incoming[p].max_size())
                throw std::length_error("Incoming MPI message is too large");
            incoming[p].resize(static_cast<std::size_t>(received[p]));
        }
    });
    // Определяем число общих раундов, чтобы каждый отдельный запрос укладывался в счётчик MPI типа int.
    const Count maximum = *std::max_element(sent.begin(), sent.end());
    Count total = 0;
    MPI_Allreduce(&maximum, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    for (Count offset = 0; offset < total;) {
        requests.clear();
        for (int p = 0; p < size; ++p) {
            const auto i = static_cast<std::size_t>(p);
            if (offset < received[i]) {
                const int n = static_cast<int>(std::min<Count>(received[i] - offset, INT_MAX));
                MPI_Request req;
                MPI_Irecv(&incoming[i][static_cast<std::size_t>(offset)], n, MPI_BYTE, p, 0, MPI_COMM_WORLD,
                          &req);
                requests.push_back(req);
            }
            if (offset < sent[i]) {
                const int n = static_cast<int>(std::min<Count>(sent[i] - offset, INT_MAX));
                MPI_Request req;
                MPI_Isend(outgoing[i].data() + static_cast<std::size_t>(offset), n, MPI_BYTE, p, 0,
                          MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }
        }
        if (!requests.empty())
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        offset += std::min<Count>(total - offset, INT_MAX);
    }
    return incoming;
#endif
}
}
