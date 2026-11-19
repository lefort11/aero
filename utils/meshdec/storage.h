#ifndef AERO_MESHDEC_STORAGE_H
#define AERO_MESHDEC_STORAGE_H
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace aero {
namespace meshdec {
using Id = std::uint64_t;
const Id none = std::numeric_limits<Id>::max();
inline void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
inline void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}
inline Id plus(Id a, Id b) {
    require(a <= none - b, "File size overflow");
    return a + b;
}
inline Id times(Id a, Id b) {
    require(!b || a <= none / b, "File size overflow");
    return a * b;
}
inline std::streamoff position(Id bytes) {
    require(bytes <= static_cast<Id>(std::numeric_limits<std::streamoff>::max()), "File offset too large");
    return static_cast<std::streamoff>(bytes);
}
inline std::ifstream input(const std::string& path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    require(f.is_open(), "Cannot open " + path);
    f.exceptions(std::ios::badbit | std::ios::failbit);
    return f;
}
inline std::ofstream output(const std::string& path) {
    std::ofstream f(path.c_str(), std::ios::binary);
    require(f.is_open(), "Cannot create " + path);
    f.exceptions(std::ios::badbit | std::ios::failbit);
    return f;
}
template <class T> void put(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}
template <class T> T get(std::istream& in) {
    T value;
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}
Id file_size(const std::string& path);
void copy_bytes(std::istream& in, std::ostream& out, Id bytes);
void word(std::ostream& out, Id value);
void real(std::ostream& out, double value);
std::string directory(const std::string& path);

class Scratch {
public:
    Scratch() = default;
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
    void create(const std::string& parent);
    ~Scratch();
    std::string path;
};
void publish(const std::string& staged, const std::string& target);

// Временные таблицы состоят из записей фиксированного размера в формате машины.
// Для последовательного чтения и случайного доступа при замене ссылок достаточно ограниченной страницы памяти.
template <class T> class Table {
public:
    explicit Table(const std::string& path) : file_(input(path)), count(file_size(path) / sizeof(T)) {}
    T at(Id index) {
        require(index < count, "Temporary table index out of range");
        const Id start = index - index % page_.size();
        if (start != start_) {
            file_.seekg(position(times(start, sizeof(T))));
            const Id n = std::min<Id>(page_.size(), count - start);
            file_.read(reinterpret_cast<char*>(page_.data()), position(n * sizeof(T)));
            start_ = start;
        }
        return page_[static_cast<std::size_t>(index - start)];
    }

private:
    std::ifstream file_;
    std::array<T, 4096 / sizeof(T)> page_;
    Id start_ = none;

public:
    const Id count;
};

// Отсортированные серии идут подряд в одном файле. На проход слияния нужны два читателя и один писатель
// при любом числе серий и любом размере сетки.
template <class T> class Sorter {
public:
    Sorter(const std::string& path, Id memory) : path_(path), file_(output(path)) {
        const Id n = std::max<Id>(1, memory / sizeof(T));
        require(n <= buffer_.max_size(), "Sort buffer too large");
        capacity_ = static_cast<std::size_t>(n);
    }
    void add(const T& v) {
        if (buffer_.size() == capacity_)
            flush();
        if (!buffer_.capacity())
            buffer_.reserve(capacity_);
        buffer_.push_back(v);
        count_ = plus(count_, 1);
    }
    std::string finish() {
        flush();
        std::vector<T>().swap(buffer_);
        file_.close();
        std::string src = path_, dst = path_ + ".merge";
        for (Id width = capacity_; width < count_;) {
            auto left = input(src), right = input(src);
            auto out = output(dst);
            for (Id first = 0; first < count_;) {
                const Id mid = first + std::min(width, count_ - first);
                const Id end = mid + std::min(width, count_ - mid);
                left.seekg(position(times(first, sizeof(T))));
                right.seekg(position(times(mid, sizeof(T))));
                Id a = mid - first, b = end - mid;
                T x{}, y{};
                if (a)
                    x = get<T>(left);
                if (b)
                    y = get<T>(right);
                while (a || b) {
                    if (!b || (a && !(y < x))) {
                        put(out, x);
                        if (--a)
                            x = get<T>(left);
                    } else {
                        put(out, y);
                        if (--b)
                            y = get<T>(right);
                    }
                }
                first = end;
            }
            out.close();
            std::swap(src, dst);
            width += std::min(width, count_ - width);
        }
        if (src != path_) {
            require(std::remove(path_.c_str()) == 0 && std::rename(src.c_str(), path_.c_str()) == 0,
                    "Cannot commit sorted temporary table");
        } else
            std::remove(dst.c_str());
        return path_;
    }

private:
    void flush() {
        if (buffer_.empty())
            return;
        std::sort(buffer_.begin(), buffer_.end());
        file_.write(reinterpret_cast<const char*>(buffer_.data()),
                    position(times(buffer_.size(), sizeof(T))));
        buffer_.clear();
    }
    std::string path_;
    std::ofstream file_;
    std::vector<T> buffer_;
    std::size_t capacity_;
    Id count_ = 0;
};

}
}
#endif
