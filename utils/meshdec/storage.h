#ifndef AERO_MESHDEC_STORAGE_H
#define AERO_MESHDEC_STORAGE_H
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
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
// Проверяет обязательное условие и сообщает ошибку при его нарушении.
inline void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
// Проверяет обязательное условие и сообщает ошибку при его нарушении.
inline void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}
// Складывает размеры с проверкой переполнения адресного пространства.
inline Id plus(Id a, Id b) {
    require(a <= none - b, "File size overflow");
    return a + b;
}
// Умножает размеры с проверкой переполнения адресного пространства.
inline Id times(Id a, Id b) {
    require(!b || a <= none / b, "File size overflow");
    return a * b;
}
// Преобразует смещение записи в допустимую позицию файлового потока.
inline std::streamoff position(Id bytes) {
    require(bytes <= static_cast<Id>(std::numeric_limits<std::streamoff>::max()), "File offset too large");
    return static_cast<std::streamoff>(bytes);
}
// Открывает двоичный входной файл и проверяет успешность открытия.
inline std::ifstream input(const std::string& path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    require(f.is_open(), "Cannot open " + path);
    f.exceptions(std::ios::badbit | std::ios::failbit);
    return f;
}
// Открывает двоичный выходной файл и проверяет успешность открытия.
inline std::ofstream output(const std::string& path, std::ios::openmode mode = std::ios::trunc) {
    std::ofstream f(path.c_str(), std::ios::binary | mode);
    if (!f.is_open()) {
        const int error = errno;
        throw std::runtime_error("Cannot open for writing " + path + ": " + std::strerror(error));
    }
    f.exceptions(std::ios::badbit | std::ios::failbit);
    return f;
}
// Записывает значение фиксированного размера в промежуточный поток.
template <class T> void put(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}
// Читает значение фиксированного размера из промежуточного потока.
template <class T> T get(std::istream& in) {
    T value;
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}
// Возвращает размер файла для проверки числа и адресов записей.
Id file_size(const std::string& path);
// Переносит указанный диапазон байтов между источником и приёмником.
void copy_bytes(std::istream& in, std::ostream& out, Id bytes);
// Читает или записывает целое слово промежуточного двоичного формата.
void word(std::ostream& out, Id value);
// Читает или записывает вещественное число промежуточного двоичного формата.
void real(std::ostream& out, double value);
// Создаёт рабочий каталог промежуточных таблиц.
std::string directory(const std::string& path);

// Управляет временным каталогом таблиц и публикацией готового архива.
class Scratch {
public:
    // Инициализирует объект стандартными значениями его полей.
    Scratch() = default;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    Scratch(const Scratch&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    Scratch& operator=(const Scratch&) = delete;
    // Создаёт отдельный рабочий каталог для файлов этого запуска.
    void create(const std::string& parent);
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~Scratch();
    // Путь к рабочему хранилищу.
    std::string path;
};
// Публикует готовый архив после успешного завершения всех этапов.
void publish(const std::string& staged, const std::string& target);

// Временные таблицы состоят из записей фиксированного размера в формате машины.
// Для последовательного чтения и случайного доступа при замене ссылок достаточно ограниченной страницы памяти.
template <class T> class Table {
public:
    // Открывает дисковую таблицу и определяет число записей по размеру файла.
    explicit Table(const std::string& path) : file_(input(path)), count(file_size(path) / sizeof(T)) {}
    // Читает запись по индексу, проверяя её попадание в таблицу.
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
    // Поток чтения дисковой таблицы.
    std::ifstream file_;
    // Кэш одной страницы записей таблицы.
    std::array<T, 4096 / sizeof(T)> page_;
    // Номер первой записи кэшированной страницы.
    Id start_ = none;

public:
    // Количество доступных записей.
    const Id count;
};

// Отсортированные серии идут подряд в одном файле. На проход слияния нужны два читателя и один писатель
// при любом числе серий и любом размере сетки.
template <class T> class Sorter {
public:
    // Задаёт выходной файл и ёмкость буфера внешней сортировки.
    Sorter(const std::string& path, Id memory) : path_(path), file_(output(path)) {
        const Id n = std::max<Id>(1, memory / sizeof(T));
        require(n <= buffer_.max_size(), "Sort buffer too large");
        capacity_ = static_cast<std::size_t>(n);
    }
    // Добавляет запись в буфер внешней сортировки и сбрасывает заполненный блок.
    void add(const T& v) {
        if (buffer_.size() == capacity_)
            flush();
        if (!buffer_.capacity())
            buffer_.reserve(capacity_);
        buffer_.push_back(v);
        count_ = plus(count_, 1);
    }
    // Сливает отсортированные серии в итоговую таблицу и завершает запись.
    std::string finish() {
        flush();
        std::vector<T>().swap(buffer_);
        file_.close();
        std::string src = path_, dst = path_ + ".merge";
        for (Id width = capacity_; width < count_;) {
            auto left = input(src), right = input(src);
            // Первый проход создаёт второй файл. Дальше оба файла уже имеют
            // ровно count_ записей: переписываем их с начала без усечения.
            // Windows может запретить усечение недавно прочитанного файла (ошибка 1224).
            auto out = output(dst, width == capacity_ ? std::ios::trunc : std::ios::in);
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
    // Сортирует и сохраняет текущий буфер как очередную упорядоченную серию.
    void flush() {
        if (buffer_.empty())
            return;
        std::sort(buffer_.begin(), buffer_.end());
        file_.write(reinterpret_cast<const char*>(buffer_.data()),
                    position(times(buffer_.size(), sizeof(T))));
        buffer_.clear();
    }
    // Базовое имя для временных серий сортировки.
    std::string path_;
    // Открытый файл промежуточных записей.
    std::ofstream file_;
    // Записи для очередной сортируемой порции.
    std::vector<T> buffer_;
    // Предельное число записей до сброса порции.
    std::size_t capacity_;
    // Количество накопленных записей.
    Id count_ = 0;
};

}
}
#endif
