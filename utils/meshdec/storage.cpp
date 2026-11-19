#include "storage.h"
#include "../../source/io/number_codec.h"
#include <cerrno>
#include <cstring>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <direct.h>
#include <process.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace aero {
namespace meshdec {
Id file_size(const std::string& path) {
    auto f = input(path);
    f.seekg(0, std::ios::end);
    return static_cast<Id>(f.tellg());
}
void word(std::ostream& out, Id value) {
    number_codec::write_word(out, value, false);
}
void real(std::ostream& out, double value) {
    word(out, number_codec::real_bits(value));
}
// Переносит диапазон через буфер фиксированного размера, в том числе когда файл больше оперативной памяти.
void copy_bytes(std::istream& in, std::ostream& out, Id bytes) {
    std::array<char, 32768> block;
    while (bytes) {
        const auto n = static_cast<std::streamsize>(std::min<Id>(bytes, block.size()));
        in.read(block.data(), n);
        out.write(block.data(), n);
        bytes -= static_cast<Id>(n);
    }
}
std::string directory(const std::string& path) {
    const auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos + 1);
}
// Создаёт отдельный рабочий каталог для файлов этого запуска.
void Scratch::create(const std::string& parent) {
#ifdef _WIN32
    const int pid = _getpid();
#else
    const auto pid = getpid();
#endif
    for (unsigned i = 0; i < 10000; ++i) {
        const std::string name = parent + "/.meshdec-" + std::to_string(pid) + "-" + std::to_string(i);
#ifdef _WIN32
        const int status = _mkdir(name.c_str());
#else
        const int status = mkdir(name.c_str(), 0700);
#endif
        if (status == 0) {
            path = name;
            return;
        }
        require(errno == EEXIST, "Cannot create scratch directory in " + parent);
    }
    throw std::runtime_error("Cannot choose a unique scratch directory");
}
Scratch::~Scratch() {
    if (path.empty())
        return;
#ifdef _WIN32
    WIN32_FIND_DATAA item;
    const auto search = FindFirstFileA((path + "/*").c_str(), &item);
    if (search != INVALID_HANDLE_VALUE) {
        do {
            if (!(item.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
                std::remove((path + "/" + item.cFileName).c_str());
        } while (FindNextFileA(search, &item));
        FindClose(search);
    }
    _rmdir(path.c_str());
#else
    DIR* dir = opendir(path.c_str());
    if (dir) {
        while (const auto* item = readdir(dir)) {
            const std::string name = item->d_name;
            if (name != "." && name != "..")
                unlink((path + "/" + name).c_str());
        }
        closedir(dir);
    }
    rmdir(path.c_str());
#endif
}
void publish(const std::string& staged, const std::string& target) {
#ifdef _WIN32
    require(MoveFileA(staged.c_str(), target.c_str()) != 0,
            "Cannot publish output (already exists or inaccessible): " + target);
#else
    require(link(staged.c_str(), target.c_str()) == 0,
            "Cannot publish output: " + target + ": " + std::strerror(errno));
#endif
}
}
}
