#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#include <cstdlib>
#endif
#include "scheduled_output.h"

namespace aero {
StagedFieldFile::StagedFieldFile(const std::string& path) : path_(path) {
    const auto directory = path.substr(0, path.find_last_of("/\\") + 1);
#ifdef _WIN32
    char temp[MAX_PATH];
    if (!GetTempFileNameA(directory.empty() ? "." : directory.c_str(), "aer", 0, temp))
        throw FieldError("Cannot create temporary file for " + path);
    temporary_ = temp;
#else
    std::string pattern = directory + ".aero-fields-XXXXXX";
    const int fd = mkstemp(&pattern[0]);
    if (fd < 0)
        throw FieldError("Cannot create temporary file for " + path);
    temporary_ = pattern;
    if (close(fd) != 0) {
        std::remove(temporary_.c_str());
        throw FieldError("Cannot close temporary file for " + path);
    }
#endif
}
StagedFieldFile::~StagedFieldFile() {
    if (!temporary_.empty())
        std::remove(temporary_.c_str());
}
void StagedFieldFile::publish(bool replace) {
#ifdef _WIN32
    if (!MoveFileExA(temporary_.c_str(), path_.c_str(), replace ? MOVEFILE_REPLACE_EXISTING : 0))
        throw FieldError("Cannot publish " + path_ + " (Windows error " + std::to_string(GetLastError()) +
                         ")");
#else
    if (replace) {
        if (std::rename(temporary_.c_str(), path_.c_str()) != 0)
            throw FieldError("Cannot replace " + path_);
    } else {
        if (link(temporary_.c_str(), path_.c_str()) != 0)
            throw FieldError("Cannot publish " + path_);
        if (std::remove(temporary_.c_str()) != 0)
            return; // Деструктор повторит попытку удалить временный файл.
    }
#endif
    temporary_.clear();
}

}
