#ifndef AERO_IO_SETTINGS_TEXT_H
#define AERO_IO_SETTINGS_TEXT_H

#include <fstream>
#include <stdexcept>
#include <string>

namespace aero { namespace settings_detail {
inline std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? "" : s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

// Strip comments outside quotes. Backslashes in Windows paths remain literal.
inline std::string clean(std::string line, const std::string& where, std::size_t row) {
    if (row == 1 && line.compare(0, 3, "\xef\xbb\xbf") == 0) line.erase(0, 3);
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '"') quoted = !quoted;
        if (!quoted && line[i] == '#') { line.resize(i); break; }
    }
    if (quoted) throw std::invalid_argument(where + ": unclosed quote");
    return trim(line);
}

inline std::string label(std::string value, const std::string& where) {
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);
        std::size_t pos=0;
        while ((pos=value.find('"',pos))!=std::string::npos) {
            if (pos+1==value.size() || value[pos+1]!='"')
                throw std::invalid_argument(where+": expected a doubled quote in quoted name");
            value.erase(pos,1);
            ++pos;
        }
    } else if (value.find('"')!=std::string::npos)
        throw std::invalid_argument(where+": unexpected quote in name or path");
    if (value.empty())
        throw std::invalid_argument(where + ": expected a name or path");
    return value;
}

inline std::ifstream open(const std::string& path) {
    std::ifstream file(path.c_str());
    if (!file) throw std::runtime_error("cannot open settings file: " + path);
    return file;
}
} }
#endif
