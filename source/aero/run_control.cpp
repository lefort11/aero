#include "run_control.h"
#include "../io/settings_text.h"
#include "../io/number_codec.h"
#include "../parallel/processes.h"
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <set>
#include <sstream>

namespace aero {
void RunControl::parse(const std::string& commands) {
    std::istringstream in(commands);
    std::set<std::string> keys;
    bool explicit_cfl = false;
    std::string line;
    std::size_t row = 0;
    while (std::getline(in, line)) {
        using namespace settings_detail;
        line = clean(line, "control.txt", ++row);
        if (line.empty()) continue;
        const auto eq = line.find('=');
        const auto key = trim(line.substr(0, eq));
        if (!keys.insert(key).second) throw std::invalid_argument("duplicate " + key);
        if (line == "STOP") { stop = true; continue; }
        if (line == "reset_fail") { reset = true; continue; }
        if (eq == std::string::npos) throw std::invalid_argument("expected key=value: " + line);
        const std::string token = trim(line.substr(eq + 1));
        if (key == "cfl:num_steps") {
            limits.interval = number_codec::parse_decimal(token, [](const char* why) { throw std::invalid_argument(why); });
            change_cfl = true;
            continue;
        }
        std::istringstream field(token);
        field.imbue(std::locale::classic());
        double value;
        if (!(field >> value) || !field.eof() || !std::isfinite(value)) throw std::invalid_argument("invalid " + key);
        if (key == "upw_coeff") upwind = value;
        else if (key == "mp6_hybrid") hybrid = value;
        else {
            change_cfl = true;
            if (key == "cfl") { cfl = value; explicit_cfl = true; }
            else if (key == "cfl_min") limits.minimum = value;
            else if (key == "cfl_max") limits.maximum = value;
            else if (key == "cfl:coeff") limits.factor = value;
            else throw std::invalid_argument("unknown command " + key);
        }
    }
    if (keys.empty()) throw std::invalid_argument("empty command file");
    if (upwind < 0 || upwind > 1 || hybrid < 0 || hybrid > 1)
        throw std::invalid_argument("upw_coeff and mp6_hybrid must be in [0,1]");
    if (!limits.active() && (keys.count("cfl_min") || keys.count("cfl_max") || keys.count("cfl:coeff") || keys.count("cfl:num_steps")))
        throw std::invalid_argument("adaptive CFL settings require active stabilization");
    if (limits.active()) {
        if (!limits.interval || limits.factor <= 1 || limits.minimum <= 0 || limits.maximum < limits.minimum)
            throw std::invalid_argument("invalid adaptive CFL settings");
        // При изменении одних границ приводим текущий CFL к новому диапазону.
        if (!explicit_cfl) cfl = std::max(limits.minimum, std::min(cfl, limits.maximum));
        if (cfl < limits.minimum || cfl > limits.maximum) throw std::invalid_argument("cfl outside limits");
    }
    if (cfl <= 0) throw std::invalid_argument("cfl must be positive");
}

bool RunControl::read(const std::string& directory, const Processes& world) {
    const std::string path = directory + "/control.txt", claimed = path + ".reading";
    std::string commands;
    // Только master забирает файл. Следующую команду уже можно публиковать под исходным именем.
    if (!world.rank && std::rename(path.c_str(), claimed.c_str()) == 0) {
        try {
            auto file = settings_detail::open(claimed);
            std::ostringstream contents;
            contents << file.rdbuf();
            if (file.bad()) throw std::runtime_error("read error");
            commands = contents.str();
            if (commands.empty()) throw std::runtime_error("empty file");
        } catch (const std::exception& e) {
            std::cerr << "Control rejected: " << e.what() << std::endl;
            commands.clear();
        }
        if (std::remove(claimed.c_str())) std::cerr << "Cannot remove " << claimed << std::endl;
    }
    if (world.size > 1) world.broadcast(commands, 0);
    if (commands.empty()) return false;
    try { parse(commands); }
    catch (const std::invalid_argument& e) {
        if (!world.rank) std::cerr << "Control rejected: " << e.what() << std::endl;
        return false;
    }
    return true;
}
}
