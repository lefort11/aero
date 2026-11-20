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
void RunControl::parse(const std::string& commands, std::ostream* warnings) {
    std::istringstream in(commands);
    std::set<std::string> keys;
    bool explicit_cfl = false, has_commands = false;
    std::string line;
    std::size_t row = 0;
    while (std::getline(in, line)) {
        using namespace settings_detail;
        line = clean(line, "control.txt", ++row);
        if (line.empty()) continue;
        const auto eq = line.find('=');
        const auto key = trim(line.substr(0, eq));
        has_commands = true;
        const auto argument = [&]() {
            if (eq == std::string::npos) throw std::invalid_argument("expected key=value: " + line);
            return trim(line.substr(eq + 1));
        };
        const auto real = [&]() { return finite_real(argument(), "control.txt (" + key + ")"); };
        if (key == "STOP" || key == "reset_fail") {
            if (eq != std::string::npos) throw std::invalid_argument(key + " takes no value");
            if (key == "STOP") stop = true;
            else reset = true;
        } else if (key == "upw_coeff") upwind = real();
        else if (key == "mp6_hybrid") hybrid = real();
        else if (key == "cfl") { cfl = real(); explicit_cfl = true; }
        else if (key == "local_time_step" || key == "local_time_step_max_ratio") {
            if (!steady) throw std::invalid_argument(key + " requires time_scheme=steady");
            const auto where = "control.txt (" + key + ")";
            if (key == "local_time_step") local_time_step = parse_local_time_step(argument(), where);
            else local_time_step_max_ratio = parse_local_step_ratio(argument(), where);
            change_steady_steps = true;
        }
        else if (key == "cfl_min") limits.minimum = real();
        else if (key == "cfl_max") limits.maximum = real();
        else if (key == "cfl_coeff") limits.factor = real();
        else if (key == "cfl_num_steps")
            limits.interval = number_codec::parse_decimal(argument(), [](const char* why) { throw std::invalid_argument(why); });
        else {
            // Опечатка не отменяет остальные команды, даже если её значение не является числом.
            if (warnings) *warnings << "WARNING: control.txt:" << row << ": unknown parameter '" << key << "'" << std::endl;
            continue;
        }
        if (!keys.insert(key).second) throw std::invalid_argument("duplicate " + key);
        accepted_.push_back(eq == std::string::npos ? key : key + " = " + argument());
    }
    if (!has_commands) throw std::invalid_argument("empty command file");
    const bool limits_changed = keys.count("cfl_min") || keys.count("cfl_max") || keys.count("cfl_coeff") || keys.count("cfl_num_steps");
    change_cfl = explicit_cfl || limits_changed;
    if (upwind < 0 || upwind > 1 || hybrid < 0 || hybrid > 1)
        throw std::invalid_argument("upw_coeff and mp6_hybrid must be in [0,1]");
    if (!limits.active() && limits_changed)
        throw std::invalid_argument("adaptive CFL settings require active stabilization");
    // Без явного cfl сохраняем текущее значение в пределах новых границ.
    if (limits.active() && !explicit_cfl) cfl = std::max(limits.minimum, std::min(cfl, limits.maximum));
    limits.validate(cfl, "control.txt");
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
    try { parse(commands, world.rank ? nullptr : &std::cerr); }
    catch (const std::invalid_argument& e) {
        if (!world.rank) std::cerr << "Control rejected: " << e.what() << std::endl;
        return false;
    }
    return !accepted_.empty();
}

void RunControl::report_accepted(std::ostream& out) const {
    for (const auto& setting : accepted_) out << "Control accepted: " << setting << '\n';
}
}
