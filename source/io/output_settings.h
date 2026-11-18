#ifndef AERO_IO_OUTPUT_SETTINGS_H
#define AERO_IO_OUTPUT_SETTINGS_H

#include "fields.h"

namespace aero {
struct OutputZone {
    std::string name, file;
    std::vector<std::string> variables;
    std::uint64_t every_steps = 0;
    double every_time = 0;
    bool multi = false;
    bool global = false;
    FieldFormat format = FieldFormat::Binary;
};

// Missing output.txt disables scheduled output for older cases.
std::vector<OutputZone> read_output_zones(const std::string& path);
void validate_output_zones(const std::vector<OutputZone>& zones);
std::string output_filename(const OutputZone& zone, std::uint64_t step);
std::string output_path_key(std::string file);
}
#endif
