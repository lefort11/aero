#ifndef AERO_STARTUP_H
#define AERO_STARTUP_H
#include "../io/settings.h"
#include "../mesh/mesh.h"

namespace aero {
void show_settings(const RunSettings&);
void show_boundary_labels(const Mesh&);
bool prepare_boundary_file(const RunSettings&,const std::vector<std::string>& names);
void create_output_directory(std::string path);
void prepare_output_directories(const RunSettings&);
void prepare_zone_directories(const std::string& directory,const std::vector<OutputZone>& zones,bool global);
unsigned create_resume_directory(const std::string& prefix);
std::string settings_text(const std::string& path);
void stage(const char* name);
void done();
}
#endif
