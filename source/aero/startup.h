#ifndef AERO_STARTUP_H
#define AERO_STARTUP_H
#include "../io/settings.h"
#include "../mesh/mesh.h"

namespace aero {
void show_settings(const RunSettings&);
void show_boundary_labels(const Mesh&);
bool prepare_boundary_file(const RunSettings&,const Mesh&);
void prepare_output_directories(const RunSettings&);
std::string settings_text(const std::string& path);
void stage(const char* name);
void done();
}
#endif
