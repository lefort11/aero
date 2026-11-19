#ifndef AERO_STENCIL_BUILDER_H
#define AERO_STENCIL_BUILDER_H
#include "parallel_stencil.h"

namespace aero {
namespace detail {
// EBR и WENO одного порядка используют одни и те же опорные грани и расстояния.
void build_layer_stencil(const Mesh& mesh, const ReconstructionOptions& options,
                        const ReconstructionSchemeInfo& scheme, ReconstructionStencilData& data);
}
}
#endif
