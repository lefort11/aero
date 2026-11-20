#ifndef AERO_MODULES_H
#define AERO_MODULES_H
// Единые признаки необязательных модулей для решателя и ядер Kokkos.
#if !defined(AERO_HAS_VISCOSITY) && defined(__has_include)
# if __has_include("../viscosity/transport.h")
#  define AERO_HAS_VISCOSITY 1
# endif
#endif
#if defined(AERO_HAS_VISCOSITY) && !defined(AERO_HAS_TURBULENCE) && defined(__has_include)
# if __has_include("../turbulence/sa.h")
#  define AERO_HAS_TURBULENCE 1
# endif
#endif
#endif
