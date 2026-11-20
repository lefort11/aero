#ifndef AERO_MODULES_H
#define AERO_MODULES_H
// Один признак для всех единиц трансляции, включая отдельно собираемые ядра Kokkos.
#if !defined(AERO_HAS_VISCOSITY) && defined(__has_include)
# if __has_include("../viscosity/transport.h")
#  define AERO_HAS_VISCOSITY 1
# endif
#endif
#endif
