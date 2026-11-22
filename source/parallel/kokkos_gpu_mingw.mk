# Clang использует заголовки CUDA через VFS, а системные функции Kokkos для
# стека и топологии CPU собираются тем же MinGW GCC, что и основной солвер.
gpu_overlay := $(gpu_objdir)/cuda-headers.json
gpu_windows_setup := source/parallel/kokkos_mingw.ps1
$(gpu_overlay): $(gpu_windows_setup) | $(gpu_objdir)
	powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(gpu_windows_setup)" \
	    -Cuda "$(CUDA_HOME)" -Directory "$(gpu_objdir)" $(if $(filter 1,$(STATIC_LINK)),-Static)

gpu_driver = $(KOKKOS_GPU_CXX)
gpu_compile_flags = -x cuda "--cuda-path=$(shell cygpath -am "$(CUDA_HOME)")" --cuda-gpu-arch=sm_$(CUDA_ARCH) \
                    -D__CRT__NO_INLINE -Wno-ignored-attributes -ivfsoverlay "$(gpu_overlay)"
gpu_host_pass = $(if $(findstring /impl/,$(gpu_unit)),--cuda-host-only)
gpu_host_objects := $(addprefix $(gpu_objdir)/,Kokkos_CPUDiscovery.o Kokkos_Stacktrace.o)
$(gpu_host_objects): private gpu_driver = $(CXX)
$(gpu_host_objects): private gpu_compile_flags = -I"$(CUDA_HOME)/include"
$(gpu_host_objects): private gpu_host_pass =

# В файлах зависимостей Clang пишет букву диска с двоеточием, которое
# MSYS make принимает за разделитель правила. Здесь задаём заголовки явно.
$(kokkos_device_objects): $(gpu_config) $(gpu_overlay) $(wildcard source/*/*.h) tests/device_cases.h source/parallel/kokkos_gpu_mingw.mk
	$(call action,Compiling,$(gpu_unit))
	PATH="$(CUDA_HOME)/bin:$$PATH" "$(gpu_driver)" $(gpu_flags) $(gpu_compile_flags) $(gpu_host_pass) -c "$(gpu_unit)" -o "$@.tmp"
	mv "$@.tmp" "$@"
