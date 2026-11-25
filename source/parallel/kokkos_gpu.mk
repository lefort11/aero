# Список .cpp появляется после распаковки Kokkos. Make перечитает этот файл
# в том же запуске, поэтому первая сборка видит те же цели, что и последующие.
gpu_core := $(kokkos_source)/core/src
gpu_backend_dir := $(if $(filter hip,$(KOKKOS_BACKEND)),HIP,Cuda)
gpu_units_file := $(kokkos_source)/$(KOKKOS_BACKEND)-units.mk
$(gpu_units_file): $(kokkos_stamp)
	@printf '%s\n' 'gpu_core_units := $(wildcard $(gpu_core)/impl/*.cpp $(gpu_core)/$(gpu_backend_dir)/*.cpp)' > "$@.tmp"
	@mv "$@.tmp" "$@"
include $(gpu_units_file)

gpu_objdir := $(kokkos_gpu_dir)/objects
gpu_config_dir := $(kokkos_gpu_dir)/config-$(KOKKOS_BACKEND)
gpu_config := $(gpu_config_dir)/KokkosCore_config.h
$(gpu_objdir):
	@mkdir -p "$@"
# Kokkos сам определяет версию компилятора и записывает KokkosCore_config.h.
# Для проверки обычного C++ в Linux он использует собственную обёртку nvcc.
gpu_cmake_flags := -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_CUDA_ARCH_NUMBER=$(CUDA_ARCH) -DCUDAToolkit_ROOT="$(CUDA_HOME)"
ifeq ($(KOKKOS_BACKEND),hip)
gpu_config_cxx := $(KOKKOS_GPU_CXX)
hip_arch_gfx900 := VEGA900
hip_arch_gfx906 := VEGA906
hip_arch_gfx908 := CDNA1
hip_arch_gfx90a := CDNA2
hip_arch_gfx942 := CDNA3
ifeq ($(hip_arch_$(HIP_ARCH)),)
$(error Supported HIP_ARCH values: gfx900 gfx906 gfx908 gfx90a gfx942)
endif
gpu_cmake_flags := -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_$(hip_arch_$(HIP_ARCH))=ON -DCMAKE_CXX_FLAGS=--offload-arch=$(HIP_ARCH)
else ifeq ($(OS),Windows_NT)
gpu_config_cxx := $(KOKKOS_GPU_CXX)
else
gpu_config_cxx := $(kokkos_source)/bin/nvcc_wrapper
gpu_config_env := NVCC_WRAPPER_DEFAULT_ARCH=sm_$(CUDA_ARCH) NVCC_WRAPPER_NVCC="$(KOKKOS_GPU_CXX)" NVCC_WRAPPER_DEFAULT_COMPILER="$(CXX)"
$(gpu_config): source/parallel/cuda_toolkit.mk
endif
$(gpu_config): $(kokkos_stamp) source/parallel/kokkos_gpu.mk | $(llvm_ready)
	$(call action,Configuring,Kokkos $(KOKKOS_BACKEND))
	CUDA_ROOT="$(CUDA_HOME)" $(gpu_config_env) cmake -G "Unix Makefiles" -S "$(kokkos_source)" -B "$(gpu_config_dir)" \
	    -DCMAKE_CXX_COMPILER="$(gpu_config_cxx)" $(gpu_cmake_flags) \
	    -DCMAKE_INSTALL_PREFIX="$(gpu_config_dir)/install" -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_STANDARD=11 -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_BUILD_TYPE=Release \
	    -DKokkos_ENABLE_SERIAL=ON \
	    -DKokkos_ENABLE_LIBDL=OFF || { $(RM) "$@"; exit 1; }
	@test -s "$@" && touch "$@"

gpu_units := $(gpu_core_units) source/parallel/device_work.cpp source/parallel/device_runtime.cpp source/parallel/device_memory.cpp tests/device_cases.cpp
# Для исходников Aero сохраняем расширение в имени объекта и его .d.
# После переименования .cu в .cpp make не подключит зависимости прежнего исходника.
gpu_object = $(gpu_objdir)/$(if $(filter $(gpu_core_units),$(1)),$(basename $(notdir $(1))),$(notdir $(1))).o
kokkos_device_objects := $(foreach unit,$(gpu_units),$(call gpu_object,$(unit)))
gpu_flags := -std=c++11 -O3 -I"$(gpu_config_dir)" -I"$(gpu_core)" \
             -I"$(kokkos_source)/containers/src" -DAERO_MIXED_LINEAR=$(MIXED_LINEAR)
ifeq ($(STATIC_LINK),1)
gpu_flags += -DKokkos=AeroDeviceKokkos -DAERO_STATIC_DEVICE_WORK
endif

# У каждого объекта свой исходник и свой список фактически включённых заголовков.
# Публикуем объект последним: прерванная компиляция не затрёт рабочую копию.
$(foreach unit,$(gpu_units),$(eval $(call gpu_object,$(unit)): $(unit)))
gpu_unit = $(firstword $(filter $(gpu_units),$^))
ifeq ($(OS),Windows_NT)
include source/parallel/kokkos_gpu_mingw.mk
else
ifeq ($(KOKKOS_BACKEND),hip)
gpu_flags += -x hip --offload-arch=$(HIP_ARCH) -fPIC -fvisibility=hidden
else
gpu_flags += -x cu -arch=sm_$(CUDA_ARCH) --extended-lambda --expt-relaxed-constexpr -Xcompiler=-fPIC,-fvisibility=hidden
endif
$(kokkos_device_objects): $(gpu_config) source/parallel/kokkos_gpu.mk | $(gpu_objdir)
	$(call action,Compiling,$(gpu_unit))
	"$(KOKKOS_GPU_CXX)" $(gpu_flags) -MD -MP -MF "$(@:.o=.d).tmp" -MT "$@" -c "$(gpu_unit)" -o "$@.tmp"
	mv "$(@:.o=.d).tmp" "$(@:.o=.d)"
	mv "$@.tmp" "$@"
gpu_deps := $(wildcard $(kokkos_device_objects:.o=.d))
-include $(gpu_deps)

# После перехода со старой сборки (или удаления .d) один раз восстановим зависимости.
.PHONY: gpu_missing_dependencies
$(filter-out $(gpu_deps:.d=.o),$(kokkos_device_objects)): gpu_missing_dependencies
endif

$(KOKKOS_DEVICE_LIBRARY): $(kokkos_device_objects) $(gpu_units_file)
ifeq ($(STATIC_LINK),1)
	$(archiving)
	$(RM) "$@.tmp"
	$(AR) rcs "$@.tmp" $(filter %.o,$^)
else
	$(call action,Linking,$@)
	$(CXX) -shared -Wl,-Bsymbolic -Wl,-soname,$(@F) $(filter %.o,$^) $(if $(filter hip,$(KOKKOS_BACKEND)),$(HIP_RUNTIME_LINK),-L"$(CUDA_LIBDIR)" -lcudart) -ldl -pthread -o "$@.tmp"
endif
	mv "$@.tmp" "$@"
