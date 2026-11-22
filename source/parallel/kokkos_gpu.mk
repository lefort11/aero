# Список .cpp появляется после распаковки Kokkos. Make перечитает этот файл
# в том же запуске, поэтому первая сборка видит те же цели, что и последующие.
gpu_core := $(kokkos_source)/core/src
gpu_units_file := $(kokkos_source)/cuda-units.mk
$(gpu_units_file): $(kokkos_stamp)
	@printf '%s\n' 'gpu_core_units := $(wildcard $(gpu_core)/impl/*.cpp $(gpu_core)/Cuda/*.cpp)' > "$@.tmp"
	@mv "$@.tmp" "$@"
include $(gpu_units_file)

gpu_objdir := $(kokkos_gpu_dir)/objects
gpu_config_dir := $(kokkos_gpu_dir)/config
gpu_config := $(gpu_config_dir)/KokkosCore_config.h
$(gpu_objdir):
	@mkdir -p "$@"
# Kokkos сам определяет версию компилятора и записывает KokkosCore_config.h.
# Для проверки обычного C++ в Linux он использует собственную обёртку nvcc.
ifeq ($(OS),Windows_NT)
gpu_config_cxx := $(KOKKOS_GPU_CXX)
else
gpu_config_cxx := $(kokkos_source)/bin/nvcc_wrapper
gpu_config_env := NVCC_WRAPPER_DEFAULT_ARCH=sm_$(CUDA_ARCH) NVCC_WRAPPER_NVCC="$(KOKKOS_GPU_CXX)" NVCC_WRAPPER_DEFAULT_COMPILER="$(CXX)"
endif
$(gpu_config): $(kokkos_stamp) source/parallel/kokkos_gpu.mk | $(llvm_ready)
	$(call action,Configuring,Kokkos CUDA)
	CUDA_ROOT="$(CUDA_HOME)" $(gpu_config_env) cmake -G "Unix Makefiles" -S "$(kokkos_source)" -B "$(gpu_config_dir)" \
	    -DCMAKE_CXX_COMPILER="$(gpu_config_cxx)" -DCUDAToolkit_ROOT="$(CUDA_HOME)" \
	    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_STANDARD=11 -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_BUILD_TYPE=Release \
	    -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON \
	    -DKokkos_ENABLE_LIBDL=OFF -DKokkos_CUDA_ARCH_NUMBER=$(CUDA_ARCH) || { $(RM) "$@"; exit 1; }
	@test -s "$@" && touch "$@"

gpu_units := $(gpu_core_units) source/parallel/device_work.cu source/parallel/device_memory.cu tests/device_cases.cu
kokkos_device_objects := $(addprefix $(gpu_objdir)/,$(addsuffix .o,$(basename $(notdir $(gpu_units)))))
gpu_flags := -std=c++11 -O3 -I"$(gpu_config_dir)" -I"$(gpu_core)" \
             -I"$(kokkos_source)/containers/src" -DAERO_MIXED_LINEAR=$(MIXED_LINEAR)
ifeq ($(STATIC_LINK),1)
gpu_flags += -DKokkos=AeroDeviceKokkos -DAERO_STATIC_DEVICE_WORK
endif

# У каждого объекта свой исходник и свой список фактически включённых заголовков.
# Публикуем объект последним: прерванная компиляция не затрёт рабочую копию.
$(foreach unit,$(gpu_units),$(eval $(gpu_objdir)/$(basename $(notdir $(unit))).o: $(unit)))
gpu_unit = $(firstword $(filter $(gpu_units),$^))
ifeq ($(OS),Windows_NT)
include source/parallel/kokkos_gpu_mingw.mk
else
gpu_flags += -x cu -arch=sm_$(CUDA_ARCH) --extended-lambda --expt-relaxed-constexpr -Xcompiler=-fPIC,-fvisibility=hidden
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
	$(CXX) -shared -Wl,-Bsymbolic -Wl,-soname,$(@F) $(filter %.o,$^) -L"$(CUDA_LIBDIR)" -lcudart -ldl -pthread -o "$@.tmp"
endif
	mv "$@.tmp" "$@"
