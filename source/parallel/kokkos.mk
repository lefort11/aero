# Варианты с Kokkos и без него получают разные объектные файлы даже при одинаковом компиляторе.
override BUILD := $(BUILD)/kokkos
# Внешняя установка должна соответствовать компилятору, ABI и WITH_OPENMP текущего проекта.
ifeq ($(origin KOKKOS_PREFIX),undefined)
kokkos_bundled := 1
KOKKOS_PREFIX := $(abspath $(BUILD))/kokkos-install
endif
kokkos_library := $(KOKKOS_PREFIX)/lib/libkokkoscore.a
kokkos_source := $(abspath $(BUILD))/kokkos-source
# В MSYS пути GCC вида C:/... в depfile воспринимаются make как лишнее двоеточие.
# Для этой пары инструментов зависимости заголовков собирает сам CMake.
kokkos_build := $(abspath $(BUILD))/kokkos-build$(if $(filter Windows_NT,$(OS)),-unix)
kokkos_inputs := third_party/kokkos-3.2.01.zip third_party/kokkos-3.2.01-compat.patch source/parallel/prepare_kokkos.py
kokkos_stamp := $(kokkos_source)/.prepared
$(kokkos_stamp): $(kokkos_inputs)
	python3 source/parallel/prepare_kokkos.py "$(kokkos_source)"
	touch "$@"
override CPPFLAGS += -DAERO_WITH_KOKKOS -isystem "$(KOKKOS_PREFIX)/include"
override LDLIBS += "$(kokkos_library)"
# Каталог для объектов Aero создаётся после установки заголовков библиотеки.
$(BUILD): | $(kokkos_library)
ifeq ($(kokkos_bundled),1)
$(kokkos_library): $(kokkos_stamp)
	cmake -S "$(kokkos_source)" -B "$(kokkos_build)" -G "Unix Makefiles" $(if $(filter Windows_NT,$(OS)),-DKokkos_ENABLE_LIBDL=OFF -DCMAKE_DEPENDS_USE_COMPILER=OFF) -DCMAKE_CXX_COMPILER="$(CXX)" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=11 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DKokkos_ENABLE_OPENMP=$(if $(filter 1,$(WITH_OPENMP)),ON,OFF) -DKokkos_ENABLE_SERIAL=$(if $(filter 1,$(WITH_OPENMP)),OFF,ON) -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX="$(KOKKOS_PREFIX)"
	+$(MAKE) -C "$(kokkos_build)" install
endif

ifeq ($(WITH_CUDA),1)
kokkos_gpu_dir := $(abspath $(BUILD))/kokkos-device
kokkos_module := $(if $(filter Windows_NT,$(OS)),aero_kokkos_cuda.dll,libaero_kokkos_cuda.so)
KOKKOS_GPU_CXX ?= $(if $(filter Windows_NT,$(OS)),clang++,$(NVCC))
ifeq ($(STATIC_LINK),1)
# Объекты GPU входят в исполняемый файл. На Windows LLD читает архив CUDA в формате MSVC.
kokkos_gpu_archive := libaero_kokkos_device.a
kokkos_gpu_flags := --static --archiver "$(AR)"
ifeq ($(OS),Windows_NT)
override LDFLAGS += -fuse-ld=lld
# CUDA вызывает строковое форматирование UCRT из состава Windows 10/11.
# Сначала разрешаем обычные функции через тот CRT, с которым собран текущий MinGW.
kokkos_system_crt := $(if $(findstring ucrt,$(shell $(CXX) -print-file-name=libstdc++.a)),ucrtbase,msvcrt)
kokkos_static_runtime := "$(CUDA_HOME)/lib/x64/cudart_static.lib" -L"$(kokkos_gpu_dir)" -lvcruntime -l$(kokkos_system_crt) -lucrtbase
else
kokkos_static_runtime := $(CUDA_RUNTIME_LINK)
endif
else
kokkos_gpu_archive := $(if $(filter Windows_NT,$(OS)),libaero_kokkos_cuda.a,$(kokkos_module))
endif
ifeq ($(origin KOKKOS_DEVICE_LIBRARY),undefined)
KOKKOS_DEVICE_LIBRARY := $(kokkos_gpu_dir)/$(kokkos_gpu_archive)
kokkos_device_sources := source/parallel/build_kokkos_gpu.py source/parallel/device_work.cu $(wildcard source/*/*.h)
$(KOKKOS_DEVICE_LIBRARY): $(kokkos_stamp) $(kokkos_device_sources)
	python3 source/parallel/build_kokkos_gpu.py --work "$(kokkos_gpu_dir)/objects" --source "$(kokkos_source)" --cuda "$(CUDA_HOME)" --arch "$(CUDA_ARCH)" --compiler "$(KOKKOS_GPU_CXX)" --host-compiler "$(CXX)" --output "$@" $(kokkos_gpu_flags)
endif
$(BUILD): | $(KOKKOS_DEVICE_LIBRARY)
ifeq ($(STATIC_LINK),0)
# При запуске из bin и при прямом вызове теста из build загрузчик находит один и тот же модуль.
.PHONY: kokkos-runtime
kokkos-runtime: $(KOKKOS_DEVICE_LIBRARY) | $(BIN) $(BUILD)
	cp -f "$(dir $(KOKKOS_DEVICE_LIBRARY))$(kokkos_module)" "$(BIN)/"
	cp -f "$(dir $(KOKKOS_DEVICE_LIBRARY))$(kokkos_module)" "$(BUILD)/"
ifeq ($(OS),Windows_NT)
	@for f in "$(CUDA_HOME)"/bin/cudart64_*.dll; do cp -f "$$f" "$(BIN)/"; cp -f "$$f" "$(BUILD)/"; done
else
override LDFLAGS += -Wl,-rpath,'$$ORIGIN'
endif
bin-runtime: kokkos-runtime
endif
endif

