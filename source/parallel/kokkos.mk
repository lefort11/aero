# В этом каталоге размещаются объектные файлы и локальная установка Kokkos.
override BUILD := $(BUILD)/kokkos
# Внешняя установка должна соответствовать компилятору, ABI и WITH_OPENMP текущего проекта.
ifeq ($(origin KOKKOS_PREFIX),undefined)
kokkos_bundled := 1
KOKKOS_PREFIX := $(abspath $(BUILD))/kokkos-install
endif
kokkos_library := $(KOKKOS_PREFIX)/lib/libkokkoscore.a
kokkos_unpack := $(abspath $(BUILD))/kokkos-source
kokkos_source := $(kokkos_unpack)/kokkos-3.2.01
# В MSYS пути GCC вида C:/... в depfile воспринимаются make как лишнее двоеточие.
# Для этой пары инструментов зависимости заголовков собирает сам CMake.
kokkos_build := $(abspath $(BUILD))/kokkos-build$(if $(filter Windows_NT,$(OS)),-unix)/3.2.01
kokkos_zip := third_party/kokkos-3.2.01.zip
kokkos_inputs := $(kokkos_zip) third_party/kokkos-3.2.01-compat.patch
kokkos_stamp := $(kokkos_source)/.prepared
$(kokkos_stamp): $(kokkos_inputs)
	$(call action,Preparing,Kokkos sources)
	@printf '%s  %s\n' 4e88d9cfab0520fd183e3dd76b1c327447f47b44d4dae24b360979e518fc5d75 "$(kokkos_zip)" | sha256sum -c -
	@mkdir -p "$(kokkos_unpack)"
ifeq ($(OS),Windows_NT)
	# Для тестовой ссылки gtest в zip на Windows могут требоваться особые права.
	bsdtar -xf "$(kokkos_zip)" -C "$(kokkos_unpack)" --exclude '*/gtest-test-part.h'
	cp "$(kokkos_source)/tpls/gtest/gtest/gtest.h" "$(kokkos_source)/tpls/gtest/gtest/gtest-test-part.h"
else
	cd "$(kokkos_unpack)" && cmake -E tar xf "$(abspath $(kokkos_zip))"
endif
	patch --binary -d "$(kokkos_source)" -p1 --batch < third_party/kokkos-3.2.01-compat.patch
	touch "$@"
override CPPFLAGS += -isystem "$(KOKKOS_PREFIX)/include"
override LDLIBS += "$(kokkos_library)"
# Установка внешней библиотеки общая. Объекты Aero и модуль GPU храним
# отдельно, чтобы при переключении точности make выбрал совместимую пару.
ifeq ($(MIXED_LINEAR),1)
override BUILD := $(BUILD)/mixed-linear
endif
# Каталог для объектов Aero создаётся после установки заголовков библиотеки.
$(BUILD): | $(kokkos_library)
ifeq ($(kokkos_bundled),1)
$(kokkos_library): $(kokkos_stamp)
	$(call action,Configuring,Kokkos)
	cmake -S "$(kokkos_source)" -B "$(kokkos_build)" -G "Unix Makefiles" $(if $(filter Windows_NT,$(OS)),-DKokkos_ENABLE_LIBDL=OFF -DCMAKE_DEPENDS_USE_COMPILER=OFF) -DCMAKE_CXX_COMPILER="$(CXX)" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=11 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DKokkos_ENABLE_OPENMP=$(if $(filter 1,$(WITH_OPENMP)),ON,OFF) -DKokkos_ENABLE_SERIAL=$(if $(filter 1,$(WITH_OPENMP)),OFF,ON) -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX="$(KOKKOS_PREFIX)"
	$(call action,Building,Kokkos host library)
	+$(MAKE) -C "$(kokkos_build)" install $(if $(filter 1,$(V)),VERBOSE=1)
endif

ifeq ($(kokkos_gpu),1)
kokkos_gpu_dir := $(abspath $(BUILD))/kokkos-device
ifeq ($(OS),Windows_NT)
# Сначала ищем инструменты в среде пользователя. В рабочем дереве также
# может быть локально распакован комплект MinGW для сборки Kokkos.
llvm_dir := tools/mingw-llvm
llvm_bin := $(abspath $(llvm_dir))/mingw64/bin
llvm_packages := $(wildcard third_party/mingw-llvm/*.pkg.tar.zst)
system_clang := $(shell command -v clang++)
system_lld := $(shell command -v ld.lld)
KOKKOS_GPU_CXX ?= $(or $(system_clang),$(llvm_bin)/clang++.exe)
# Архивы входят в проект. Параллельные цели ждут окончания распаковки всех DLL и заголовков.
use_local_clang := $(filter $(llvm_bin)/clang++.exe,$(KOKKOS_GPU_CXX))
use_local_lld := $(if $(system_lld),,$(filter 1,$(STATIC_LINK)))
ifneq ($(strip $(use_local_clang) $(use_local_lld)),)
llvm_ready := $(llvm_dir)/.ready
$(llvm_ready): $(llvm_packages)
	$(call action,Preparing,MinGW Clang and LLD)
	@mkdir -p "$(llvm_dir)"
	@set -e; for pkg in $(llvm_packages); do tar -xf "$$pkg" -C "$(llvm_dir)" mingw64; done
	@test -f "$(llvm_bin)/clang++.exe" && test -f "$(llvm_bin)/ld.lld.exe"
	@touch "$@"
export PATH := $(PATH):$(llvm_bin)
endif
else
KOKKOS_GPU_CXX ?= $(NVCC)
endif
ifeq ($(STATIC_LINK),1)
# Объекты GPU входят в исполняемый файл. На Windows LLD читает архив CUDA в формате MSVC.
kokkos_gpu_archive := libaero_kokkos_device.a
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
kokkos_module := libaero_kokkos_cuda.so
kokkos_gpu_archive := $(kokkos_module)
endif
ifeq ($(origin KOKKOS_DEVICE_LIBRARY),undefined)
KOKKOS_DEVICE_LIBRARY := $(kokkos_gpu_dir)/$(kokkos_gpu_archive)
include source/parallel/kokkos_gpu.mk
endif
$(BUILD): | $(KOKKOS_DEVICE_LIBRARY)
ifeq ($(STATIC_LINK),0)
# При запуске из bin и при прямом вызове теста из build загрузчик находит один и тот же модуль.
.PHONY: kokkos-runtime
kokkos-runtime: $(KOKKOS_DEVICE_LIBRARY) | $(BIN) $(BUILD)
	cp -f "$(dir $(KOKKOS_DEVICE_LIBRARY))$(kokkos_module)" "$(BIN)/"
	cp -f "$(dir $(KOKKOS_DEVICE_LIBRARY))$(kokkos_module)" "$(BUILD)/"
override LDFLAGS += -Wl,-rpath,'$$ORIGIN'
bin-runtime: kokkos-runtime
endif
endif
