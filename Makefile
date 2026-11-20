# Сборка GNU Make для C++11. Для варианта CPU достаточно компилятора C++.
# Основные проверки запускаются целями test, debug-test и cuda-check.
CXX = g++
# Исполнитель Kokkos задаётся один раз для солвера, библиотек и тестов.
KOKKOS_BACKEND ?= cpu
ifeq ($(KOKKOS_BACKEND),cuda)
kokkos_gpu := 1
else ifeq ($(KOKKOS_BACKEND),cpu)
kokkos_gpu := 0
else
$(error KOKKOS_BACKEND must be cpu or cuda)
endif
ifneq ($(filter-out undefined,$(origin WITH_CUDA) $(origin WITH_KOKKOS)),)
$(error Choose KOKKOS_BACKEND=cpu or KOKKOS_BACKEND=cuda)
endif
CUDA_ARCH ?= 86
ifeq ($(OS)/$(kokkos_gpu),Windows_NT/1)
ifneq ($(filter %cygwin %msys,$(MAKE_HOST)),)
MINGW_CUDA := 1
# Для совместимости с C++11 при наличии нескольких версий выбираем CUDA 12.
cuda12_env := $(lastword $(sort $(filter CUDA_PATH_V12_%,$(.VARIABLES))))
CUDA_HOME ?= $(shell cygpath -u "$(or $($(cuda12_env)),$(CUDA_PATH))")
endif
endif
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(if $(wildcard $(CUDA_HOME)/bin/nvcc),$(CUDA_HOME)/bin/nvcc,nvcc)
CPPFLAGS ?=
CXXFLAGS ?= -std=c++11 -pedantic-errors -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
LDFLAGS ?=
LDLIBS ?=
WITH_OPENMP ?= 1
WITH_MPI ?= 1
build_optimization := -O2
STATIC_LINK ?= 1
CUDA_LIBDIR ?= $(if $(wildcard $(CUDA_HOME)/lib64),$(CUDA_HOME)/lib64,$(CUDA_HOME)/lib)
ifeq ($(STATIC_LINK),1)
ifeq ($(OS),Windows_NT)
override LDFLAGS += -static
else ifneq ($(filter 1,$(kokkos_gpu) $(WITH_MPI)),)
# MPI и драйвер NVIDIA использует системную glibc при загрузке в процесс.
override LDFLAGS += -static-libgcc -static-libstdc++
else
override LDFLAGS += -static
endif
CUDA_RUNTIME_LINK = -L$(CUDA_LIBDIR) -lcudart_static -ldl -lrt -pthread
else ifeq ($(STATIC_LINK),0)
CUDA_RUNTIME_LINK = -L$(CUDA_LIBDIR) -lcudart
else
$(error STATIC_LINK must be 0 or 1)
endif
ifeq ($(WITH_OPENMP),1)
override CXXFLAGS += -fopenmp
ifneq ($(OS),Windows_NT)
ifeq ($(kokkos_gpu)/$(STATIC_LINK),1/1)
override LDLIBS += -Wl,-Bstatic -lgomp -Wl,-Bdynamic
endif
endif
override LDLIBS += -fopenmp
else ifneq ($(WITH_OPENMP),0)
$(error WITH_OPENMP must be 0 or 1)
endif

ifeq ($(OS),Windows_NT)
ifneq ($(filter %cygwin %msys,$(MAKE_HOST)),)
SHELL := /bin/sh
windows_bin = $(shell cygpath -am "$(BIN)")
else
$(error Use GNU Make in the MSYS2 MinGW64 shell for Kokkos)
endif
EXE := .exe
BUILD := build
ifneq ($(filter %cygwin %msys,$(MAKE_HOST)),)
# MINGW64 и UCRT64 используют разные типы в libstdc++. Путь библиотеки
# различает эти среды, а версия компилятора отделяет результаты после обновления GCC.
mingw_toolchain_id := $(shell { $(CXX) -print-file-name=libstdc++.a && $(CXX) -dumpfullversion; } | cksum | cut -d' ' -f1)
BUILD := build/mingw-cpu-$(mingw_toolchain_id)
endif
ifeq ($(MINGW_CUDA),1)
# Разделяем объектные файлы разных установок CUDA Toolkit.
cuda_toolkit_id := $(shell printf '%s' "$(CUDA_HOME)" | cksum | cut -d' ' -f1)
BUILD := build/mingw-cuda-sm$(CUDA_ARCH)-$(cuda_toolkit_id)-$(mingw_toolchain_id)
endif
else
EXE :=
BUILD := build/linux
ifeq ($(kokkos_gpu),1)
BUILD := $(BUILD)/cuda-sm$(CUDA_ARCH)
endif
override LDLIBS += -pthread
endif

BIN := bin
ifeq ($(MAKECMDGOALS),clean)
include utils/clean.mk
else
ifneq ($(filter clean,$(MAKECMDGOALS)),)
$(error Run make clean separately from a build)
endif
.DEFAULT_GOAL := all
ifeq ($(STATIC_LINK),1)
BUILD := $(BUILD)/static
endif
ifeq ($(WITH_OPENMP),0)
BUILD := $(BUILD)/serial
endif
BUILD_TYPE ?= $(if $(filter programs,$(MAKECMDGOALS)),Release,Default)
ifeq ($(BUILD_TYPE),Release)
build_optimization := -O3
override BUILD := $(BUILD)/Release/O3
override CPPFLAGS += -DNDEBUG
override CXXFLAGS := $(filter-out -O%,$(CXXFLAGS)) $(build_optimization)
else ifeq ($(BUILD_TYPE),Debug)
build_optimization := -O0
override BUILD := $(BUILD)/Debug
override CXXFLAGS := $(filter-out -O%,$(CXXFLAGS)) $(build_optimization) -g
else ifneq ($(BUILD_TYPE),Default)
$(error BUILD_TYPE must be Default, Debug or Release)
endif
include source/parallel/kokkos.mk
MESH_SOURCE := source/mesh
MESH_HEADERS := $(MESH_SOURCE)/coordinate.h $(MESH_SOURCE)/mesh.h $(MESH_SOURCE)/mesh_index.h source/parallel/kernel_math.h
ifeq ($(kokkos_gpu),1)
# Обычный C++ компилирует вызывающую сторону. Только модуль Kokkos
# собирается компилятором устройства; статическая сборка включает его в EXE.
compile_device_host = $(CXX) $(CPPFLAGS) $(CXXFLAGS) -DAERO_WITH_KOKKOS_GPU -c "$<" -o "$@"
override LDLIBS += "$(KOKKOS_DEVICE_LIBRARY)" $(kokkos_static_runtime)
endif
TEST := $(BUILD)/mesh_tests$(EXE)
DEBUG_TEST := $(BUILD)/mesh_tests_debug$(EXE)

.PHONY: all test debug-test cuda-check
all: $(BIN)/$(notdir $(TEST))
all: converter
.PHONY: programs
programs: converter tecplot

.PHONY: publish-programs bin-runtime
$(BIN):
	mkdir -p "$(BIN)"
bin-runtime: | $(BIN)
ifeq ($(OS)-$(STATIC_LINK),Windows_NT-0)
	@MSYS2_ARG_CONV_EXCL='*' cmd.exe /d /c 'msvs\copy_mingw_runtime.cmd' "$(windows_bin)"
endif

publish_exe = cp -f "$<" "$@"
$(BIN)/%: $(BUILD)/% publish-programs | bin-runtime
	$(publishing)
	$(publish_exe)

$(BUILD):
	mkdir -p "$(BUILD)"

$(BUILD)/mesh_tests.o $(DEBUG_TEST): qa/test_checks.h

include source/io/io.mk

$(BUILD)/mesh.o: $(MESH_SOURCE)/mesh.cpp $(MESH_SOURCE)/mesh_reader.h $(MESH_HEADERS) $(IO_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(MESH_SOURCE)/mesh.cpp -o $@

$(BUILD)/mesh_tests.o: $(MESH_SOURCE)/mesh_tests.cpp $(MESH_HEADERS) source/io/lzma_stream.h | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(MESH_SOURCE)/mesh_tests.cpp -o $@

$(TEST): $(BUILD)/mesh.o $(BUILD)/mesh_tests.o $(IO_LIBRARY)
	$(call action,Linking,$@)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

.PHONY: mesh mesh-test mesh-debug-test
mesh: $(BUILD)/mesh.o
test: mesh-test
mesh-test: $(BIN)/$(notdir $(TEST))
	$(running_test)
	./$(TEST)

$(DEBUG_TEST): $(MESH_SOURCE)/mesh.cpp $(MESH_SOURCE)/mesh_reader.h $(MESH_HEADERS) $(MESH_SOURCE)/mesh_tests.cpp source/io/lzma_stream.cpp $(IO_HEADERS) $(LZMA_OBJECTS) | $(BUILD)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) -std=c++11 -pedantic-errors -O0 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(MESH_SOURCE)/mesh.cpp $(MESH_SOURCE)/mesh_tests.cpp source/io/lzma_stream.cpp $(LZMA_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

debug-test: mesh-debug-test
mesh-debug-test: $(BIN)/$(notdir $(DEBUG_TEST))
	$(running_test)
	./$(DEBUG_TEST)

# Компиляция модуля проверяет доступ к данным через Kokkos на выбранном устройстве.
ifeq ($(kokkos_gpu),1)
cuda-check: $(KOKKOS_DEVICE_LIBRARY)
else
cuda-check:
	$(MAKE) KOKKOS_BACKEND=cuda cuda-check
endif

include utils/converter/converter.mk
test: converter-test
debug-test: converter-debug-test

include utils/tecplot/tecplot.mk
include source/parallel/mpi.mk
include utils/meshdec/meshdec.mk

include source/reconstruction/reconstruction.mk

include source/eulersolver/eulersolver.mk

-include source/cv/cv.mk
-include source/bc/bc.mk
-include source/explicit/explicit.mk
ifneq ($(and $(EXPLICIT_LIBRARY),$(REPORT_OBJECTS)),)
include source/io/report_tests.mk
endif
-include source/timestep/timestep.mk
-include source/initial/initial.mk
ifneq ($(and $(BC_LIB),$(CV_LIB),$(EBR_LIBRARY),$(EULER_LIBRARY)),)
-include source/residual/residual.mk
endif
include source/implicit/implicit.mk
ifneq ($(and $(INITIAL_LIB),$(BC_LIB),$(CV_LIB)),)
include source/io/settings_tests.mk
endif
ifneq ($(and $(RES_LIB),$(INITIAL_LIB),$(EXPLICIT_LIBRARY),$(TIMESTEP_LIBRARY)),)
include source/aero/aero.mk
endif

# Цели отладки компилируют общие параллельные циклы непосредственно из исходников.
PARALLEL_HEADER := source/parallel/parallel_for.h
$(BUILD)/mesh.o $(BUILD)/cv.o $(BUILD)/periodic.o $(BUILD)/initial.o $(BUILD)/io_fields.o $(RECONSTRUCTION_OBJECT) $(WENO_OBJECT) $(BUILD)/converter_topology_$(WITH_CGNS).o: $(PARALLEL_HEADER)
ifeq ($(kokkos_gpu),1)
gpu_objects := $(AERO_GPU) $(EBR_CUDA_OBJECT) $(RECONSTRUCTION_CUDA_TEST_OBJECT) $(EULER_CUDA_TEST_OBJECT) $(BC_GPU) $(CV_GPU) $(RES_GPU) $(EXPLICIT_GPU) $(EXPLICIT_TEST_GPU) $(TIMESTEP_GPU) $(TIMESTEP_TEST_GPU)
$(gpu_objects): source/parallel/device_api.h source/parallel/device_memory.h source/parallel/device_check.h source/parallel/device_work.h
$(RECONSTRUCTION_CUDA_TEST_OBJECT) $(EULER_CUDA_TEST_OBJECT) $(BC_GPU) $(CV_GPU) $(EXPLICIT_TEST_GPU) $(TIMESTEP_TEST_GPU): qa/device_cases.h qa/device_test_buffer.h
$(AERO_PROGRAM) $(IMPLICIT_TEST) $(RES_TEST) $(RES_DEBUG) $(TIMESTEP_TEST) $(TIMESTEP_DEBUG) $(EXPLICIT_TEST) $(EXPLICIT_DEBUG) $(EBR_TEST) $(EBR_DEBUG_TEST) $(EULER_TEST) $(EULER_DEBUG_TEST) $(BC_TEST) $(CV_TEST) $(REPORT_TEST): $(KOKKOS_DEVICE_LIBRARY)
endif
endif

# Печатаем параметры после подключения модулей. К этому моменту известны
# фактические пути к библиотекам и добавленные модулями флаги.
include utils/build_output.mk
