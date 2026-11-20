# Сборка GNU Make для C++11. Для варианта CPU достаточно компилятора C++.
# Основные проверки запускаются целями test, debug-test и cuda-check.
CXX = g++
WITH_CUDA ?= 0
CUDA_ARCH ?= 86
ifeq ($(OS)/$(WITH_CUDA),Windows_NT/1)
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
NVCCFLAGS ?= -std=c++11
WITH_OPENMP ?= 1
WITH_MPI ?= 1
WITH_KOKKOS ?= 0
build_optimization := -O2
STATIC_LINK ?= 1
CUDA_LIBDIR ?= $(if $(wildcard $(CUDA_HOME)/lib64),$(CUDA_HOME)/lib64,$(CUDA_HOME)/lib)
ifeq ($(STATIC_LINK),1)
ifeq ($(OS),Windows_NT)
override LDFLAGS += -static
else ifneq ($(filter 1,$(WITH_CUDA) $(WITH_MPI)),)
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
ifeq ($(WITH_CUDA)/$(STATIC_LINK),1/1)
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
# Ключи MSBuild должны дойти до Windows без преобразования MSYS2 в пути.
msbuild_switches := /t:;/p:;/m:;/nr:;/nologo;/v:
ifneq ($(MSYS2_ARG_CONV_EXCL),*)
export MSYS2_ARG_CONV_EXCL := $(if $(MSYS2_ARG_CONV_EXCL),$(MSYS2_ARG_CONV_EXCL);)$(msbuild_switches)
endif
msbuild_bin = $(shell cygpath -am "$(BIN)")
else
SHELL := cmd.exe
.SHELLFLAGS := /C
USE_CMD := 1
msbuild_bin = $(abspath $(BIN))
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
# Вычисляем отдельный каталог сборки для каждого набора библиотек NVRTC.
cuda_toolkit_id := $(shell printf '%s' "$(CUDA_HOME)" | cksum | cut -d' ' -f1)
BUILD := build/mingw-cuda-sm$(CUDA_ARCH)-$(cuda_toolkit_id)-$(mingw_toolchain_id)
endif
else
EXE :=
BUILD := build/linux
ifeq ($(WITH_CUDA),1)
BUILD := $(BUILD)/cuda-sm$(CUDA_ARCH)
endif
override LDLIBS += -pthread
endif

BIN := bin
ifeq ($(WITH_KOKKOS),1)
# Изолированные программы позволяют измерять исходный и перенесённый варианты на одной машине.
BIN := bin/kokkos
ifeq ($(USE_CMD),1)
$(error Use the MSYS2 MinGW64 shell for the C++11 Kokkos build on Windows)
endif
endif
ifeq ($(MAKECMDGOALS),clean)
include utils/clean.mk
else
ifneq ($(filter clean,$(MAKECMDGOALS)),)
$(error Run make clean separately from a build)
endif
ifeq ($(OS)/$(WITH_CUDA)/$(USE_CMD)/$(MAKECMDGOALS),Windows_NT/1/1/programs)
# На Windows nvcc использует ABI MSVC. Здесь собираются программы и их библиотеки.
# Тестовые проекты запускаются отдельными целями.
MSBUILD ?= MSBuild.exe
CUDA_ARCH ?= 86
.PHONY: programs
programs:
	"$(MSBUILD)" msvs/aero.sln "/t:aero;mesh_converter;fields_to_tecplot;meshdec" /m:1 /nr:false /nologo /v:minimal /p:Platform=x64 /p:Configuration=Release /p:WithCuda=true /p:CudaArchitecture=$(CUDA_ARCH) /p:WithOpenMP=$(if $(filter 1,$(WITH_OPENMP)),true,false) $(if $(WITH_CGNS),/p:WithCGNS=$(if $(filter 1,$(WITH_CGNS)),true,false)) /p:WithMPI=$(if $(filter 1,$(WITH_MPI)),true,false) "/p:AeroBin=$(msbuild_bin)" $(MSBUILD_CUDA_FLAGS)
else
.DEFAULT_GOAL := all
ifeq ($(STATIC_LINK),1)
BUILD := $(BUILD)/static
endif
ifeq ($(WITH_OPENMP),0)
BUILD := $(BUILD)/serial
endif
ifneq ($(filter programs,$(MAKECMDGOALS)),)
build_optimization := -O3
override BUILD := $(BUILD)/Release/O3
override CPPFLAGS += -DNDEBUG
override CXXFLAGS := $(filter-out -O%,$(CXXFLAGS)) $(build_optimization)
override NVCCFLAGS := $(filter-out -O%,$(NVCCFLAGS)) $(build_optimization) -DNDEBUG
endif
ifeq ($(WITH_KOKKOS),1)
include source/parallel/kokkos.mk
endif
MESH_SOURCE := source/mesh
MESH_HEADERS := $(MESH_SOURCE)/mesh.h $(MESH_SOURCE)/mesh_index.h source/parallel/kernel_math.h
ifeq ($(MINGW_CUDA),1)
include source/cuda/mingw.mk
else
compile_cuda = $(NVCC) $(NVCCFLAGS) -arch=sm_$(CUDA_ARCH) -c $< -o $@
endif
compile_solver_cuda = $(compile_cuda)
ifeq ($(WITH_KOKKOS)$(WITH_CUDA),11)
# Файлы интерфейса принимают представления солвера. GPU-обходы компилируются один раз в модуле Kokkos.
compile_solver_cuda = $(CXX) $(CPPFLAGS) $(CXXFLAGS) -DAERO_WITH_CUDA -DAERO_WITH_KOKKOS_GPU -isystem "$(CUDA_HOME)/include" -x c++ -c "$<" -o "$@"
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
ifeq ($(USE_CMD),1)
	mkdir "$(BIN)"
else
	mkdir -p "$(BIN)"
endif
bin-runtime: | $(BIN)
ifeq ($(OS)-$(STATIC_LINK),Windows_NT-0)
ifeq ($(USE_CMD),1)
	@msvs\copy_mingw_runtime.cmd "$(BIN)"
else
	@MSYS2_ARG_CONV_EXCL='*' cmd.exe /d /c 'msvs\copy_mingw_runtime.cmd' "$(msbuild_bin)"
endif
endif

ifeq ($(USE_CMD),1)
publish_exe = copy /Y "$(subst /,\,$<)" "$(subst /,\,$@)" >nul
else
publish_exe = cp -f "$<" "$@"
endif
$(BIN)/%: $(BUILD)/% publish-programs | bin-runtime
	$(publish_exe)

$(BUILD):
ifeq ($(USE_CMD),1)
	if not exist "$(BUILD)" mkdir "$(BUILD)"
else
	mkdir -p "$(BUILD)"
endif

$(BUILD)/mesh_tests.o $(DEBUG_TEST): qa/test_checks.h

include source/io/io.mk

$(BUILD)/mesh.o: $(MESH_SOURCE)/mesh.cpp $(MESH_SOURCE)/mesh_reader.h $(MESH_HEADERS) $(IO_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(MESH_SOURCE)/mesh.cpp -o $@

$(BUILD)/mesh_tests.o: $(MESH_SOURCE)/mesh_tests.cpp $(MESH_HEADERS) source/io/lzma_stream.h | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(MESH_SOURCE)/mesh_tests.cpp -o $@

$(TEST): $(BUILD)/mesh.o $(BUILD)/mesh_tests.o $(IO_LIBRARY)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

test: $(BIN)/$(notdir $(TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(TEST))
else
	./$(TEST)
endif

$(DEBUG_TEST): $(MESH_SOURCE)/mesh.cpp $(MESH_SOURCE)/mesh_reader.h $(MESH_HEADERS) $(MESH_SOURCE)/mesh_tests.cpp source/io/lzma_stream.cpp $(IO_HEADERS) $(LZMA_OBJECTS) | $(BUILD)
	$(CXX) $(CPPFLAGS) -std=c++11 -pedantic-errors -O0 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(MESH_SOURCE)/mesh.cpp $(MESH_SOURCE)/mesh_tests.cpp source/io/lzma_stream.cpp $(LZMA_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

debug-test: $(BIN)/$(notdir $(DEBUG_TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(DEBUG_TEST))
else
	./$(DEBUG_TEST)
endif

# Ядро компилирует методы доступа к сетке в режиме CUDA без запуска на устройстве.
# Через NVCCFLAGS можно указать компилятор хоста, подходящий для установленного CUDA.
cuda-check: | $(BUILD)
ifeq ($(MINGW_CUDA),1)
cuda-check: $(BUILD)/mesh_cuda_probe.ptx
$(BUILD)/mesh_cuda_probe.ptx: $(MESH_SOURCE)/mesh_probe.cuh $(MESH_HEADERS) $(ptx_compiler) | $(BUILD)
	PATH="$(CUDA_HOME)/bin/x64:$(CUDA_HOME)/bin:$$PATH" "$(ptx_compiler)" "$<" "$(CUDA_ARCH)" "$@"
else ifeq ($(OS)/$(USE_CMD),Windows_NT/)
	$(MAKE) WITH_CUDA=1 cuda-check
else
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -x cu -c $(MESH_SOURCE)/mesh_tests.cpp -o $(BUILD)/mesh_cuda_probe.obj
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
ifneq ($(and $(EXPLICIT_LIBRARY),$(FLOWREPORT_OBJECTS)),)
include source/io/flowreport_tests.mk
endif
-include source/timestep/timestep.mk
-include source/initial/initial.mk
ifneq ($(and $(BC_LIB),$(CV_LIB),$(EBR_LIBRARY),$(EULER_LIBRARY)),)
-include source/residual/residual.mk
endif
ifneq ($(and $(INITIAL_LIB),$(BC_LIB),$(CV_LIB)),)
include source/io/settings_tests.mk
endif
ifneq ($(and $(RES_LIB),$(INITIAL_LIB),$(EXPLICIT_LIBRARY),$(TIMESTEP_LIBRARY)),)
include source/aero/aero.mk
endif

# Цели отладки компилируют общие параллельные циклы непосредственно из исходников.
PARALLEL_HEADER := source/parallel/parallel_for.h
$(BUILD)/mesh.o $(BUILD)/cv.o $(BUILD)/periodic.o $(BUILD)/initial.o $(BUILD)/io_fields.o $(RECONSTRUCTION_OBJECT) $(WENO_OBJECT) $(BUILD)/converter_topology_$(WITH_CGNS).o: $(PARALLEL_HEADER)
ifneq ($(filter 1,$(WITH_CUDA)),)
cuda_objects := $(AERO_GPU) $(EBR_CUDA_OBJECT) $(RECONSTRUCTION_CUDA_TEST_OBJECT) $(EULER_CUDA_TEST_OBJECT) $(BC_GPU) $(CV_GPU) $(RES_GPU) $(EXPLICIT_GPU) $(EXPLICIT_TEST_GPU) $(TIMESTEP_GPU) $(TIMESTEP_TEST_GPU)
$(cuda_objects): source/cuda/check.h source/cuda/runtime.h source/parallel/kernel_math.h source/mesh/mesh_index.h
ifeq ($(WITH_KOKKOS),1)
$(AERO_GPU) $(TIMESTEP_GPU) $(EXPLICIT_GPU) $(RES_GPU): source/parallel/device_work.h
ifeq ($(STATIC_LINK),1)
# У статического теста код ядра находится в EXE. Обновление архива требует повторной линковки.
$(RES_TEST) $(RES_DEBUG) $(TIMESTEP_TEST) $(TIMESTEP_DEBUG) $(EXPLICIT_TEST) $(EXPLICIT_DEBUG) $(EBR_TEST) $(EBR_DEBUG_TEST) $(EULER_TEST) $(EULER_DEBUG_TEST): $(KOKKOS_DEVICE_LIBRARY)
endif
endif
$(AERO_PROGRAM) $(RES_OBJ) $(EXPLICIT_OBJ) $(EBR_TEST) $(RES_TEST) $(REPORT_TEST): source/cuda/check.h source/cuda/runtime.h
$(CV_GPU) $(BC_GPU) $(TIMESTEP_TEST_GPU) $(EBR_TEST) $(EBR_DEBUG_TEST) $(RES_TEST) $(RES_DEBUG): qa/cuda_test_buffer.h
ifeq ($(MINGW_CUDA),1)
$(cuda_objects): source/cuda/mingw.mk $(ptx_compiler) $(CUDA_RUNTIME_LINK)
endif
endif
endif
endif
