# Собирает реконструкцию и отдельные программы проверки геометрии, схем и GPU.
RECONSTRUCTION_SOURCE := source/reconstruction
WITH_CUDA ?= 0
EBR_TAG := cuda$(WITH_CUDA)_omp$(WITH_OPENMP)
EBR_FLAGS :=
EBR_LIBS :=
RECONSTRUCTION_HEADERS := $(RECONSTRUCTION_SOURCE)/reconstruction.h $(RECONSTRUCTION_SOURCE)/ebr.h $(RECONSTRUCTION_SOURCE)/weno.h
RECONSTRUCTION_CPP := $(addprefix $(RECONSTRUCTION_SOURCE)/,reconstruction.cpp ray_geometry.cpp stencil_builder.cpp)
RECONSTRUCTION_OBJECTS := $(patsubst $(RECONSTRUCTION_SOURCE)/%.cpp,$(BUILD)/%_$(EBR_TAG).o,$(RECONSTRUCTION_CPP))
RECONSTRUCTION_HEADERS += $(RECONSTRUCTION_SOURCE)/scheme.h $(RECONSTRUCTION_SOURCE)/stencil_builder.h
RECONSTRUCTION_HEADERS += $(RECONSTRUCTION_SOURCE)/parallel_stencil.h source/parallel/parallel_for.h source/parallel/kernel_math.h
EBR_LIBRARY := $(BUILD)/libaero_reconstruction_$(EBR_TAG).a
EBR_TEST := $(BUILD)/reconstruction_tests_$(EBR_TAG)$(EXE)
EBR_DEBUG_TEST := $(BUILD)/reconstruction_tests_debug_$(EBR_TAG)$(EXE)
EBR_CUDA_OBJECT :=
RECONSTRUCTION_CUDA_TEST_OBJECT :=
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 86

ifeq ($(WITH_CUDA),1)
ifeq ($(USE_CMD),1)
$(error Windows CUDA uses the MSVC ABI: use make reconstruction-cuda-test (see source/reconstruction/README.txt))
else
EBR_FLAGS += -DAERO_WITH_CUDA -isystem "$(CUDA_HOME)/include"
EBR_LIBS += $(CUDA_RUNTIME_LINK)
EBR_CUDA_OBJECT := $(BUILD)/ebr_cuda.o
RECONSTRUCTION_CUDA_TEST_OBJECT := $(BUILD)/reconstruction_cuda_tests.o
$(RECONSTRUCTION_CUDA_TEST_OBJECT): $(RECONSTRUCTION_SOURCE)/reconstruction_cuda_tests.cu $(RECONSTRUCTION_SOURCE)/reconstruction_cuda_tests.cuh $(RECONSTRUCTION_HEADERS) source/cuda/device_scope.h source/cuda/runtime.h | $(BUILD)
	$(compile_cuda)
$(EBR_CUDA_OBJECT): $(RECONSTRUCTION_SOURCE)/reconstruction_cuda.cu $(RECONSTRUCTION_HEADERS) source/cuda/device_scope.h source/cuda/runtime.h | $(BUILD)
ifeq ($(MINGW_CUDA),1)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -x c++ -c $< -o $@
else
	$(compile_cuda)
endif
endif
endif

$(RECONSTRUCTION_OBJECTS): $(BUILD)/%_$(EBR_TAG).o: $(RECONSTRUCTION_SOURCE)/%.cpp $(RECONSTRUCTION_HEADERS) $(MESH_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EBR_FLAGS) -c $< -o $@
$(EBR_LIBRARY): $(RECONSTRUCTION_OBJECTS) $(EBR_CUDA_OBJECT)
	$(AR) rcs $@ $^
$(EBR_TEST): $(RECONSTRUCTION_SOURCE)/reconstruction_tests.cpp $(RECONSTRUCTION_HEADERS) $(EBR_LIBRARY) $(BUILD)/mesh.o $(IO_LIBRARY) $(RECONSTRUCTION_CUDA_TEST_OBJECT)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EBR_FLAGS) $< $(EBR_LIBRARY) $(BUILD)/mesh.o $(IO_LIBRARY) $(RECONSTRUCTION_CUDA_TEST_OBJECT) $(LDFLAGS) $(LDLIBS) $(EBR_LIBS) -o $@
$(EBR_DEBUG_TEST): $(RECONSTRUCTION_CPP) $(RECONSTRUCTION_SOURCE)/reconstruction_tests.cpp $(RECONSTRUCTION_HEADERS) $(MESH_SOURCE)/mesh.cpp $(MESH_HEADERS) source/io/lzma_stream.cpp $(IO_HEADERS) $(LZMA_OBJECTS) $(EBR_CUDA_OBJECT) $(RECONSTRUCTION_CUDA_TEST_OBJECT) | $(BUILD)
	$(CXX) $(CPPFLAGS) -std=c++11 -pedantic-errors -O0 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(EBR_FLAGS) $(RECONSTRUCTION_CPP) $(RECONSTRUCTION_SOURCE)/reconstruction_tests.cpp $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp $(LZMA_OBJECTS) $(EBR_CUDA_OBJECT) $(RECONSTRUCTION_CUDA_TEST_OBJECT) $(LDFLAGS) $(LDLIBS) $(EBR_LIBS) -o $@

.PHONY: reconstruction reconstruction-test reconstruction-debug-test reconstruction-cuda-test
reconstruction: $(EBR_LIBRARY)
reconstruction-test: $(BIN)/$(notdir $(EBR_TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(EBR_TEST))
else
	./$(EBR_TEST)
endif
reconstruction-debug-test: $(BIN)/$(notdir $(EBR_DEBUG_TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(EBR_DEBUG_TEST))
else
	./$(EBR_DEBUG_TEST)
endif

# На Windows все объекты теста CUDA собираем в проектах Visual Studio с MSVC.
# Это сохраняет совместимость представления объектов C++ при линковке.
MSBUILD ?= MSBuild.exe
MSBUILD_CUDA_FLAGS ?=
ifeq ($(USE_CMD),1)
reconstruction-cuda-test:
	"$(MSBUILD)" msvs/aero.sln /t:reconstruction_tests /m:1 /nr:false /nologo /v:minimal /p:Platform=x64 /p:Configuration=Release /p:WithCuda=true /p:RunReconstructionTests=true /p:RequireCudaTests=true $(MSBUILD_CUDA_FLAGS)
else
reconstruction-cuda-test:
	$(MAKE) WITH_CUDA=1 $(BIN)/reconstruction_tests_cuda1_omp$(WITH_OPENMP)$(EXE)
	./$(BIN)/reconstruction_tests_cuda1_omp$(WITH_OPENMP)$(EXE) --require-cuda
endif
all: $(BIN)/$(notdir $(EBR_TEST))
test: reconstruction-test
debug-test: reconstruction-debug-test
$(EBR_TEST) $(EBR_DEBUG_TEST): qa/test_checks.h
