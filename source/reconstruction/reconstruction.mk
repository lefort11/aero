# Собирает реконструкцию и отдельные программы проверки геометрии, схем и GPU.
RECONSTRUCTION_SOURCE := source/reconstruction
EBR_TAG := cuda$(kokkos_gpu)_omp$(WITH_OPENMP)
EBR_FLAGS :=
EBR_LIBS :=
RECONSTRUCTION_HEADERS := $(RECONSTRUCTION_SOURCE)/reconstruction.h $(RECONSTRUCTION_SOURCE)/ebr.h $(RECONSTRUCTION_SOURCE)/weno.h $(RECONSTRUCTION_SOURCE)/mp5.h $(RECONSTRUCTION_SOURCE)/mpweno5.h $(RECONSTRUCTION_SOURCE)/mp6_hybrid.h
RECONSTRUCTION_CPP := $(addprefix $(RECONSTRUCTION_SOURCE)/,reconstruction.cpp ray_geometry.cpp stencil_builder.cpp curved_stencil.cpp mp5.cpp)
RECONSTRUCTION_OBJECTS := $(patsubst $(RECONSTRUCTION_SOURCE)/%.cpp,$(BUILD)/%_$(EBR_TAG).o,$(RECONSTRUCTION_CPP))
RECONSTRUCTION_HEADERS += $(RECONSTRUCTION_SOURCE)/scheme.h $(RECONSTRUCTION_SOURCE)/stencil_builder.h $(RECONSTRUCTION_SOURCE)/curved_stencil.h
RECONSTRUCTION_HEADERS += $(RECONSTRUCTION_SOURCE)/parallel_stencil.h source/parallel/parallel_for.h source/parallel/kernel_math.h
EBR_LIBRARY := $(BUILD)/libaero_reconstruction_$(EBR_TAG).a
EBR_TEST := $(BUILD)/reconstruction_tests_$(EBR_TAG)$(EXE)
EBR_DEBUG_TEST := $(BUILD)/reconstruction_tests_debug_$(EBR_TAG)$(EXE)
EBR_CUDA_OBJECT :=
RECONSTRUCTION_CUDA_TEST_OBJECT :=
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 86

ifeq ($(kokkos_gpu),1)
EBR_FLAGS += -DAERO_WITH_KOKKOS_GPU
EBR_CUDA_OBJECT := $(BUILD)/ebr_cuda.o
RECONSTRUCTION_CUDA_TEST_OBJECT := $(BUILD)/reconstruction_cuda_tests.o
$(RECONSTRUCTION_CUDA_TEST_OBJECT): $(RECONSTRUCTION_SOURCE)/reconstruction_gpu_tests.cpp $(RECONSTRUCTION_HEADERS) source/parallel/device_scope.h source/parallel/device_memory.h | $(BUILD)
	$(compiling)
	$(compile_device_host)
$(EBR_CUDA_OBJECT): $(RECONSTRUCTION_SOURCE)/reconstruction_gpu.cpp $(RECONSTRUCTION_SOURCE)/device_stencil.h source/parallel/device_arrays.h source/parallel/device_check.h $(RECONSTRUCTION_HEADERS) source/parallel/device_scope.h source/parallel/device_memory.h | $(BUILD)
	$(compiling)
	$(compile_device_host)
endif

$(RECONSTRUCTION_OBJECTS): $(BUILD)/%_$(EBR_TAG).o: $(RECONSTRUCTION_SOURCE)/%.cpp $(RECONSTRUCTION_HEADERS) $(MESH_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EBR_FLAGS) -c $< -o $@
$(EBR_LIBRARY): $(RECONSTRUCTION_OBJECTS) $(EBR_CUDA_OBJECT)
	$(archiving)
	$(AR) rcs $@ $^
$(EBR_TEST): $(RECONSTRUCTION_SOURCE)/reconstruction_tests.cpp $(RECONSTRUCTION_HEADERS) $(EBR_LIBRARY) $(BUILD)/mesh.o $(IO_LIBRARY) $(RECONSTRUCTION_CUDA_TEST_OBJECT)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EBR_FLAGS) $< $(EBR_LIBRARY) $(BUILD)/mesh.o $(IO_LIBRARY) $(RECONSTRUCTION_CUDA_TEST_OBJECT) $(LDFLAGS) $(LDLIBS) $(EBR_LIBS) -o $@
$(EBR_DEBUG_TEST): $(RECONSTRUCTION_CPP) $(RECONSTRUCTION_SOURCE)/reconstruction_tests.cpp $(RECONSTRUCTION_HEADERS) $(MESH_SOURCE)/mesh.cpp $(MESH_HEADERS) source/io/lzma_stream.cpp $(IO_HEADERS) $(LZMA_OBJECTS) $(EBR_CUDA_OBJECT) $(RECONSTRUCTION_CUDA_TEST_OBJECT) | $(BUILD)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) -std=c++11 -pedantic-errors -O0 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(EBR_FLAGS) $(RECONSTRUCTION_CPP) $(RECONSTRUCTION_SOURCE)/reconstruction_tests.cpp $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp $(LZMA_OBJECTS) $(EBR_CUDA_OBJECT) $(RECONSTRUCTION_CUDA_TEST_OBJECT) $(LDFLAGS) $(LDLIBS) $(EBR_LIBS) -o $@

.PHONY: reconstruction reconstruction-test reconstruction-debug-test reconstruction-cuda-test
reconstruction: $(EBR_LIBRARY)
reconstruction-test: $(BIN)/$(notdir $(EBR_TEST))
	$(running_test)
	"$(EBR_TEST)"
reconstruction-debug-test: $(BIN)/$(notdir $(EBR_DEBUG_TEST))
	$(running_test)
	"$(EBR_DEBUG_TEST)"

reconstruction-cuda-test:
	$(MAKE) KOKKOS_BACKEND=cuda $(BIN)/reconstruction_tests_cuda1_omp$(WITH_OPENMP)$(EXE)
	$(running_test)
	"$(BIN)/reconstruction_tests_cuda1_omp$(WITH_OPENMP)$(EXE)" --require-cuda
all: $(BIN)/$(notdir $(EBR_TEST))
test: reconstruction-test
debug-test: reconstruction-debug-test
$(EBR_TEST) $(EBR_DEBUG_TEST): tests/test_checks.h
