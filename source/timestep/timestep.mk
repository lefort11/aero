# Критерий CFL собирается с нужным вариантом OpenMP и, по запросу, с ядрами редукции для CUDA.
TIMESTEP_DIR := source/timestep
WITH_CUDA ?= 0
TIMESTEP_TAG := cuda$(WITH_CUDA)_omp$(WITH_OPENMP)
TIMESTEP_SRC := $(addprefix $(TIMESTEP_DIR)/,timestep.cpp)
TIMESTEP_OBJ := $(patsubst $(TIMESTEP_DIR)/%.cpp,$(BUILD)/%_$(TIMESTEP_TAG).o,$(TIMESTEP_SRC))
TIMESTEP_HEADERS := $(TIMESTEP_DIR)/timestep.h $(TIMESTEP_DIR)/cfl_detail.h
TIMESTEP_HEADERS += source/parallel/kernel_math.h
TIMESTEP_LIBRARY := $(BUILD)/libaero_timestep_$(TIMESTEP_TAG).a
TIMESTEP_TEST := $(BUILD)/timestep_tests_$(TIMESTEP_TAG)$(EXE)
TIMESTEP_DEBUG := $(BUILD)/timestep_tests_debug_$(TIMESTEP_TAG)$(EXE)
TIMESTEP_TEST_HEADERS := $(EXPLICIT_HEADERS) $(BC_HEADERS) source/eulersolver/lax_friedrichs.h
TIMESTEP_DEPS := $(BC_LIB) $(CV_LIB) $(BUILD)/mesh.o $(IO_LIBRARY) $(EXPLICIT_LIBRARY)
TIMESTEP_DEBUG_SRC := $(BC_DIR)/periodic.cpp $(CV_DIR)/cv.cpp $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp $(EXPLICIT_SRC)
TIMESTEP_HEADERS += $(BC_DIR)/periodic.h $(CV_HEADERS) $(RECONSTRUCTION_HEADERS) $(BC_HEADERS)
ifeq ($(WITH_CUDA),1)
TIMESTEP_FLAGS += -DAERO_WITH_CUDA -isystem "$(CUDA_HOME)/include"
TIMESTEP_LIBS := $(CUDA_RUNTIME_LINK)
TIMESTEP_GPU := $(BUILD)/timestep_kernel.o
TIMESTEP_TEST_GPU := $(BUILD)/timestep_cuda_tests.o
$(TIMESTEP_GPU): $(TIMESTEP_DIR)/timestep_cuda.cu $(TIMESTEP_DIR)/timestep_cuda.cuh $(TIMESTEP_HEADERS) source/timestep/cfl_block_min.cuh source/cuda/device_scope.h source/cuda/runtime.h | $(BUILD)
	$(compile_cuda)
$(TIMESTEP_TEST_GPU): $(TIMESTEP_DIR)/timestep_cuda_tests.cu $(TIMESTEP_DIR)/timestep_cuda_tests.cuh $(TIMESTEP_TEST_HEADERS) $(TIMESTEP_HEADERS) | $(BUILD)
	$(compile_cuda)
endif
$(TIMESTEP_OBJ): $(BUILD)/%_$(TIMESTEP_TAG).o: $(TIMESTEP_DIR)/%.cpp $(TIMESTEP_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TIMESTEP_FLAGS) -c $< -o $@
$(TIMESTEP_LIBRARY): $(TIMESTEP_OBJ) $(TIMESTEP_GPU)
	$(AR) rcs $@ $^
$(TIMESTEP_TEST): $(TIMESTEP_DIR)/timestep_tests.cpp $(TIMESTEP_TEST_HEADERS) $(TIMESTEP_HEADERS) $(TIMESTEP_LIBRARY) $(TIMESTEP_DEPS) $(TIMESTEP_TEST_GPU)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TIMESTEP_FLAGS) $< $(TIMESTEP_LIBRARY) $(TIMESTEP_DEPS) $(TIMESTEP_TEST_GPU) $(LDFLAGS) $(LDLIBS) $(TIMESTEP_LIBS) -o $@
$(TIMESTEP_DEBUG): $(TIMESTEP_DIR)/timestep_tests.cpp $(TIMESTEP_TEST_HEADERS) $(TIMESTEP_SRC) $(TIMESTEP_HEADERS) $(TIMESTEP_GPU) $(TIMESTEP_TEST_GPU) $(TIMESTEP_DEBUG_SRC) $(IO_HEADERS) $(LZMA_OBJECTS) $(EXPLICIT_GPU) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -g -O0 -D_GLIBCXX_DEBUG $(TIMESTEP_FLAGS) $(filter %.cpp %.o,$^) $(LDFLAGS) $(LDLIBS) $(TIMESTEP_LIBS) -o $@
.PHONY: timestep timestep-test timestep-debug-test timestep-cuda-test
all: timestep
test: timestep-test
debug-test: timestep-debug-test
timestep: $(TIMESTEP_LIBRARY)
timestep-test: $(BIN)/$(notdir $(TIMESTEP_TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(TIMESTEP_TEST))
else
	./$(TIMESTEP_TEST)
endif
timestep-debug-test: $(BIN)/$(notdir $(TIMESTEP_DEBUG))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(TIMESTEP_DEBUG))
else
	./$(TIMESTEP_DEBUG)
endif
timestep-cuda-test:
ifeq ($(USE_CMD),1)
	"$(MSBUILD)" msvs/aero.sln /t:timestep_tests /m:1 /nr:false /nologo /v:minimal /p:Platform=x64 /p:Configuration=Release /p:WithCuda=true /p:RequireCudaTests=true $(MSBUILD_CUDA_FLAGS)
else
	$(MAKE) WITH_CUDA=1 $(BIN)/timestep_tests_cuda1_omp$(WITH_OPENMP)$(EXE)
	./$(BIN)/timestep_tests_cuda1_omp$(WITH_OPENMP)$(EXE) --require-cuda
endif
$(TIMESTEP_TEST) $(TIMESTEP_DEBUG): qa/test_checks.h
