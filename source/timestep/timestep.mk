# Критерий CFL собирается с нужным вариантом OpenMP и, по запросу, с ядрами редукции для CUDA.
TIMESTEP_DIR := source/timestep
TIMESTEP_TAG := cuda$(kokkos_gpu)_omp$(WITH_OPENMP)
TIMESTEP_SRC := $(addprefix $(TIMESTEP_DIR)/,timestep.cpp)
TIMESTEP_OBJ := $(patsubst $(TIMESTEP_DIR)/%.cpp,$(BUILD)/%_$(TIMESTEP_TAG).o,$(TIMESTEP_SRC))
TIMESTEP_HEADERS := $(TIMESTEP_DIR)/stabilization.h $(TIMESTEP_DIR)/timestep.h $(TIMESTEP_DIR)/cfl_detail.h $(TIMESTEP_DIR)/adaptive_step.h $(TIMESTEP_DIR)/local_rate_limit.h $(TIMESTEP_DIR)/local_step_mode.h
TIMESTEP_HEADERS += source/parallel/kernel_math.h
TIMESTEP_LIBRARY := $(BUILD)/libaero_timestep_$(TIMESTEP_TAG).a
TIMESTEP_TEST := $(BUILD)/timestep_tests_$(TIMESTEP_TAG)$(EXE)
TIMESTEP_DEBUG := $(BUILD)/timestep_tests_debug_$(TIMESTEP_TAG)$(EXE)
TIMESTEP_TEST_HEADERS := $(EXPLICIT_HEADERS) $(BC_HEADERS) source/eulersolver/lax_friedrichs.h
TIMESTEP_DEPS := $(BC_LIB) $(CV_LIB) $(BUILD)/mesh.o $(IO_LIBRARY) $(EXPLICIT_LIBRARY)
TIMESTEP_DEBUG_SRC := $(BC_DIR)/periodic.cpp $(CV_DIR)/cv.cpp $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp $(EXPLICIT_SRC)
TIMESTEP_HEADERS += $(BC_DIR)/periodic.h $(CV_HEADERS) $(RECONSTRUCTION_HEADERS) $(BC_HEADERS)
ifeq ($(kokkos_gpu),1)
TIMESTEP_FLAGS += -DAERO_WITH_KOKKOS_GPU
TIMESTEP_GPU := $(BUILD)/timestep_kernel.o
TIMESTEP_TEST_GPU := $(BUILD)/timestep_cuda_tests.o
$(TIMESTEP_GPU): $(TIMESTEP_DIR)/timestep_gpu.cpp $(TIMESTEP_HEADERS) source/parallel/device_arrays.h source/parallel/device_scope.h source/parallel/device_memory.h | $(BUILD)
	$(compiling)
	$(compile_device_host)
$(TIMESTEP_TEST_GPU): $(TIMESTEP_DIR)/timestep_gpu_tests.cpp $(TIMESTEP_TEST_HEADERS) $(TIMESTEP_HEADERS) | $(BUILD)
	$(compiling)
	$(compile_device_host)
endif
$(TIMESTEP_OBJ): $(BUILD)/%_$(TIMESTEP_TAG).o: $(TIMESTEP_DIR)/%.cpp $(TIMESTEP_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TIMESTEP_FLAGS) -c $< -o $@
$(TIMESTEP_LIBRARY): $(TIMESTEP_OBJ) $(TIMESTEP_GPU)
	$(archiving)
	$(AR) rcs $@ $^
$(TIMESTEP_TEST): $(TIMESTEP_DIR)/timestep_tests.cpp $(TIMESTEP_TEST_HEADERS) $(TIMESTEP_HEADERS) $(TIMESTEP_LIBRARY) $(TIMESTEP_DEPS) $(TIMESTEP_TEST_GPU)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TIMESTEP_FLAGS) $< $(TIMESTEP_LIBRARY) $(TIMESTEP_DEPS) $(TIMESTEP_TEST_GPU) $(LDFLAGS) $(LDLIBS) $(TIMESTEP_LIBS) -o $@
$(TIMESTEP_DEBUG): $(TIMESTEP_DIR)/timestep_tests.cpp $(TIMESTEP_TEST_HEADERS) $(TIMESTEP_SRC) $(TIMESTEP_HEADERS) $(TIMESTEP_GPU) $(TIMESTEP_TEST_GPU) $(TIMESTEP_DEBUG_SRC) $(IO_HEADERS) $(LZMA_OBJECTS) $(EXPLICIT_GPU) | $(BUILD)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -g -O0 -D_GLIBCXX_DEBUG $(TIMESTEP_FLAGS) $(filter %.cpp %.o,$^) $(LDFLAGS) $(LDLIBS) $(TIMESTEP_LIBS) -o $@
.PHONY: timestep timestep-test timestep-debug-test timestep-cuda-test
all: timestep
test: timestep-test
debug-test: timestep-debug-test
timestep: $(TIMESTEP_LIBRARY)
timestep-test: $(BIN)/$(notdir $(TIMESTEP_TEST))
	$(running_test)
	./$(TIMESTEP_TEST)
timestep-debug-test: $(BIN)/$(notdir $(TIMESTEP_DEBUG))
	$(running_test)
	./$(TIMESTEP_DEBUG)
timestep-cuda-test:
	$(MAKE) KOKKOS_BACKEND=cuda $(BIN)/timestep_tests_cuda1_omp$(WITH_OPENMP)$(EXE)
	$(running_test)
	./$(BIN)/timestep_tests_cuda1_omp$(WITH_OPENMP)$(EXE) --require-cuda
$(TIMESTEP_TEST) $(TIMESTEP_DEBUG): tests/test_checks.h
