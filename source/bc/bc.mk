# Проверки границ используют ту же реализацию потока и ту же геометрию площадок, что и солвер.
BC_DIR := source/bc
BC_HEADERS := source/parallel/kernel_math.h $(BC_DIR)/bc.h source/eulersolver/euler.h source/io/freestream.h
BC_TEST_HEADERS := $(BC_HEADERS) $(BC_DIR)/bc_test_case.h
BC_LIB := $(BUILD)/libaero_bc.a
PERIODIC_HEADERS := $(BC_DIR)/periodic.h $(CV_HEADERS) $(RECONSTRUCTION_HEADERS)
PERIODIC_SOURCES := $(BC_DIR)/periodic.cpp $(BC_DIR)/periodic_stencil.cpp
PERIODIC_OBJECTS := $(BUILD)/periodic.o $(BUILD)/periodic_stencil.o
$(PERIODIC_OBJECTS): $(BUILD)/%.o: $(BC_DIR)/%.cpp $(PERIODIC_HEADERS) $(BC_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
BC_DEPS := $(CV_LIB) $(EBR_LIBRARY) $(BUILD)/mesh.o $(IO_LIBRARY)
BC_TEST := $(BUILD)/bc_tests_cuda$(kokkos_gpu)$(EXE)
BC_DEBUG := $(BUILD)/bc_tests_debug_cuda$(kokkos_gpu)$(EXE)

ifeq ($(kokkos_gpu),1)
BC_GPU := $(BUILD)/bc_cuda_tests.o
BC_FLAGS := -DAERO_WITH_KOKKOS_GPU
$(BC_GPU): $(BC_DIR)/bc_gpu_tests.cpp $(BC_TEST_HEADERS) $(PERIODIC_HEADERS) | $(BUILD)
	$(compiling)
	$(compile_device_host)
endif
$(BUILD)/bc.o: $(BC_DIR)/bc.cpp $(BC_HEADERS) source/io/settings_text.h | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BC_LIB): $(BUILD)/bc.o $(PERIODIC_OBJECTS)
	$(archiving)
	$(AR) rcs $@ $^
$(BC_TEST): $(BC_DIR)/bc_tests.cpp $(BC_DIR)/periodic_tests.cpp $(PERIODIC_HEADERS) $(BC_TEST_HEADERS) $(BC_LIB) $(BC_DEPS) $(BC_GPU)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(BC_FLAGS) $(filter %.cpp,$^) $(BC_LIB) $(BC_DEPS) $(BC_GPU) $(LDFLAGS) $(EBR_LIBS) $(BC_LIBS) $(LDLIBS) -o $@
$(BC_DEBUG): $(BC_DIR)/bc_tests.cpp $(BC_DIR)/periodic_tests.cpp $(BC_DIR)/bc.cpp $(PERIODIC_SOURCES) $(CV_DIR)/cv.cpp $(RECONSTRUCTION_CPP) $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp $(LZMA_OBJECTS) $(PERIODIC_HEADERS) $(BC_TEST_HEADERS) $(IO_HEADERS) $(BC_GPU) $(EBR_CUDA_OBJECT)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -g -O0 -D_GLIBCXX_DEBUG $(EBR_FLAGS) $(BC_FLAGS) $(filter %.cpp %.o,$^) $(LDFLAGS) $(EBR_LIBS) $(BC_LIBS) $(LDLIBS) -o $@

.PHONY: bc bc-test bc-debug-test bc-cuda-test
bc: $(BC_LIB)
all: bc
test: bc-test
debug-test: bc-debug-test
bc-test: $(BIN)/$(notdir $(BC_TEST))
	$(running_test)
	./$(BC_TEST)
bc-debug-test: $(BIN)/$(notdir $(BC_DEBUG))
	$(running_test)
	./$(BC_DEBUG)
.PHONY: bc-hip-test
bc-cuda-test bc-hip-test:
	$(MAKE) KOKKOS_BACKEND=$(if $(filter %-hip-test,$@),hip,cuda) $(BIN)/bc_tests_cuda1$(EXE)
	$(running_test)
	./$(BIN)/bc_tests_cuda1$(EXE) --require-gpu
$(BC_DEBUG): source/io/settings_text.h
$(BC_TEST) $(BC_DEBUG): tests/test_checks.h
$(BC_TEST) $(BC_DEBUG): source/implicit/boundary_jacobian.h source/implicit/edge_jacobian.h
