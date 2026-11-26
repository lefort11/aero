# Собирает явный интегратор и его тестовый оператор, которым можно заменить невязку солвера.
EXPLICIT_DIR := source/explicit
EXPLICIT_TAG := cuda$(kokkos_gpu)_omp$(WITH_OPENMP)
EXPLICIT_SRC := $(addprefix $(EXPLICIT_DIR)/,explicit.cpp explicit_euler.cpp ssprk3.cpp)
EXPLICIT_OBJ := $(patsubst $(EXPLICIT_DIR)/%.cpp,$(BUILD)/%_$(EXPLICIT_TAG).o,$(EXPLICIT_SRC))
EXPLICIT_HEADERS := $(EXPLICIT_DIR)/explicit.h $(EXPLICIT_DIR)/explicit_stage.h
EXPLICIT_HEADERS += $(EXPLICIT_DIR)/stage_sequence.h $(EXPLICIT_DIR)/stage_execution.h source/parallel/row_loop.h source/parallel/kernel_math.h
EXPLICIT_LIBRARY := $(BUILD)/libaero_explicit_$(EXPLICIT_TAG).a
EXPLICIT_TEST := $(BUILD)/explicit_tests_$(EXPLICIT_TAG)$(EXE)
EXPLICIT_DEBUG := $(BUILD)/explicit_tests_debug_$(EXPLICIT_TAG)$(EXE)
ifeq ($(kokkos_gpu),1)
EXPLICIT_HEADERS += source/parallel/device_scope.h source/parallel/device_memory.h
EXPLICIT_FLAGS += -DAERO_WITH_KOKKOS_GPU
EXPLICIT_GPU := $(BUILD)/explicit_kernel.o
EXPLICIT_TEST_GPU := $(BUILD)/explicit_cuda_tests.o
$(EXPLICIT_GPU): $(EXPLICIT_DIR)/explicit_gpu.cpp $(EXPLICIT_HEADERS) | $(BUILD)
	$(compiling)
	$(compile_device_host)
$(EXPLICIT_TEST_GPU): $(EXPLICIT_DIR)/explicit_gpu_tests.cpp $(EXPLICIT_DIR)/explicit_test_problem.h $(EXPLICIT_HEADERS) | $(BUILD)
	$(compiling)
	$(compile_device_host)
endif
$(EXPLICIT_OBJ): $(BUILD)/%_$(EXPLICIT_TAG).o: $(EXPLICIT_DIR)/%.cpp $(EXPLICIT_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXPLICIT_FLAGS) -c $< -o $@
$(EXPLICIT_LIBRARY): $(EXPLICIT_OBJ) $(EXPLICIT_GPU)
	$(archiving)
	$(AR) rcs $@ $^
$(EXPLICIT_TEST): $(EXPLICIT_DIR)/explicit_tests.cpp $(EXPLICIT_DIR)/explicit_test_problem.h $(EXPLICIT_HEADERS) $(EXPLICIT_LIBRARY) $(EXPLICIT_TEST_GPU)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXPLICIT_FLAGS) $< $(EXPLICIT_LIBRARY) $(EXPLICIT_TEST_GPU) $(LDFLAGS) $(LDLIBS) $(EXPLICIT_LIBS) -o $@
$(EXPLICIT_DEBUG): $(EXPLICIT_DIR)/explicit_tests.cpp $(EXPLICIT_DIR)/explicit_test_problem.h $(EXPLICIT_SRC) $(EXPLICIT_HEADERS) $(EXPLICIT_GPU) $(EXPLICIT_TEST_GPU) | $(BUILD)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -g -O0 -D_GLIBCXX_DEBUG $(EXPLICIT_FLAGS) $(filter %.cpp %.o,$^) $(LDFLAGS) $(LDLIBS) $(EXPLICIT_LIBS) -o $@
.PHONY: explicit explicit-test explicit-debug-test explicit-cuda-test
all: explicit
test: explicit-test
debug-test: explicit-debug-test
explicit: $(EXPLICIT_LIBRARY)
explicit-test: $(BIN)/$(notdir $(EXPLICIT_TEST))
	$(running_test)
	./$(EXPLICIT_TEST)
explicit-debug-test: $(BIN)/$(notdir $(EXPLICIT_DEBUG))
	$(running_test)
	./$(EXPLICIT_DEBUG)
.PHONY: explicit-hip-test
explicit-cuda-test explicit-hip-test:
	$(MAKE) KOKKOS_BACKEND=$(if $(filter %-hip-test,$@),hip,cuda) $(BIN)/explicit_tests_cuda1_omp$(WITH_OPENMP)$(EXE)
	$(running_test)
	./$(BIN)/explicit_tests_cuda1_omp$(WITH_OPENMP)$(EXE) --require-gpu
$(EXPLICIT_TEST) $(EXPLICIT_DEBUG): tests/test_checks.h
