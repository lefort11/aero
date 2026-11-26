# Модуль потоков Эйлера собирается независимо от сетки, реконструкции и ввода-вывода.
EULERSOLVER_SOURCE := source/eulersolver
EULER_TAG := cuda$(kokkos_gpu)_omp$(WITH_OPENMP)
EULER_HEADERS := $(EULERSOLVER_SOURCE)/linear_euler.h $(EULERSOLVER_SOURCE)/euler.h $(EULERSOLVER_SOURCE)/characteristic.h $(EULERSOLVER_SOURCE)/lax_friedrichs.h $(EULERSOLVER_SOURCE)/roe.h
EULER_HEADERS += source/parallel/kernel_math.h
EULER_SOURCES := $(addprefix $(EULERSOLVER_SOURCE)/,euler.cpp characteristic.cpp lax_friedrichs.cpp roe.cpp)
EULER_OBJECTS := $(patsubst $(EULERSOLVER_SOURCE)/%.cpp,$(BUILD)/eulersolver_%_$(EULER_TAG).o,$(EULER_SOURCES))
EULER_LIBRARY := $(BUILD)/libaero_eulersolver_$(EULER_TAG).a
EULER_TEST := $(BUILD)/eulersolver_tests_$(EULER_TAG)$(EXE)
EULER_DEBUG_TEST := $(BUILD)/eulersolver_tests_debug_$(EULER_TAG)$(EXE)
EULER_FLAGS :=
EULER_LIBS :=
EULER_CUDA_TEST_OBJECT :=
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 86
ifeq ($(kokkos_gpu),1)
EULER_FLAGS += -DAERO_WITH_KOKKOS_GPU
EULER_CUDA_TEST_OBJECT := $(BUILD)/eulersolver_cuda_tests.o
$(EULER_CUDA_TEST_OBJECT): $(EULERSOLVER_SOURCE)/eulersolver_gpu_tests.cpp $(EULER_HEADERS) source/parallel/device_check.h source/parallel/device_scope.h source/parallel/device_memory.h | $(BUILD)
	$(compiling)
	$(compile_device_host)
endif

$(EULER_OBJECTS): $(BUILD)/eulersolver_%_$(EULER_TAG).o: $(EULERSOLVER_SOURCE)/%.cpp $(EULER_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EULER_FLAGS) -c $< -o $@
$(EULER_LIBRARY): $(EULER_OBJECTS)
	$(archiving)
	$(AR) rcs $@ $^
$(EULER_TEST): $(EULERSOLVER_SOURCE)/eulersolver_tests.cpp $(EULER_HEADERS) $(EULER_LIBRARY) $(EULER_CUDA_TEST_OBJECT)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EULER_FLAGS) $< $(EULER_LIBRARY) $(EULER_CUDA_TEST_OBJECT) $(LDFLAGS) $(LDLIBS) $(EULER_LIBS) -o $@
$(EULER_DEBUG_TEST): $(EULERSOLVER_SOURCE)/eulersolver_tests.cpp $(EULER_SOURCES) $(EULER_HEADERS) $(EULER_CUDA_TEST_OBJECT) | $(BUILD)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) -std=c++11 -pedantic-errors -O0 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(EULER_FLAGS) $(EULERSOLVER_SOURCE)/eulersolver_tests.cpp $(EULER_SOURCES) $(EULER_CUDA_TEST_OBJECT) $(LDFLAGS) $(LDLIBS) $(EULER_LIBS) -o $@

.PHONY: eulersolver eulersolver-test eulersolver-debug-test eulersolver-cuda-test
eulersolver: $(EULER_LIBRARY)
eulersolver-test: $(BIN)/$(notdir $(EULER_TEST))
	$(running_test)
	"$<"
eulersolver-debug-test: $(BIN)/$(notdir $(EULER_DEBUG_TEST))
	$(running_test)
	"$<"
.PHONY: eulersolver-hip-test
eulersolver-cuda-test eulersolver-hip-test:
	$(MAKE) KOKKOS_BACKEND=$(if $(filter %-hip-test,$@),hip,cuda) $(BIN)/eulersolver_tests_cuda1_omp$(WITH_OPENMP)$(EXE)
	$(running_test)
	./$(BIN)/eulersolver_tests_cuda1_omp$(WITH_OPENMP)$(EXE) --require-gpu
all: $(BIN)/$(notdir $(EULER_TEST))
test: eulersolver-test
debug-test: eulersolver-debug-test
$(EULER_TEST) $(EULER_DEBUG_TEST): tests/test_checks.h tests/test_suite.h
