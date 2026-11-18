EXPLICIT_DIR := source/explicit
WITH_CUDA ?= 0
EXPLICIT_TAG := cuda$(WITH_CUDA)_omp$(WITH_OPENMP)
EXPLICIT_SRC := $(addprefix $(EXPLICIT_DIR)/,explicit.cpp explicit_euler.cpp ssprk3.cpp)
EXPLICIT_OBJ := $(patsubst $(EXPLICIT_DIR)/%.cpp,$(BUILD)/%_$(EXPLICIT_TAG).o,$(EXPLICIT_SRC))
EXPLICIT_HEADERS := $(EXPLICIT_DIR)/explicit.h $(EXPLICIT_DIR)/explicit_stage.h
EXPLICIT_HEADERS += $(EXPLICIT_DIR)/stage_sequence.h source/parallel/kernel_math.h
EXPLICIT_LIBRARY := $(BUILD)/libaero_explicit_$(EXPLICIT_TAG).a
EXPLICIT_TEST := $(BUILD)/explicit_tests_$(EXPLICIT_TAG)$(EXE)
EXPLICIT_DEBUG := $(BUILD)/explicit_tests_debug_$(EXPLICIT_TAG)$(EXE)
ifeq ($(WITH_CUDA),1)
EXPLICIT_HEADERS += source/cuda/device_scope.h source/cuda/runtime.h
EXPLICIT_FLAGS += -DAERO_WITH_CUDA -isystem "$(CUDA_HOME)/include"
EXPLICIT_LIBS := $(CUDA_RUNTIME_LINK)
EXPLICIT_GPU := $(BUILD)/explicit_kernel.o
EXPLICIT_TEST_GPU := $(BUILD)/explicit_cuda_tests.o
$(EXPLICIT_GPU): $(EXPLICIT_DIR)/explicit_cuda.cu $(EXPLICIT_DIR)/explicit_cuda.cuh $(EXPLICIT_HEADERS) | $(BUILD)
	$(compile_cuda)
$(EXPLICIT_TEST_GPU): $(EXPLICIT_DIR)/explicit_cuda_tests.cu $(EXPLICIT_DIR)/explicit_cuda_tests.cuh $(EXPLICIT_DIR)/explicit_test_problem.h $(EXPLICIT_HEADERS) | $(BUILD)
	$(compile_cuda)
endif
$(EXPLICIT_OBJ): $(BUILD)/%_$(EXPLICIT_TAG).o: $(EXPLICIT_DIR)/%.cpp $(EXPLICIT_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXPLICIT_FLAGS) -c $< -o $@
$(EXPLICIT_LIBRARY): $(EXPLICIT_OBJ) $(EXPLICIT_GPU)
	$(AR) rcs $@ $^
$(EXPLICIT_TEST): $(EXPLICIT_DIR)/explicit_tests.cpp $(EXPLICIT_DIR)/explicit_test_problem.h $(EXPLICIT_HEADERS) $(EXPLICIT_LIBRARY) $(EXPLICIT_TEST_GPU)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXPLICIT_FLAGS) $< $(EXPLICIT_LIBRARY) $(EXPLICIT_TEST_GPU) $(LDFLAGS) $(LDLIBS) $(EXPLICIT_LIBS) -o $@
$(EXPLICIT_DEBUG): $(EXPLICIT_DIR)/explicit_tests.cpp $(EXPLICIT_DIR)/explicit_test_problem.h $(EXPLICIT_SRC) $(EXPLICIT_HEADERS) $(EXPLICIT_GPU) $(EXPLICIT_TEST_GPU) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -g -O0 -D_GLIBCXX_DEBUG $(EXPLICIT_FLAGS) $(filter %.cpp %.o,$^) $(LDFLAGS) $(LDLIBS) $(EXPLICIT_LIBS) -o $@
.PHONY: explicit explicit-test explicit-debug-test explicit-cuda-test
all: explicit
test: explicit-test
debug-test: explicit-debug-test
explicit: $(EXPLICIT_LIBRARY)
explicit-test: $(BIN)/$(notdir $(EXPLICIT_TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(EXPLICIT_TEST))
else
	./$(EXPLICIT_TEST)
endif
explicit-debug-test: $(BIN)/$(notdir $(EXPLICIT_DEBUG))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(EXPLICIT_DEBUG))
else
	./$(EXPLICIT_DEBUG)
endif
explicit-cuda-test:
ifeq ($(USE_CMD),1)
	"$(MSBUILD)" msvs/aero.sln /t:explicit_tests /m:1 /nr:false /nologo /v:minimal /p:Platform=x64 /p:Configuration=Release /p:WithCuda=true /p:RequireCudaTests=true $(MSBUILD_CUDA_FLAGS)
else
	$(MAKE) WITH_CUDA=1 $(BIN)/explicit_tests_cuda1_omp$(WITH_OPENMP)$(EXE)
	./$(BIN)/explicit_tests_cuda1_omp$(WITH_OPENMP)$(EXE) --require-cuda
endif
