# Optional gas-dynamics module, independent of mesh, reconstruction and I/O.
EULERSOLVER_SOURCE := source/eulersolver
WITH_CUDA ?= 0
EULER_TAG := cuda$(WITH_CUDA)_omp$(WITH_OPENMP)
EULER_HEADERS := $(EULERSOLVER_SOURCE)/euler.h $(EULERSOLVER_SOURCE)/characteristic.h $(EULERSOLVER_SOURCE)/lax_friedrichs.h $(EULERSOLVER_SOURCE)/roe.h
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
ifeq ($(WITH_CUDA),1)
ifeq ($(USE_CMD),1)
$(error Windows CUDA uses MSVC: use make eulersolver-cuda-test)
else
EULER_FLAGS += -DAERO_WITH_CUDA
EULER_LIBS += $(CUDA_RUNTIME_LINK)
EULER_CUDA_TEST_OBJECT := $(BUILD)/eulersolver_cuda_tests.o
$(EULER_CUDA_TEST_OBJECT): $(EULERSOLVER_SOURCE)/eulersolver_cuda_tests.cu $(EULERSOLVER_SOURCE)/eulersolver_cuda_tests.cuh $(EULER_HEADERS) | $(BUILD)
	$(compile_cuda)
endif
endif

$(EULER_OBJECTS): $(BUILD)/eulersolver_%_$(EULER_TAG).o: $(EULERSOLVER_SOURCE)/%.cpp $(EULER_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EULER_FLAGS) -c $< -o $@
$(EULER_LIBRARY): $(EULER_OBJECTS)
	$(AR) rcs $@ $^
$(EULER_TEST): $(EULERSOLVER_SOURCE)/eulersolver_tests.cpp $(EULER_HEADERS) $(EULER_LIBRARY) $(EULER_CUDA_TEST_OBJECT)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EULER_FLAGS) $< $(EULER_LIBRARY) $(EULER_CUDA_TEST_OBJECT) $(LDFLAGS) $(LDLIBS) $(EULER_LIBS) -o $@
$(EULER_DEBUG_TEST): $(EULERSOLVER_SOURCE)/eulersolver_tests.cpp $(EULER_SOURCES) $(EULER_HEADERS) $(EULER_CUDA_TEST_OBJECT) | $(BUILD)
	$(CXX) $(CPPFLAGS) -std=c++11 -pedantic-errors -O0 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(EULER_FLAGS) $(EULERSOLVER_SOURCE)/eulersolver_tests.cpp $(EULER_SOURCES) $(EULER_CUDA_TEST_OBJECT) $(LDFLAGS) $(LDLIBS) $(EULER_LIBS) -o $@

.PHONY: eulersolver eulersolver-test eulersolver-debug-test eulersolver-cuda-test
eulersolver: $(EULER_LIBRARY)
eulersolver-test: $(BIN)/$(notdir $(EULER_TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(EULER_TEST))
else
	./$(EULER_TEST)
endif
eulersolver-debug-test: $(BIN)/$(notdir $(EULER_DEBUG_TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(EULER_DEBUG_TEST))
else
	./$(EULER_DEBUG_TEST)
endif
MSBUILD ?= MSBuild.exe
MSBUILD_CUDA_FLAGS ?=
ifeq ($(USE_CMD),1)
eulersolver-cuda-test:
	"$(MSBUILD)" msvs/aero.sln /t:eulersolver_tests /m:1 /nr:false /nologo /v:minimal /p:Platform=x64 /p:Configuration=Release /p:WithCuda=true /p:RunEulersolverTests=true /p:RequireCudaTests=true $(MSBUILD_CUDA_FLAGS)
else
eulersolver-cuda-test:
	$(MAKE) WITH_CUDA=1 $(BIN)/eulersolver_tests_cuda1_omp$(WITH_OPENMP)$(EXE)
	./$(BIN)/eulersolver_tests_cuda1_omp$(WITH_OPENMP)$(EXE) --require-cuda
endif
all: $(BIN)/$(notdir $(EULER_TEST))
test: eulersolver-test
debug-test: eulersolver-debug-test
