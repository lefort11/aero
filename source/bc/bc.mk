# Проверки границ используют ту же реализацию потока и ту же геометрию площадок, что и солвер.
BC_DIR := source/bc
BC_HEADERS := source/parallel/kernel_math.h $(BC_DIR)/bc.h source/eulersolver/euler.h
BC_TEST_HEADERS := $(BC_HEADERS) $(BC_DIR)/bc_test_case.h
BC_LIB := $(BUILD)/libaero_bc.a
PERIODIC_HEADERS := $(BC_DIR)/periodic.h $(CV_HEADERS) $(RECONSTRUCTION_HEADERS)
PERIODIC_SOURCES := $(BC_DIR)/periodic.cpp $(BC_DIR)/periodic_stencil.cpp
PERIODIC_OBJECTS := $(BUILD)/periodic.o $(BUILD)/periodic_stencil.o
$(PERIODIC_OBJECTS): $(BUILD)/%.o: $(BC_DIR)/%.cpp $(PERIODIC_HEADERS) $(BC_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
BC_DEPS := $(CV_LIB) $(EBR_LIBRARY) $(BUILD)/mesh.o $(IO_LIBRARY)
BC_TEST := $(BUILD)/bc_tests_cuda$(WITH_CUDA)$(EXE)
BC_DEBUG := $(BUILD)/bc_tests_debug_cuda$(WITH_CUDA)$(EXE)

ifeq ($(WITH_CUDA),1)
BC_GPU := $(BUILD)/bc_cuda_tests.o
BC_FLAGS := -DAERO_WITH_CUDA
BC_LIBS := $(CUDA_RUNTIME_LINK)
$(BC_GPU): $(BC_DIR)/bc_cuda_tests.cu $(BC_DIR)/bc_cuda_tests.cuh $(BC_TEST_HEADERS) $(PERIODIC_HEADERS) | $(BUILD)
	$(compile_cuda)
endif
$(BUILD)/bc.o: $(BC_DIR)/bc.cpp $(BC_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BC_LIB): $(BUILD)/bc.o $(PERIODIC_OBJECTS)
	$(AR) rcs $@ $^
$(BC_TEST): $(BC_DIR)/bc_tests.cpp $(BC_DIR)/periodic_tests.cpp $(PERIODIC_HEADERS) $(BC_TEST_HEADERS) $(BC_LIB) $(BC_DEPS) $(BC_GPU)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(BC_FLAGS) $(filter %.cpp,$^) $(BC_LIB) $(BC_DEPS) $(BC_GPU) $(LDFLAGS) $(EBR_LIBS) $(BC_LIBS) $(LDLIBS) -o $@
$(BC_DEBUG): $(BC_DIR)/bc_tests.cpp $(BC_DIR)/periodic_tests.cpp $(BC_DIR)/bc.cpp $(PERIODIC_SOURCES) $(CV_DIR)/cv.cpp $(RECONSTRUCTION_CPP) $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp $(LZMA_OBJECTS) $(PERIODIC_HEADERS) $(BC_TEST_HEADERS) $(IO_HEADERS) $(BC_GPU) $(EBR_CUDA_OBJECT)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -g -O0 -D_GLIBCXX_DEBUG $(EBR_FLAGS) $(BC_FLAGS) $(filter %.cpp %.o,$^) $(LDFLAGS) $(EBR_LIBS) $(BC_LIBS) $(LDLIBS) -o $@

.PHONY: bc bc-test bc-debug-test bc-cuda-test
bc: $(BC_LIB)
all: bc
test: bc-test
debug-test: bc-debug-test
bc-test: $(BIN)/$(notdir $(BC_TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(BC_TEST))
else
	./$(BC_TEST)
endif
bc-debug-test: $(BIN)/$(notdir $(BC_DEBUG))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(BC_DEBUG))
else
	./$(BC_DEBUG)
endif
bc-cuda-test:
ifeq ($(USE_CMD),1)
	"$(MSBUILD)" msvs/aero.sln /t:bc_tests /m:1 /nr:false /nologo /v:minimal /p:Platform=x64 /p:Configuration=Release /p:WithCuda=true /p:RequireCudaTests=true $(MSBUILD_CUDA_FLAGS)
else
	$(MAKE) WITH_CUDA=1 $(BIN)/bc_tests_cuda1$(EXE)
	./$(BIN)/bc_tests_cuda1$(EXE) --require-cuda
endif
$(BC_TEST) $(BC_DEBUG): qa/test_checks.h
