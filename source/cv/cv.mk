CV_DIR := source/cv
CV_HEADERS := $(CV_DIR)/cv.h $(MESH_SOURCE)/mesh.h $(MESH_SOURCE)/mesh_index.h
CV_TEST_HEADERS := $(CV_HEADERS) source/eulersolver/lax_friedrichs.h source/eulersolver/euler.h
CV_LIB := $(BUILD)/libaero_cv.a
CV_TEST := $(BUILD)/cv_tests_cuda$(WITH_CUDA)$(EXE)
CV_DEBUG := $(BUILD)/cv_tests_debug_cuda$(WITH_CUDA)$(EXE)

ifeq ($(WITH_CUDA),1)
CV_GPU := $(BUILD)/cv_cuda_tests.o
CV_FLAGS := -DAERO_WITH_CUDA
CV_LIBS := $(CUDA_RUNTIME_LINK)
$(CV_GPU): $(CV_DIR)/cv_cuda_tests.cu $(CV_DIR)/cv_cuda_tests.cuh $(CV_DIR)/cv.h | $(BUILD)
	$(compile_cuda)
endif
$(BUILD)/cv.o: $(CV_DIR)/cv.cpp $(CV_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CV_LIB): $(BUILD)/cv.o
	$(AR) rcs $@ $^
$(CV_TEST): $(CV_DIR)/cv_tests.cpp $(CV_TEST_HEADERS) $(CV_LIB) $(BUILD)/mesh.o $(IO_LIBRARY) $(CV_GPU)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CV_FLAGS) $< $(CV_LIB) $(BUILD)/mesh.o $(IO_LIBRARY) $(CV_GPU) $(LDFLAGS) $(CV_LIBS) $(LDLIBS) -o $@
$(CV_DEBUG): $(CV_DIR)/cv_tests.cpp $(CV_DIR)/cv.cpp $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp $(CV_TEST_HEADERS) $(IO_HEADERS) $(LZMA_OBJECTS) $(CV_GPU)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -g -O0 -D_GLIBCXX_DEBUG $(CV_FLAGS) $(filter %.cpp %.o,$^) $(LDFLAGS) $(CV_LIBS) $(LDLIBS) -o $@

.PHONY: cv cv-test cv-debug-test cv-cuda-test
cv: $(CV_LIB)
all: cv
test: cv-test
debug-test: cv-debug-test
cv-test: $(BIN)/$(notdir $(CV_TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(CV_TEST))
else
	./$(CV_TEST)
endif
cv-debug-test: $(BIN)/$(notdir $(CV_DEBUG))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(CV_DEBUG))
else
	./$(CV_DEBUG)
endif
cv-cuda-test:
ifeq ($(USE_CMD),1)
	"$(MSBUILD)" msvs/aero.sln /t:cv_tests /m:1 /nr:false /nologo /v:minimal /p:Platform=x64 /p:Configuration=Release /p:WithCuda=true /p:RequireCudaTests=true $(MSBUILD_CUDA_FLAGS)
else
	$(MAKE) WITH_CUDA=1 $(BIN)/cv_tests_cuda1$(EXE)
	./$(BIN)/cv_tests_cuda1$(EXE) --require-cuda
endif
