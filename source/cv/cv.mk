# Библиотека строит контрольные объёмы на CPU, CUDA-тест проверяет чтение готовых массивов на карте.
CV_DIR := source/cv
CV_HEADERS := $(CV_DIR)/cv.h $(MESH_HEADERS)
CV_TEST_HEADERS := $(CV_HEADERS) source/eulersolver/lax_friedrichs.h source/eulersolver/euler.h
CV_LIB := $(BUILD)/libaero_cv.a
CV_TEST := $(BUILD)/cv_tests_cuda$(kokkos_gpu)$(EXE)
CV_DEBUG := $(BUILD)/cv_tests_debug_cuda$(kokkos_gpu)$(EXE)

ifeq ($(kokkos_gpu),1)
CV_GPU := $(BUILD)/cv_cuda_tests.o
CV_FLAGS := -DAERO_WITH_KOKKOS_GPU
$(CV_GPU): $(CV_DIR)/cv_gpu_tests.cpp $(CV_DIR)/cv.h | $(BUILD)
	$(compiling)
	$(compile_device_host)
endif
$(BUILD)/cv.o: $(CV_DIR)/cv.cpp $(CV_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CV_LIB): $(BUILD)/cv.o
	$(archiving)
	$(AR) rcs $@ $^
$(CV_TEST): $(CV_DIR)/cv_tests.cpp $(CV_TEST_HEADERS) $(CV_LIB) $(BUILD)/mesh.o $(IO_LIBRARY) $(CV_GPU)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CV_FLAGS) $< $(CV_LIB) $(BUILD)/mesh.o $(IO_LIBRARY) $(CV_GPU) $(LDFLAGS) $(CV_LIBS) $(LDLIBS) -o $@
$(CV_DEBUG): $(CV_DIR)/cv_tests.cpp $(CV_DIR)/cv.cpp $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp $(CV_TEST_HEADERS) $(IO_HEADERS) $(LZMA_OBJECTS) $(CV_GPU)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -g -O0 -D_GLIBCXX_DEBUG $(CV_FLAGS) $(filter %.cpp %.o,$^) $(LDFLAGS) $(CV_LIBS) $(LDLIBS) -o $@

.PHONY: cv cv-test cv-debug-test cv-cuda-test
cv: $(CV_LIB)
all: cv
test: cv-test
debug-test: cv-debug-test
cv-test: $(BIN)/$(notdir $(CV_TEST))
	$(running_test)
	"$(CV_TEST)"
cv-debug-test: $(BIN)/$(notdir $(CV_DEBUG))
	$(running_test)
	"$(CV_DEBUG)"
cv-cuda-test:
	$(MAKE) KOKKOS_BACKEND=cuda $(BIN)/cv_tests_cuda1$(EXE)
	$(running_test)
	"$(BIN)/cv_tests_cuda1$(EXE)" --require-cuda
$(CV_TEST) $(CV_DEBUG): tests/test_checks.h
