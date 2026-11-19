# Библиотека невязки связывает поток, реконструкцию и геометрию. Тесты проверяют также выбор шага.
RES_DIR := source/residual
RES_HDR := $(RES_DIR)/residual.h $(RES_DIR)/residual_detail.h $(RES_DIR)/packed_weno.h $(RES_DIR)/flux_cfl.h source/timestep/cfl_detail.h source/timestep/timestep.h source/parallel/kernel_math.h source/mesh/mesh_index.h source/parallel/parallel_for.h source/parallel/wall_timer.h $(BC_HEADERS) $(CV_HEADERS) $(PERIODIC_HEADERS) $(EULER_HEADERS) $(MESH_HEADERS)
# Формула WENO встраивается в поток ребра на CPU и GPU. При изменении весов пересобираем оба варианта.
RES_HDR += $(RECONSTRUCTION_HEADERS)
RES_LIB := $(BUILD)/libaero_residual_$(EBR_TAG).a
RES_OBJ := $(BUILD)/residual_$(EBR_TAG).o
RES_DEPS := $(BC_LIB) $(CV_LIB) $(EBR_LIBRARY) $(EULER_LIBRARY) $(BUILD)/mesh.o $(IO_LIBRARY)
RES_TEST := $(BUILD)/residual_tests_$(EBR_TAG)$(EXE)
ifneq ($(and $(EXPLICIT_LIBRARY),$(TIMESTEP_LIBRARY)),)
RES_TIME_LIBS := $(TIMESTEP_LIBRARY) $(EXPLICIT_LIBRARY)
RES_TIME_SRC := $(TIMESTEP_SRC) $(EXPLICIT_SRC)
RES_TIME_HEADERS := $(TIMESTEP_HEADERS) $(EXPLICIT_HEADERS)
RES_TIME_GPU := $(TIMESTEP_GPU) $(EXPLICIT_GPU)
else
RES_TEST_FLAGS := -DAERO_RESIDUAL_NO_TIME_TEST
endif
ifeq ($(WITH_CUDA),1)
RES_HDR += source/cuda/device_scope.h source/cuda/runtime.h source/timestep/cfl_block_min.cuh
RES_GPU := $(BUILD)/residual_cuda.o
$(RES_GPU): $(RES_DIR)/residual_cuda.cu $(RES_DIR)/residual_cuda.cuh $(RES_HDR) | $(BUILD)
	$(compile_cuda)
endif
# Ошибки расчёта потока передаются через EulerStatus. Для sqrt в цикле CPU не нужен
# побочный эффект записи в errno. Проверки плотности, давления и конечности сохраняются.
$(RES_OBJ): $(RES_DIR)/residual.cpp $(RES_HDR) $(RES_DIR)/residual.mk | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fno-math-errno $(EBR_FLAGS) -c $< -o $@
$(RES_LIB): $(RES_OBJ) $(RES_GPU)
	$(AR) rcs $@ $^
$(RES_TEST): $(RES_DIR)/residual_tests.cpp $(RES_HDR) $(RES_TIME_HEADERS) $(RES_LIB) $(RES_DEPS) $(RES_TIME_LIBS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EBR_FLAGS) $(RES_TEST_FLAGS) $< $(RES_LIB) $(RES_TIME_LIBS) $(RES_DEPS) $(LDFLAGS) $(EBR_LIBS) $(LDLIBS) -o $@
.PHONY: residual residual-test residual-debug-test residual-cuda-test
all: residual
residual: $(RES_LIB)
test: residual-test
residual-test: $(BIN)/$(notdir $(RES_TEST))
	"$(RES_TEST)"
RES_DEBUG := $(BUILD)/residual_tests_debug_$(EBR_TAG)$(EXE)
RES_DEBUG_SRC := $(RES_DIR)/residual_tests.cpp $(RES_DIR)/residual.cpp $(RES_TIME_SRC) $(PERIODIC_SOURCES) $(CV_DIR)/cv.cpp $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp $(RECONSTRUCTION_CPP)
$(RES_DEBUG): $(RES_DEBUG_SRC) $(RES_HDR) $(RES_TIME_HEADERS) $(IO_HEADERS) $(LZMA_OBJECTS) $(RES_GPU) $(RES_TIME_GPU) $(EBR_CUDA_OBJECT)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EBR_FLAGS) $(RES_TEST_FLAGS) -O0 -g -D_GLIBCXX_DEBUG $(filter %.cpp %.o,$^) $(LDFLAGS) $(EBR_LIBS) $(LDLIBS) -o $@
debug-test: residual-debug-test
residual-debug-test: $(BIN)/$(notdir $(RES_DEBUG))
	"$(RES_DEBUG)"
residual-cuda-test:
ifeq ($(USE_CMD),1)
	"$(MSBUILD)" msvs/aero.sln /t:residual_tests /m:1 /nr:false /nologo /v:minimal /p:Platform=x64 /p:Configuration=Release /p:WithCuda=true /p:RequireCudaTests=true $(MSBUILD_CUDA_FLAGS)
else
	$(MAKE) WITH_CUDA=1 $(BIN)/residual_tests_cuda1_omp$(WITH_OPENMP)$(EXE)
	"$(BIN)/residual_tests_cuda1_omp$(WITH_OPENMP)$(EXE)" --require-cuda
endif
$(RES_TEST) $(RES_DEBUG): qa/test_checks.h
