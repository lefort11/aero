# Для проверки отчёта собираем интегратор и проверяем его поля на принятых шагах.
REPORT_SRC := $(IO_DIR)/flowreport_tests.cpp
REPORT_TEST := $(BUILD)/io_flowreport_tests_$(EXPLICIT_TAG)$(EXE)
REPORT_DEBUG := $(BUILD)/io_flowreport_tests_debug_$(EXPLICIT_TAG)$(EXE)
REPORT_HEADERS := $(IO_DIR)/flowreport.h $(EXPLICIT_HEADERS) source/eulersolver/euler.h source/parallel/parallel_for.h
$(REPORT_TEST): $(REPORT_SRC) $(REPORT_HEADERS) $(IO_LIBRARY) $(EXPLICIT_LIBRARY)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXPLICIT_FLAGS) $< $(IO_LIBRARY) $(EXPLICIT_LIBRARY) $(LDFLAGS) $(LDLIBS) $(EXPLICIT_LIBS) -o $@
$(REPORT_DEBUG): $(REPORT_SRC) $(IO_DIR)/flowreport.cpp $(IO_DIR)/flowreport_explicit.cpp $(EXPLICIT_SRC) $(REPORT_HEADERS) $(EXPLICIT_GPU) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O0 -g -D_GLIBCXX_DEBUG $(EXPLICIT_FLAGS) $(filter %.cpp %.o,$^) $(LDFLAGS) $(LDLIBS) $(EXPLICIT_LIBS) -o $@

.PHONY: io-flowreport-test io-flowreport-debug-test io-flowreport-cuda-test
test: io-flowreport-test
debug-test: io-flowreport-debug-test
io-flowreport-test: $(BIN)/$(notdir $(REPORT_TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$<)
else
	./$<
endif
io-flowreport-debug-test: $(BIN)/$(notdir $(REPORT_DEBUG))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$<)
else
	./$<
endif
io-flowreport-cuda-test:
ifeq ($(USE_CMD),1)
	"$(MSBUILD)" msvs/aero.sln /t:io_flowreport_tests /m:1 /nr:false /nologo /v:minimal /p:Platform=x64 /p:Configuration=Release /p:WithCuda=true /p:RequireCudaTests=true $(MSBUILD_CUDA_FLAGS)
else
	$(MAKE) WITH_CUDA=1 $(BIN)/io_flowreport_tests_cuda1_omp$(WITH_OPENMP)$(EXE)
	./$(BIN)/io_flowreport_tests_cuda1_omp$(WITH_OPENMP)$(EXE) --require-cuda
endif
$(REPORT_TEST) $(REPORT_DEBUG): qa/test_checks.h
