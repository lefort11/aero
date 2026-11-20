# Для проверки отчёта собираем интегратор и проверяем его поля на принятых шагах.
REPORT_SRC := $(IO_DIR)/flowreport_tests.cpp
REPORT_TEST := $(BUILD)/io_flowreport_tests_$(EXPLICIT_TAG)$(EXE)
REPORT_DEBUG := $(BUILD)/io_flowreport_tests_debug_$(EXPLICIT_TAG)$(EXE)
REPORT_HEADERS := $(IO_DIR)/flowreport.h $(EXPLICIT_HEADERS) source/eulersolver/euler.h source/parallel/parallel_for.h
$(REPORT_TEST): $(REPORT_SRC) $(REPORT_HEADERS) $(IO_LIBRARY) $(EXPLICIT_LIBRARY)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXPLICIT_FLAGS) $< $(IO_LIBRARY) $(EXPLICIT_LIBRARY) $(LDFLAGS) $(LDLIBS) $(EXPLICIT_LIBS) -o $@
$(REPORT_DEBUG): $(REPORT_SRC) $(IO_DIR)/flowreport.cpp $(IO_DIR)/flowreport_explicit.cpp $(EXPLICIT_SRC) $(REPORT_HEADERS) $(EXPLICIT_GPU) | $(BUILD)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O0 -g -D_GLIBCXX_DEBUG $(EXPLICIT_FLAGS) $(filter %.cpp %.o,$^) $(LDFLAGS) $(LDLIBS) $(EXPLICIT_LIBS) -o $@

.PHONY: io-flowreport-test io-flowreport-debug-test io-flowreport-cuda-test
test: io-flowreport-test
debug-test: io-flowreport-debug-test
io-flowreport-test: $(BIN)/$(notdir $(REPORT_TEST))
	$(running_test)
	"$<"
io-flowreport-debug-test: $(BIN)/$(notdir $(REPORT_DEBUG))
	$(running_test)
	"$<"
io-flowreport-cuda-test:
	$(MAKE) KOKKOS_BACKEND=cuda $(BIN)/io_flowreport_tests_cuda1_omp$(WITH_OPENMP)$(EXE)
	$(running_test)
	./$(BIN)/io_flowreport_tests_cuda1_omp$(WITH_OPENMP)$(EXE) --require-cuda
$(REPORT_TEST) $(REPORT_DEBUG): qa/test_checks.h
