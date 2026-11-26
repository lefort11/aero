# Для проверки отчёта собираем интегратор и проверяем его поля на принятых шагах.
REPORT_SRC := $(IO_DIR)/report_tests.cpp
REPORT_TEST := $(BUILD)/io_report_tests_$(EXPLICIT_TAG)$(EXE)
REPORT_DEBUG := $(BUILD)/io_report_tests_debug_$(EXPLICIT_TAG)$(EXE)
REPORT_HEADERS := source/io/diagnostic_fields.h source/io/flow_values.h source/eulersolver/linear_euler.h $(IO_DIR)/report.h $(IO_DIR)/steady_report.h $(EXPLICIT_HEADERS) source/eulersolver/euler.h source/parallel/parallel_for.h
$(REPORT_TEST): $(REPORT_SRC) $(REPORT_HEADERS) $(IO_LIBRARY) $(EXPLICIT_LIBRARY)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXPLICIT_FLAGS) $< $(IO_LIBRARY) $(EXPLICIT_LIBRARY) $(LDFLAGS) $(LDLIBS) $(EXPLICIT_LIBS) -o $@
$(REPORT_DEBUG): $(REPORT_SRC) $(IO_DIR)/report.cpp $(IO_DIR)/report_explicit.cpp $(EXPLICIT_SRC) $(REPORT_HEADERS) $(EXPLICIT_GPU) | $(BUILD)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O0 -g -D_GLIBCXX_DEBUG $(EXPLICIT_FLAGS) $(filter %.cpp %.o,$^) $(LDFLAGS) $(LDLIBS) $(EXPLICIT_LIBS) -o $@

.PHONY: io-report-test io-report-debug-test io-report-cuda-test
test: io-report-test
debug-test: io-report-debug-test
io-report-test: $(BIN)/$(notdir $(REPORT_TEST))
	$(running_test)
	"$<"
io-report-debug-test: $(BIN)/$(notdir $(REPORT_DEBUG))
	$(running_test)
	"$<"
.PHONY: io-report-hip-test
io-report-cuda-test io-report-hip-test:
	$(MAKE) KOKKOS_BACKEND=$(if $(filter %-hip-test,$@),hip,cuda) $(BIN)/io_report_tests_cuda1_omp$(WITH_OPENMP)$(EXE)
	$(running_test)
	./$(BIN)/io_report_tests_cuda1_omp$(WITH_OPENMP)$(EXE) --require-gpu
$(REPORT_TEST) $(REPORT_DEBUG): tests/test_checks.h
