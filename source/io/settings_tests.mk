# Тесты читают реальные настройки и проверяют согласование параметров между модулями.
SETTINGS_TEST := $(BUILD)/io_settings_tests$(if $(filter 1,$(kokkos_gpu)),_cuda)$(EXE)
SETTINGS_DEBUG_TEST := $(BUILD)/io_settings_tests_debug$(EXE)
SETTINGS_TEST_SRC := $(IO_DIR)/settings_tests.cpp
SETTINGS_TEST_HEADERS := $(IO_DIR)/settings.h $(IO_DIR)/settings_boundary.h $(OUTPUT_HEADERS) $(INITIAL_HEADERS) $(PERIODIC_HEADERS) $(BC_HEADERS)
SETTINGS_TEST_LIBS := $(IO_LIBRARY) $(INITIAL_LIB) $(BC_LIB) $(CV_LIB) $(BUILD)/mesh.o $(IO_LIBRARY)
ifeq ($(kokkos_gpu),1)
$(SETTINGS_TEST): $(KOKKOS_DEVICE_LIBRARY)
ifneq ($(EXPLICIT_LIBRARY),)
SETTINGS_TEST_FLAGS := -DAERO_TEST_OUTPUT_DEVICE
SETTINGS_TEST_LIBS += $(EXPLICIT_LIBRARY)
endif
endif

$(SETTINGS_TEST): $(SETTINGS_TEST_SRC) $(SETTINGS_TEST_HEADERS) $(SETTINGS_TEST_LIBS)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SETTINGS_TEST_FLAGS) $(SETTINGS_TEST_SRC) $(SETTINGS_TEST_LIBS) $(LDFLAGS) $(LDLIBS) $(EXPLICIT_LIBS) -o $@

SETTINGS_DEBUG_SRCS := $(filter-out $(INITIAL_DIR)/initial_tests.cpp,$(INITIAL_DEBUG_SRC)) $(BC_DIR)/bc.cpp $(IO_DIR)/settings.cpp $(IO_DIR)/settings_boundary.cpp $(SETTINGS_TEST_SRC)
SETTINGS_DEBUG_SRCS += $(IO_DIR)/output_settings.cpp $(IO_DIR)/scheduled_output.cpp $(IO_DIR)/staged_field_file.cpp
$(SETTINGS_DEBUG_TEST): $(SETTINGS_DEBUG_SRCS) $(SETTINGS_TEST_HEADERS) $(IO_DIR)/settings_text.h $(IO_HEADERS) $(LZMA_OBJECTS) | $(BUILD)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O0 -g -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(SETTINGS_DEBUG_SRCS) $(LZMA_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

.PHONY: io-settings-test io-settings-debug-test
test: io-settings-test
debug-test: io-settings-debug-test
io-settings-test: $(BIN)/$(notdir $(SETTINGS_TEST))
	$(running_test)
	"$(SETTINGS_TEST)"
io-settings-debug-test: $(BIN)/$(notdir $(SETTINGS_DEBUG_TEST))
	$(running_test)
	"$(SETTINGS_DEBUG_TEST)"
$(SETTINGS_TEST) $(SETTINGS_DEBUG_TEST): qa/test_checks.h
