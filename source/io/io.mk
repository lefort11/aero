# Shared host I/O library, available to the solver and all utilities.
LZMA_SOURCE := third_party/lzma
IO_HEADERS := source/io/lzma_stream.h $(wildcard $(LZMA_SOURCE)/*.h)
LZMA_OBJECTS := $(addprefix $(BUILD)/lzma_,LzmaDec.o LzmaEnc.o LzFind.o CpuArch.o)
IO_LIBRARY := $(BUILD)/libaero_io.a
IO_DIR := source/io
FIELD_HEADERS := $(IO_DIR)/fields.h $(IO_DIR)/sha256.h
FIELD_SOURCES := $(IO_DIR)/fields.cpp $(IO_DIR)/fields_mesh.cpp $(IO_DIR)/sha256.cpp
FIELD_OBJECTS := $(patsubst $(IO_DIR)/%.cpp,$(BUILD)/io_%.o,$(FIELD_SOURCES))
FIELD_TEST := $(BUILD)/io_fields_tests$(EXE)
FIELD_DEBUG_TEST := $(BUILD)/io_fields_tests_debug$(EXE)
SETTINGS_OBJECTS := $(BUILD)/settings.o $(BUILD)/io_output_settings.o
OUTPUT_HEADERS := $(IO_DIR)/output_settings.h $(IO_DIR)/scheduled_output.h $(IO_DIR)/fields.h
$(BUILD)/io_output_settings.o: $(OUTPUT_HEADERS) $(IO_DIR)/settings_text.h
ifneq ($(wildcard source/eulersolver/euler.h),)
SETTINGS_OBJECTS += $(BUILD)/io_scheduled_output.o
$(BUILD)/io_scheduled_output.o: $(OUTPUT_HEADERS) $(MESH_SOURCE)/mesh.h source/eulersolver/euler.h source/parallel/parallel_for.h
FLOWREPORT_OBJECTS := $(BUILD)/flowreport.o
ifneq ($(wildcard source/explicit/explicit.h),)
FLOWREPORT_OBJECTS += $(BUILD)/flowreport_explicit.o
endif
endif
$(BUILD)/flowreport.o: $(IO_DIR)/flowreport.cpp $(IO_DIR)/flowreport.h source/eulersolver/euler.h source/parallel/parallel_for.h | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD)/flowreport_explicit.o: $(IO_DIR)/flowreport_explicit.cpp $(IO_DIR)/flowreport.h source/explicit/explicit.h | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
ifneq ($(wildcard source/bc/bc.h),)
SETTINGS_OBJECTS += $(BUILD)/settings_boundary.o
endif

$(BUILD)/settings.o: $(IO_DIR)/settings.cpp $(IO_DIR)/settings.h $(IO_DIR)/settings_text.h $(IO_DIR)/output_settings.h | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/settings_boundary.o: $(IO_DIR)/settings_boundary.cpp $(IO_DIR)/settings_boundary.h $(IO_DIR)/settings_text.h source/bc/bc.h source/eulersolver/euler.h $(MESH_SOURCE)/mesh.h | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/lzma_%.o: $(LZMA_SOURCE)/%.c $(IO_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) -x c -DZ7_ST $(build_optimization) -c $< -o $@

$(BUILD)/lzma_stream.o: source/io/lzma_stream.cpp $(IO_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/expression.o: source/io/expression.cpp source/io/expression.h | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/io_%.o: $(IO_DIR)/%.cpp $(FIELD_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/io_fields.o $(BUILD)/io_fields_tests.o: $(IO_DIR)/lzma_stream.h
$(BUILD)/io_fields_mesh.o $(BUILD)/io_fields_tests.o: $(MESH_SOURCE)/mesh.h

$(IO_LIBRARY): $(BUILD)/lzma_stream.o $(BUILD)/expression.o $(FIELD_OBJECTS) $(SETTINGS_OBJECTS) $(FLOWREPORT_OBJECTS) $(LZMA_OBJECTS)
	$(AR) rcs $@ $^

$(FIELD_TEST): $(BUILD)/io_fields_tests.o $(BUILD)/mesh.o $(IO_LIBRARY)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

FIELD_DEBUG_SOURCES := $(IO_DIR)/fields_tests.cpp $(FIELD_SOURCES) $(MESH_SOURCE)/mesh.cpp $(IO_DIR)/lzma_stream.cpp
$(FIELD_DEBUG_TEST): $(FIELD_DEBUG_SOURCES) $(FIELD_HEADERS) $(MESH_SOURCE)/mesh.h $(IO_HEADERS) $(LZMA_OBJECTS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O0 -g -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(FIELD_DEBUG_SOURCES) $(LZMA_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

.PHONY: io io-fields-test io-fields-debug-test output output-test output-debug-test
io: $(IO_LIBRARY)
io-fields-test: $(BIN)/$(notdir $(FIELD_TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(FIELD_TEST))
else
	./$(FIELD_TEST)
endif
io-fields-debug-test: $(BIN)/$(notdir $(FIELD_DEBUG_TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(FIELD_DEBUG_TEST))
else
	./$(FIELD_DEBUG_TEST)
endif

# Compatibility with existing build scripts.
output: io
output-test: io-fields-test
output-debug-test: io-fields-debug-test
all: $(BIN)/$(notdir $(FIELD_TEST))
test: io-fields-test
debug-test: io-fields-debug-test
