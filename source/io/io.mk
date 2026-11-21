# Общая библиотека ввода и вывода используется солвером и утилитами.
LZMA_SOURCE := third_party/lzma
$(BUILD)/report.o $(BUILD)/io_scheduled_output.o: source/io/flow_values.h source/eulersolver/linear_euler.h
NUMBER_CODEC := source/io/number_codec.h
NATIVE_FORMAT := source/io/native_format.h $(NUMBER_CODEC)
LABEL_HEADER := source/io/utf8_label.h
IO_HEADERS := $(LABEL_HEADER) $(NATIVE_FORMAT) source/io/lzma_stream.h $(wildcard $(LZMA_SOURCE)/*.h)
LZMA_OBJECTS := $(addprefix $(BUILD)/lzma_,LzmaDec.o LzmaEnc.o LzFind.o CpuArch.o)
IO_LIBRARY := $(BUILD)/libaero_io.a
IO_DIR := source/io
FIELD_HEADERS := $(LABEL_HEADER) $(NATIVE_FORMAT) $(IO_DIR)/fields.h $(IO_DIR)/sha256.h
FIELD_SOURCES := $(IO_DIR)/fields.cpp $(IO_DIR)/fields_mesh.cpp $(IO_DIR)/sha256.cpp
FIELD_OBJECTS := $(patsubst $(IO_DIR)/%.cpp,$(BUILD)/io_%.o,$(FIELD_SOURCES))
FIELD_TEST := $(BUILD)/io_fields_tests$(EXE)
FIELD_DEBUG_TEST := $(BUILD)/io_fields_tests_debug$(EXE)
SETTINGS_OBJECTS := $(BUILD)/settings.o $(BUILD)/io_output_settings.o
$(BUILD)/settings.o $(BUILD)/report_explicit.o: source/eulersolver/linear_euler.h
$(BUILD)/settings.o: source/timestep/stabilization.h source/implicit/implicit_settings.h source/implicit/bdf1_step.h
OUTPUT_HEADERS := $(IO_DIR)/output_settings.h $(IO_DIR)/scheduled_output.h $(IO_DIR)/fields.h
$(BUILD)/io_output_settings.o: $(NUMBER_CODEC) $(OUTPUT_HEADERS) $(IO_DIR)/settings_text.h
ifneq ($(wildcard source/eulersolver/euler.h),)
SETTINGS_OBJECTS += $(BUILD)/io_staged_field_file.o $(BUILD)/io_scheduled_output.o
$(BUILD)/io_scheduled_output.o: $(OUTPUT_HEADERS) $(MESH_HEADERS) source/eulersolver/euler.h source/parallel/parallel_for.h
REPORT_OBJECTS := $(BUILD)/report.o
ifneq ($(wildcard source/explicit/explicit.h),)
REPORT_OBJECTS += $(BUILD)/report_explicit.o
endif
endif
$(BUILD)/report.o: $(IO_DIR)/report.cpp $(IO_DIR)/report.h source/eulersolver/euler.h source/parallel/parallel_for.h | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD)/report_explicit.o: $(IO_DIR)/report_explicit.cpp $(IO_DIR)/report.h source/explicit/explicit.h | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
ifneq ($(wildcard source/bc/bc.h),)
SETTINGS_OBJECTS += $(BUILD)/settings_boundary.o
endif

$(BUILD)/settings.o: source/parallel/modules.h $(wildcard source/viscosity/settings.h source/turbulence/sa.h source/turbulence/sigma.h source/turbulence/ddes.h)
$(BUILD)/settings.o: $(IO_DIR)/settings.cpp $(IO_DIR)/settings.h $(IO_DIR)/settings_text.h $(IO_DIR)/output_settings.h source/reconstruction/scheme.h | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/settings_boundary.o: $(IO_DIR)/settings_boundary.cpp $(IO_DIR)/settings_boundary.h $(IO_DIR)/settings_text.h source/bc/bc.h source/eulersolver/euler.h $(MESH_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/lzma_%.o: $(LZMA_SOURCE)/%.c $(IO_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) -x c -DZ7_ST $(build_optimization) -c $< -o $@

$(BUILD)/lzma_stream.o: source/io/lzma_stream.cpp $(IO_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/expression.o: source/io/expression.cpp source/io/expression.h | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/io_%.o: $(IO_DIR)/%.cpp $(FIELD_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/io_fields.o $(BUILD)/io_fields_tests.o: $(IO_DIR)/lzma_stream.h
$(BUILD)/io_fields_mesh.o $(BUILD)/io_fields_tests.o: $(MESH_HEADERS)

$(IO_LIBRARY): source/io/io.mk $(BUILD)/lzma_stream.o $(BUILD)/expression.o $(FIELD_OBJECTS) $(SETTINGS_OBJECTS) $(REPORT_OBJECTS) $(LZMA_OBJECTS)
	$(archiving)
	# Создаём архив заново: ar сохраняет старые объектные файлы после переименования.
	$(RM) "$@.tmp"
	$(AR) rcs "$@.tmp" $(filter %.o,$^)
	mv -f "$@.tmp" "$@"

$(FIELD_TEST): $(BUILD)/io_fields_tests.o $(BUILD)/mesh.o $(IO_LIBRARY)
	$(call action,Linking,$@)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

FIELD_DEBUG_SOURCES := $(IO_DIR)/fields_tests.cpp $(FIELD_SOURCES) $(MESH_SOURCE)/mesh.cpp $(IO_DIR)/lzma_stream.cpp
$(FIELD_DEBUG_TEST): $(FIELD_DEBUG_SOURCES) $(FIELD_HEADERS) $(MESH_HEADERS) $(IO_HEADERS) $(LZMA_OBJECTS) | $(BUILD)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O0 -g -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(FIELD_DEBUG_SOURCES) $(LZMA_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

.PHONY: io io-fields-test io-fields-debug-test output output-test output-debug-test
io: $(IO_LIBRARY)
io-fields-test: $(BIN)/$(notdir $(FIELD_TEST))
	$(running_test)
	./$(FIELD_TEST)
io-fields-debug-test: $(BIN)/$(notdir $(FIELD_DEBUG_TEST))
	$(running_test)
	./$(FIELD_DEBUG_TEST)

# Сохраняем прежние имена целей для существующих сценариев сборки.
output: io
output-test: io-fields-test
output-debug-test: io-fields-debug-test
all: $(BIN)/$(notdir $(FIELD_TEST))
test: io-fields-test
debug-test: io-fields-debug-test
$(BUILD)/io_fields_tests.o $(FIELD_DEBUG_TEST): qa/test_checks.h
