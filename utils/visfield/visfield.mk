# Утилита и решатель используют одни писатели VTK и Tecplot из библиотеки ввода-вывода.
TECPLOT_SOURCE := utils/visfield
TECPLOT_HEADERS := $(VIS_HEADERS) $(TECPLOT_SOURCE)/cli.h $(MESH_HEADERS) $(FIELD_HEADERS)
TECPLOT_HEADERS += source/io/submesh.h source/io/scheduled_output.h source/io/output_settings.h
TECPLOT_OBJECTS := $(BUILD)/tecplot_cli_omp$(WITH_OPENMP).o

.PHONY: generate-tecplot-examples
generate-tecplot-examples: $(BIN)/generate_tecplot_examples$(EXE)
$(BUILD)/generate_tecplot_examples$(EXE): $(TECPLOT_SOURCE)/examples/generate_examples.cpp $(BUILD)/mesh.o $(IO_LIBRARY)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD)/tecplot_%_omp$(WITH_OPENMP).o: $(TECPLOT_SOURCE)/%.cpp $(TECPLOT_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: tecplot tecplot-test tecplot-debug-test
visfield: $(BUILD)/tecplot_main_omp$(WITH_OPENMP).o $(TECPLOT_OBJECTS) $(BUILD)/mesh.o $(IO_LIBRARY) | bin-runtime
	$(call action,Linking,$(BIN)/visfield$(EXE))
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $(BIN)/visfield$(EXE)
tecplot-test: $(BUILD)/tecplot_tecplot_tests_omp$(WITH_OPENMP).o $(TECPLOT_OBJECTS) $(BUILD)/mesh.o $(IO_LIBRARY) | bin-runtime
	$(call action,Linking,$(BIN)/tecplot_tests$(EXE))
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $(BIN)/tecplot_tests$(EXE)
	$(running_test)
	"$(BIN)/tecplot_tests$(EXE)"
tecplot-debug-test: $(LZMA_OBJECTS) | $(BUILD) bin-runtime
	$(call action,Compiling and linking,$(BIN)/tecplot_tests_debug$(EXE))
	$(CXX) $(CPPFLAGS) -std=c++11 -pedantic-errors -O0 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(VIS_WRITERS) $(TECPLOT_SOURCE)/cli.cpp $(TECPLOT_SOURCE)/tecplot_tests.cpp $(FIELD_SOURCES) source/io/staged_field_file.cpp $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp $(LZMA_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $(BIN)/tecplot_tests_debug$(EXE)
	$(running_test)
	"$(BIN)/tecplot_tests_debug$(EXE)"
# Старые цели Make по-прежнему собирают переименованную программу.
.PHONY: visfield visfield-test visfield-debug-test
tecplot: visfield
visfield-test: tecplot-test
visfield-debug-test: tecplot-debug-test
all: visfield
test: tecplot-test
debug-test: tecplot-debug-test
$(BUILD)/tecplot_tecplot_tests_omp$(WITH_OPENMP).o: tests/test_checks.h tests/mesh_fixture.h tests/test_files.h tests/test_suite.h
$(BUILD)/tecplot_cli_omp$(WITH_OPENMP).o: source/io/cli_help.h
