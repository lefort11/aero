# Конвертер Tecplot использует общие библиотеки сетки и полей и может выгружать зоны параллельно.
TECPLOT_SOURCE := utils/tecplot
TECPLOT_HEADERS := $(TECPLOT_SOURCE)/tecplot.h $(MESH_HEADERS) $(FIELD_HEADERS)
TECPLOT_OBJECTS := $(BUILD)/tecplot_writer_omp$(WITH_OPENMP).o $(BUILD)/tecplot_cli_omp$(WITH_OPENMP).o

.PHONY: generate-tecplot-examples
generate-tecplot-examples: $(BIN)/generate_tecplot_examples$(EXE)
$(BUILD)/generate_tecplot_examples$(EXE): $(TECPLOT_SOURCE)/examples/generate_examples.cpp $(BUILD)/mesh.o $(IO_LIBRARY)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD)/tecplot_writer_omp$(WITH_OPENMP).o: $(TECPLOT_SOURCE)/tecplot.cpp $(TECPLOT_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD)/tecplot_%_omp$(WITH_OPENMP).o: $(TECPLOT_SOURCE)/%.cpp $(TECPLOT_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: tecplot tecplot-test tecplot-debug-test
tecplot: $(BUILD)/tecplot_main_omp$(WITH_OPENMP).o $(TECPLOT_OBJECTS) $(BUILD)/mesh.o $(IO_LIBRARY) | bin-runtime
	$(call action,Linking,$(BIN)/fields_to_tecplot$(EXE))
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $(BIN)/fields_to_tecplot$(EXE)
tecplot-test: $(BUILD)/tecplot_tecplot_tests_omp$(WITH_OPENMP).o $(TECPLOT_OBJECTS) $(BUILD)/mesh.o $(IO_LIBRARY) | bin-runtime
	$(call action,Linking,$(BIN)/tecplot_tests$(EXE))
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $(BIN)/tecplot_tests$(EXE)
	$(running_test)
	./$(BIN)/tecplot_tests$(EXE)
tecplot-debug-test: $(LZMA_OBJECTS) | $(BUILD) bin-runtime
	$(call action,Compiling and linking,$(BIN)/tecplot_tests_debug$(EXE))
	$(CXX) $(CPPFLAGS) -std=c++11 -pedantic-errors -O0 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(TECPLOT_SOURCE)/tecplot.cpp $(TECPLOT_SOURCE)/cli.cpp $(TECPLOT_SOURCE)/tecplot_tests.cpp $(FIELD_SOURCES) source/io/staged_field_file.cpp $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp $(LZMA_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $(BIN)/tecplot_tests_debug$(EXE)
	$(running_test)
	./$(BIN)/tecplot_tests_debug$(EXE)
all: tecplot
test: tecplot-test
debug-test: tecplot-debug-test
$(BUILD)/tecplot_tecplot_tests_omp$(WITH_OPENMP).o: qa/test_checks.h
