TECPLOT_SOURCE := utils/tecplot
TECPLOT_HEADERS := $(TECPLOT_SOURCE)/tecplot.h $(MESH_SOURCE)/mesh.h $(FIELD_HEADERS)
TECPLOT_OBJECTS := $(BUILD)/tecplot_writer_omp$(WITH_OPENMP).o $(BUILD)/tecplot_cli_omp$(WITH_OPENMP).o

$(BUILD)/tecplot_writer_omp$(WITH_OPENMP).o: $(TECPLOT_SOURCE)/tecplot.cpp $(TECPLOT_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD)/tecplot_%_omp$(WITH_OPENMP).o: $(TECPLOT_SOURCE)/%.cpp $(TECPLOT_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: tecplot tecplot-test tecplot-debug-test
tecplot: $(BUILD)/tecplot_main_omp$(WITH_OPENMP).o $(TECPLOT_OBJECTS) $(BUILD)/mesh.o $(IO_LIBRARY) | bin-runtime
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $(BIN)/fields_to_tecplot$(EXE)
tecplot-test: $(BUILD)/tecplot_tecplot_tests_omp$(WITH_OPENMP).o $(TECPLOT_OBJECTS) $(BUILD)/mesh.o $(IO_LIBRARY) | bin-runtime
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $(BIN)/tecplot_tests$(EXE)
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(BIN))\tecplot_tests$(EXE)
else
	./$(BIN)/tecplot_tests$(EXE)
endif
tecplot-debug-test: $(LZMA_OBJECTS) | $(BUILD) bin-runtime
	$(CXX) $(CPPFLAGS) -std=c++11 -pedantic-errors -O0 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(TECPLOT_SOURCE)/tecplot.cpp $(TECPLOT_SOURCE)/cli.cpp $(TECPLOT_SOURCE)/tecplot_tests.cpp $(FIELD_SOURCES) $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp $(LZMA_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $(BIN)/tecplot_tests_debug$(EXE)
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(BIN))\tecplot_tests_debug$(EXE)
else
	./$(BIN)/tecplot_tests_debug$(EXE)
endif
all: tecplot
test: tecplot-test
debug-test: tecplot-debug-test
