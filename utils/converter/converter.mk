# Included by the root Makefile. CGNS is isolated from the mesh/solver library.
CONVERTER_SOURCE := utils/converter
ifeq ($(OS),Windows_NT)
CGNS_PREFIX ?= build/deps/install
WITH_CGNS ?= $(if $(wildcard $(CGNS_PREFIX)/include/cgnslib.h),1,0)
else
include $(CONVERTER_SOURCE)/linux_dependencies.mk
CGNS_PREFIX ?= /usr/local
endif
CGNS_CPPFLAGS ?= -I$(CGNS_PREFIX)/include
CGNS_LDLIBS ?= -L$(CGNS_PREFIX)/lib -lcgns -lhdf5
ifeq ($(OS),Windows_NT)
CGNS_LDLIBS += -lws2_32 -lshlwapi
else
CGNS_LDLIBS += -ldl -lm
endif
ifeq ($(WITH_CGNS),1)
CONVERTER_CPPFLAGS := -DAERO_WITH_CGNS $(CGNS_CPPFLAGS)
CONVERTER_LDLIBS := $(CGNS_LDLIBS)
endif
CONVERTER_SOURCES := $(CONVERTER_SOURCE)/topology.cpp $(CONVERTER_SOURCE)/gmsh.cpp $(CONVERTER_SOURCE)/cgns.cpp $(CONVERTER_SOURCE)/another.cpp $(CONVERTER_SOURCE)/cli.cpp
CONVERTER_HEADERS := $(CONVERTER_SOURCE)/converter.h $(CONVERTER_SOURCE)/detail.h $(CONVERTER_SOURCE)/cli.h $(MESH_SOURCE)/mesh.h
CONVERTER_OBJECTS := $(patsubst $(CONVERTER_SOURCE)/%.cpp,$(BUILD)/converter_%_$(WITH_CGNS).o,$(CONVERTER_SOURCES))
$(BUILD)/converter_another_$(WITH_CGNS).o: $(wildcard $(LZMA_SOURCE)/*.h)
$(BUILD)/converter_cgns_$(WITH_CGNS).o $(BUILD)/converter_converter_tests_$(WITH_CGNS).o: $(CGNS_BUILD_DEPS)

$(BUILD)/converter_%_$(WITH_CGNS).o: $(CONVERTER_SOURCE)/%.cpp $(CONVERTER_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CONVERTER_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: converter converter-test converter-debug-test
# Always link the selected backend, including after WITH_CGNS changes.
converter: $(BUILD)/mesh.o $(CONVERTER_OBJECTS) $(BUILD)/converter_main_$(WITH_CGNS).o $(IO_LIBRARY) | bin-runtime
	$(CXX) $(LDFLAGS) $^ $(CONVERTER_LDLIBS) $(LDLIBS) -o $(BIN)/mesh_converter$(EXE)

converter-test: $(BUILD)/mesh.o $(CONVERTER_OBJECTS) $(BUILD)/converter_converter_tests_$(WITH_CGNS).o $(IO_LIBRARY) | bin-runtime
	$(CXX) $(LDFLAGS) $^ $(CONVERTER_LDLIBS) $(LDLIBS) -o $(BIN)/converter_tests$(EXE)
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(BIN))\converter_tests$(EXE)
else
	./$(BIN)/converter_tests$(EXE)
endif

converter-debug-test: $(LZMA_OBJECTS) | $(BUILD) bin-runtime $(CGNS_BUILD_DEPS)
	$(CXX) $(CPPFLAGS) $(CONVERTER_CPPFLAGS) -std=c++11 -pedantic-errors -O0 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(MESH_SOURCE)/mesh.cpp $(CONVERTER_SOURCES) $(CONVERTER_SOURCE)/converter_tests.cpp source/io/lzma_stream.cpp $(LZMA_OBJECTS) $(LDFLAGS) $(CONVERTER_LDLIBS) $(LDLIBS) -o $(BIN)/converter_tests_debug$(EXE)
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(BIN))\converter_tests_debug$(EXE)
else
	./$(BIN)/converter_tests_debug$(EXE)
endif
