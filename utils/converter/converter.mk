# CGNS нужен конвертеру, библиотека сетки и солвер собираются самостоятельно.
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
CONVERTER_HEADERS := $(CONVERTER_SOURCE)/converter.h $(CONVERTER_SOURCE)/detail.h $(CONVERTER_SOURCE)/cli.h $(MESH_HEADERS)
CONVERTER_OBJECTS := $(patsubst $(CONVERTER_SOURCE)/%.cpp,$(BUILD)/converter_%_$(WITH_CGNS).o,$(CONVERTER_SOURCES))
$(BUILD)/converter_another_$(WITH_CGNS).o: $(wildcard $(LZMA_SOURCE)/*.h)
$(BUILD)/converter_cgns_$(WITH_CGNS).o $(BUILD)/converter_converter_tests_$(WITH_CGNS).o: $(CGNS_BUILD_DEPS)

$(BUILD)/converter_%_$(WITH_CGNS).o: $(CONVERTER_SOURCE)/%.cpp $(CONVERTER_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CONVERTER_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: converter converter-test converter-debug-test
# Повторно линкуем выбранный вариант при смене WITH_CGNS.
converter: $(BUILD)/mesh.o $(CONVERTER_OBJECTS) $(BUILD)/converter_main_$(WITH_CGNS).o $(IO_LIBRARY) | bin-runtime
	$(call action,Linking,$(BIN)/mesh_converter$(EXE))
	$(CXX) $(LDFLAGS) $^ $(CONVERTER_LDLIBS) $(LDLIBS) -o $(BIN)/mesh_converter$(EXE)

converter-test: $(BUILD)/mesh.o $(CONVERTER_OBJECTS) $(BUILD)/converter_converter_tests_$(WITH_CGNS).o $(IO_LIBRARY) | bin-runtime
	$(call action,Linking,$(BIN)/converter_tests$(EXE))
	$(CXX) $(LDFLAGS) $^ $(CONVERTER_LDLIBS) $(LDLIBS) -o $(BIN)/converter_tests$(EXE)
	$(running_test)
	./$(BIN)/converter_tests$(EXE)

converter-debug-test: $(LZMA_OBJECTS) | $(BUILD) bin-runtime $(CGNS_BUILD_DEPS)
	$(call action,Compiling and linking,$(BIN)/converter_tests_debug$(EXE))
	$(CXX) $(CPPFLAGS) $(CONVERTER_CPPFLAGS) -std=c++11 -pedantic-errors -O0 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(MESH_SOURCE)/mesh.cpp $(CONVERTER_SOURCES) $(CONVERTER_SOURCE)/converter_tests.cpp source/io/lzma_stream.cpp $(LZMA_OBJECTS) $(LDFLAGS) $(CONVERTER_LDLIBS) $(LDLIBS) -o $(BIN)/converter_tests_debug$(EXE)
	$(running_test)
	./$(BIN)/converter_tests_debug$(EXE)
$(BUILD)/converter_converter_tests_$(WITH_CGNS).o: qa/test_checks.h
