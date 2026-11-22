# Утилита читает все листья .mdec в одном процессе через штатный загрузчик разделов.
SLICE_SOURCES := utils/submesh/main.cpp source/mesh/mdec.cpp source/mesh/partition.cpp source/mesh/leaf_balance.cpp source/parallel/processes.cpp
SLICE_HEADERS := source/io/submesh.h source/io/scheduled_output.h source/mesh/partition.h source/mesh/mdec.h source/mesh/leaf_balance.h source/mesh/archive_format.h source/parallel/processes.h $(MESH_HEADERS) $(FIELD_HEADERS)
SLICE_OBJECTS := $(addprefix $(BUILD)/slice/,$(SLICE_SOURCES:.cpp=.o))
$(BUILD)/slice/%.o: %.cpp $(SLICE_HEADERS) | $(BUILD)
	@mkdir -p "$(@D)"
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c "$<" -o "$@"
$(BUILD)/mesh_slice$(EXE): $(SLICE_OBJECTS) $(BUILD)/mesh.o $(IO_LIBRARY)
	$(call action,Linking,$@)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o "$@"
.PHONY: mesh_slice
mesh_slice: $(BIN)/mesh_slice$(EXE)
programs all: mesh_slice
