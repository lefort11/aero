MESHDEC_DIR := utils/meshdec
MESHDEC_HEADERS := $(MESHDEC_DIR)/meshdec.h $(MESHDEC_DIR)/storage.h $(MESH_SOURCE)/mesh_reader.h $(MESH_SOURCE)/mesh.h $(IO_HEADERS)
MESHDEC_SOURCES := $(addprefix $(MESHDEC_DIR)/,storage.cpp ingest.cpp order.cpp archive.cpp)
MESHDEC_OBJECTS := $(patsubst $(MESHDEC_DIR)/%.cpp,$(BUILD)/meshdec_%.o,$(MESHDEC_SOURCES))
WITH_MPI ?= 0
MPI_CXX ?= mpicxx
ifeq ($(WITH_MPI),1)
ifeq ($(OS),Windows_NT)
MESHDEC_CXX = $(CXX)
MPI_CPPFLAGS ?= -I"$(subst \,/,$(MSMPI_INC))."
MPI_LIBS ?= "$(subst \,/,$(MSMPI_LIB64))msmpi.lib"
else
MESHDEC_CXX = $(MPI_CXX)
endif
MESHDEC_MPI_FLAGS = -DAERO_MESHDEC_MPI $(MPI_CPPFLAGS)
else
MESHDEC_CXX = $(CXX)
endif

$(BUILD)/meshdec_%.o: $(MESHDEC_DIR)/%.cpp $(MESHDEC_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/meshdec_main_mpi$(WITH_MPI).o: $(MESHDEC_DIR)/main.cpp $(MESHDEC_HEADERS) | $(BUILD)
	$(MESHDEC_CXX) $(CPPFLAGS) $(MESHDEC_MPI_FLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: meshdec meshdec-test meshdec-debug-test
meshdec: $(BUILD)/meshdec_main_mpi$(WITH_MPI).o $(MESHDEC_OBJECTS) $(IO_LIBRARY) | bin-runtime
	$(MESHDEC_CXX) $(LDFLAGS) $^ $(if $(filter 1,$(WITH_MPI)),$(MPI_LIBS)) $(LDLIBS) -o $(BIN)/meshdec$(EXE)

$(BUILD)/meshdec_tests$(EXE): $(BUILD)/meshdec_meshdec_tests.o $(MESHDEC_OBJECTS) $(BUILD)/mesh.o $(IO_LIBRARY)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/meshdec_tests_debug$(EXE): $(MESHDEC_DIR)/meshdec_tests.cpp $(MESHDEC_SOURCES) $(MESHDEC_HEADERS) $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp $(LZMA_OBJECTS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O0 -g -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS $(filter %.cpp %.o,$^) $(LDFLAGS) $(LDLIBS) -o $@

meshdec-test: $(BIN)/meshdec_tests$(EXE)
	"$(BUILD)/meshdec_tests$(EXE)"
meshdec-debug-test: $(BIN)/meshdec_tests_debug$(EXE)
	"$(BUILD)/meshdec_tests_debug$(EXE)"
all programs: meshdec
test: meshdec-test
debug-test: meshdec-debug-test
