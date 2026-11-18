AERO_SOURCES := source/aero/main.cpp source/aero/startup.cpp
AERO_HEADERS := source/aero/startup.h source/io/settings.h source/io/settings_text.h source/io/settings_boundary.h source/io/flowreport.h $(OUTPUT_HEADERS) $(INITIAL_HEADERS) $(RES_HDR) $(TIMESTEP_HEADERS) $(EXPLICIT_HEADERS)
AERO_LIBS := $(RES_LIB) $(INITIAL_LIB) $(TIMESTEP_LIBRARY) $(EXPLICIT_LIBRARY) $(IO_LIBRARY) $(RES_DEPS)
AERO_PROGRAM := $(BUILD)/aero_$(EBR_TAG)$(EXE)
$(AERO_PROGRAM): $(AERO_SOURCES) $(AERO_HEADERS) $(AERO_LIBS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EBR_FLAGS) $(AERO_SOURCES) $(LDFLAGS) $(AERO_LIBS) $(EBR_LIBS) $(LDLIBS) -o $@
$(BIN)/aero$(EXE): $(AERO_PROGRAM) publish-programs | bin-runtime
	$(publish_exe)
.PHONY: aero
aero: $(BIN)/aero$(EXE)
all: aero
programs: aero

AERO_TEST := $(BUILD)/aero_tests$(EXE)
$(AERO_TEST): source/aero/aero_tests.cpp $(MESH_SOURCE)/mesh.h $(FIELD_HEADERS) $(BUILD)/mesh.o $(IO_LIBRARY)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/mesh.o $(IO_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@
.PHONY: aero-test
aero-test: aero $(BIN)/aero_tests$(EXE)
	"$(BIN)/aero_tests$(EXE)" "$(BIN)/aero$(EXE)"
ifeq ($(MINGW_CUDA),1)
	CUDA_VISIBLE_DEVICES=-1 "$(BIN)/aero_tests$(EXE)" "$(BIN)/aero$(EXE)" --expect-cpu-fallback
endif
test: aero-test
