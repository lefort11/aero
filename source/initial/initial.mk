INITIAL_DIR := source/initial
INITIAL_HEADERS := $(INITIAL_DIR)/initial.h source/io/expression.h $(FIELD_HEADERS) $(MESH_SOURCE)/mesh.h source/bc/periodic.h source/eulersolver/euler.h
INITIAL_LIB := $(BUILD)/libaero_initial.a
INITIAL_TEST := $(BUILD)/initial_tests$(EXE)
INITIAL_DEBUG := $(BUILD)/initial_tests_debug$(EXE)
$(BUILD)/initial.o: $(INITIAL_DIR)/initial.cpp $(INITIAL_HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(INITIAL_LIB): $(BUILD)/initial.o
	$(AR) rcs $@ $^
$(INITIAL_TEST): $(INITIAL_DIR)/initial_tests.cpp $(INITIAL_HEADERS) $(INITIAL_LIB) $(BC_LIB) $(CV_LIB) $(BUILD)/mesh.o $(IO_LIBRARY)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(filter %.cpp %.a %.o,$^) $(LDFLAGS) $(LDLIBS) -o $@
INITIAL_DEBUG_SRC := $(INITIAL_DIR)/initial_tests.cpp $(INITIAL_DIR)/initial.cpp source/io/expression.cpp $(BC_DIR)/periodic.cpp $(CV_DIR)/cv.cpp $(FIELD_SOURCES) $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp
$(INITIAL_DEBUG): $(INITIAL_DEBUG_SRC) $(INITIAL_HEADERS) $(CV_HEADERS) $(IO_HEADERS) $(LZMA_OBJECTS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O0 -g -D_GLIBCXX_DEBUG $(INITIAL_DEBUG_SRC) $(LZMA_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@
.PHONY: initial initial-test initial-debug-test
initial: $(INITIAL_LIB)
all: initial
test: initial-test
debug-test: initial-debug-test
initial-test: $(BIN)/$(notdir $(INITIAL_TEST))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(INITIAL_TEST))
else
	./$(INITIAL_TEST)
endif
initial-debug-test: $(BIN)/$(notdir $(INITIAL_DEBUG))
ifeq ($(USE_CMD),1)
	.\$(subst /,\,$(INITIAL_DEBUG))
else
	./$(INITIAL_DEBUG)
endif
