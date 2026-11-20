# Начальные поля читаются из снимка или вычисляются по формулам. Проверки используют также периодические узлы.
INITIAL_DIR := source/initial
INITIAL_HEADERS := $(INITIAL_DIR)/initial.h source/io/expression.h $(FIELD_HEADERS) $(MESH_HEADERS) source/bc/periodic.h source/eulersolver/euler.h source/eulersolver/linear_euler.h
INITIAL_LIB := $(BUILD)/libaero_initial.a
INITIAL_TEST := $(BUILD)/initial_tests$(EXE)
INITIAL_DEBUG := $(BUILD)/initial_tests_debug$(EXE)
$(BUILD)/initial.o: $(INITIAL_DIR)/initial.cpp $(INITIAL_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(INITIAL_LIB): $(BUILD)/initial.o
	$(archiving)
	$(AR) rcs $@ $^
$(INITIAL_TEST): $(INITIAL_DIR)/initial_tests.cpp $(INITIAL_HEADERS) $(INITIAL_LIB) $(BC_LIB) $(CV_LIB) $(BUILD)/mesh.o $(IO_LIBRARY)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(filter %.cpp %.a %.o,$^) $(LDFLAGS) $(LDLIBS) -o $@
INITIAL_DEBUG_SRC := $(INITIAL_DIR)/initial_tests.cpp $(INITIAL_DIR)/initial.cpp source/io/expression.cpp $(BC_DIR)/periodic.cpp $(CV_DIR)/cv.cpp $(FIELD_SOURCES) $(MESH_SOURCE)/mesh.cpp source/io/lzma_stream.cpp
$(INITIAL_DEBUG): $(INITIAL_DEBUG_SRC) $(INITIAL_HEADERS) $(CV_HEADERS) $(IO_HEADERS) $(LZMA_OBJECTS) | $(BUILD)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O0 -g -D_GLIBCXX_DEBUG $(INITIAL_DEBUG_SRC) $(LZMA_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@
.PHONY: initial initial-test initial-debug-test
initial: $(INITIAL_LIB)
all: initial
test: initial-test
debug-test: initial-debug-test
initial-test: $(BIN)/$(notdir $(INITIAL_TEST))
	$(running_test)
	./$(INITIAL_TEST)
initial-debug-test: $(BIN)/$(notdir $(INITIAL_DEBUG))
	$(running_test)
	./$(INITIAL_DEBUG)
$(INITIAL_TEST) $(INITIAL_DEBUG): qa/test_checks.h
