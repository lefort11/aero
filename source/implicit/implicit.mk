# Предобусловливатель собирается отдельно от физики потока. Для теста достаточно диагонали якобиана.
IMPLICIT_HEADERS := source/timestep/node_step.h source/implicit/implicit_settings.h source/implicit/partition.h source/implicit/block_jacobi.h source/implicit/block_jacobi_detail.h source/parallel/parallel_for.h source/parallel/kernel_math.h source/mesh/mesh_index.h
IMPLICIT_OBJECT := $(BUILD)/block_jacobi_$(EBR_TAG).o
SPATIAL_OBJECT := $(BUILD)/spatial_jacobian_$(EBR_TAG).o
BICGSTAB_OBJECT := $(BUILD)/bicgstab_$(EBR_TAG).o
NEWTON_OBJECT := $(BUILD)/bdf1_step_$(EBR_TAG).o
BICGSTAB_HEADERS := source/implicit/bdf1_linear_system.h source/implicit/bicgstab_vectors.h
SPATIAL_HEADERS := source/implicit/edge_jacobian.h source/implicit/csgs_order.h source/implicit/csgs_sweep.h source/implicit/roe_edge_jacobian.h source/implicit/spatial_jacobian.h source/implicit/spatial_jacobian_detail.h source/implicit/boundary_jacobian.h source/implicit/lf_edge_jacobian.h source/parallel/device_arrays.h $(RES_HDR)
IMPLICIT_LIBRARY := $(BUILD)/libaero_implicit_$(EBR_TAG).a
IMPLICIT_TEST := $(BUILD)/implicit_tests_$(EBR_TAG)$(EXE)
ifeq ($(kokkos_gpu),1)
IMPLICIT_FLAGS := -DAERO_WITH_KOKKOS_GPU
IMPLICIT_TEST_FLAGS := $(EBR_FLAGS)
IMPLICIT_TEST_LIBS := $(EBR_LIBS)
IMPLICIT_HEADERS += source/parallel/device_api.h
IMPLICIT_GPU_DEPS := $(KOKKOS_DEVICE_LIBRARY)
endif
$(IMPLICIT_OBJECT): source/implicit/block_jacobi.cpp $(IMPLICIT_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(IMPLICIT_FLAGS) -c $< -o $@
$(SPATIAL_OBJECT): source/implicit/spatial_jacobian.cpp $(SPATIAL_HEADERS) $(IMPLICIT_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(IMPLICIT_FLAGS) -c $< -o $@
$(BICGSTAB_OBJECT): source/implicit/bicgstab.cpp $(BICGSTAB_HEADERS) $(SPATIAL_HEADERS) $(IMPLICIT_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(IMPLICIT_FLAGS) -c $< -o $@
$(NEWTON_OBJECT): source/implicit/bdf1_step.cpp source/implicit/bdf1_step.h source/implicit/newton_vectors.h $(BICGSTAB_HEADERS) $(SPATIAL_HEADERS) $(IMPLICIT_HEADERS) | $(BUILD)
	$(compiling)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(IMPLICIT_FLAGS) -c $< -o $@
$(IMPLICIT_LIBRARY): $(IMPLICIT_OBJECT) $(SPATIAL_OBJECT) $(BICGSTAB_OBJECT) $(NEWTON_OBJECT)
	$(archiving)
	$(AR) rcs $@ $^
IMPLICIT_TEST_SOURCES := source/implicit/csgs_tests.cpp source/implicit/block_jacobi_tests.cpp source/implicit/lf_edge_jacobian_tests.cpp
IMPLICIT_TEST_SOURCES += source/implicit/spatial_jacobian_tests.cpp
IMPLICIT_TEST_SOURCES += source/implicit/bdf1_linear_tests.cpp
IMPLICIT_TEST_SOURCES += source/implicit/bdf1_step_tests.cpp
$(IMPLICIT_TEST): $(BICGSTAB_HEADERS)
$(IMPLICIT_TEST): qa/test_field.h $(SPATIAL_HEADERS) $(RES_LIB) $(RES_DEPS)
$(IMPLICIT_TEST): $(IMPLICIT_TEST_SOURCES) source/implicit/lf_edge_jacobian.h source/eulersolver/euler.h source/eulersolver/lax_friedrichs.h qa/device_cases.h qa/test_checks.h qa/device_test_buffer.h $(IMPLICIT_HEADERS) $(IMPLICIT_LIBRARY) $(IMPLICIT_GPU_DEPS)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(IMPLICIT_TEST_FLAGS) $(IMPLICIT_TEST_SOURCES) $(IMPLICIT_LIBRARY) $(RES_LIB) $(RES_DEPS) $(LDFLAGS) $(IMPLICIT_TEST_LIBS) $(LDLIBS) -o $@
.PHONY: implicit implicit-test
implicit: $(IMPLICIT_LIBRARY)
implicit-test: $(IMPLICIT_TEST) | bin-runtime
	$(running_test)
	"$(IMPLICIT_TEST)"
test: implicit-test
