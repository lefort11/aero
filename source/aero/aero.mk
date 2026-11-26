# Сборка солвера объединяет геометрию, невязку и интегратор. Цели mpi-*-test запускают сквозные расчёты.
AERO_SOURCES := source/aero/exact_error.cpp source/aero/main.cpp source/aero/startup.cpp source/aero/solver_timings.cpp source/aero/time_loop.cpp
AERO_SOURCES += source/aero/run_control.cpp source/aero/recovery.cpp source/aero/indexed_io.cpp source/aero/mpi_fields.cpp source/aero/solver.cpp source/aero/momentum_frame.cpp
AERO_SOURCES += source/aero/mpi_periodic.cpp source/aero/mpi_periodic_match.cpp
AERO_SOURCES += source/aero/partition_state.cpp source/aero/partition_output.cpp source/aero/partition_operator.cpp
AERO_SOURCES += source/aero/mpi_stencil.cpp source/aero/partition_report.cpp source/aero/mpi_balance.cpp source/mesh/leaf_balance.cpp
AERO_SOURCES += source/aero/geometry.cpp source/aero/case_geometry.cpp source/mesh/mdec.cpp source/mesh/partition.cpp source/parallel/processes.cpp
AERO_HEADERS := source/parallel/backend.h source/aero/run_control.h source/aero/exact_error.h source/io/expression.h $(MESH_SOURCE)/archive_format.h $(NUMBER_CODEC) source/aero/solver_timings.h source/aero/startup.h source/io/settings.h source/io/settings_text.h source/io/settings_boundary.h source/io/report.h $(OUTPUT_HEADERS) $(INITIAL_HEADERS) $(RES_HDR) $(TIMESTEP_HEADERS) $(EXPLICIT_HEADERS)
AERO_LIBS := $(IMPLICIT_LIBRARY) $(RES_LIB) $(INITIAL_LIB) $(TIMESTEP_LIBRARY) $(EXPLICIT_LIBRARY) $(IO_LIBRARY) $(RES_DEPS)
AERO_SOURCES += $(wildcard source/viscosity/aes.cpp source/viscosity/grad.cpp source/viscosity/affine_hex.cpp) $(if $(wildcard source/viscosity/transport.h),$(wildcard source/turbulence/*.cpp))
AERO_HEADERS += source/io/flow_values.h
AERO_HEADERS += source/io/submesh.h source/aero/partition_output.h
AERO_HEADERS += source/aero/console_log.h
AERO_HEADERS += source/implicit/implicit_settings.h source/implicit/bdf1_step.h source/implicit/partition.h
AERO_HEADERS += source/aero/geometry.h source/aero/case_geometry.h source/mesh/mdec.h source/mesh/partition.h source/mesh/mesh_reader.h source/parallel/processes.h
AERO_HEADERS += source/aero/recovery.h source/aero/indexed_io.h source/aero/mpi_fields.h source/aero/solver.h source/timestep/cfl_detail.h source/explicit/explicit_stage.h source/residual/residual_detail.h
AERO_HEADERS += source/aero/partition_execution.h source/parallel/row_loop.h
AERO_HEADERS += source/aero/partition_kernels.h source/aero/time_loop.h source/aero/mpi_periodic.h source/aero/mpi_periodic_detail.h
AERO_HEADERS += source/aero/partition_state.h source/aero/partition_output.h source/aero/partition_operator.h source/aero/partition_transfers.h
AERO_HEADERS += source/aero/mpi_stencil.h source/aero/partition_report.h source/aero/mpi_balance.h source/mesh/leaf_balance.h
AERO_HEADERS += source/timestep/steady_step_spread.h
AERO_HEADERS += source/aero/compact_slopes.h $(wildcard source/viscosity/affine_hex.h)
ifeq ($(kokkos_gpu),1)
AERO_GPU := $(BUILD)/partition_kernels.o
AERO_SOURCES += source/aero/partition_gpu.cpp
AERO_HEADERS += source/aero/partition_gpu.h
$(AERO_GPU): source/aero/partition_kernels.cpp source/aero/partition_kernels.h $(RES_HDR) $(TIMESTEP_HEADERS) source/timestep/cfl_detail.h source/explicit/explicit_stage.h | $(BUILD)
	$(compiling)
	$(compile_device_host)
endif
AERO_VARIANT := aero_$(EBR_TAG)$(if $(filter 1,$(WITH_MPI)),_mpi)
AERO_PROGRAM := $(BUILD)/$(AERO_VARIANT)$(EXE)
# Раздельные объекты позволяют Make распределить исходники по ядрам и показать
# имя каждого файла. Отдельный каталог защищает от смешивания вариантов MPI.
AERO_OBJECT_DIR := $(BUILD)/$(AERO_VARIANT)_objects
AERO_OBJECTS := $(patsubst source/%.cpp,$(AERO_OBJECT_DIR)/%.o,$(AERO_SOURCES))
$(AERO_OBJECT_DIR)/aero/solver.o $(AERO_OBJECT_DIR)/aero/partition_operator.o: source/io/steady_report.h source/io/diagnostic_fields.h
$(AERO_OBJECT_DIR)/aero/partition_operator.o: source/residual/step_execution.h
$(AERO_OBJECT_DIR)/aero/partition_gpu.o: source/reconstruction/device_stencil.h
$(AERO_OBJECT_DIR)/mesh/mdec.o: $(NATIVE_FORMAT)
$(AERO_OBJECT_DIR)/turbulence/partition.o: source/timestep/bdf_weights.h
$(AERO_OBJECT_DIR)/%.o: source/%.cpp $(AERO_HEADERS) | $(BUILD)
	mkdir -p "$(@D)"
	$(compiling)
	$(MPI_HOST_CXX) $(CPPFLAGS) $(if $(filter 1,$(WITH_MPI)),-DAERO_WITH_MPI $(MPI_CPPFLAGS)) $(CXXFLAGS) $(EBR_FLAGS) -c $< -o $@
AERO_MATRIX_TEST := $(BUILD)/mpi_matrix_tests_$(EBR_TAG)$(EXE)
$(AERO_PROGRAM): $(AERO_OBJECT_DIR)/aero/main.o
$(AERO_MATRIX_TEST): $(AERO_OBJECT_DIR)/implicit/mpi_matrix_tests.o
$(AERO_OBJECT_DIR)/implicit/mpi_matrix_tests.o: source/implicit/bdf1_linear_system.h tests/test_field.h
$(AERO_PROGRAM) $(AERO_MATRIX_TEST): $(filter-out $(AERO_OBJECT_DIR)/aero/main.o,$(AERO_OBJECTS)) $(AERO_LIBS) $(AERO_GPU) $(KOKKOS_DEVICE_LIBRARY)
	$(call action,Linking,$@)
	$(MPI_HOST_CXX) $(CXXFLAGS) $(filter-out $(AERO_LIBS),$(filter %.o,$^)) $(LDFLAGS) $(AERO_LIBS) $(EBR_LIBS) $(if $(filter 1,$(WITH_MPI)),$(MPI_LIBS)) $(LDLIBS) $(if $(filter Windows_NT,$(OS)),-lpsapi) -o $@
$(BIN)/aero$(EXE): $(AERO_PROGRAM) publish-programs | bin-runtime
	$(publishing)
	$(publish_exe)
.PHONY: aero
aero: $(BIN)/aero$(EXE)
all: aero
programs: aero

AERO_TEST := $(BUILD)/aero_tests$(EXE)
$(AERO_TEST): source/aero/aero_tests.cpp source/aero/partition_kernels.h $(RES_HDR) source/explicit/explicit_stage.h source/mesh/leaf_balance.cpp source/mesh/leaf_balance.h source/mesh/partition.h source/mesh/mdec.h $(MESH_HEADERS) $(FIELD_HEADERS) $(BUILD)/mesh.o $(IO_LIBRARY)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< source/mesh/leaf_balance.cpp $(BUILD)/mesh.o $(IO_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@
.PHONY: aero-test
aero-test: aero $(BIN)/aero_tests$(EXE)
	$(running_test)
	"$(BIN)/aero_tests$(EXE)" "$(BIN)/aero$(EXE)"
ifeq ($(MINGW_CUDA),1)
	$(running_test)
	CUDA_VISIBLE_DEVICES=-1 "$(BIN)/aero_tests$(EXE)" "$(BIN)/aero$(EXE)" --expect-cpu-fallback
endif
test: aero-test

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)
MPIEXEC ?= mpiexec
.PHONY: mpi-gpu-test
ifeq ($(kokkos_gpu)$(WITH_MPI),11)
mpi-gpu-test: aero meshdec
	$(running_test)
	"$(PYTHON)" qa/mpi_gpu.py --aero "$(BIN)/aero$(EXE)" --meshdec "$(BIN)/meshdec$(EXE)" --mpiexec "$(MPIEXEC)"
	$(running_test)
	"$(PYTHON)" qa/mpi_timing.py --devices=0,1 --aero "$(BIN)/aero$(EXE)" --meshdec "$(BIN)/meshdec$(EXE)" --mpiexec "$(MPIEXEC)"
else
mpi-gpu-test:
	$(MAKE) KOKKOS_BACKEND=$(if $(filter hip,$(KOKKOS_BACKEND)),hip,cuda) WITH_MPI=1 $@
endif
MPI_RECONSTRUCTION_TESTS := mpi-ebr3-test mpi-ebr5-test mpi-weno3-test mpi-weno5-test mpi-mp5-test mpi-mpweno5-test
.PHONY: mpi-timing-test mpi-fields-test mpi-recovery-test mpi-geometry-test mpi-balance-test mpi-euler-test mpi-ssprk3-test mpi-bdf1-test mpi-bdf2-test mpi-linear-test mpi-periodic-test $(MPI_RECONSTRUCTION_TESTS)
ifeq ($(WITH_MPI),1)
# Для проверок с одноимённым Python-сценарием берём его имя из выбранной цели Make.
MPI_QA_ARGS := --aero "$(BIN)/aero$(EXE)" --meshdec "$(BIN)/meshdec$(EXE)" --mpiexec "$(MPIEXEC)"
mpi-timing-test mpi-fields-test mpi-recovery-test mpi-balance-test mpi-geometry-test: aero meshdec
	$(running_test)
	"$(PYTHON)" qa/mpi_$(patsubst mpi-%-test,%,$@).py $(MPI_QA_ARGS)
mpi-linear-test: aero meshdec
	$(running_test)
	"$(PYTHON)" qa/linear_euler.py $(MPI_QA_ARGS) $(if $(MPI_BDF1_DEVICES),--devices=$(MPI_BDF1_DEVICES))
mpi-bdf1-test mpi-bdf2-test: $(AERO_MATRIX_TEST)
mpi-bdf1-test mpi-bdf2-test: matrix_args = --matrix-test "$(AERO_MATRIX_TEST)" $(if $(MPI_BDF1_DEVICES),--devices=$(MPI_BDF1_DEVICES))
mpi-euler-test mpi-ssprk3-test mpi-bdf1-test mpi-bdf2-test: aero meshdec
	$(running_test)
	"$(PYTHON)" qa/mpi_euler.py $(MPI_QA_ARGS) --time-scheme $(patsubst mpi-%-test,%,$@) $(matrix_args)
$(MPI_RECONSTRUCTION_TESTS): aero meshdec
	$(running_test)
	"$(PYTHON)" qa/mpi_periodic.py $(MPI_QA_ARGS) --$(patsubst mpi-%-test,%,$@)
mpi-periodic-test: aero meshdec
	$(running_test)
	"$(PYTHON)" qa/mpi_periodic.py $(MPI_QA_ARGS)
test: mpi-timing-test mpi-fields-test mpi-recovery-test mpi-geometry-test mpi-balance-test mpi-euler-test mpi-ssprk3-test mpi-bdf1-test mpi-bdf2-test mpi-linear-test mpi-periodic-test $(MPI_RECONSTRUCTION_TESTS)
else
mpi-timing-test mpi-fields-test mpi-recovery-test mpi-geometry-test mpi-balance-test mpi-euler-test mpi-ssprk3-test mpi-bdf1-test mpi-bdf2-test mpi-linear-test mpi-periodic-test $(MPI_RECONSTRUCTION_TESTS):
	$(MAKE) WITH_MPI=1 $@
endif
$(AERO_TEST): source/parallel/backend.h tests/test_checks.h source/aero/steady_forecast.h
$(AERO_OBJECT_DIR)/aero/time_loop.o: source/aero/steady_forecast.h
.PHONY: completion-forecast-test curved-stencil-test
completion-forecast-test: aero
	$(running_test)
	"$(PYTHON)" qa/completion_forecast.py --aero "$(BIN)/aero$(EXE)"
ifeq ($(WITH_MPI),1)
curved-stencil-test: aero meshdec
	$(running_test)
	"$(PYTHON)" qa/curved_stencil.py $(MPI_QA_ARGS)
test: curved-stencil-test
else
curved-stencil-test:
	$(MAKE) WITH_MPI=1 $@
endif
test: completion-forecast-test
-include source/viscosity/tests.mk
ifneq ($(wildcard source/viscosity/transport.h),)
-include source/turbulence/tests.mk
endif

# Сверка физических величин на выходе, в том числе при продолжении на других рангах.
.PHONY: output-diagnostics-qa
output-diagnostics-qa: aero meshdec
	$(running_test)
	"$(PYTHON)" qa/output_diagnostics.py --aero "$(BIN)/aero$(EXE)" --meshdec "$(BIN)/meshdec$(EXE)" $(if $(filter 1,$(WITH_MPI)),--mpi --mpiexec "$(MPIEXEC)") $(if $(filter 1,$(kokkos_gpu)),--gpu $(if $(MPI_BDF1_DEVICES),--devices=$(MPI_BDF1_DEVICES)))
test: output-diagnostics-qa

.PHONY: local-step-neighbors-qa
local-step-neighbors-qa: aero $(if $(filter 1,$(WITH_MPI)),meshdec)
	$(running_test)
	"$(PYTHON)" qa/local_step_neighbors.py $(if $(filter 1,$(WITH_MPI)),--mpi $(MPI_QA_ARGS),--aero "$(BIN)/aero$(EXE)") $(if $(filter 1,$(kokkos_gpu)),--gpu)
test: local-step-neighbors-qa

ifneq ($(wildcard source/multigrid/coarse_step.h),)
.PHONY: steady-multigrid-qa
steady-multigrid-qa: aero $(if $(filter 1,$(WITH_MPI)),meshdec)
	$(running_test)
	"$(PYTHON)" qa/steady_multigrid.py $(if $(filter 1,$(WITH_MPI)),--mpi $(MPI_QA_ARGS),--aero "$(BIN)/aero$(EXE)") $(if $(filter 1,$(kokkos_gpu)),--gpu)
test: steady-multigrid-qa
endif
$(AERO_OBJECT_DIR)/aero/main.o: source/io/cli_help.h

$(AERO_OBJECT_DIR)/viscosity/aes.o: source/viscosity/row_execution.h source/parallel/device_affine_aes.h source/parallel/device_volume_coordinates.h source/parallel/row_loop.h
$(AERO_OBJECT_DIR)/turbulence/partition.o: source/turbulence/step_execution.h source/turbulence/convection.h source/parallel/row_loop.h
