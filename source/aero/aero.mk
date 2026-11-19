# Сборка солвера объединяет геометрию, невязку и интегратор. Цели mpi-*-test запускают сквозные расчёты.
AERO_SOURCES := source/aero/main.cpp source/aero/startup.cpp source/aero/solver_timings.cpp source/aero/time_loop.cpp
AERO_SOURCES += source/aero/recovery.cpp source/aero/indexed_io.cpp source/aero/mpi_fields.cpp source/aero/solver.cpp source/aero/momentum_frame.cpp
AERO_SOURCES += source/aero/mpi_periodic.cpp source/aero/mpi_periodic_match.cpp
AERO_SOURCES += source/aero/partition_state.cpp source/aero/partition_output.cpp source/aero/partition_operator.cpp
AERO_SOURCES += source/aero/mpi_stencil.cpp source/aero/partition_report.cpp source/aero/mpi_balance.cpp source/mesh/leaf_balance.cpp
AERO_SOURCES += source/aero/geometry.cpp source/aero/case_geometry.cpp source/mesh/mdec.cpp source/mesh/partition.cpp source/parallel/processes.cpp
AERO_HEADERS := $(MESH_SOURCE)/archive_format.h $(NUMBER_CODEC) source/aero/solver_timings.h source/aero/startup.h source/io/settings.h source/io/settings_text.h source/io/settings_boundary.h source/io/flowreport.h $(OUTPUT_HEADERS) $(INITIAL_HEADERS) $(RES_HDR) $(TIMESTEP_HEADERS) $(EXPLICIT_HEADERS)
AERO_LIBS := $(RES_LIB) $(INITIAL_LIB) $(TIMESTEP_LIBRARY) $(EXPLICIT_LIBRARY) $(IO_LIBRARY) $(RES_DEPS)
AERO_HEADERS += source/aero/geometry.h source/aero/case_geometry.h source/mesh/mdec.h source/mesh/partition.h source/mesh/mesh_reader.h source/parallel/processes.h
AERO_HEADERS += source/aero/recovery.h source/aero/indexed_io.h source/aero/mpi_fields.h source/aero/solver.h source/timestep/cfl_detail.h source/explicit/explicit_stage.h source/residual/residual_detail.h
AERO_HEADERS += source/aero/partition_kernels.h source/aero/time_loop.h source/aero/mpi_periodic.h source/aero/mpi_periodic_detail.h
AERO_HEADERS += source/aero/partition_state.h source/aero/partition_output.h source/aero/partition_operator.h source/aero/partition_transfers.h
AERO_HEADERS += source/aero/mpi_stencil.h source/aero/partition_report.h source/aero/mpi_balance.h source/mesh/leaf_balance.h
ifeq ($(WITH_CUDA),1)
AERO_GPU := $(BUILD)/partition_kernels.o
AERO_SOURCES += source/aero/partition_gpu.cpp
AERO_HEADERS += source/aero/partition_gpu.h
$(AERO_GPU): source/aero/partition_kernels.cu source/aero/partition_kernels.cuh source/aero/partition_kernels.h $(RES_HDR) $(TIMESTEP_HEADERS) source/timestep/cfl_detail.h source/explicit/explicit_stage.h | $(BUILD)
	$(compile_cuda)
endif
AERO_PROGRAM := $(BUILD)/aero_$(EBR_TAG)$(if $(filter 1,$(WITH_MPI)),_mpi)$(EXE)
$(AERO_PROGRAM): $(AERO_SOURCES) $(AERO_HEADERS) $(AERO_LIBS) $(AERO_GPU)
	$(MPI_HOST_CXX) $(CPPFLAGS) $(if $(filter 1,$(WITH_MPI)),-DAERO_WITH_MPI $(MPI_CPPFLAGS)) $(CXXFLAGS) $(EBR_FLAGS) $(AERO_SOURCES) $(AERO_GPU) $(LDFLAGS) $(AERO_LIBS) $(EBR_LIBS) $(if $(filter 1,$(WITH_MPI)),$(MPI_LIBS)) $(LDLIBS) $(if $(filter Windows_NT,$(OS)),-lpsapi) -o $@
$(BIN)/aero$(EXE): $(AERO_PROGRAM) publish-programs | bin-runtime
	$(publish_exe)
.PHONY: aero
aero: $(BIN)/aero$(EXE)
all: aero
programs: aero

AERO_TEST := $(BUILD)/aero_tests$(EXE)
$(AERO_TEST): source/aero/aero_tests.cpp source/aero/partition_kernels.h $(RES_HDR) source/explicit/explicit_stage.h source/mesh/leaf_balance.cpp source/mesh/leaf_balance.h source/mesh/partition.h source/mesh/mdec.h $(MESH_HEADERS) $(FIELD_HEADERS) $(BUILD)/mesh.o $(IO_LIBRARY)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< source/mesh/leaf_balance.cpp $(BUILD)/mesh.o $(IO_LIBRARY) $(LDFLAGS) $(LDLIBS) -o $@
.PHONY: aero-test
aero-test: aero $(BIN)/aero_tests$(EXE)
	"$(BIN)/aero_tests$(EXE)" "$(BIN)/aero$(EXE)"
ifeq ($(MINGW_CUDA),1)
	CUDA_VISIBLE_DEVICES=-1 "$(BIN)/aero_tests$(EXE)" "$(BIN)/aero$(EXE)" --expect-cpu-fallback
endif
test: aero-test

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)
MPIEXEC ?= mpiexec
.PHONY: mpi-gpu-test
ifeq ($(WITH_CUDA)$(WITH_MPI),11)
mpi-gpu-test: aero meshdec
	"$(PYTHON)" qa/mpi_gpu.py --aero "$(BIN)/aero$(EXE)" --meshdec "$(BIN)/meshdec$(EXE)" --mpiexec "$(MPIEXEC)"
	"$(PYTHON)" qa/mpi_timing.py --devices=0,1 --aero "$(BIN)/aero$(EXE)" --meshdec "$(BIN)/meshdec$(EXE)" --mpiexec "$(MPIEXEC)"
else
mpi-gpu-test:
	$(MAKE) WITH_CUDA=1 WITH_MPI=1 $@
endif
MPI_RECONSTRUCTION_TESTS := mpi-ebr3-test mpi-ebr5-test mpi-weno3-test mpi-weno5-test
.PHONY: mpi-timing-test mpi-fields-test mpi-recovery-test mpi-geometry-test mpi-balance-test mpi-euler-test mpi-ssprk3-test mpi-periodic-test $(MPI_RECONSTRUCTION_TESTS)
ifeq ($(WITH_MPI),1)
# Для проверок с одноимённым Python-сценарием берём его имя из выбранной цели Make.
MPI_QA_ARGS := --aero "$(BIN)/aero$(EXE)" --meshdec "$(BIN)/meshdec$(EXE)" --mpiexec "$(MPIEXEC)"
mpi-timing-test mpi-fields-test mpi-recovery-test mpi-balance-test mpi-geometry-test: aero meshdec
	"$(PYTHON)" qa/mpi_$(patsubst mpi-%-test,%,$@).py $(MPI_QA_ARGS)
mpi-euler-test mpi-ssprk3-test: aero meshdec
	"$(PYTHON)" qa/mpi_euler.py $(MPI_QA_ARGS) --time-scheme $(patsubst mpi-%-test,%,$@)
$(MPI_RECONSTRUCTION_TESTS): aero meshdec
	"$(PYTHON)" qa/mpi_periodic.py $(MPI_QA_ARGS) --$(patsubst mpi-%-test,%,$@)
mpi-periodic-test: aero meshdec
	"$(PYTHON)" qa/mpi_periodic.py $(MPI_QA_ARGS)
test: mpi-timing-test mpi-fields-test mpi-recovery-test mpi-geometry-test mpi-balance-test mpi-euler-test mpi-ssprk3-test mpi-periodic-test $(MPI_RECONSTRUCTION_TESTS)
else
mpi-timing-test mpi-fields-test mpi-recovery-test mpi-geometry-test mpi-balance-test mpi-euler-test mpi-ssprk3-test mpi-periodic-test $(MPI_RECONSTRUCTION_TESTS):
	$(MAKE) WITH_MPI=1 $@
endif
$(AERO_TEST): qa/test_checks.h
