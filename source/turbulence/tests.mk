# Модель можно полностью удалить вместе с её тестовыми целями.
SA_TEST := $(BUILD)/sa_tests$(EXE)
$(SA_TEST): source/turbulence/diffusion_geometry.h source/viscosity/grad.h
$(SA_TEST): qa/sa_tests.cpp qa/test_checks.h source/turbulence/convection.h source/turbulence/transport.h source/turbulence/sa.h source/turbulence/sigma.h source/turbulence/ddes.h source/turbulence/hex_filter.h source/turbulence/model_math.h | $(BUILD)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) "$<" -o "$@"
.PHONY: turbulence-test turbulence-qa sa-convection-qa
turbulence-test: $(SA_TEST)
	$(running_test)
	"$(SA_TEST)"
sa_qa_args = --aero "$(BIN)/aero$(EXE)" --meshdec "$(BIN)/meshdec$(EXE)" $(if $(filter 1,$(WITH_MPI)),--mpi --mpiexec "$(MPIEXEC)") $(if $(filter 1,$(kokkos_gpu)),--gpu $(if $(MPI_BDF1_DEVICES),--devices=$(MPI_BDF1_DEVICES)))
sa-convection-qa: aero meshdec
	$(running_test)
	"$(PYTHON)" qa/sa_convection.py $(sa_qa_args)
turbulence-qa: aero meshdec sa-convection-qa
	$(running_test)
	"$(PYTHON)" qa/turbulence.py $(sa_qa_args)
test: turbulence-test turbulence-qa
