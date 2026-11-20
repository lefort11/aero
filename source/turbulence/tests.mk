# Модель можно полностью удалить вместе с её тестовыми целями.
SA_TEST := $(BUILD)/sa_tests$(EXE)
$(SA_TEST): qa/sa_tests.cpp qa/test_checks.h source/turbulence/sa.h | $(BUILD)
	$(call action,Compiling and linking,$@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) "$<" -o "$@"
.PHONY: turbulence-test turbulence-qa
turbulence-test: $(SA_TEST)
	$(running_test)
	"$(SA_TEST)"
turbulence-qa: aero meshdec
	$(running_test)
	"$(PYTHON)" qa/turbulence.py --aero "$(BIN)/aero$(EXE)" --meshdec "$(BIN)/meshdec$(EXE)" $(if $(filter 1,$(WITH_MPI)),--mpi --mpiexec "$(MPIEXEC)") $(if $(filter 1,$(kokkos_gpu)),--gpu $(if $(MPI_BDF1_DEVICES),--devices=$(MPI_BDF1_DEVICES)))
test: turbulence-test turbulence-qa
