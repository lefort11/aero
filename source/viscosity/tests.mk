# Проверяем сам оператор отдельно от интегратора и формата файлов постановки.
AES_TEST := $(BUILD)/viscosity_tests_$(AERO_VARIANT)$(EXE)
$(AERO_OBJECT_DIR)/viscosity/viscosity_tests.o: qa/test_checks.h qa/test_field.h qa/device_test_buffer.h $(SPATIAL_HEADERS)
$(AES_TEST): $(AERO_OBJECT_DIR)/viscosity/viscosity_tests.o $(AERO_OBJECT_DIR)/viscosity/aes.o $(AERO_OBJECT_DIR)/viscosity/grad.o $(AERO_OBJECT_DIR)/viscosity/affine_hex.o $(filter %/turbulence/diffusion.o,$(AERO_OBJECTS)) $(AERO_OBJECT_DIR)/parallel/processes.o $(IMPLICIT_LIBRARY) $(RES_LIB) $(RES_DEPS)
	$(call action,Linking,$@)
	$(MPI_HOST_CXX) $(CXXFLAGS) $^ $(LDFLAGS) $(if $(filter 1,$(WITH_MPI)),$(MPI_LIBS)) $(LDLIBS) -o $@
.PHONY: viscosity-test viscosity-qa
viscosity-test: $(AES_TEST)
	$(running_test)
	"$(AES_TEST)"
viscosity-qa: aero meshdec
	$(running_test)
	"$(PYTHON)" qa/viscosity.py --aero "$(BIN)/aero$(EXE)" --meshdec "$(BIN)/meshdec$(EXE)" $(if $(filter 1,$(WITH_MPI)),--mpi --mpiexec "$(MPIEXEC)") $(if $(filter 1,$(kokkos_gpu)),--gpu $(if $(MPI_BDF1_DEVICES),--devices=$(MPI_BDF1_DEVICES)))
test: viscosity-test viscosity-qa
