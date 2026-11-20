# Очищаем только известные результаты Make. В build также находятся сетки,
# записанные решения и установка CGNS/HDF5.
clean_roots := $(BUILD)
ifeq ($(OS)/$(origin BUILD),Windows_NT/file)
clean_roots := build $(wildcard build/mingw-cpu-* build/mingw-cuda-sm*)
endif
ifneq ($(OS),Windows_NT)
ifeq ($(origin BUILD),file)
clean_roots := build/linux $(wildcard build/linux/cuda-sm*)
endif
endif
clean_variants := $(foreach root,$(clean_roots),$(root) $(root)/serial $(root)/static $(root)/static/serial)
clean_dirs := $(clean_variants) $(addsuffix /Debug,$(clean_variants)) $(addsuffix /Release,$(clean_variants)) $(addsuffix /Release/O3,$(clean_variants))
clean_dirs += $(addsuffix /kokkos,$(clean_dirs))
clean_modules := aero bc converter cv eulersolver explicit implicit initial io_fields io_report io_settings mesh meshdec reconstruction residual tecplot timestep
clean_tags := cuda0 cuda1 $(foreach gpu,0 1,$(foreach omp,0 1,cuda$(gpu)_omp$(omp)))
clean_suffixes := $(addprefix _,$(clean_tags) cuda)
clean_tests := $(foreach mod,$(clean_modules),$(foreach base,$(mod)_tests $(mod)_tests_debug,$(base) $(addprefix $(base),$(clean_suffixes))))
clean_programs := aero mesh_converter fields_to_tecplot mesh_to_tecplot meshdec
clean_exes := $(addsuffix $(EXE),$(clean_programs) $(clean_tests) $(addprefix aero_,$(clean_tags) $(addsuffix _mpi,$(clean_tags))))
clean_runtime :=
ifeq ($(OS),Windows_NT)
clean_runtime := libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll libgomp-1.dll vcomp140.dll
endif
clean_files = $(wildcard $(addprefix $(1)/,$(2)))
clean_bin_files := $(call clean_files,$(BIN),$(clean_exes) $(clean_runtime))
remove_file = rm -f -- "$(1)"

define remove_files
$(foreach f,$(1),$(call remove_file,$(f))
)
endef

.PHONY: clean
clean:
	$(call action,Cleaning,generated build files)
	$(call remove_files,$(clean_bin_files))
	$(call remove_files,$(foreach dir,$(clean_dirs),$(call clean_files,$(dir),*.o aero_*_objects/*/*.o libaero*.a $(clean_exes))))
