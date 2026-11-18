# Only expand files in known Make output directories. build also holds meshes,
# solver results and the CGNS/HDF5 installation.
clean_roots := $(BUILD)
ifeq ($(OS)/$(origin BUILD),Windows_NT/file)
clean_roots := build $(wildcard build/mingw-cuda-sm*)
endif
ifneq ($(OS),Windows_NT)
ifeq ($(origin BUILD),file)
clean_roots := build/linux $(wildcard build/linux/cuda-sm*)
endif
endif
clean_variants := $(foreach root,$(clean_roots),$(root) $(root)/serial $(root)/static $(root)/static/serial)
clean_dirs := $(clean_variants) $(addsuffix /Release,$(clean_variants)) $(addsuffix /Release/O3,$(clean_variants))
clean_modules := aero bc converter cv eulersolver explicit initial io_fields io_flowreport io_settings mesh meshdec reconstruction residual tecplot timestep
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
ifeq ($(USE_CMD),1)
remove_file = del /f /q "$(subst /,\,$(1))"
else
remove_file = rm -f -- "$(1)"
endif

define remove_files
$(foreach f,$(1),$(call remove_file,$(f))
)
endef

.PHONY: clean
clean:
ifeq ($(USE_CMD)/$(WITH_CUDA),1/1)
	"$(or $(MSBUILD),MSBuild.exe)" msvs/aero.sln "/t:aero:Clean;mesh_converter:Clean;fields_to_tecplot:Clean;meshdec:Clean" /m:1 /nr:false /nologo /v:minimal /p:Platform=x64 /p:Configuration=Release /p:WithCuda=true /p:CudaArchitecture=$(CUDA_ARCH) /p:WithOpenMP=$(if $(filter 1,$(WITH_OPENMP)),true,false) $(MSBUILD_CUDA_FLAGS)
endif
	$(call remove_files,$(clean_bin_files))
	$(call remove_files,$(foreach dir,$(clean_dirs),$(call clean_files,$(dir),*.o *.obj *.fatbin *.cubin *.ptx compile_ptx.exe libaero*.a $(clean_exes))))
