# Очищаем только известные результаты Make. В build также находятся сетки,
# записанные решения и установка CGNS/HDF5.
clean_roots := $(BUILD)
ifeq ($(OS)/$(origin BUILD),Windows_NT/file)
clean_roots := build $(wildcard build/mingw-cpu-* build/mingw-cuda-sm*)
endif
ifneq ($(OS),Windows_NT)
ifeq ($(origin BUILD),file)
clean_roots := build/linux $(wildcard build/linux/cuda-sm* build/linux/hip-gfx*)
endif
endif
clean_variants := $(foreach root,$(clean_roots),$(root) $(root)/serial $(root)/static $(root)/static/serial)
# Make добавляет набор физических модулей перед типом сборки.
clean_variants += $(addsuffix /inviscid,$(clean_variants)) $(addsuffix /rans,$(clean_variants))
clean_dirs := $(clean_variants) $(addsuffix /Debug,$(clean_variants)) $(addsuffix /Release,$(clean_variants)) $(addsuffix /Release/O3,$(clean_variants))
clean_dirs += $(addsuffix /kokkos,$(clean_dirs))
clean_dirs := $(wildcard $(clean_dirs))
clean_modules := aero bc converter cv eulersolver explicit implicit initial io_fields io_report io_settings mesh meshdec mpi_matrix reconstruction residual sa tecplot timestep
clean_tags := cuda0 cuda1 $(foreach gpu,0 1,$(foreach omp,0 1,cuda$(gpu)_omp$(omp)))
clean_suffixes := $(addprefix _,$(clean_tags) cuda)
clean_tests := $(foreach mod,$(clean_modules),$(foreach base,$(mod)_tests $(mod)_tests_debug,$(base) $(addprefix $(base),$(clean_suffixes))))
clean_programs := aero mesh_converter fields_to_tecplot mesh_to_tecplot meshdec mesh_slice
clean_exes := $(addsuffix $(EXE),$(clean_programs) $(clean_tests) $(addprefix aero_,$(clean_tags) $(addsuffix _mpi,$(clean_tags))))
clean_exes += $(addsuffix $(EXE),$(addprefix viscosity_tests_aero_,$(clean_tags) $(addsuffix _mpi,$(clean_tags))))
# Только эти подкаталоги создаются самой сборкой. Здесь лежат в том числе
# установка Kokkos, его кэш CMake и объекты GPU, которые иначе переживали clean.
clean_generated_dirs := $(foreach dir,$(filter %/kokkos,$(clean_dirs)),$(wildcard $(addprefix $(dir)/,kokkos-source kokkos-build kokkos-build-unix kokkos-install kokkos-device)))
clean_runtime :=
ifeq ($(OS),Windows_NT)
clean_runtime := libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll libgomp-1.dll vcomp140.dll
endif
# Возвращает только существующие файлы из заданного каталога и списка известных шаблонов.
clean_files = $(wildcard $(addprefix $(1)/,$(2)))
clean_bin_files := $(call clean_files,$(BIN),$(clean_exes) $(clean_runtime) aero_kokkos_cuda.dll libaero_kokkos_cuda.so libaero_kokkos_hip.so)
# Удаляет один известный файл; кавычки сохраняют пробелы, а -- завершает список опций rm.
remove_file = rm -f -- "$(1)"

# Разворачивает список файлов в отдельные команды удаления, сохраняя границы каждого пути.
define remove_files
$(foreach f,$(1),$(call remove_file,$(f))
)
endef

.PHONY: clean
clean:
	$(call action,Cleaning,generated build files)
	$(call remove_files,$(clean_bin_files))
	$(call remove_files,$(foreach dir,$(clean_dirs),$(call clean_files,$(dir),*.o slice/*/*/*.o aero_*_objects/*/*.o libaero*.a libaero_kokkos_cuda.so libaero_kokkos_hip.so aero_kokkos_cuda.dll $(clean_exes))))
	$(if $(clean_generated_dirs),$(RM) -r -- $(clean_generated_dirs))
