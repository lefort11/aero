# V=0 оставляет сообщения о работе и диагностику инструментов. V=1 дополнительно
# показывает команды рецептов и подробный вывод системы сборки зависимости.
V ?= 0
ifeq ($(V),0)
.SILENT:
else ifneq ($(V),1)
$(error V must be 0 or 1)
endif
MAKEFLAGS += --no-print-directory

# Сообщение выполняется непосредственно перед командой. Например, если линковка
# теста завершилась ошибкой, строка о запуске этого теста не будет напечатана.
action = @printf '%-12s %s\n' '$(1)' "$(2)"
compiling = $(call action,Compiling,$<)
archiving = $(call action,Archiving,$@)
publishing = $(call action,Publishing,$@)
running_test = $(call action,Testing,$@)

# Итоговые параметры доступны только после подключения файлов модулей. Выводим
# отдельную сводку во вложенном Make, если он собирает другую конфигурацию.
ifneq ($(MAKECMDGOALS),clean)
$(info Aero build configuration)
$(info Goals       : $(or $(MAKECMDGOALS),$(.DEFAULT_GOAL)))
$(info Build type  : $(BUILD_TYPE))
$(info Kokkos      : $(KOKKOS_BACKEND))
$(info Linear solve: $(if $(filter 1,$(MIXED_LINEAR)),mixed float/double,double) (MIXED_LINEAR=$(MIXED_LINEAR)))
ifeq ($(kokkos_gpu),1)
$(info GPU         : $(KOKKOS_GPU_CXX)  sm_$(CUDA_ARCH)  CUDA_HOME=$(CUDA_HOME))
endif
$(info Options     : MPI=$(WITH_MPI)  OpenMP=$(WITH_OPENMP)  CGNS=$(WITH_CGNS)  STATIC_LINK=$(STATIC_LINK))
$(info Build dir   : $(BUILD))
$(info Output dir  : $(BIN))
$(info C++         : $(CXX))
$(info CPPFLAGS    : $(strip $(CPPFLAGS)))
$(info CXXFLAGS    : $(strip $(CXXFLAGS)))
ifneq ($(strip $(EBR_FLAGS)),)
$(info Solver flags: $(strip $(EBR_FLAGS)))
endif
$(info LDFLAGS     : $(strip $(LDFLAGS)))
$(info LDLIBS      : $(strip $(LDLIBS)))
ifeq ($(WITH_MPI),1)
$(info MPI         : $(MPI_HOST_CXX) -DAERO_WITH_MPI $(strip $(MPI_CPPFLAGS) $(MPI_LIBS)))
endif
ifeq ($(WITH_CGNS),1)
$(info CGNS        : $(strip $(CONVERTER_CPPFLAGS) $(CONVERTER_LDLIBS)))
endif
$(info )
endif
