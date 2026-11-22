# Компилятор устройства и CMake должны использовать одну установку Toolkit.
# Явный корень имеет приоритет. Без него учитываем nvcc из текущего окружения.
cuda_root_hint := $(or $(CUDA_HOME),$(CUDAToolkit_ROOT),$(CUDA_ROOT))
ifeq ($(origin KOKKOS_DEVICE_LIBRARY),undefined)
cuda_compiler_hint := $(or $(KOKKOS_GPU_CXX),$(NVCC),$(if $(cuda_root_hint),$(cuda_root_hint)/bin/nvcc,nvcc))
cuda_nvcc := $(shell command -v "$(cuda_compiler_hint)")
ifeq ($(strip $(cuda_nvcc)$(cuda_root_hint)$(NVCC)$(KOKKOS_GPU_CXX)),)
# Стандартная установка NVIDIA может отсутствовать в PATH. Если версий
# несколько, выбираем только через общую ссылку или явную настройку пользователя.
cuda_nvcc := $(wildcard /usr/local/cuda/bin/nvcc)
ifeq ($(cuda_nvcc),)
cuda_versions := $(sort $(realpath $(wildcard /usr/local/cuda-*/bin/nvcc)))
ifeq ($(words $(cuda_versions)),1)
cuda_nvcc := $(cuda_versions)
endif
endif
endif
ifeq ($(cuda_nvcc),)
$(error Cannot find CUDA compiler '$(cuda_compiler_hint)'. Set CUDA_HOME to a Toolkit containing bin/nvcc or add the desired nvcc to PATH)
endif

# У NVIDIA nvcc может быть ссылкой в другой каталог. У системного пакета
# Ubuntu компилятор доступен через /usr/bin, а заголовки лежат в /usr/include.
ifeq ($(cuda_root_hint),)
cuda_root_hint := $(shell compiler=$$(readlink -f "$(cuda_nvcc)") && root=$$(dirname "$$(dirname "$$compiler")"); \
    if test -f "$$root/include/cuda_runtime.h"; then printf '%s' "$$root"; \
    else dirname "$$(dirname "$(cuda_nvcc)")"; fi)
endif
NVCC ?= $(cuda_nvcc)
endif
ifneq ($(cuda_root_hint),)
CUDA_HOME ?= $(cuda_root_hint)
# У системных пакетов библиотеки лежат в lib/<архитектура>, у NVIDIA в lib64.
cuda_multiarch := $(shell $(CXX) -print-multiarch)
CUDA_LIBDIR ?= $(shell for dir in "$(CUDA_HOME)/lib64" "$(CUDA_HOME)/lib/$(cuda_multiarch)" "$(CUDA_HOME)/lib"; do \
    if test -f "$$dir/$(if $(filter 0,$(STATIC_LINK)),libcudart.so,libcudart_static.a)"; then printf '%s' "$$dir"; exit; fi; \
    done; printf '%s' "$(CUDA_HOME)/lib")
endif
