# MSYS2 x64. The build helper compiles device sources with NVRTC. ptxas and
# fatbinary package native code and PTX for compatible and future devices.
CUDA_RUNTIME_LINK = $(BUILD)/libaero_cuda_driver.a
override CPPFLAGS += -DAERO_MINGW_CUDA_HOST -isystem "$(CUDA_HOME)/include"
mingw_fatbin = $(basename $@).fatbin
mingw_image = aero_image_$(basename $(notdir $@))
cuda_tools := $(if $(shell test -f "$(CUDA_HOME)/bin/x64/ptxas.exe" && echo yes),$(CUDA_HOME)/bin/x64,$(CUDA_HOME)/bin)
ptx_compiler := $(BUILD)/compile_ptx.exe
$(ptx_compiler): source/cuda/compile_ptx.cpp | $(BUILD)
	$(CXX) -std=c++11 $(build_optimization) -static -isystem "$(CUDA_HOME)/include" "$<" "$(CUDA_HOME)/lib/x64/nvrtc.lib" -o "$@"

define compile_cuda
PATH="$(CUDA_HOME)/bin/x64:$(CUDA_HOME)/bin:$$PATH" "$(ptx_compiler)" "$(basename $<).cuh" "$(CUDA_ARCH)" "$(basename $@).ptx"
"$(cuda_tools)/ptxas.exe" -arch=sm_$(CUDA_ARCH) "$(basename $@).ptx" -o "$(basename $@).cubin"
"$(cuda_tools)/fatbinary.exe" --64 --create="$(mingw_fatbin)" "--image3=kind=elf,sm=$(CUDA_ARCH),file=$(basename $@).cubin" "--image3=kind=ptx,sm=$(CUDA_ARCH),file=$(basename $@).ptx"
$(CXX) $(CPPFLAGS) $(CXXFLAGS) -x c++ -DAERO_CUDA_IMAGE=$(mingw_image) '-DAERO_CUDA_IMAGE_PATH="$(mingw_fatbin)"' -c "$<" -o "$@"
endef

$(BUILD)/mingw_driver.o: source/cuda/mingw_driver.cpp source/cuda/runtime.h | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD)/libaero_cuda_driver.a: $(BUILD)/mingw_driver.o
	$(AR) rcs $@ $^
