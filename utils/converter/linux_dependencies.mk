# По умолчанию конвертер использует архивы зависимостей из third_party.
# Префикс или флаги сборки позволяют выбрать установленную библиотеку.
WITH_CGNS ?= 1
ifeq ($(WITH_CGNS),1)
ifeq ($(origin CGNS_PREFIX)/$(origin CGNS_CPPFLAGS)/$(origin CGNS_LDLIBS),undefined/undefined/undefined)
CMAKE ?= cmake
DEPS_DIR := build/deps/linux
CGNS_PREFIX := $(abspath $(DEPS_DIR)/install)
HDF5_ARCHIVE := third_party/HDF5-1.14.6.zip
CGNS_ARCHIVE := third_party/CGNS-4.4.0.zip
HDF5_STAMP := $(DEPS_DIR)/install/.hdf5-1.14.6
CGNS_BUILD_DEPS := $(DEPS_DIR)/install/.cgns-4.4.0
DEPS_CONFIG := -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX="$(CGNS_PREFIX)" -DCMAKE_C_COMPILER="$(CC)" -DCMAKE_C_STANDARD=11 -DCMAKE_C_STANDARD_REQUIRED=ON

$(HDF5_STAMP): $(HDF5_ARCHIVE) third_party/hdf5-1.14.6-linux.patch utils/converter/linux_dependencies.mk
	printf '%s  %s\n' e78b518da9ff8bcec3a095a3ddbb9c906dfc99024593cf3d16db200d97a10b09 "$<" | sha256sum -c -
	mkdir -p "$(DEPS_DIR)"
	cd "$(DEPS_DIR)" && $(CMAKE) -E tar xf "$(abspath $<)"
	patch --binary -d "$(DEPS_DIR)/hdf5-hdf5_1.14.6" -p1 --batch < third_party/hdf5-1.14.6-linux.patch
	$(call action,Configuring,HDF5)
	$(CMAKE) -S "$(DEPS_DIR)/hdf5-hdf5_1.14.6" -B "$(DEPS_DIR)/hdf5-build" $(DEPS_CONFIG) \
	    -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_EXAMPLES=OFF \
	    -DHDF5_BUILD_HL_LIB=OFF -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF -DHDF5_ENABLE_SZIP_SUPPORT=OFF -DHDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16=OFF
	$(call action,Building,HDF5)
	$(MAKE) -C "$(DEPS_DIR)/hdf5-build" install $(if $(filter 1,$(V)),VERBOSE=1)
	touch "$@"

$(CGNS_BUILD_DEPS): $(CGNS_ARCHIVE) third_party/cgns-4.4.0-linux.patch $(HDF5_STAMP) utils/converter/linux_dependencies.mk
	printf '%s  %s\n' 72ad499936089102c1d77e3cdd78a623d61d1c6ad94c4df3b23f36db9ddfad6a "$<" | sha256sum -c -
	cd "$(DEPS_DIR)" && $(CMAKE) -E tar xf "$(abspath $<)"
	patch --binary -d "$(DEPS_DIR)/CGNS-4.4.0" -p1 --batch < third_party/cgns-4.4.0-linux.patch
	$(call action,Configuring,CGNS)
	$(CMAKE) -S "$(DEPS_DIR)/CGNS-4.4.0" -B "$(DEPS_DIR)/cgns-build" $(DEPS_CONFIG) \
	    -DCMAKE_PREFIX_PATH="$(CGNS_PREFIX)" -DCGNS_BUILD_SHARED=OFF -DCGNS_BUILD_CGNSTOOLS=OFF \
	    -DCGNS_ENABLE_FORTRAN=OFF -DCGNS_ENABLE_HDF5=ON -DCGNS_ENABLE_TESTS=OFF
	$(call action,Building,CGNS)
	$(MAKE) -C "$(DEPS_DIR)/cgns-build" install $(if $(filter 1,$(V)),VERBOSE=1)
	touch "$@"
endif
endif
