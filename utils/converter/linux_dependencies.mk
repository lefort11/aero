# The default converter uses the bundled source archives. An explicit prefix or
# flags select a caller-provided installation instead.
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
DEPS_CONFIG := -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX="$(CGNS_PREFIX)" -DCMAKE_C_COMPILER="$(CC)"

$(HDF5_STAMP): $(HDF5_ARCHIVE) utils/converter/linux_dependencies.mk
	printf '%s  %s\n' e78b518da9ff8bcec3a095a3ddbb9c906dfc99024593cf3d16db200d97a10b09 "$<" | sha256sum -c -
	mkdir -p "$(DEPS_DIR)"
	cd "$(DEPS_DIR)" && $(CMAKE) -E tar xf "$(abspath $<)"
	$(CMAKE) -S "$(DEPS_DIR)/hdf5-hdf5_1.14.6" -B "$(DEPS_DIR)/hdf5-build" $(DEPS_CONFIG) \
	    -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_EXAMPLES=OFF \
	    -DHDF5_BUILD_HL_LIB=OFF -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF -DHDF5_ENABLE_SZIP_SUPPORT=OFF -DHDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16=OFF
	$(MAKE) -C "$(DEPS_DIR)/hdf5-build" install
	touch "$@"

$(CGNS_BUILD_DEPS): $(CGNS_ARCHIVE) $(HDF5_STAMP) utils/converter/linux_dependencies.mk
	printf '%s  %s\n' 72ad499936089102c1d77e3cdd78a623d61d1c6ad94c4df3b23f36db9ddfad6a "$<" | sha256sum -c -
	cd "$(DEPS_DIR)" && $(CMAKE) -E tar xf "$(abspath $<)"
	$(CMAKE) -S "$(DEPS_DIR)/CGNS-4.4.0" -B "$(DEPS_DIR)/cgns-build" $(DEPS_CONFIG) \
	    -DCMAKE_PREFIX_PATH="$(CGNS_PREFIX)" -DCGNS_BUILD_SHARED=OFF -DCGNS_BUILD_CGNSTOOLS=OFF \
	    -DCGNS_ENABLE_FORTRAN=OFF -DCGNS_ENABLE_HDF5=ON -DCGNS_ENABLE_TESTS=OFF
	$(MAKE) -C "$(DEPS_DIR)/cgns-build" install
	touch "$@"
endif
endif
