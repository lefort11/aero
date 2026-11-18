WITH_MPI ?= 0
MPI_CXX ?= mpicxx
ifeq ($(WITH_MPI),1)
ifeq ($(OS),Windows_NT)
MPI_HOST_CXX = $(CXX)
MPI_CPPFLAGS ?= -I"$(subst \,/,$(MSMPI_INC))."
MPI_LIBS ?= "$(subst \,/,$(MSMPI_LIB64))msmpi.lib"
else
MPI_HOST_CXX = $(MPI_CXX)
endif
else
MPI_HOST_CXX = $(CXX)
endif
