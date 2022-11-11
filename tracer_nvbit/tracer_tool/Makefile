# -ccbin=$(CXX) 
NVCC=nvcc -D_FORCE_INLINES

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif

NVBIT_PATH=../nvbit_release/core
INCLUDES=-I$(NVBIT_PATH)

LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES=$(wildcard *.cu)

OBJECTS=$(SOURCES:.cu=.o)
ARCH=35

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_TOOL=$(current_dir).so

all: $(NVBIT_TOOL)

# -arch=sm_$(ARCH) 
# -lcudart_static 
# $(NVCC_PATH) 
$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -O3 $(OBJECTS) $(LIBS) -lcuda -shared -o $@

# -Wall
# -arch=sm_$(ARCH)
%.o: %.cu common.h
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -w  -O3 -Xcompiler -fPIC $< -o $@

# -Wall 
# -arch=sm_$(ARCH) 
# -Xcompiler -w -Xcompiler -fPIC 
inject_funcs.o: inject_funcs.cu common.h
	$(NVCC) $(INCLUDES) -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions -c $< -o $@

clean:
	rm -f *.so *.o
