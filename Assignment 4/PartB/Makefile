### Makefile

SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

CXX = g++
CXXFLAGS = -std=c++11

### Define the list of executable names to be built from corresponding source files
EXECS := vecaddcpu vecaddgpu00 vecaddgpu01 
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################

vecaddcpu: vecaddcpu.cpp
	$(CXX) -o vecaddcpu vecaddcpu.cpp $(CXXFLAGS)

#######################################################################

vecaddgpu00: vecaddgpu00.cu
	${NVCC} -o vecaddgpu00 vecaddgpu00.cu $(LIB) $(OPTIONS)

#######################################################################

vecaddgpu01: vecaddgpu01.cu
	${NVCC} -o vecaddgpu01 vecaddgpu01.cu $(LIB) $(OPTIONS)

#######################################################################

