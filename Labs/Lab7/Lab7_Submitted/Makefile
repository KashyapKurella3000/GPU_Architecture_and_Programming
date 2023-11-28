CC=nvcc
ARCH=-arch=sm_52
SOURCES=gpu_op_warp.cu
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=gpu_op2
all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(ARCH) $(OBJECTS) -o $@
.PHONY : clean
clean :
	-rm $(EXECUTABLE)

