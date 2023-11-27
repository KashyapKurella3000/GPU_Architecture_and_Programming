CC=nvcc
ARCH=-arch=sm_75
L1ENB=-Xptxas -dlcm=ca
L1DIS=-Xptxas -dlcm=cg
SOURCES=main.cu
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=lab4
all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(ARCH) $(L1ENB) $(OBJECTS) -o $@
	$(CC) $(ARCH) $(L1DIS) $(OBJECTS) -o $@L2
.PHONY : clean
clean :
	-rm $(EXECUTABLE)
