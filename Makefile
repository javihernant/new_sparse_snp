ODIR = obj
SDIR = src
LDIR = lib
BDIR = bin
IDIR = include

CC=g++
RM=rm

OBJ=main.o 
OBJ+=snp_model.o snp_static.o tests.o
BIN=ssnp
#OBJ_LIB = snp_model.o
#LIB = 
#OMP=-fopenmp

CFlags=-c $(OMP) #-Wall
LDFlags=-lm $(OMP)

############ NVIDIA specifics
CUDA_PATH=/usr/local/cuda-11.3

NCC=nvcc -ccbin=$(CC)
#GENCODE_SM20    := -gencode arch=compute_20,code=\"sm_20,compute_20\"
#GENCODE_SM50    := -gencode arch=compute_50,code=\"sm_50,compute_50\"
#GENCODE_SM60    := -gencode arch=compute_60,code=\"sm_60,compute_60\"
#GENCODE_SM61    := -gencode arch=compute_61,code=\"sm_61,compute_61\"
GENCODE_SM75    := -gencode arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM35) $(GENCODE_SM60)\
                   $(GENCODE_SM61) $(GENCODE_SM75)
#NCFlags=-c --compiler-options -Wall -Xcompiler $(OMP) $(GENCODE_FLAGS)
NCFlags=-c $(GENCODE_FLAGS) -I$(CUDA_PATH)/include 
NLDFlags=-lm -Xcompiler $(OMP) -L$(CUDA_PATH)/lib64
############

############ Options for GPU and debugging
XCC=$(NCC) 	
XLD=$(NLDFlags)

ifeq ($(dbg),1)
	CFlags += -O0 -g -Wall
	NCFlags += -O0 -g
else	
	CFlags += -O3
	NCFlags += -O3
endif
############

all: $(OBJ) $(BIN) $(LIB)

$(LIB): $(patsubst %,$(ODIR)/%,$(OBJ_LIB))
	@mkdir -p $(LDIR)
	ar rcs $(LDIR)/$@ $^ 

$(BIN): $(patsubst %,$(ODIR)/%,$(OBJ))
	@mkdir -p $(BDIR)
	$(XCC) $^ $(XLD) -o $(BDIR)/$@ 

%.o: $(SDIR)/%.cpp
	@mkdir -p $(ODIR)
	$(CC) $(CFlags) -I$(IDIR) -I$(CUDA_PATH)/include -o $(ODIR)/$@ $<

%.o: $(SDIR)/%.cu
	@mkdir -p $(ODIR)
	$(NCC) $(NCFlags) -I$(IDIR) -o $(ODIR)/$@ $<

clean:
	$(RM) $(patsubst %,$(ODIR)/%,$(OBJ)) $(BDIR)/$(BIN) $(LDIR)/$(LIB)
