CC=g++
MKDIR=mkdir -p
RM=rm -rf
SRC_DIR=src
OBJ_DIR=obj
BIN_DIR=bin
SRCS=$(wildcard $(SRC_DIR)/*.cc)
OBJS=$(SRCS:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)
CFLAGS=-Wall -std=c++11 -O2 -mavx
TARGET=$(BIN_DIR)/QuanCNN

# choose different BLAS libraries
BLAS=atlas
ifeq ($(BLAS), atlas)
  CPPFLAGS=-I/usr/include/atlas -I/opt/OpenVML/include
  LDFLAGS=-L/usr/lib/atlas-base -L/opt/OpenVML/lib
  LDLIBS=-lcblas -latlas -lopenvml
  DFLAGS=-D ENABLE_ATLAS -D ENABLE_OPENVML
endif
ifeq ($(BLAS), mkl)
  CPPFLAGS=-I/opt/intel/mkl/include
  LDFLAGS=-L/opt/intel/mkl/lib/intel64
  LDLIBS=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -pthread
  DFLAGS=-D ENABLE_MKL
endif
ifeq ($(BLAS), openblas)
  CPPFLAGS=-I/opt/OpenBLAS/include -I/opt/OpenVML/include
  LDFLAGS=-L/opt/OpenBLAS/lib -L/opt/OpenVML/lib
  LDLIBS=-lopenblas -lopenvml
  DFLAGS=-D ENABLE_OPENBLAS -D ENABLE_OPENVML
endif

.PHONY: all run clean

all: $(BIN_DIR) $(OBJ_DIR) $(TARGET)

$(BIN_DIR):
	$(MKDIR) $(BIN_DIR)
$(OBJ_DIR):
	$(MKDIR) $(OBJ_DIR)
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DFLAGS) -c $< -o $@

run:
	$(TARGET)

clean:
	$(RM) $(BIN_DIR) $(OBJ_DIR) $(OBJS) $(TARGET)
