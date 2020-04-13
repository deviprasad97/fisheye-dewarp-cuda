CUDA_INSTALL_PATH := /usr/local/cuda
CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC  := nvcc
# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I/opt/arrayfire/include -I/opt/arrayfire-3/include
# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)
OPENCV = `pkg-config opencv --cflags --libs`
FFMPEG = `pkg-config --cflags --libs libavformat libavcodec libavutil libswscale`
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcudart
LFLAGS = -L/opt/local/lib -L/opt/arrayfire-3/lib
LIBS = $(OPENCV) $(FFMPEG) -lpng -ltiff -ljpeg -lm -lafcpu -lforge
OBJS = cudaWarp-fisheye.cu.o main.cpp.o
TARGET = vs_fisheye_stack
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA) $(LFLAGS) $(LIBS) -lm
.SUFFIXES: .c .cpp .cu .o
%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LFLAGS) $(LIBS) -lm
$(TARGET): $(OBJS) Makefile
	$(LINKLINE)
clean:
	rm *.o vs_fisheye_stack