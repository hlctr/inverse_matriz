CC=gcc
CFLAGS= -Wall -Wextra -O3
LDFLAGS= -lOpenCL -lm

.PHONY: all clean

all: im_opencl

im_opencl: im_opencl.c wtime.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f im_opencl *.o results_opencl.csv
