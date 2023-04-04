all:
	nvcc -ccbin=$(CXX) -O3 -lineinfo -arch=sm_35 vectoradd.cu -o vectoradd

clean:
	rm -f vectoradd
