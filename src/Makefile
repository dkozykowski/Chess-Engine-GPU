engine: main.o uci.o position.o
	nvcc -o engine main.o uci.o position.o

main.o: main.cu
	nvcc -o main.o -c main.cu

uci.o: uci.cu
	nvcc -o uci.o -c uci.cu

position.o: position.cu
	nvcc -o position.o -c position.cu

.PHONY: clean

clean:
	rm *.o
