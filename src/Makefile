engine: main.o uci.o position.o evaluate.o search.o
	nvcc -o engine main.o uci.o position.o evaluate.o search.o

main.o: main.cu
	nvcc -o main.o -c main.cu

uci.o: uci.cu uci.cuh
	nvcc -o uci.o -c uci.cu

position.o: position.cu position.cuh
	nvcc -o position.o -c position.cu

evaluate.o: evaluate.cu evaluate.cuh
	nvcc -o evaluate.o -c evaluate.cu

search.o: search.cu search.cuh
	nvcc -o search.o -c search.cu

.PHONY: clean

clean:
	rm *.o
