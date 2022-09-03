engine: main.o uci.o position.o evaluate.o search.o moves.o
	nvcc -o engine main.o uci.o position.o evaluate.o search.o moves.o

main.o: main.cu
	nvcc -o main.o -c main.cu -dc

uci.o: uci.cu uci.cuh
	nvcc -o uci.o -c uci.cu -dc

position.o: position.cu position.cuh
	nvcc -o position.o -c position.cu -dc

evaluate.o: evaluate.cu evaluate.cuh
	nvcc -o evaluate.o -c evaluate.cu -dc

search.o: search.cu search.cuh
	nvcc -o search.o -c search.cu -dc

moves.o: moves.cu moves.cuh
	nvcc -o moves.o -c moves.cu -dc

.PHONY: clean

clean:
	rm *.o
