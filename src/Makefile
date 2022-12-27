engine: main.o uci.o position.o evaluate.o search.o moves.o scan.o macros.o
	nvcc -o engine main.o uci.o position.o evaluate.o search.o moves.o scan.o macros.o -std=c++11

main.o: main.cu
	nvcc -o main.o -c main.cu -dc -std=c++11

uci.o: uci.cu uci.cuh
	nvcc -o uci.o -c uci.cu -dc -std=c++11

position.o: position.cu position.cuh
	nvcc -o position.o -c position.cu -dc -std=c++11

evaluate.o: evaluate.cu evaluate.cuh
	nvcc -o evaluate.o -c evaluate.cu -dc -std=c++11

search.o: search.cu search.cuh
	nvcc -o search.o -c search.cu -dc -std=c++11

moves.o: moves.cu moves.cuh
	nvcc -o moves.o -c moves.cu -dc -std=c++11

scan.o: ../src/scan.cu ../src/scan.cuh
	nvcc -o scan.o -c scan.cu -dc -std=c++11

macros.o: ../src/macros.cu ../src/macros.cuh
	nvcc -o macros.o -c macros.cu -dc --std=c++11

.PHONY: clean

clean:
	rm *.o
