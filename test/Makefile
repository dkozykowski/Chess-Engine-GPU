runTests: tests.o ../src/evaluate.o ../src/moves.o ../src/search.o ../src/scan.o ../src/macros.o
	nvcc -o runTests ../src/evaluate.o ../src/moves.o ../src/search.o ../src/scan.o tests.o ../src/macros.o -lgtest_main  -lgtest

tests.o: tests.cu ../src/evaluate.o 
	nvcc -o tests.o -c tests.cu -dc --std=c++11

../src/moves.o: ../src/moves.cu ../src/moves.cuh
	nvcc -o ../src/moves.o -c ../src/moves.cu -dc --std=c++11

../src/search.o: ../src/search.cu ../src/search.cuh
	nvcc -o ../src/search.o -c ../src/search.cu -dc --std=c++11

../src/evaluate.o: ../src/evaluate.cu ../src/evaluate.cuh
	nvcc -o ../src/evaluate.o -c ../src/evaluate.cu -dc --std=c++11

../src/scan.o: ../src/scan.cu ../src/scan.cuh
	nvcc -o ../src/scan.o -c ../src/scan.cu -dc --std=c++11

../src/macros.o: ../src/macros.cu ../src/macros.cuh
	nvcc -o ../src/macros.o -c ../src/macros.cu -dc --std=c++11