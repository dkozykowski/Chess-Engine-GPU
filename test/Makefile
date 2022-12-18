runTests: tests.o ../src/evaluate.o ../src/moves.o ../src/search.o ../src/scan.o
	nvcc -o runTests ../src/evaluate.o ../src/moves.o ../src/search.o ../src/scan.o tests.o   -lgtest_main  -lgtest

tests.o: tests.cu ../src/evaluate.o 
	nvcc -o tests.o -c tests.cu -dc

../src/moves.o: ../src/moves.cu ../src/moves.cuh
	nvcc -o ../src/moves.o -c ../src/moves.cu -dc

../src/search.o: ../src/search.cu ../src/search.cuh
	nvcc -o ../src/search.o -c ../src/search.cu -dc

../src/evaluate.o: ../src/evaluate.cu ../src/evaluate.cuh
	nvcc -o ../src/evaluate.o -c ../src/evaluate.cu -dc

../src/scan.o: ../src/scan.cu ../src/scan.cuh
	nvcc -o ../src/scan.o -c ../src/scan.cu -dc