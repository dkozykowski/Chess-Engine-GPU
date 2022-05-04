engine: main.o uci.o position.o
	g++ -Wall -o engine main.o uci.o position.o

main.o: main.cpp
	g++ -Wall -o main.o -c main.cpp

uci.o: uci.cpp
	g++ -Wall -o uci.o -c uci.cpp

position.o: position.cpp
	g++ -Wall -o position.o -c position.cpp


.PHONY: clean

clean:
	rm *.o
