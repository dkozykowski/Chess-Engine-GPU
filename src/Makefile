engine: main.o uci.o position.o
	g++ -Wall -o engine main.o uci.o position.o

main.o:
	g++ -Wall -o main.o -c main.cpp

uci.o:
	g++ -Wall -o uci.o -c uci.cpp

position.o:
	g++ -Wall -o position.o -c position.cpp


.PHONY: clean

clean:
	rm *.o
