CC=g++ -std=c++11 -O3
all: SCVI 
SCVI.o: SCVI.cpp
	$(CC) -c -I./ SCVI.cpp -o SCVI.o -Wl,--stack=16777216 -mavx2

Graph.o: Graph.cpp Graph.h
	$(CC) -c Graph.cpp -o Graph.o

Query.o: Query.cpp Query.h
	$(CC) -c Query.cpp -o Query.o

QCSR.o: QCSR.cpp QCSR.h Graph.o Query.o
	$(CC) -c QCSR.cpp -o QCSR.o	

SCVI: SCVI.o QCSR.o Graph.o Query.o
	$(CC) SCVI.o QCSR.o Graph.o Query.o -o SCVI

clean:
	rm -rf *.o SCVI
