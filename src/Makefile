CC=g++ -std=c++11 -O3
all: SCVI QueryGenerator
SCVI.o: SCVI.cpp
	$(CC) -c -I./ SCVI.cpp -o SCVI.o -Wl,--stack=16777216 -mavx2

QueryGenerator.o: QueryGenerator.cpp
	$(CC) -c -I./ QueryGenerator.cpp -o QueryGenerator.o

Graph.o: Graph.cpp Graph.h
	$(CC) -c Graph.cpp -o Graph.o

Query.o: Query.cpp Query.h
	$(CC) -c Query.cpp -o Query.o

QCSR.o: QCSR.cpp QCSR.h Graph.o Query.o
	$(CC) -c QCSR.cpp -o QCSR.o	

SCVI: SCVI.o QCSR.o Graph.o Query.o
	$(CC) SCVI.o QCSR.o Graph.o Query.o -o SCVI

QueryGenerator: QueryGenerator.o
	$(CC) QueryGenerator.o -o QueryGenerator

clean:
	rm -rf *.o SCVI QueryGenerator
