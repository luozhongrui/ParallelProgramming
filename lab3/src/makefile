all: matmul
.PHONY : all clean

matmul: main.o function.o
	mpicxx main.o function.o  -o matmul

main.o: main.cc
	mpicxx -c main.cc -O3 -o main.o
function.o: function.cc
	mpicxx -c function.cc -O3 -o function.o


clean:
	-rm -f matmul main.o function.o
