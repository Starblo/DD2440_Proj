OBJS = main.o genetic_algorithm.o two_opt.o
CC = g++
CFLAGS = -std=c++11 -Wall

all: main

main: $(OBJS)
	$(CC) $(CFLAGS) -o main $(OBJS)

main.o: main.cpp main.h genetic_algorithm.h two_opt.h
	$(CC) $(CFLAGS) -c main.cpp

genetic_algorithm.o: genetic_algorithm.cpp genetic_algorithm.h main.h
	$(CC) $(CFLAGS) -c genetic_algorithm.cpp

two_opt.o: two_opt.cpp two_opt.h main.h
	$(CC) $(CFLAGS) -c two_opt.cpp

clean:
	rm -f *.o main
