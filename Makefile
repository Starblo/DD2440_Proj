OBJS = main.o genetic_algorithm.o two_opt.o christofides.o
CC = g++
CFLAGS = -g -O0 -std=c++11 -Wall  # 添加 -g 调试符号，并使用 -O0 禁用优化

all: main

main: $(OBJS)
	$(CC) $(CFLAGS) -o main $(OBJS)

main.o: main.cpp main.h genetic_algorithm.h two_opt.h
	$(CC) $(CFLAGS) -c main.cpp

genetic_algorithm.o: genetic_algorithm.cpp genetic_algorithm.h main.h christofides.h
	$(CC) $(CFLAGS) -c genetic_algorithm.cpp

two_opt.o: two_opt.cpp two_opt.h main.h
	$(CC) $(CFLAGS) -c two_opt.cpp

christofides.o: christofides.cpp christofides.h main.h
	$(CC) $(CFLAGS) -c christofides.cpp

clean:
	rm -f *.o main
