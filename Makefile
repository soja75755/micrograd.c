CC = gcc
CFLAGS = -Wall -g -I./src -I./test
LDFLAGS = -lm

SOURCES = main.c src/engine.c test/test_engine.c
OBJECTS = $(SOURCES:.c=.o)
EXECUTABLES = main

.PHONY: all clean

all: $(EXECUTABLES)

main: main.o src/engine.o test/test_engine.o
	$(CC) $^ $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLES)
