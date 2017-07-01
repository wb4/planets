
planets: planets.c
	gcc -O -Wall -o planets planets.c `sdl-config --cflags` -lm -lGL -lGLU `sdl-config --libs` -lpthread
