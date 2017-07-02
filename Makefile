
planets: planets.c
	gcc -O -Wall -o planets planets.c `sdl-config --cflags` -lm -lGL -lGLU -lpng `sdl-config --libs` -lpthread
