
#include <GL/gl.h>
#include <GL/glu.h>
#include "SDL.h"

#include <errno.h>
#include <libgen.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/select.h>
#include <sys/sysinfo.h>
#include <sys/time.h>


/* #define USE_PLANET_MUTEX */


#define G 50000.0  /* gravitation constant */

#define MASS_MIN 0.1
#define MASS_MAX 15.0

#define MIN_BRIGHTNESS (0.35)
#define MAX_BRIGHTNESS (2.0)

#define VEL_INIT_MAX 0.1

#define SPLIT_COUNT_MIN 5
#define SPLIT_COUNT_MAX 30

#define SPLIT_DISTANCE 1000.0

#define TOTAL_MASS 1500
#define PLANET_COUNT_INITIAL 100

#define PLANET_COUNT_MAX ((size_t) (TOTAL_MASS / MASS_MIN))

#define PLANET_DENSITY 0.03

#define WORLD_SCALE 0.75
#define WORLD_WIDTH (3277.0 * WORLD_SCALE)
#define WORLD_HEIGHT (2048.0 * WORLD_SCALE)

#define FRAMES_PER_SECOND 60
#define TICKS_PER_FRAME 5

#define COLLISION_ITERATION_MAX 30


#define rand_normal() (rand() / (RAND_MAX + 1.0))


/* Stuff dealing with display. */

#define SCREEN_WIDTH_INIT 1680  /* initial screen height is calculated from this and the world dimensions */
#define SCREEN_DEPTH 24 /* color depth */

#define CIRCLE_POLY_COUNT 10

int video_flags = 0;

SDL_Surface *main_window = NULL;

unsigned quitting = 0;


typedef struct planet {
  double x_pos, y_pos;
  double x_vel, y_vel;

  double mass;
  double radius;
  double radius_squared;

  volatile double x_force, y_force;

  double energy;

  double hue;
  size_t hue_tick;

  size_t collision_list;

#ifdef USE_PLANET_MUTEX
  pthread_mutex_t mutex;
#endif
} planet_t;


typedef struct planet_node {
  planet_t *planet;
  struct planet_node *next;
} planet_node_t;


typedef struct planet_list {
  planet_node_t *first;
  size_t size;
} planet_list_t;


typedef struct {
  planet_list_t *planets;
  planet_node_t *planet_node;
  size_t tick;
  size_t working_planet_count;
  volatile int running;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
} thread_arg_t;


typedef struct {
  pthread_t *array;
  size_t size;
} pthread_list_t;


void die_usage(const char *prog);
size_t parse_ticks(char *spec);
int initialize_display(void);
int set_up_pixel_format(void);
int create_window(const char *window_name, int width, int height, int video_flags);
int initialize_openGL(int width, int height);
void size_openGL_screen(int width, int height);
void run_simulation(void);
void start_threads(pthread_list_t *threads, thread_arg_t *arg);
void stop_threads(pthread_list_t *threads, thread_arg_t *arg);
void pthread_list_init(pthread_list_t *list, size_t size);
void pthread_list_delete(pthread_list_t *list);
void *t_planet_ticker(void *void_arg);
int handle_sdl_event(SDL_Event *event);
void handle_key_press_event(SDL_keysym *keysym);
void initialize_planets(planet_list_t *planets);
void display_planets(const planet_list_t *planets);
void draw_planet(planet_t *planet);
void color_planet(planet_t *planet);
void hue_to_rgb(double hue, double *r, double *g, double *b);
void scale_color(double brightness, double *r, double *g, double *b);
void draw_circle(double cx, double cy, double radius);
void draw_rect(double x, double y, double w, double h);
void tick_planets(thread_arg_t *thread_arg);
void resolve_collisions(planet_list_t *planets, size_t tick);
void resolve_collision_group(planet_list_t *planets, planet_list_t *collision, planet_list_t *new_planets, size_t tick);
void find_oldest_hue(const planet_list_t *planets, double *hue, size_t *hue_tick);
void split_planet(planet_list_t *planets, planet_t *planet, planet_list_t *new_planets, size_t tick);
double calculate_split_energy(double mass, size_t count, double r1, double r2);
void delete_planets(planet_list_t *planets);
void find_collision_groups(planet_list_t *planets, planet_list_t *new_planets, planet_list_t *collision_lists, size_t *collision_count);
void resolve_collision_pair(planet_t *p1, planet_t *p2, planet_list_t *collision_lists, size_t *collision_count);
void merge_collision_lists(planet_list_t *collision_lists, size_t list_a, size_t list_b);
void calculate_forces(thread_arg_t *thread_arg);
void calculate_planet_forces(planet_node_t *node);
void calculate_force_pair(planet_t *p1, planet_t *p2);
void position_mod(planet_t *p1, planet_t *p2, double *p2_x, double *p2_y);
void move_planets(planet_list_t *planets);
double mod_double(double value, double min, double max);
void wait_for_next_tick(struct timeval *start);
planet_t *planet_new(double x_pos, double y_pos, double x_vel, double y_vel, double mass, double hue, size_t tick);
void planet_init(planet_t *planet, double x_pos, double y_pos, double x_vel, double y_vel, double mass, double hue, size_t tick);
void planet_reinit(planet_t *planet, double x_pos, double y_pos, double x_vel, double y_vel, double mass, double hue, size_t tick);
void planet_delete(planet_t *planet);
void planet_add_force(planet_t *planet, double x_force, double y_force);
double radius_for_mass(double mass);
void list_init(planet_list_t *list);
void list_add(planet_list_t *list, planet_t *planet);
planet_t *list_remove_first(planet_list_t *list);
void list_remove_all(planet_list_t *list, planet_list_t *removing);
int list_contains(planet_list_t *list, planet_t *planet);
void list_delete(planet_list_t *list);
void thread_arg_init(thread_arg_t *arg, planet_list_t *planets);
planet_node_t *thread_arg_get_planet_node(thread_arg_t *arg, int finished_one);
void thread_arg_reset_planet_node(thread_arg_t *arg);
void thread_arg_wait_till_zero_working(thread_arg_t *arg);
void thread_arg_stop_running(thread_arg_t *arg);
void *my_malloc(size_t size);


int main(int argc, char **argv) {
  size_t anim_ticks = 0;
  const char *anim_dir = NULL;
  int c;
  const char *prog_name;

  prog_name = basename(argv[0]);

  while ((c = getopt(argc, argv, "t:d:")) != -1) {
    switch (c) {
      case 't':
        if (anim_ticks > 0) {
          die_usage(prog_name);
        }
        anim_ticks = parse_ticks(optarg);
        if (anim_ticks == 0) {
          die_usage(prog_name);
        }
        break;
      case 'd':
        anim_dir = optarg;
        if (strlen(anim_dir) == 0) {
          die_usage(prog_name);
        }
        break;
      default:
        die_usage(prog_name);
    }
  }

  if ((anim_ticks > 0) ^ (anim_dir != NULL)) {
    fputs("-t and -d must be provided together.\n", stderr);
    die_usage(prog_name);
  }

  argv += optind;
  argc -= optind;

  printf("Anim dir = \"%s\", tick count = %lu\n", anim_dir, anim_ticks);

  if (argc != 0) {
    die_usage(prog_name);
    return 1;
  }

  srand(time(NULL));

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    fprintf(stderr, "SDL_Init() failed: %s\n", SDL_GetError());
    return 1;
  }

  atexit(SDL_Quit);

  SDL_ShowCursor(SDL_DISABLE);

  if (initialize_display() < 0) {
    return 1;
  }

  run_simulation();

  return 0;
}


void die_usage(const char *prog) {
  fprintf(stderr, "Usage: %s [-d <anim_dir> -t <anim_duration>]\n", prog);
  fprintf(stderr, "\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -d <anim_dir>    Directory to save animation frames in.  It will be created if it does not exist.\n");
  fprintf(stderr, "  -t <num>[s|m|h]  Duration of animation.  Units are s=seconds, m=minutes, h=hours, or <omitted>=frames.\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "If either -d or -t is given, then the other must be provided as well.\n");
  fprintf(stderr, "If neither option is provided, then no animation frames are saved.\n");

  exit(1);
}


size_t parse_ticks(char *spec) {
  char unit;
  size_t factor = 1;
  size_t count;
  char *endptr;

  if (strlen(spec) == 0) {
    return 0;
  }

  unit = spec[strlen(spec) - 1];
  if (unit == 'h') {
    factor *= 60;
    unit = 'm';
  }
  if (unit == 'm') {
    factor *= 60;
    unit = 's';
  }
  if (unit == 's') {
    factor *= FRAMES_PER_SECOND;
  }

  if (factor > 1) {
    spec[strlen(spec) - 1] = '\0';
  }

  count = strtoul(spec, &endptr, 10);
  if (*endptr || strchr(spec, '-')) {
    return 0;
  }
  if (errno == ERANGE || (size_t) count != count) {
    fprintf(stderr, "Tick count %s too large!\n", spec);
    return 0;
  }
  if ((count * factor) / factor != count) {
    fprintf(stderr, "Tick count too large!\n");
    return 0;
  }
  return count * factor;
}


void run_simulation(void) {
  struct timeval start_time;
  SDL_Event event;
  size_t i;
  pthread_list_t threads;
  thread_arg_t thread_arg;

  planet_list_t planets;

  initialize_planets(&planets);

  thread_arg_init(&thread_arg, &planets);

  start_threads(&threads, &thread_arg);

  for (thread_arg.tick = 0;  !quitting;  ) {
    gettimeofday(&start_time, NULL);

    display_planets(&planets);

    for (i = 0;  i < TICKS_PER_FRAME;  ++i) {
      tick_planets(&thread_arg);
      ++thread_arg.tick;
    }

    while (SDL_PollEvent(&event)) {
      handle_sdl_event(&event);
    }

    wait_for_next_tick(&start_time);
  }

  stop_threads(&threads, &thread_arg);
}


void start_threads(pthread_list_t *threads, thread_arg_t *arg) {
  size_t thread_count;
  size_t i;

  thread_count = (size_t) get_nprocs();

  pthread_list_init(threads, thread_count);

  arg->running = 1;

  for (i = 0;  i < thread_count;  ++i) {
    pthread_create(&threads->array[i], 0, t_planet_ticker, arg);
  }
}


void stop_threads(pthread_list_t *threads, thread_arg_t *arg) {
  size_t i;

  thread_arg_stop_running(arg);

  for (i = 0;  i < threads->size;  ++i) {
    pthread_join(threads->array[i], 0);
  }

  pthread_list_delete(threads);
}


void pthread_list_init(pthread_list_t *list, size_t size) {
  list->array = calloc(size, sizeof(list->array[0]));
  list->size = size;
}


void pthread_list_delete(pthread_list_t *list) {
  free(list->array);
  list->array = 0;
  list->size = 0;
}


void *t_planet_ticker(void *void_arg) {
  thread_arg_t *arg;
  planet_node_t *node = NULL;

  arg = (thread_arg_t *) void_arg;

  while (arg->running) {
    node = thread_arg_get_planet_node(arg, node != NULL);
    if (node) {
      calculate_planet_forces(node);
    }
  }

  return 0;
}


int handle_sdl_event(SDL_Event *event) {
  switch (event->type) {
    case SDL_QUIT:
      quitting = 1;
      break;

    case SDL_KEYDOWN:
      handle_key_press_event(&event->key.keysym);
      break;

    default:
      break;
  }

  return 0;
}


void handle_key_press_event(SDL_keysym *keysym) {
  switch (keysym->sym) {
    case SDLK_ESCAPE:
      quitting = 1;
      break;

    default:
      break;
  }
}


int initialize_display(void) {
  int screen_width, screen_height;
  double ratio;

  screen_width = SCREEN_WIDTH_INIT;

  ratio = WORLD_HEIGHT / WORLD_WIDTH;

  screen_height = (int) (ratio * screen_width + 0.5);

  if (set_up_pixel_format() < 0) {
    return -1;
  }

  if (create_window("Planets", screen_width, screen_height, video_flags) < 0) {
    return -1;
  }

  if (initialize_openGL(screen_width, screen_height) < 0) {
    return -1;
  }

  return 0;
}


int set_up_pixel_format(void) {
  const SDL_VideoInfo *video_info;

  video_flags = SDL_OPENGL;
  video_flags |= SDL_HWPALETTE;

  video_info = SDL_GetVideoInfo();
  if (video_info == NULL) {
    fprintf(stderr, "Failed to get video info: %s", SDL_GetError());
    return -1;
  }

  if (video_info->hw_available) {
    video_flags |= SDL_HWSURFACE;
  } else {
    video_flags |= SDL_SWSURFACE;
  }

  if (video_info->blit_hw) {
    video_flags |= SDL_HWACCEL;
  }

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, SCREEN_DEPTH);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);
  SDL_GL_SetAttribute(SDL_GL_ACCUM_RED_SIZE, 0);
  SDL_GL_SetAttribute(SDL_GL_ACCUM_GREEN_SIZE, 0);
  SDL_GL_SetAttribute(SDL_GL_ACCUM_BLUE_SIZE, 0);
  SDL_GL_SetAttribute(SDL_GL_ACCUM_ALPHA_SIZE, 0);

  return 0;
}


int create_window(const char *window_name, int width, int height, int video_flags) {
  main_window = SDL_SetVideoMode(width, height, SCREEN_DEPTH, video_flags);
  if ( main_window == NULL ) {
    fprintf(stderr, "Failed to create window: %s", SDL_GetError());
    return -1;
  }

  SDL_WM_SetCaption(window_name, window_name);

  return 0;
}


int initialize_openGL(int width, int height) {
  size_openGL_screen(width, height);

  return 0;
}


void size_openGL_screen(int width, int height) {
  const float world_aspect = (float) (WORLD_WIDTH / WORLD_HEIGHT);
  float screen_aspect;

  if ( height == 0 ) {
    height = 1;
  }

  screen_aspect = (float) width / height;

  glViewport(0, 0, width, height);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  if (screen_aspect > world_aspect) {
    glScalef(2.0f/(WORLD_HEIGHT*screen_aspect), 2.0f/WORLD_HEIGHT, 1.0f);
  } else {
    glScalef (2.0f/WORLD_WIDTH, 2.0f/(WORLD_WIDTH/screen_aspect), 1.0f);
  }
  glTranslatef(-WORLD_WIDTH/2.0f, -WORLD_HEIGHT/2.0f, 0.0f);
}


void initialize_planets(planet_list_t *planets) {
  size_t i;
  planet_t *planet;
  double x_pos, y_pos;
  double x_vel, y_vel;
  double speed, angle;
  double mass;

  list_init(planets);

  mass = TOTAL_MASS / PLANET_COUNT_INITIAL;

  for (i = 0;  i < PLANET_COUNT_INITIAL;  ++i) {
    x_pos = rand_normal() * WORLD_WIDTH;
    y_pos = rand_normal() * WORLD_HEIGHT;

    speed = rand_normal() * VEL_INIT_MAX;
    angle = rand_normal() * 2.0 * M_PI;

    x_vel = speed * cos(angle);
    y_vel = speed * sin(angle);

    planet = planet_new(x_pos, y_pos, x_vel, y_vel, mass, 0.0, 0);
    list_add(planets, planet);
  }
}


void display_planets(const planet_list_t *planets) {
  planet_node_t *node;

  glClear(GL_COLOR_BUFFER_BIT);

  for (node = planets->first;  node;  node = node->next) {
    draw_planet(node->planet);
  }

  SDL_GL_SwapBuffers();
}


void draw_planet(planet_t *planet) {
  color_planet(planet);

  draw_circle(planet->x_pos, planet->y_pos, planet->radius);

  if (planet->x_pos - planet->radius < 0.0) {
    draw_circle(planet->x_pos + WORLD_WIDTH, planet->y_pos, planet->radius);
    if (planet->y_pos - planet->radius < 0.0) {
      draw_circle(planet->x_pos + WORLD_WIDTH, planet->y_pos + WORLD_HEIGHT, planet->radius);
    }
    if (planet->y_pos + planet->radius >= WORLD_HEIGHT) {
      draw_circle(planet->x_pos + WORLD_WIDTH, planet->y_pos - WORLD_HEIGHT, planet->radius);
    }
  }
  if (planet->x_pos + planet->radius >= WORLD_WIDTH) {
    draw_circle(planet->x_pos - WORLD_WIDTH, planet->y_pos, planet->radius);
    if (planet->y_pos - planet->radius < 0.0) {
      draw_circle(planet->x_pos - WORLD_WIDTH, planet->y_pos + WORLD_HEIGHT, planet->radius);
    }
    if (planet->y_pos + planet->radius >= WORLD_HEIGHT) {
      draw_circle(planet->x_pos - WORLD_WIDTH, planet->y_pos - WORLD_HEIGHT, planet->radius);
    }
  }
  if (planet->y_pos - planet->radius < 0.0) {
    draw_circle(planet->x_pos, planet->y_pos + WORLD_HEIGHT, planet->radius);
  }
  if (planet->y_pos + planet->radius >= WORLD_HEIGHT) {
    draw_circle(planet->x_pos, planet->y_pos - WORLD_HEIGHT, planet->radius);
  }
}


void color_planet(planet_t *planet) {
  float value;
  double r, g, b;

  value = MIN_BRIGHTNESS + (MAX_BRIGHTNESS - MIN_BRIGHTNESS) * (planet->mass - MASS_MIN) / (MASS_MAX - MASS_MIN);

  if (value < 0.0f) {
    value = 0.0f;
  }
  if (value > 2.0f) {
    value = 2.0f;
  }

  hue_to_rgb(planet->hue, &r, &g, &b);

  scale_color(value, &r, &g, &b);

  glColor3f((float) r, (float) g, (float) b);
}


void hue_to_rgb(double hue, double *r, double *g, double *b) {
  size_t step;

  while (hue < 0.0) {
    hue += 360.0;
  }
  while (hue >= 360.0) {
    hue -= 360.0;
  }

  step = (size_t) (hue / 60.0);

  *r = *g = *b = 0.0;

  switch (step) {
    case 0:  /* 0 - 60 */
      *r = 1.0;
      *g = hue / 60.0;
      break;
    case 1: /* 60 - 120 */
      *r = (120.0 - hue) / 60.0;
      *g = 1.0;
      break;
    case 2: /* 120 - 180 */
      *g = 1.0;
      *b = (hue - 120.0) / 60.0;
      break;
    case 3: /* 180 - 240 */
      *g = (240.0 - hue) / 60.0;
      *b = 1.0;
      break;
    case 4: /* 240 - 300 */
      *r = (hue - 240.0) / 60.0;
      *b = 1.0;
      break;
    case 5:  /* 300 - 360 */
      *r = 1.0;
      *b = (360.0 - hue) / 60.0;
      break;
  }
}


void scale_color(double brightness, double *r, double *g, double *b) {
  if (brightness < 1.0) {
    *r *= brightness;
    *g *= brightness;
    *b *= brightness;
  } else {
    brightness = 2.0 - brightness;
    *r = (1.0 - ((1.0 - *r) * brightness));
    *g = (1.0 - ((1.0 - *g) * brightness));
    *b = (1.0 - ((1.0 - *b) * brightness));
  }
}


void draw_circle(double cx, double cy, double radius) {
  float angle;
  const float angle_diff = 2 * M_PI / CIRCLE_POLY_COUNT;
  size_t i;
  float x, y;

  glBegin(GL_TRIANGLE_FAN);
    for (angle = 0.0f, i = 0;  i < CIRCLE_POLY_COUNT;  angle += angle_diff, ++i) {
      x = (float) (cx + radius * cos(angle));
      y = (float) (cy + radius * sin(angle));
      glVertex2f(x, y);
    }
  glEnd();
}


void draw_rect(double x, double y, double w, double h) {
  glBegin(GL_QUADS);
    glVertex2f(x, y);
    glVertex2f(x+w, y);
    glVertex2f(x+w, y+h);
    glVertex2f(x, y+h);
  glEnd();
}


void tick_planets(thread_arg_t *thread_arg) {
  resolve_collisions(thread_arg->planets, thread_arg->tick);
  calculate_forces(thread_arg);
  move_planets(thread_arg->planets);
}


void resolve_collisions(planet_list_t *planets, size_t tick) {
  planet_list_t collision_lists[PLANET_COUNT_MAX];
  planet_list_t new_planets;
  size_t collision_count;
  size_t i;
  size_t col_it;

  list_init(&new_planets);

  memset(collision_lists, 0, sizeof(collision_lists));

  for (col_it = 0;  col_it < COLLISION_ITERATION_MAX;  ++col_it) {
    collision_count = 0;

    find_collision_groups(planets, &new_planets, collision_lists, &collision_count);

    list_delete(&new_planets);

    for (i = 0;  i < collision_count;  ++i) {
      resolve_collision_group(planets, &collision_lists[i], &new_planets, tick);
      list_delete(&collision_lists[i]);
    }

    if (!new_planets.size) {
      break;
    }
  }
}


void resolve_collision_group(planet_list_t *planets, planet_list_t *collision, planet_list_t *new_planets, size_t tick) {
  /* We need to merge all the planets involved in the collision into a single planet
     with the same mass and momentum as the entire group, located at the group's
     center of mass. */
  planet_node_t *node;
  planet_t *planet;

  double total_x_pos, total_y_pos;
  double total_x_vel, total_y_vel;
  double total_mass;

  double hue;
  size_t hue_tick;

  double x_pos, y_pos;
  double x_vel, y_vel;

/*
  double x_pos_diff, y_pos_diff;
  double x_vel_diff, y_vel_diff;
  double speed_proj;
*/

  double first_x, first_y;

  if (!collision->size) {
    return;
  }

  total_x_pos = total_y_pos = 0.0;
  total_x_vel = total_y_vel = 0.0;
  total_mass = 0.0;

  find_oldest_hue(collision, &hue, &hue_tick);

  first_x = collision->first->planet->x_pos;
  first_y = collision->first->planet->y_pos;

  for (node = collision->first;  node;  node = node->next) {
    planet = node->planet;

    x_pos = planet->x_pos;
    y_pos = planet->y_pos;

    if (x_pos - first_x > 0.5 * WORLD_WIDTH) {
      x_pos -= WORLD_WIDTH;
    }
    if (first_x - x_pos > 0.5 * WORLD_WIDTH) {
      x_pos += WORLD_WIDTH;
    }
    if (y_pos - first_y > 0.5 * WORLD_HEIGHT) {
      y_pos -= WORLD_HEIGHT;
    }
    if (first_y - y_pos > 0.5 * WORLD_HEIGHT) {
      y_pos += WORLD_HEIGHT;
    }

    total_x_pos += x_pos * planet->mass;
    total_y_pos += y_pos * planet->mass;

    total_x_vel += planet->x_vel * planet->mass;
    total_y_vel += planet->y_vel * planet->mass;

    total_mass += planet->mass;

    if ((hue == -1.0) || (planet->hue_tick > hue_tick && rand_normal() < 0.35)) {
      hue = planet->hue;
      hue_tick = planet->hue_tick;
    }
  }

  x_pos = mod_double(total_x_pos / total_mass, 0, WORLD_WIDTH);
  y_pos = mod_double(total_y_pos / total_mass, 0, WORLD_HEIGHT);

  x_vel = total_x_vel / total_mass;
  y_vel = total_y_vel / total_mass;

/*
  for (node = collision->first;  node;  node = node->next) {
    planet = node->planet;

    x_pos_diff = planet->x_pos - x_pos;
    y_pos_diff = planet->y_pos - y_pos;

    x_vel_diff = planet->x_vel - x_vel;
    y_vel_diff = planet->y_vel - y_vel;

    speed_proj = x_pos_diff * y_vel_diff - y_pos_diff * x_vel_diff;
  }
*/

  planet = list_remove_first(collision);

  list_remove_all(planets, collision);
  delete_planets(collision);

  planet_reinit(planet, x_pos, y_pos, x_vel, y_vel, total_mass, hue, hue_tick);

  if (total_mass > MASS_MAX) {
    /* It's too big!  We have to break it up. */
    split_planet(planets, planet, new_planets, tick);
  }
}


void find_oldest_hue(const planet_list_t *planets, double *hue, size_t *hue_tick) {
  const planet_node_t *node;
  const planet_t *planet;

  *hue_tick = (size_t) -1;

  for (node = planets->first;  node;  node = node->next) {
    planet = node->planet;

    if (planet->hue_tick < *hue_tick) {
      *hue = planet->hue;
      *hue_tick = planet->hue_tick;
    }
  }
}


void split_planet(planet_list_t *planets, planet_t *planet, planet_list_t *new_planets, size_t tick) {
  size_t child_count;
  size_t i;

  double angle_diff;
  double angle;
  double move_dir;

  double distance;

  double radius;

  double x_pos, y_pos;
  double x_vel, y_vel;

  double child_x_pos, child_y_pos;
  double child_x_vel, child_y_vel;
  double child_mass;
  double child_hue;
  size_t child_hue_tick;

  double this_child_hue;
  size_t this_child_hue_tick;

  double speed;

  double energy;

  child_count = SPLIT_COUNT_MIN + (size_t) ((SPLIT_COUNT_MAX - SPLIT_COUNT_MIN + 1) * rand_normal());
  child_mass = planet->mass / child_count;
  child_hue = planet->hue;
  child_hue_tick = planet->hue_tick;

/*
  for (;;) {
    child_mass = planet->mass / child_count;
    if (child_mass <= MASS_MAX) {
      break;
    }
    ++child_count;
  }
*/

  angle_diff = 2 * M_PI / child_count;
  angle = 2 * M_PI * rand_normal();

  radius = radius_for_mass(child_mass);

  distance = sqrt((radius + 5.0) / (1 - cos(angle_diff)));

  x_pos = planet->x_pos;
  y_pos = planet->y_pos;
  x_vel = planet->x_vel;
  y_vel = planet->y_vel;

  energy = calculate_split_energy(child_mass, child_count, distance, SPLIT_DISTANCE);

  /*
   * e = 1/2 m * v^2
   * 2e = m * v^2
   * v^2 = 2e / m
   * v = sqrt(2e / m)
   */
  speed = sqrt(2 * energy / child_mass);

  move_dir = 0.0175 * (2.0 * M_PI);

  for (i = 0;  i < child_count;  ++i, angle += angle_diff) {
    child_x_pos = x_pos + distance * cos(angle);
    child_y_pos = y_pos + distance * sin(angle);

    child_x_vel = x_vel + speed * cos(angle + move_dir);
    child_y_vel = y_vel + speed * sin(angle + move_dir);

    this_child_hue = child_hue;
    this_child_hue_tick = child_hue_tick;

    if (rand_normal() < 0.0005) {
      this_child_hue = 360.0 * rand_normal();
      this_child_hue_tick = tick;
    }

    if (!i) {
      planet_reinit(planet, child_x_pos, child_y_pos, child_x_vel, child_y_vel, child_mass, this_child_hue, this_child_hue_tick);
    } else {
      planet = planet_new(child_x_pos, child_y_pos, child_x_vel, child_y_vel, child_mass, this_child_hue, this_child_hue_tick);
      list_add(planets, planet);
    }

    list_add(new_planets, planet);

    if (planet->mass > MASS_MAX) {
      /* It's too big!  We have to break it up. */
      split_planet(planets, planet, new_planets, tick);
    }
  }
}


double calculate_split_energy(double mass, size_t count, double r1, double r2) {
  double constant;
  double sum;
  double alpha;
  double alpha_diff;
  size_t i;

  constant = 0.5 * G * mass * mass * (1/r1 - 1/r2);

  sum = 0.0;

  alpha_diff = 2 * M_PI / count;

  for (i = 1;  i < count;  ++i) {
    alpha = i * alpha_diff;
    sum += sin(0.5 * alpha) / (1 - cos(alpha));
  }

  return constant * sum;
}


void delete_planets(planet_list_t *planets) {
  planet_node_t *node;

  for (node = planets->first;  node;  node = node->next) {
    free(node->planet);
  }
}


void find_collision_groups(planet_list_t *planets, planet_list_t *new_planets, planet_list_t *collision_lists, size_t *collision_count) {
  planet_node_t *node_a;
  planet_node_t *node_b;

  planet_t *planet_a;
  planet_t *planet_b;

  for (node_a = new_planets->size ? new_planets->first : planets->first;  node_a;  node_a = node_a->next) {
    planet_a = node_a->planet;
    for (node_b = new_planets->size ? planets->first : node_a->next;  node_b;  node_b = node_b->next) {
      planet_b = node_b->planet;

      if (planet_a == planet_b) {
        continue;
      }

      if (planet_a->collision_list == PLANET_COUNT_MAX ||
          planet_a->collision_list != planet_b->collision_list) {
        resolve_collision_pair(planet_a, planet_b, collision_lists, collision_count);
      }
    }
  }
}


void resolve_collision_pair(planet_t *p1, planet_t *p2, planet_list_t *collision_lists, size_t *collision_count) {
  double p2_x;
  double p2_y;

  double x_diff, y_diff;
  double distance_squared;

  position_mod(p1, p2, &p2_x, &p2_y);

  x_diff = p2_x - p1->x_pos;
  y_diff = p2_y - p1->y_pos;

  distance_squared = x_diff * x_diff + y_diff * y_diff;

  if (distance_squared < p1->radius_squared + p2->radius_squared) {
    /* Collision! */
    /* We've got to figure out which collision list to put it on. */
    if (p1->collision_list == PLANET_COUNT_MAX) {
      if (p2->collision_list == PLANET_COUNT_MAX) {
        /* It's a new collision group. */
        list_add(&collision_lists[*collision_count], p1);
        list_add(&collision_lists[*collision_count], p2);
        p1->collision_list = *collision_count;
        p2->collision_list = *collision_count;
        ++*collision_count;
      } else {
        /* p2 is already involved in a collision, while p1 is not,
           so we just add p1 to p2's collision list. */
        list_add(&collision_lists[p2->collision_list], p1);
        p1->collision_list = p2->collision_list;
      }
    } else {
      if (p2->collision_list == PLANET_COUNT_MAX) {
        /* p1 is already involved in a collision, while p2 is not,
           so we just add p2 to p1's collision list. */
        list_add(&collision_lists[p1->collision_list], p2);
        p2->collision_list = p1->collision_list;
      } else {
        /* Both planets are already involed in two different collision lists.
           We have to merge the two lists into one. */
        merge_collision_lists(collision_lists, p1->collision_list, p2->collision_list);
      }
    }
  }
}


void merge_collision_lists(planet_list_t *collision_lists, size_t list_a, size_t list_b) {
  planet_node_t *node;
  planet_node_t *b_last;

  b_last = NULL;

  for (node = collision_lists[list_b].first;  node;  b_last = node, node = node->next) {
    node->planet->collision_list = list_a;
  }

  b_last->next = collision_lists[list_a].first;
  collision_lists[list_a].first = collision_lists[list_b].first;

  list_init(&collision_lists[list_b]);
}


void calculate_forces(thread_arg_t *thread_arg) {
  thread_arg->working_planet_count = thread_arg->planets->size;
  thread_arg_reset_planet_node(thread_arg);
  thread_arg_wait_till_zero_working(thread_arg);
}


void calculate_planet_forces(planet_node_t *node) {
  planet_node_t *node_b;
  planet_t *planet_a;

  planet_a = node->planet;

  for (node_b = node->next;  node_b;  node_b = node_b->next) {
    calculate_force_pair(planet_a, node_b->planet);
  }
}


void calculate_force_pair(planet_t *p1, planet_t *p2) {
  double p2_x, p2_y;
  double distance_squared;
  double x_diff, y_diff;
  double distance;
  double force_magnitude;
  double force_ratio;
  double x_force, y_force;

  position_mod(p1, p2, &p2_x, &p2_y);

  x_diff = p2_x - p1->x_pos;
  y_diff = p2_y - p1->y_pos;

  distance_squared = x_diff * x_diff + y_diff * y_diff;
  distance = sqrt(distance_squared);

  if (distance < p1->radius + p2->radius) {
    return;
  }

  force_magnitude = G * p1->mass * p2->mass / distance_squared;

  force_ratio = force_magnitude / distance;

  x_force = force_ratio * x_diff;
  y_force = force_ratio * y_diff;

  planet_add_force(p1, x_force, y_force);
  planet_add_force(p2, -x_force, -y_force);
}


void position_mod(planet_t *p1, planet_t *p2, double *p2_x, double *p2_y) {
  *p2_x = p2->x_pos;
  *p2_y = p2->y_pos;

  if (*p2_x - p1->x_pos > 0.5 * WORLD_WIDTH) {
    *p2_x -= WORLD_WIDTH;
  }
  if (p1->x_pos - *p2_x > 0.5 * WORLD_WIDTH) {
    *p2_x += WORLD_WIDTH;
  }
  if (*p2_y - p1->y_pos > 0.5 * WORLD_HEIGHT) {
    *p2_y -= WORLD_HEIGHT;
  }
  if (p1->y_pos - *p2_y > 0.5 * WORLD_HEIGHT) {
    *p2_y += WORLD_HEIGHT;
  }
}


void move_planets(planet_list_t *planets) {
  planet_node_t *node;
  planet_t *planet;
  double x_accel, y_accel;

  for (node = planets->first;  node;  node = node->next) {
    planet = node->planet;
    x_accel = planet->x_force / planet->mass / (FRAMES_PER_SECOND * TICKS_PER_FRAME);
    y_accel = planet->y_force / planet->mass / (FRAMES_PER_SECOND * TICKS_PER_FRAME);

    planet->x_vel += x_accel;
    planet->y_vel += y_accel;

    planet->x_pos += planet->x_vel / (FRAMES_PER_SECOND * TICKS_PER_FRAME);
    planet->y_pos += planet->y_vel / (FRAMES_PER_SECOND * TICKS_PER_FRAME);

    planet->x_pos = mod_double(planet->x_pos, 0.0, WORLD_WIDTH);
    planet->y_pos = mod_double(planet->y_pos, 0.0, WORLD_HEIGHT);

    planet->x_force = 0.0;
    planet->y_force = 0.0;
  }
}


double mod_double(double value, double min, double max) {
  const double diff = max - min;

  while (value < min) {
    value += diff;
  }
  while (value >= max) {
    value -= diff;
  }

  return value;
}


void wait_for_next_tick(struct timeval *start) {
  struct timeval now;
  struct timeval diff;
  struct timeval interval;
  struct timeval interval_remaining;

  interval.tv_sec = 0;
  interval.tv_usec = 1000000.0 / FRAMES_PER_SECOND;

  gettimeofday(&now, NULL);

  timersub(&now, start, &diff);
  timersub(&interval, &diff, &interval_remaining);

  select(0, NULL, NULL, NULL, &interval_remaining);
}


planet_t *planet_new(double x_pos, double y_pos, double x_vel, double y_vel, double mass, double hue, size_t tick) {
  planet_t *planet;

  planet = my_malloc(sizeof(*planet));

  planet_init(planet, x_pos, y_pos, x_vel, y_vel, mass, hue, tick);

  return planet;
}


void planet_init(planet_t *planet, double x_pos, double y_pos, double x_vel, double y_vel, double mass, double hue, size_t tick) {
  planet->x_pos = x_pos;
  planet->y_pos = y_pos;
  planet->x_vel = x_vel;
  planet->y_vel = y_vel;
  planet->mass = mass;

  planet->radius = radius_for_mass(mass);
  planet->radius_squared = planet->radius * planet->radius;

  planet->x_force = 0.0;
  planet->y_force = 0.0;

  planet->hue = hue;
  planet->hue_tick = tick;

  planet->collision_list = PLANET_COUNT_MAX;

#ifdef USE_PLANET_MUTEX
  pthread_mutex_init(&planet->mutex, 0);
#endif
}


void planet_reinit(planet_t *planet, double x_pos, double y_pos, double x_vel, double y_vel, double mass, double hue, size_t tick) {
  planet_delete(planet);
  planet_init(planet, x_pos, y_pos, x_vel, y_vel, mass, hue, tick);
}


void planet_delete(planet_t *planet) {
#ifdef USE_PLANET_MUTEX
  pthread_mutex_destroy(&planet->mutex);
#endif
}


void planet_add_force(planet_t *planet, double x_force, double y_force) {
#ifdef USE_PLANET_MUTEX
  pthread_mutex_lock(&planet->mutex);
#endif
    planet->x_force += x_force;
    planet->y_force += y_force;
#ifdef USE_PLANET_MUTEX
  pthread_mutex_unlock(&planet->mutex);
#endif
}


double radius_for_mass(double mass) {
  double volume;

  volume = mass / PLANET_DENSITY;

  /* Volume of a sphere is 4/3 * PI * r^3
     r^3 = 3/4 * V/PI
  */
  return pow(0.75 * M_1_PI * volume, 1.0/3.0);
}


void list_init(planet_list_t *list) {
  list->first = NULL;
  list->size = 0;
}


void list_add(planet_list_t *list, planet_t *planet) {
  planet_node_t *node;

  node = my_malloc(sizeof(*node));

  node->planet = planet;
  node->next = list->first;
  list->first = node;

  ++list->size;
}


planet_t *list_remove_first(planet_list_t *list) {
  planet_node_t *node;
  planet_t *planet;

  node = list->first;
  planet = node->planet;

  list->first = node->next;
  --list->size;

  free(node);

  return planet;
}


void list_remove_all(planet_list_t *list, planet_list_t *removing) {
  planet_node_t *node;
  planet_node_t *prev;
  planet_node_t *next;

  prev = NULL;
  for (node = list->first;  node;  node = next) {
    next = node->next;
    if (list_contains(removing, node->planet)) {
      if (prev) {
        prev->next = next;
      } else {
        list->first = next;
      }
      free(node);
      --list->size;
    } else {
      prev = node;
    }
  }
}


int list_contains(planet_list_t *list, planet_t *planet) {
  planet_node_t *node;

  for (node = list->first;  node;  node = node->next) {
    if (node->planet == planet) {
      return 1;
    }
  }

  return 0;
}


void list_delete(planet_list_t *list) {
  planet_node_t *node;
  planet_node_t *next;

  if (!list->size) {
    return;
  }

  for (node = list->first;  node;  node = next) {
    next = node->next;
    free(node);
  }

  list_init(list);
}


void thread_arg_init(thread_arg_t *arg, planet_list_t *planets) {
  arg->planets = planets;
  arg->planet_node = 0;
  pthread_mutex_init(&arg->mutex, 0);
  pthread_cond_init(&arg->cond, 0);
}


planet_node_t *thread_arg_get_planet_node(thread_arg_t *arg, int finished_one) {
  planet_node_t *node;

  pthread_mutex_lock(&arg->mutex);
    if (finished_one) {
      --arg->working_planet_count;
      if (arg->working_planet_count == 0) {
        pthread_cond_broadcast(&arg->cond);
      }
    }
    if (arg->running) {
      if (!arg->planet_node) {
        pthread_cond_wait(&arg->cond, &arg->mutex);
      }
      node = arg->planet_node;
      if (node) {
        arg->planet_node = node->next;
      }
    } else {
      node = 0;
      pthread_cond_broadcast(&arg->cond);
    }
  pthread_mutex_unlock(&arg->mutex);

  return node;
}


void thread_arg_reset_planet_node(thread_arg_t *arg) {
  pthread_mutex_lock(&arg->mutex);
    arg->planet_node = arg->planets->first;
    pthread_cond_broadcast(&arg->cond);
  pthread_mutex_unlock(&arg->mutex);
}


void thread_arg_wait_till_zero_working(thread_arg_t *arg) {
  pthread_mutex_lock(&arg->mutex);
    while (arg->working_planet_count > 0) {
      pthread_cond_wait(&arg->cond, &arg->mutex);
    }
  pthread_mutex_unlock(&arg->mutex);
}


void thread_arg_stop_running(thread_arg_t *arg) {
  pthread_mutex_lock(&arg->mutex);
    arg->running = 0;
    pthread_cond_broadcast(&arg->cond);
  pthread_mutex_unlock(&arg->mutex);
}


void *my_malloc(size_t size) {
  void *ptr;

  ptr = malloc(size);

  if (!ptr) {
    perror("malloc()");
    exit(1);
  }

  return ptr;
}
