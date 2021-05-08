#ifndef CONSTANTS_H_
#define CONSTANTS_H_

typedef struct SimulationParameters
{

    //bool debug;
    //bool benchmark;
    //bool fullscreen;
    bool timeKernels;
    int iterations;
    int numberOfParticles;
    float timestep;
    float gravity;
    float dampening;
    int gridSize;
    int blockSize;
    int warp;
    int stackSize;
    int renderInterval;

} SimulationParameters;

/// Physical constants
const double PI = 3.14159265358979323846;   //! Pi
const double TO_METERS = 1.496e11;          //! AU to meters
const double G = 6.67408e-11;               //! Gravitational constant


/// Rendering related
const int WIDTH = 1024;
const int HEIGHT = 1024;
const double RENDER_SCALE = 2.5;
const double MAX_VEL_COLOR = 1500.0;
const double MIN_VEL_COLOR = 0;
const double PARTICLE_BRIGHTNESS = 0.35;
const double PARTICLE_SHARPNESS = 1.0;
const int DOT_SIZE = 8;
const int RENDER_INTERVAL = 2; // How many timesteps to simulate in between each frame rendered


#endif /* CONSTANTS_H_ */
