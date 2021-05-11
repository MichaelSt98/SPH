#ifndef CUDA_NBODY_BARNESHUT_H
#define CUDA_NBODY_BARNESHUT_H

#include <random>
#include "Constants.h"
#include "Body.h"
#include "Logger.h"
#include "KernelsWrapper.cuh"

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <climits> // for ulong_max
#define KEY_MAX ULONG_MAX

#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

class BarnesHut {

private:

    SimulationParameters parameters;
    KernelsWrapper KernelHandler;

    int step;
    int numParticles;
    int numNodes;

    float *h_min_x;
    float *h_max_x;
    float *h_min_y;
    float *h_max_y;
    float *h_min_z;
    float *h_max_z;

    float *h_mass;

    /*float *h_x;
    float *h_y;
    float *h_z;

    float *h_vx;
    float *h_vy;
    float *h_vz;*/

    float *h_ax;
    float *h_ay;
    float *h_az;

    int *h_child;
    int *h_start;
    int *h_sorted;
    int *h_count;

    SubDomainKeyTree *h_subDomainHandler;

    float *d_min_x;
    float *d_max_x;
    float *d_min_y;
    float *d_max_y;
    float *d_min_z;
    float *d_max_z;

    float *d_mass;

    float *d_x;
    float *d_y;
    float *d_z;

    float *d_vx;
    float *d_vy;
    float *d_vz;

    float *d_ax;
    float *d_ay;
    float *d_az;

    int *d_index;
    int *d_child;
    int *d_start;
    int *d_sorted;
    int *d_count;

    SubDomainKeyTree *d_subDomainHandler;
    unsigned long *d_range;

    int *d_mutex;  //used for locking

    cudaEvent_t start, stop; // used for timing
    cudaEvent_t start_global, stop_global; // used for timing

    float *h_output;  //host output array for visualization
    float *d_output;  //device output array for visualization

    void plummerModel(float *mass, float *x, float* y, float* z,
                      float *x_vel, float *y_vel, float *z_vel,
                      float *x_acc, float *y_acc, float *z_acc, int n);

    void diskModel(float *mass, float *x, float* y, float* z,
                   float *x_vel, float *y_vel, float *z_vel,
                   float *x_acc, float *y_acc, float *z_acc, int n);

public:

    bool timeKernels;

    float *time_resetArrays;
    float *time_computeBoundingBox;
    float *time_buildTree;
    float *time_centreOfMass;
    float *time_sort;
    float *time_computeForces;
    float *time_update;
    float *time_copyDeviceToHost;
    float *time_all;

    float *h_x;
    float *h_y;
    float *h_z;

    float *h_vx;
    float *h_vy;
    float *h_vz;

    BarnesHut(const SimulationParameters p);
    ~BarnesHut();

    void update(int step);
    void reset();
    float getSystemSize();

};


#endif //CUDA_NBODY_BARNESHUT_H
