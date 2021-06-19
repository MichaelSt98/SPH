#ifndef CUDA_NBODY_BARNESHUT_H
#define CUDA_NBODY_BARNESHUT_H

#include <random>
#include "Constants.h"
#include "Body.h"
#include "Logger.h"
#include "KernelsWrapper.cuh"

#include <mpi.h>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits> // for ulong_max
#include <algorithm>
#include <cmath>
//#define KEY_MAX ULONG_MAX

#define TESTING 0
#define CUDA_AWARE_MPI_TESTING 0

#define SafeCudaCall(call) CheckCudaCall(call, #call, __FILE__, __LINE__)
#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
void CheckCudaCall(cudaError_t command, const char * commandName, const char * fileName, int line);

class BarnesHut {

private:

    SimulationParameters parameters;
    KernelsWrapper KernelHandler;

    int step;
    int numParticles;
    int numParticlesLocal;
    int numNodes;

    float *h_min_x;
    float *h_max_x;
    float *h_min_y;
    float *h_max_y;
    float *h_min_z;
    float *h_max_z;

    float *h_mass;

    int *h_domainListIndices;
    unsigned long *h_domainListKeys;
    int *h_domainListLevels;
    int *h_domainListIndex;

    int *h_procCounter;

    //changed to public ...
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

    //int *h_tempIntArray;

    SubDomainKeyTree *h_subDomainHandler;

    float *d_min_x;
    float *d_max_x;
    float *d_min_y;
    float *d_max_y;
    float *d_min_z;
    float *d_max_z;

    float *d_mass;

    int *d_domainListIndices;
    unsigned long *d_domainListKeys;
    int *d_domainListLevels;
    int *d_domainListIndex;

    int *d_lowestDomainListIndex;
    int *d_lowestDomainListIndices;
    unsigned long *d_lowestDomainListKeys;
    unsigned long *d_sortedLowestDomainListKeys;
    int *d_lowestDomainListCounter;

    int *d_tempIntArray;

    float *d_tempArray;
    float *d_tempArray_2;
    int *d_sortArray;
    int *d_sortArrayOut;

    int *d_procCounter;
    int *d_procCounterTemp;

    int *d_domainListCounter;
    int *d_relevantDomainListIndices;
    int *d_sendIndices;
    int *d_sendIndicesTemp;

    int *d_to_delete_cell;
    int *d_to_delete_leaf;

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

    float *all_x;
    float *all_y;
    float *all_z;

    float *all_vx;
    float *all_vy;
    float *all_vz;

    BarnesHut(const SimulationParameters p);
    ~BarnesHut();

    void update(int step);
    void reset();
    float getSystemSize();
    void globalizeBoundingBox();

    void sortArrayRadix(float *arrayToSort, float *tempArray, int *keyIn, int *keyOut, int n);
    void sortArrayRadix(float *arrayToSort, float *tempArray, unsigned long *keyIn, unsigned long *keyOut, int n);
    //int gatherParticles(float *xAll, float *yAll, float *zAll, bool deviceToHost=false);
    int gatherParticles(bool velocities=true, bool deviceToHost=false);

    int sendParticlesEntry(int *sendLengths, int *receiveLengths, float *entry);

    void exchangeParticleEntry(int *sendLengths, int *receiveLengths, float *entry);

    void compPseudoParticlesParallel();

    float parallelForce();

    int deleteDuplicates(int numItems);
};


#endif //CUDA_NBODY_BARNESHUT_H
