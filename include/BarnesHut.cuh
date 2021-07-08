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
#include <utility>
//#define KEY_MAX ULONG_MAX

#include <set>
#include <fstream>
#include <iomanip>
#include <highfive/H5File.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5DataSet.hpp>

#define TESTING 0
#define CUDA_AWARE_MPI_TESTING 0

#define SafeCudaCall(call) CheckCudaCall(call, #call, __FILE__, __LINE__)
#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
void CheckCudaCall(cudaError_t command, const char * commandName, const char * fileName, int line);

class BarnesHut {

private:

    //---- HOST VARIABLES ----------------------------------------------------------------------------------------------
    // Simulation parameters/settings
    SimulationParameters parameters;

    // Instance to call CUDA Kernels via Wrapper
    KernelsWrapper KernelHandler;

    // process/subdomain handler (MPI rank, number of processes, ...)
    //SubDomainKeyTree *h_subDomainHandler;

    // current iteration step
    int step;

    // total number of particles (on all processes)
    int numParticles;
    // number of particles on process
    int numParticlesLocal;
    // number of nodes on process
    int numNodes;

    // bounding box, domain size
    float *h_min_x;
    float *h_max_x;
    float *h_min_y;
    float *h_max_y;
    float *h_min_z;
    float *h_max_z;

    int *h_domainListIndices;
    unsigned long *h_domainListKeys;
    int *h_domainListLevels;
    int *h_domainListIndex;

    int *h_procCounter;

    // particle masses
    float *h_mass;

    // particle positions
    float *h_x;
    float *h_y;
    float *h_z;

    // particle velocities
    float *h_vx;
    float *h_vy;
    float *h_vz;

    // particle accelerations
    float *h_ax;
    float *h_ay;
    float *h_az;

    unsigned long *h_keys;

    // children (needed for tree construction)
    int *h_child;
    int *h_start;
    int *h_sorted;
    // number of children (in order to optimize performance)
    int *h_count;

    //---- DEVICE VARIABLES --------------------------------------------------------------------------------------------

    // process/subdomain handler (MPI rank, number of processes, ...)
    //SubDomainKeyTree *d_subDomainHandler;
    unsigned long *d_range;

    // lock/mutex
    int *d_mutex;  //used for locking

    // CUDA performance analysis
    cudaEvent_t start, stop; // used for timing
    cudaEvent_t start_global, stop_global; // used for timing

    // bounding box, domain size
    float *d_min_x;
    float *d_max_x;
    float *d_min_y;
    float *d_max_y;
    float *d_min_z;
    float *d_max_z;

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

    // particles masses
    float *d_mass;

    // particles positions
    float *d_x;
    float *d_y;
    float *d_z;

    // particles velocities
    float *d_vx;
    float *d_vy;
    float *d_vz;

    // particles accelerations
    float *d_ax;
    float *d_ay;
    float *d_az;

    unsigned long *d_keys;

    int *d_index;
    int *d_child;
    int *d_start;
    int *d_sorted;
    int *d_count;


    //float *h_output;  //host output array for visualization
    //float *d_output;  //device output array for visualization

    //---- PRIVATE FUNCTIONS -------------------------------------------------------------------------------------------

    void plummerModel(float *mass, float *x, float* y, float* z,
                      float *x_vel, float *y_vel, float *z_vel,
                      float *x_acc, float *y_acc, float *z_acc, int n);

    void diskModel(float *mass, float *x, float* y, float* z,
                   float *x_vel, float *y_vel, float *z_vel,
                   float *x_acc, float *y_acc, float *z_acc, int n);

    void initRange();
    void initRange(int binSize);

    float globalizeBoundingBox(bool timing=false);

    void sortArrayRadix(float *arrayToSort, float *tempArray, int *keyIn, int *keyOut, int n);
    void sortArrayRadix(float *arrayToSort, float *tempArray, unsigned long *keyIn, unsigned long *keyOut, int n);

    int sendParticlesEntry(int *sendLengths, int *receiveLengths, float *entry);

    void exchangeParticleEntry(int *sendLengths, int *receiveLengths, float *entry);

    void compPseudoParticlesParallel();

    float parallelForce();

    int deleteDuplicates(int numItems);

    void newLoadDistribution();
    void updateRange();
    void updateRangeApproximately(int aimedParticlesPerProcess, int bins=4000);

public:

    // time kernels y/n, comes with some overhead
    bool timeKernels;

    // arrays to store the time for each iteration (step), for postprocessing...
    float *time_resetArrays;
    float *time_computeBoundingBox;
    float *time_buildTree;
    float *time_centreOfMass;
    float *time_sort;
    float *time_computeForces;
    float *time_update;
    float *time_copyDeviceToHost;
    float *time_all;

    // (all) particle positions for visualization/output/...
    float *all_x;
    float *all_y;
    float *all_z;

    // (all) particle velocities for visualization/output/...
    float *all_vx;
    float *all_vy;
    float *all_vz;

    SubDomainKeyTree *h_subDomainHandler;
    SubDomainKeyTree *d_subDomainHandler;

    //---- CONSTRUCTOR -------------------------------------------------------------------------------------------------

    BarnesHut(const SimulationParameters p);

    //---- DESTRUCTOR --------------------------------------------------------------------------------------------------

    ~BarnesHut();

    //---- PUBLIC FUNCTIONS --------------------------------------------------------------------------------------------
    // getter
    int getNumParticlesLocal();
    float getSystemSize();

    // "main" function
    void update(int step);

    // gathering/collecting particles from all processes
    int gatherParticles(bool velocities=true, bool deviceToHost=false);

    void particles2file(HighFive::DataSet *pos, HighFive::DataSet *vel, HighFive::DataSet *key);

    };


#endif //CUDA_NBODY_BARNESHUT_H
