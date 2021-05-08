/**
 * Wrapping CUDA Kernel functions.
 */

#ifndef CUDA_NBODY_KERNELSWRAPPER_H
#define CUDA_NBODY_KERNELSWRAPPER_H

#include <iostream>
#include <cuda.h>

#include "Kernels.cuh"
#include "Constants.h"


class KernelsWrapper {

public:

    dim3 gridSize;  //= 1024; //2048; //1024; //512;
    dim3 blockSize; //256; //256;
    int blockSizeInt;
    int warp;
    int stackSize;

    KernelsWrapper();
    KernelsWrapper(SimulationParameters p);

    float resetArrays(int *mutex, float *x, float *y, float *z, float *mass, int *count,
                      int *start, int *sorted, int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                      float *minZ, float *maxZ, int n, int m, bool timing=false);

    float computeBoundingBox(int *mutex, float *x, float *y, float *z, float *minX,
                             float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int n, bool timing=false);

    float buildTree(float *x, float *y, float *z, float *mass, int *count, int *start,
                    int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                    float *minZ, float *maxZ, int n, int m, bool timing=false);

    void getParticleKey(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                                   float *minZ, float *maxZ, unsigned long *key, int maxLevel, int n);

    float centreOfMass(float *x, float *y, float *z, float *mass, int *index, int n, bool timing=false);

    float sort(int *count, int *start, int *sorted, int *child, int *index, int n, bool timing=false);

    float computeForces(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                        float *ax, float *ay, float *az, float *mass, int *sorted, int *child,
                        float *minX, float *maxX, int n, float g, bool timing=false);

    float update(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                 float *ax, float *ay, float *az, int n, float dt, float d, bool timing=false);

};

#endif //CUDA_NBODY_KERNELSWRAPPER_H
