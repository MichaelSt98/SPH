/**
 * CUDA Kernel functions
 *
 * See
 * * [Summary: An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm](../resources/NBodyCUDA.md)
 * * [An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm](https://iss.oden.utexas.edu/Publications/Papers/burtscher11.pdf)
 */

#ifndef CUDA_NBODY_KERNELS_CUH
#define CUDA_NBODY_KERNELS_CUH

//#define KEY_MAX ULONG_MAX

#include "Constants.h"
#include "SubDomainKeyTree.cuh"

#include <iostream>
#include <stdio.h>
#include <cuda.h>

/**
 * Reset the arrays/pointers.
 */
__global__ void resetArraysKernel(int *mutex, float *x, float *y, float *z, float *mass, int *count, int *start,
                                  int *sorted, int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                  float *minZ, float *maxZ, int n, int m);

__global__ void resetArraysParallelKernel(int *domainListIndex, unsigned long *domainListKeys,
                                          unsigned long *domainListIndices, int *domainListLevels);

/**
 * Kernel 1: computes bounding box around all bodies
 */
__global__ void computeBoundingBoxKernel(int *mutex, float *x, float *y, float *z, float *minX, float *maxX,
                                         float *minY, float *maxY, float *minZ, float *maxZ, int n, int blockSize);

__global__ void buildDomainTreeKernel(int *domainListIndex, unsigned long *domainListKeys, int *domainListLevels,
                                      int *count, int *start, int *child, int *index, int n, int m);

/**
 * Kernel 2: hierarchically subdivides the root cells
 */
__global__ void buildTreeKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int n, int m);

__device__ void key2Char(unsigned long key, int maxLevel, char *keyAsChar);

__global__ void getParticleKeyKernel(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                               float *minZ, float *maxZ, unsigned long *key, int maxLevel, int n, SubDomainKeyTree *s);

__device__ unsigned long getParticleKeyPerParticle(float x, float y, float z,
                                                   float *minX, float *maxX, float *minY,
                                                   float *maxY, float *minZ, float *maxZ,
                                                   int maxLevel);

__device__ int key2proc(unsigned long k, SubDomainKeyTree *s);

__global__ void traverseIterativeKernel(float *x, float *y, float *z, float *mass, int *child, int n, int m,
                         SubDomainKeyTree *s, int maxLevel);

/*__global__ void createDomainListKernel(float *x, float *y, float *z, float *mass, int *child, int n,
                                       SubDomainKeyTree *s, int maxLevel);*/

/*__global__ void createDomainListKernel(float *x, float *y, float *z, float *mass, float *minX, float *maxX,
                                       float *minY, float *maxY, float *minZ, float *maxZ, int *child, int n,
                                       SubDomainKeyTree *s, int maxLevel);*/

__global__ void createDomainListKernel(SubDomainKeyTree *s, int maxLevel, unsigned long *domainListKeys, int *levels,
                                       int *index);

__device__ bool isDomainListNode(unsigned long key, int maxLevel, int level, SubDomainKeyTree *s);

__device__ unsigned long keyMaxLevel(unsigned long key, int maxLevel, int level, SubDomainKeyTree *s);

/**
 * Kernel 3: computes the COM for each cell
 */
__global__ void centreOfMassKernel(float *x, float *y, float *z, float *mass, int *index, int n);

/**
 * Kernel 4: sorts the bodies
 */
__global__ void sortKernel(int *count, int *start, int *sorted, int *child, int *index, int n);

/**
 * Kernel 5: computes the (gravitational) forces
 */
__global__ void computeForcesKernel(float* x, float *y, float *z, float *vx, float *vy, float *vz,
                                    float *ax, float *ay, float *az, float *mass, int *sorted, int *child,
                                    float *minX, float *maxX, int n, float g, int blockSize, int warp, int stackSize);

/**
 * Kernel 6: updates the bodies
 */
__global__ void updateKernel(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                             float *ax, float *ay, float *az, int n, float dt, float d);


#endif //CUDA_NBODY_KERNELS_CUH
