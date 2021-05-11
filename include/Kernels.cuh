/**
 * CUDA Kernel functions
 *
 * See
 * * [Summary: An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm](../resources/NBodyCUDA.md)
 * * [An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm](https://iss.oden.utexas.edu/Publications/Papers/burtscher11.pdf)
 */

#ifndef CUDA_NBODY_KERNELS_CUH
#define CUDA_NBODY_KERNELS_CUH

#include "Constants.h"
#include "SubDomainKeyTree.cuh"

#include <iostream>
#include <stdio.h>
#include <cuda.h>

/**
 * Reset the arrays/pointers.
 *
 * @param mutex lock
 * @param x x-coordinate
 * @param y y-coordinate
 * @param z z-coordinate
 * @param mass mass
 * @param count counter
 * @param start start index
 * @param sorted sorted particles (array)
 * @param child child
 * @param index index
 * @param minX min(x-coordinate)
 * @param maxX max(x-coordinate)
 * @param minY min(x-coordinate)
 * @param maxY max(y-coordinate)
 * @param minZ min(z-coordinate)
 * @param maxZ max(z-coordinate)
 * @param n number of particles
 * @param m number of nodes
 */
__global__ void resetArraysKernel(int *mutex, float *x, float *y, float *z, float *mass, int *count, int *start,
                                  int *sorted, int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                  float *minZ, float *maxZ, int n, int m);

/**
 * Kernel 1: computes bounding box around all bodies
 *
 * @param mutex lock
 * @param x x-coordinate
 * @param y y-coordinate
 * @param z z-coordinate
 * @param[out] minX min(x-coordinate)
 * @param[out] maxX max(x-coordinate)
 * @param[out] minY min(x-coordinate)
 * @param[out] maxY max(y-coordinate)
 * @param[out] minZ min(z-coordinate)
 * @param[out] maxZ max(z-coordinate)
 * @param n number of particles
 */
__global__ void computeBoundingBoxKernel(int *mutex, float *x, float *y, float *z, float *minX, float *maxX,
                                         float *minY, float *maxY, float *minZ, float *maxZ, int n, int blockSize);

/**
 * Kernel 2: hierarchically subdivides the root cells
 *
 * @param x x-coordinate
 * @param y y-coordinate
 * @param z z-coordinate
 * @param mass mass
 * @param count counter
 * @param start start index
 * @param sorted sorted particles (array)
 * @param child child
 * @param index index
 * @param minX min(x-coordinate)
 * @param maxX max(x-coordinate)
 * @param minY min(x-coordinate)
 * @param maxY max(y-coordinate)
 * @param minZ min(z-coordinate)
 * @param maxZ max(z-coordinate)
 * @param n number of particles
 * @param m number of nodes
 */
__global__ void buildTreeKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int n, int m);

__device__ void key2Char(unsigned long key, int maxLevel, char *keyAsChar);

__global__ void getParticleKeyKernel(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                               float *minZ, float *maxZ, unsigned long *key, int maxLevel, int n, SubDomainKeyTree *s);


__device__ int key2proc(unsigned long k, SubDomainKeyTree *s);

/**
 * Kernel 3: computes the COM for each cell
 *
 * @param x x-coordinate
 * @param y y-coordinate
 * @param z z-coordinate
 * @param mass mass
 * @param index
 * @param n number of particles
 */
__global__ void centreOfMassKernel(float *x, float *y, float *z, float *mass, int *index, int n);

/**
 * Kernel 4: sorts the bodies
 *
 * @param count counter
 * @param start start index
 * @param sorted sorted sorted particles (array)
 * @param child
 * @param index
 * @param n number of particles
 */
__global__ void sortKernel(int *count, int *start, int *sorted, int *child, int *index, int n);

/**
 * Kernel 5: computes the (gravitational) forces
 *
 * @param x x-coordinate
 * @param y y-coordinate
 * @param z z-coordinate
 * @param vx x-velocity
 * @param vy y-velocity
 * @param vz z-velocity
 * @param ax x-acceleration
 * @param ay y-acceleration
 * @param az z-acceleration
 * @param mass mass
 * @param sorted
 * @param child
 * @param minX min(x-coordinate)
 * @param maxX max(x-coordinate)
 * @param n number of particles
 * @param g gravitational constant (not needed!?)
 */
__global__ void computeForcesKernel(float* x, float *y, float *z, float *vx, float *vy, float *vz,
                                    float *ax, float *ay, float *az, float *mass, int *sorted, int *child,
                                    float *minX, float *maxX, int n, float g, int blockSize, int warp, int stackSize);

/**
 * Kernel 6: updates the bodies
 *
 * @param x x-coordinate
 * @param y y-coordinate
 * @param z z-coordinate
 * @param vx x-velocity
 * @param vy y-velocity
 * @param vz z-velocity
 * @param ax x-acceleration
 * @param ay y-acceleration
 * @param az z-acceleration
 * @param n number of particles
 * @param dt time step
 * @param d distance
 */
__global__ void updateKernel(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                             float *ax, float *ay, float *az, int n, float dt, float d);


#endif //CUDA_NBODY_KERNELS_CUH
