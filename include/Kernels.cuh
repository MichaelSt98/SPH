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

#include "cub/cub.cuh"
#include <iostream>
#include <stdio.h>
#include <cuda.h>

#include <thrust/device_vector.h>

/**
 * Reset the arrays/pointers.
 */
__global__ void resetArraysKernel(int *mutex, float *x, float *y, float *z, float *mass, int *count, int *start,
                                  int *sorted, int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                  float *minZ, float *maxZ, int n, int m, int *procCounter, int *procCounterTemp);

__global__ void resetArraysParallelKernel(int *domainListIndex, unsigned long *domainListKeys,
                                          int *domainListIndices, int *domainListLevels,
                                          int *lowestDomainListIndices, int *lowestDomainListIndex,
                                          unsigned long *lowestDomainListKeys, unsigned long *sortedLowestDomainListKeys,
                                          float *tempArray, int *to_delete_cell,
                                          int *to_delete_leaf, int n, int m);

/**
 * Kernel 1: computes bounding box around all bodies
 */
__global__ void computeBoundingBoxKernel(int *mutex, float *x, float *y, float *z, float *minX, float *maxX,
                                         float *minY, float *maxY, float *minZ, float *maxZ, int n, int blockSize);

__global__ void buildDomainTreeKernel(int *domainListIndex, unsigned long *domainListKeys, int *domainListLevels,
                                      int *domainListIndices, float *x, float *y, float *z, float *mass, float *minX,
                                      float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int *count,
                                      int *start, int *child, int *index, int n, int m);

__global__ void lowestDomainListNodesKernel(int *domainListIndices, int *domainListIndex,
                                      unsigned long *domainListKeys,
                                      int *lowestDomainListIndices, int *lowestDomainListIndex,
                                      unsigned long *lowestDomainListKeys,
                                      float *x, float *y, float *z, float *mass, int *count, int *start,
                                      int *child, int n, int m, int *procCounter);

__global__ void particlesPerProcessKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                    int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                    float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                                    int *procCounterTemp);

__global__ void markParticlesProcessKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                           int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                           float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                                           int *procCounterTemp, int *sortArray);

__global__ void copyArrayKernel(float *targetArray, float *sourceArray, int n);

__global__ void resetFloatArrayKernel(float *array, float value, int n);

__global__ void debugKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                    int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                    float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                                    float *tempArray, int *sortArray, int *sortArrayOut);

/**
 * Kernel 2: hierarchically subdivides the root cells
 */
__global__ void buildTreeKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int n, int m);

__global__ void treeInfoKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                               int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                               float *minZ, float *maxZ, int n, int m, int *procCounter, SubDomainKeyTree *s,
                               int *sortArray, int *sortArrayOut);

__global__ void domainListInfoKernel(float *x, float *y, float *z, float *mass, int *child, int *index, int n,
                                     int *domainListIndices, int *domainListIndex,
                                     int *domainListLevels, int *lowestDomainListIndices,
                                     int *lowestDomainListIndex, SubDomainKeyTree *s);

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

__global__ void createDomainListKernel(SubDomainKeyTree *s, int maxLevel, unsigned long *domainListKeys, int *levels,
                                       int *index);

__global__ void prepareLowestDomainExchangeKernel(float *entry, float *mass, float *tempArray, int *lowestDomainListIndices,
                                                  int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                                  int *counter);

__global__ void prepareLowestDomainExchangeMassKernel(float *mass, float *tempArray, int *lowestDomainListIndices,
                                                      int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                                      int *counter);

__global__ void updateLowestDomainListNodesKernel(float *tempArray, float *entry, int *lowestDomainListIndices,
                                                  int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                                  unsigned long *sortedLowestDomainListKeys, int *counter);

__global__ void compLowestDomainListNodesKernel(float *x, float *y, float *z, float *mass, int *lowestDomainListIndices,
                                                int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                                unsigned long *sortedLowestDomainListKeys, int *counter);

__global__ void zeroDomainListNodesKernel(int *domainListIndex, int *domainListIndices, int *lowestDomainListIndex,
                                          int *lowestDomainListIndices, float *x, float *y, float *z, float *mass);

__global__ void compLocalPseudoParticlesParKernel(float *x, float *y, float *z, float *mass, int *index, int n,
                                                  int *domainListIndices, int *domainListIndex,
                                                  int *lowestDomainListIndices, int *lowestDomainListIndex);

__global__ void compDomainListPseudoParticlesParKernel(float *x, float *y, float *z, float *mass, int *child, int *index, int n,
                                                       int *domainListIndices, int *domainListIndex,
                                                       int *domainListLevels, int *lowestDomainListIndices,
                                                       int *lowestDomainListIndex);

__device__ bool isDomainListNode(unsigned long key, int maxLevel, int level, SubDomainKeyTree *s);

__device__ unsigned long keyMaxLevel(unsigned long key, int maxLevel, int level, SubDomainKeyTree *s);

/**
 * Kernel 3: computes the COM for each cell
 */
__global__ void centreOfMassKernel(float *x, float *y, float *z, float *mass, int *index, int n);

/**
 * Kernel 4: sorts the bodies
 */
__global__ void sortKernel(int *count, int *start, int *sorted, int *child, int *index, int n, int m);

/**
 * Kernel 5: computes the (gravitational) forces
 */
__global__ void computeForcesKernel(float* x, float *y, float *z, float *vx, float *vy, float *vz,
                                    float *ax, float *ay, float *az, float *mass, int *sorted, int *child,
                                    float *minX, float *maxX, float *minY, float *maxY,
                                    float *minZ, float *maxZ, int n, int m, float g, int blockSize, int warp,
                                    int stackSize, SubDomainKeyTree *s);


__device__ float smallestDistance(float* x, float *y, float *z, int node1, int node2);

__global__ void collectSendIndicesKernel(int *sendIndices, float *entry, float *tempArray, int *domainListCounter,
                                   int sendCount);

__global__ void symbolicForceKernel(int relevantIndex, float *x, float *y, float *z, float *minX, float *maxX, float *minY,
                                    float *maxY, float *minZ, float *maxZ, int *child, int *domainListIndex,
                                    unsigned long *domainListKeys, int *domainListIndices, int *domainListLevels,
                                    int *domainListCounter, int *sendIndices, int *index, int *particleCounter,
                                    SubDomainKeyTree *s, int n, int m, float diam, float theta_, int *mutex,
                                    int *relevantDomainListIndices);

__global__ void compThetaKernel(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int *domainListIndex, int *domainListCounter,
                                unsigned long *domainListKeys, int *domainListIndices, int *domainListLevels,
                                int *relevantDomainListIndices, SubDomainKeyTree *s);

/**
 * Kernel 6: updates the bodies
 */
__global__ void updateKernel(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                             float *ax, float *ay, float *az, int n, float dt, float d);

__global__ void insertReceivedParticlesKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                        int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                        float *minZ, float *maxZ, int *to_delete_leaf, int *domainListIndices,
                                        int *domainListIndex, int *lowestDomainListIndices, int *lowestDomainListIndex,
                                        int n, int m);

__global__ void centreOfMassReceivedParticlesKernel(float *x, float *y, float *z, float *mass, int *startIndex, int *endIndex, int n);

__global__ void repairTreeKernel(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                           float *ax, float *ay, float *az, float *mass, int *count, int *start,
                           int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                           float *minZ, float *maxZ, int *to_delete_cell, int *to_delete_leaf,
                           int *domainListIndices, int n, int m);

__device__ int getTreeLevel(int index, int *child, float *x, float *y, float *z, float *minX, float *maxX, float *minY,
                            float *maxY, float *minZ, float *maxZ);

__global__ void findDuplicatesKernel(float *array, float *array_2, int length, SubDomainKeyTree *s, int *duplicateCounter);

__global__ void markDuplicatesKernel(int *indices, float *x, float *y, float *z,
                                     float *mass, SubDomainKeyTree *s, int *counter, int length);

__global__ void removeDuplicatesKernel(int *indices, int *removedDuplicatesIndices, int *counter, int length);

__global__ void getParticleCount(int *child, int *count, int *particleCount);

__global__ void createKeyHistRangesKernel(int bins, unsigned long *keyHistRanges);

__global__ void keyHistCounterKernel(unsigned long *keyHistRanges, int *keyHistCounts, int bins, int n,
                                     float *x, float *y, float *z, float *mass, int *count, int *start,
                                     int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                     float *minZ, float *maxZ, SubDomainKeyTree *s);

__global__ void calculateNewRangeKernel(unsigned long *keyHistRanges, int *keyHistCounts, int bins, int n,
                                        float *x, float *y, float *z, float *mass, int *count, int *start,
                                        int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                        float *minZ, float *maxZ, SubDomainKeyTree *s);
#endif //CUDA_NBODY_KERNELS_CUH
