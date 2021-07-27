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
#define MAXDEPTH 128
#define MAX_NUM_INTERACTIONS 180

#include "Constants.h"
#include "SubDomainKeyTree.cuh"

#include "cub/cub.cuh"
#include <iostream>
#include <stdio.h>
#include <cuda.h>

#include <thrust/device_vector.h>

// table needed to convert from Lebesgue to Hilbert keys
__device__ const unsigned char DirTable[12][8] =
        { { 8,10, 3, 3, 4, 5, 4, 5}, { 2, 2,11, 9, 4, 5, 4, 5},
          { 7, 6, 7, 6, 8,10, 1, 1}, { 7, 6, 7, 6, 0, 0,11, 9},
          { 0, 8, 1,11, 6, 8, 6,11}, {10, 0, 9, 1,10, 7, 9, 7},
          {10, 4, 9, 4,10, 2, 9, 3}, { 5, 8, 5,11, 2, 8, 3,11},
          { 4, 9, 0, 0, 7, 9, 2, 2}, { 1, 1, 8, 5, 3, 3, 8, 6},
          {11, 5, 0, 0,11, 6, 2, 2}, { 1, 1, 4,10, 3, 3, 7,10} };

// table needed to convert from Lebesgue to Hilbert keys
__device__ const unsigned char HilbertTable[12][8] = { {0,7,3,4,1,6,2,5}, {4,3,7,0,5,2,6,1}, {6,1,5,2,7,0,4,3},
                                                       {2,5,1,6,3,4,0,7}, {0,1,7,6,3,2,4,5}, {6,7,1,0,5,4,2,3},
                                                       {2,3,5,4,1,0,6,7}, {4,5,3,2,7,6,0,1}, {0,3,1,2,7,4,6,5},
                                                       {2,1,3,0,5,6,4,7}, {4,7,5,6,3,0,2,1}, {6,5,7,4,1,2,0,3} };


__global__ void MyKernel(int *array, int arrayCount);

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
                                          int *lowestDomainListLevels,
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
                                      int *domainListLevels, int *lowestDomainListLevels,
                                      float *x, float *y, float *z, float *mass, int *count, int *start,
                                      int *child, int n, int m, int *procCounter);

__global__ void particlesPerProcessKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                    int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                    float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                                    int *procCounterTemp, int curveType=0);

__global__ void markParticlesProcessKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                           int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                           float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                                           int *procCounterTemp, int *sortArray, int curveType=0);

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

__device__ int key2proc(unsigned long k, SubDomainKeyTree *s, int curveType=0);

__device__ unsigned long Lebesgue2Hilbert(unsigned long lebesgue, int maxLevel);

__global__ void traverseIterativeKernel(float *x, float *y, float *z, float *mass, int *child, int n, int m,
                         SubDomainKeyTree *s, int maxLevel);

__global__ void createDomainListKernel(SubDomainKeyTree *s, int maxLevel, unsigned long *domainListKeys, int *levels,
                                       int *index, int curveType=0);

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

__device__ bool isDomainListNode(unsigned long key, int maxLevel, int level, SubDomainKeyTree *s, int curveType=0);

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

__global__ void symbolicForceKernel(int relevantIndex, float *x, float *y, float *z, float *mass, float *minX, float *maxX, float *minY,
                                    float *maxY, float *minZ, float *maxZ, int *child, int *domainListIndex,
                                    unsigned long *domainListKeys, int *domainListIndices, int *domainListLevels,
                                    int *domainListCounter, int *sendIndices, int *index, int *particleCounter,
                                    SubDomainKeyTree *s, int n, int m, float diam, float theta_, int *mutex,
                                    int *relevantDomainListIndices);

__global__ void compThetaKernel(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int *domainListIndex, int *domainListCounter,
                                unsigned long *domainListKeys, int *domainListIndices, int *domainListLevels,
                                int *relevantDomainListIndices, SubDomainKeyTree *s, int curveType=0);

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
                                     float *minZ, float *maxZ, SubDomainKeyTree *s, int curveType=0);

__global__ void calculateNewRangeKernel(unsigned long *keyHistRanges, int *keyHistCounts, int bins, int n,
                                        float *x, float *y, float *z, float *mass, int *count, int *start,
                                        int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                        float *minZ, float *maxZ, SubDomainKeyTree *s);

__global__ void fixedRadiusNNKernel(int *interactions, int *numberOfInteractions, float *x, float *y, float *z, int *child, float *minX, float *maxX,
                              float *minY, float *maxY, float *minZ, float *maxZ, float sml,
                              int numParticlesLocal, int numParticles, int numNodes);

__global__ void sphDebugKernel(int *interactions, int *numberOfInteractions, float *x, float *y, float *z, int *child, float *minX, float *maxX,
                               float *minY, float *maxY, float *minZ, float *maxZ, float sml,
                               int numParticlesLocal, int numParticles, int numNodes);

__global__ void sphParticles2SendKernel(int numParticlesLocal, int numParticles, int numNodes, float radius,
                                        float *x, float *y, float *z,
                                        float *minX, float *maxX, float *minY, float *maxY, float *minZ, float *maxZ,
                                        SubDomainKeyTree *s, int *domainListIndex, unsigned long *domainListKeys,
                                        int *domainListIndices, int *domainListLevels,
                                        int *lowestDomainListIndices, int *lowestDomainListIndex,
                                        unsigned long *lowestDomainListKeys, int *lowestDomainListLevels,
                                        float sml, int maxLevel, int curveType,
                                        int *toSend, int *sendCount, int *alreadyInserted, int insertOffset);

__global__ void collectSendIndicesSPHKernel(int *toSend, int *toSendCollected, int count);

__global__ void collectSendEntriesSPHKernel(float *entry, float *toSend, int *sendIndices, int *sendCount,
                                            int totalSendCount, int insertOffset, SubDomainKeyTree *s);

#endif //CUDA_NBODY_KERNELS_CUH
