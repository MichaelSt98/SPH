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

    float resetArrays(int *mutex, float *x, float *y, float *z, float *mass, int *count, int *start, int *sorted,
                      int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                      float *minZ, float *maxZ, int n, int m, int *procCounter, int *procCounterTemp,
                      bool timing=false);

    float resetArraysParallel(int *domainListIndex, unsigned long *domainListKeys,  int *domainListIndices,
                             int *domainListLevels, int *lowestDomainListIndices, int *lowestDomainListIndex,
                             unsigned long *lowestDomainListKeys, unsigned long *sortedLowestDomainListKeys,
                             float *tempArray, int *to_delete_cell, int *to_delete_leaf, int n, int m,
                             bool timing=false);

    float computeBoundingBox(int *mutex, float *x, float *y, float *z, float *minX, float *maxX, float *minY,
                             float *maxY, float *minZ, float *maxZ, int n, bool timing=false);

    float buildDomainTree(int *domainListIndex, unsigned long *domainListKeys, int *domainListLevels,
                          int *domainListIndices, float *x, float *y, float *z, float *mass, float *minX,
                          float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int *count,
                          int *start, int *child, int *index, int n, int m, bool timing=false);

    float particlesPerProcess(float *x, float *y, float *z, float *mass, int *count, int *start, int *child,
                              int *index, float *minX, float *maxX, float *minY, float *maxY, float *minZ,
                              float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                              int *procCounterTemp, int curveType=0, bool timing=false);

    float markParticlesProcess(float *x, float *y, float *z, float *mass, int *count, int *start, int *child, int *index,
                               float *minX, float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int n,
                               int m, SubDomainKeyTree *s, int *procCounter, int *procCounterTemp, int *sortArray,
                               int curveType=0, bool timing=false);

    float copyArray(float *targetArray, float *sourceArray, int n, bool timing=false);

    float resetFloatArray(float *array, float value, int n, bool timing=false);

    float debug(float *x, float *y, float *z, float *mass, int *count, int *start, int *child, int *index,
                        float *minX, float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int n, int m,
                        SubDomainKeyTree *s, int *procCounter, float *tempArray, int *sortArray, int *sortArrayOut,
                        bool timing=false);

    float buildTree(float *x, float *y, float *z, float *mass, int *count, int *start, int *child, int *index,
                    float *minX, float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int n, int m,
                    bool timing=false);

    float treeInfo(float *x, float *y, float *z, float *mass, int *count, int *start, int *child, int *index,
                   float *minX, float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int n, int m,
                   int *procCounter, SubDomainKeyTree *s, int *sortArray, int *sortArrayOut, bool timing=false);

    float domainListInfo(float *x, float *y, float *z, float *mass, int *child, int *index, int n,
                         int *domainListIndices, int *domainListIndex,
                         int *domainListLevels, int *lowestDomainListIndices,
                         int *lowestDomainListIndex, SubDomainKeyTree *s, bool timing=false);

    float getParticleKey(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY, float *minZ,
                         float *maxZ, unsigned long *key, int maxLevel, int n, SubDomainKeyTree *s, bool timing=false);

    float traverseIterative(float *x, float *y, float *z, float *mass, int *child, int n, int m, SubDomainKeyTree *s,
                            int maxLevel, bool timing=false);

    float createDomainList(SubDomainKeyTree *s, int maxLevel, unsigned long *domainListKeys, int *levels, int *index,
                           int curveType=0, bool timing=false);

    float centreOfMass(float *x, float *y, float *z, float *mass, int *index, int n, bool timing=false);

    float sort(int *count, int *start, int *sorted, int *child, int *index, int n, int m, bool timing=false);

    float computeForces(float *x, float *y, float *z, float *vx, float *vy, float *vz, float *ax, float *ay, float *az,
                        float *mass, int *sorted, int *child, float *minX, float *maxX, float *minY, float *maxY,
                        float *minZ, float *maxZ, int n, int m, float g, SubDomainKeyTree *s,
                        bool timing=false);

    float update(float *x, float *y, float *z, float *vx, float *vy, float *vz, float *ax, float *ay, float *az, int n,
                 float dt, float d, bool timing=false);

    float lowestDomainListNodes(int *domainListIndices, int *domainListIndex, unsigned long *domainListKeys,
                                                int *lowestDomainListIndices, int *lowestDomainListIndex,
                                                unsigned long *lowestDomainListKeys,
                                                float *x, float *y, float *z, float *mass, int *count, int *start,
                                                int *child, int n, int m, int *procCounter, bool timing=false);

    float prepareLowestDomainExchange(float *entry, float *mass, float *tempArray, int *lowestDomainListIndices,
                                           int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                           int *counter, bool timing=false);

    float prepareLowestDomainExchangeMass(float *mass, float *tempArray, int *lowestDomainListIndices,
                                               int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                               int *counter, bool timing=false);

    float updateLowestDomainListNodes(float *tempArray, float *entry, int *lowestDomainListIndices,
                                           int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                           unsigned long *sortedLowestDomainListKeys, int *counter,
                                           bool timing=false);

    float compLowestDomainListNodes(float *x, float *y, float *z, float *mass, int *lowestDomainListIndices,
                                         int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                         unsigned long *sortedLowestDomainListKeys, int *counter,
                                         bool timing=false);

    float zeroDomainListNodes(int *domainListIndex, int *domainListIndices,
                              int *lowestDomainListIndex, int *lowestDomainListIndices,
                              float *x, float *y, float *z,
                              float *mass, bool timing=false);


    float compLocalPseudoParticlesPar(float *x, float *y, float *z, float *mass, int *index, int n,
                                      int *domainListIndices, int *domainListIndex,
                                      int *lowestDomainListIndices, int *lowestDomainListIndex, bool timing=false);

    float compDomainListPseudoParticlesPar(float *x, float *y, float *z, float *mass, int *child, int *index, int n,
                                                int *domainListIndices, int *domainListIndex,
                                                int *domainListLevels, int *lowestDomainListIndices,
                                                int *lowestDomainListIndex, bool timing=false);

    float collectSendIndices(int *sendIndices, float *entry, float *tempArray, int *domainListCounter,
                                             int sendCount, bool timing=false);

    float symbolicForce(int relevantIndex, float *x, float *y, float *z, float *mass, float *minX, float *maxX, float *minY,
                                        float *maxY, float *minZ, float *maxZ, int *child, int *domainListIndex,
                                        unsigned long *domainListKeys, int *domainListIndices, int *domainListLevels,
                                        int *domainListCounter, int *sendIndices, int *index, int *particleCounter,
                                        SubDomainKeyTree *s, int n, int m, float diam, float theta, int *mutex,
                                        int *relevantDomainListIndices, bool timing=false);

    float compTheta(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                                    float *minZ, float *maxZ, int *domainListIndex, int *domainListCounter,
                                    unsigned long *domainListKeys, int *domainListIndices, int *domainListLevels,
                                    int *relevantDomainListIndices, SubDomainKeyTree *s, int curveType=0,
                                    bool timing=false);

    float insertReceivedParticles(float *x, float *y, float *z, float *mass, int *count, int *start,
                                  int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                  float *minZ, float *maxZ, int *to_delete_leaf, int *domainListIndices,
                                  int *domainListIndex, int *lowestDomainListIndices, int *lowestDomainListIndex,
                                  int n, int m, bool timing=false);

    float centreOfMassReceivedParticles(float *x, float *y, float *z, float *mass, int *startIndex, int *endIndex,
                                             int n, bool timing=false);


    float repairTree(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                               float *ax, float *ay, float *az, float *mass, int *count, int *start,
                               int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                               float *minZ, float *maxZ, int *to_delete_cell, int *to_delete_leaf,
                               int *domainListIndices, int n, int m, bool timing=false);

    float findDuplicates(float *array, float *array_2, int length, SubDomainKeyTree *s, int *duplicateCounter, bool timing=false);

    float markDuplicates(int *indices, float *x, float *y, float *z, float *mass, SubDomainKeyTree *s, int *counter,
                         int length, bool timing=false);

    float removeDuplicates(int *indices, int *removedDuplicatesIndices, int *counter, int length,
                                 bool timing=false);

    float createKeyHistRanges(int bins, unsigned long *keyHistRanges, bool timing=false);

    float keyHistCounter(unsigned long *keyHistRanges, int *keyHistCounts, int bins, int n,
                                         float *x, float *y, float *z, float *mass, int *count, int *start,
                                         int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                         float *minZ, float *maxZ, SubDomainKeyTree *s, bool timing=false);

    float calculateNewRange(unsigned long *keyHistRanges, int *keyHistCounts, int bins, int n,
                                            float *x, float *y, float *z, float *mass, int *count, int *start,
                                            int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                            float *minZ, float *maxZ, SubDomainKeyTree *s, bool timing=false);

};

#endif //CUDA_NBODY_KERNELSWRAPPER_H
