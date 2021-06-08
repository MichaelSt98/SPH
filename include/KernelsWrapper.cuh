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

    void resetArraysParallel(int *domainListIndex, unsigned long *domainListKeys,  int *domainListIndices,
                             int *domainListLevels, float *tempArray, int *to_delete_cell, int *to_delete_leaf,
                             int n, int m);

    float computeBoundingBox(int *mutex, float *x, float *y, float *z, float *minX, float *maxX, float *minY,
                             float *maxY, float *minZ, float *maxZ, int n, bool timing=false);

    float buildDomainTree(int *domainListIndex, unsigned long *domainListKeys, int *domainListLevels, int *domainListIndices,
                          int *count, int *start, int *child, int *index, int n, int m, bool timing=false);

    float particlesPerProcess(float *x, float *y, float *z, float *mass, int *count, int *start, int *child,
                              int *index, float *minX, float *maxX, float *minY, float *maxY, float *minZ,
                              float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                              int *procCounterTemp, bool timing=false);

    float sortParticlesProc(float *x, float *y, float *z, float *mass, int *count, int *start, int *child, int *index,
                            float *minX, float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int n,
                            int m, SubDomainKeyTree *s, int *procCounter, int *procCounterTemp, int *sortArray,
                            bool timing=false);

    float copyArray(float *targetArray, float *sourceArray, int n, bool timing=false);

    float resetFloatArray(float *array, float value, int n, bool timing=false);

    float reorderArray(float *array, float *tempArray, SubDomainKeyTree *s, int *procCounter, int *receiveLengths,
                       bool timing=false);

    float sendParticles(float *x, float *y, float *z, float *mass, int *count, int *start, int *child, int *index,
                        float *minX, float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int n, int m,
                        SubDomainKeyTree *s, int *procCounter, float *tempArray, int *sortArray, int *sortArrayOut,
                        bool timing=false);

    float buildTree(float *x, float *y, float *z, float *mass, int *count, int *start, int *child, int *index,
                    float *minX, float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int n, int m,
                    bool timing=false);

    float treeInfo(float *x, float *y, float *z, float *mass, int *count, int *start, int *child, int *index,
                   float *minX, float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int n, int m,
                   int *procCounter, SubDomainKeyTree *s, int *sortArray, int *sortArrayOut, bool timing=false);

    float getParticleKey(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY, float *minZ,
                         float *maxZ, unsigned long *key, int maxLevel, int n, SubDomainKeyTree *s, bool timing=false);

    float traverseIterative(float *x, float *y, float *z, float *mass, int *child, int n, int m, SubDomainKeyTree *s,
                            int maxLevel, bool timing=false);

    void createDomainList(float *x, float *y, float *z, float *mass, int *child, int n, SubDomainKeyTree *s,
                          int maxLevel);

    /*void createDomainList(float *x, float *y, float *z, float *mass, float *minX, float *maxX,
                                                float *minY, float *maxY, float *minZ, float *maxZ, int *child, int n,
                                                SubDomainKeyTree *s, int maxLevel);*/

    float createDomainList(SubDomainKeyTree *s, int maxLevel, unsigned long *domainListKeys, int *levels, int *index,
                           bool timing=false);

    float centreOfMass(float *x, float *y, float *z, float *mass, int *index, int n, bool timing=false);

    float sort(int *count, int *start, int *sorted, int *child, int *index, int n, bool timing=false);

    float computeForces(float *x, float *y, float *z, float *vx, float *vy, float *vz, float *ax, float *ay, float *az,
                        float *mass, int *sorted, int *child, float *minX, float *maxX, int n, float g,
                        bool timing=false);

    float update(float *x, float *y, float *z, float *vx, float *vy, float *vz, float *ax, float *ay, float *az, int n,
                 float dt, float d, bool timing=false);


    void collectSendIndices(int *sendIndices, float *entry, float *tempArray, int *domainListCounter,
                                             int sendCount, timing=false);

};

#endif //CUDA_NBODY_KERNELSWRAPPER_H
