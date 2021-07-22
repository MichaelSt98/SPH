/**
 * Wrapping CUDA Kernel functions.
 */

#include "../include/KernelsWrapper.cuh"

ExecutionPolicy::ExecutionPolicy() : gridSize(256), blockSize(256), sharedMemBytes(0) {};

ExecutionPolicy::ExecutionPolicy(dim3 _gridSize, dim3 _blockSize, size_t _sharedMemBytes)
                : gridSize(_gridSize), blockSize(_blockSize), sharedMemBytes(_sharedMemBytes) {};

ExecutionPolicy::ExecutionPolicy(dim3 _gridSize, dim3 _blockSize)
                : gridSize(_gridSize), blockSize(_blockSize), sharedMemBytes(0) {};

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) getchar();
    }
}

void CheckCudaCall(cudaError_t command, const char * commandName, const char * fileName, int line)
{
    if (command != cudaSuccess)
    {
        fprintf(stderr, "Error: CUDA result \"%s\" for call \"%s\" in file \"%s\" at line %d. Terminating...\n",
                cudaGetErrorString(command), commandName, fileName, line);
        exit(0);
    }
}

/*
dim3 gridSize  = 1024; //2048; //1024; //512;
dim3 blockSize = 256; //256;
 */

KernelsWrapper::KernelsWrapper() {
    gridSize = 0;
    blockSize = 0;
    blockSizeInt = 0;
    blockSizeInt = 0;
    warp = 0;
    stackSize = 0;
}

KernelsWrapper::KernelsWrapper(SimulationParameters p) {
    gridSize = p.gridSize;
    blockSize = p.blockSize;
    blockSizeInt = p.blockSize;
    warp = p.warp;
    stackSize = p.stackSize;
}

float KernelsWrapper::resetArrays(int *mutex, float *x, float *y, float *z, float *mass, int *count, int *start,
                                  int *sorted, int *child, int *index, float *minX, float *maxX, float *minY,
                                  float *maxY, float *minZ, float *maxZ, int n, int m, int *procCounter,
                                  int *procCounterTemp, bool timing) {

    float elapsedTime = 0.f;
    /*if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        resetArraysKernel<<< gridSize, blockSize >>>(mutex, x, y, z, mass, count, start, sorted, child, index,
                minX, maxX, minY, maxY, minZ, maxZ, n, m, procCounter, procCounterTemp);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        resetArraysKernel<<< gridSize, blockSize >>>(mutex, x, y, z, mass, count, start, sorted, child, index,
                                                     minX, maxX, minY, maxY, minZ, maxZ, n, m, procCounter,
                                                     procCounterTemp);
    }*/

    elapsedTime = cudaLaunch(true, ExecutionPolicy(gridSize, blockSize), resetArraysKernel, mutex, x, y, z, mass, count, start, sorted, child, index,
               minX, maxX, minY, maxY, minZ, maxZ, n, m, procCounter,
               procCounterTemp);

    printf("variadic elapsed time: %f\n", elapsedTime);

    return elapsedTime;
}

float KernelsWrapper::resetArraysParallel(int *domainListIndex, unsigned long *domainListKeys, int *domainListIndices,
                                          int *domainListLevels, int *lowestDomainListIndices,
                                          int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                          unsigned long *sortedLowestDomainListKeys, int *lowestDomainListLevels,
                                          float *tempArray,
                                          int *to_delete_cell, int *to_delete_leaf, int n, int m, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        resetArraysParallelKernel<<< gridSize, blockSize >>>(domainListIndex, domainListKeys, domainListIndices,
                                                         domainListLevels, lowestDomainListIndices, lowestDomainListIndex,
                                                         lowestDomainListKeys, sortedLowestDomainListKeys, lowestDomainListLevels,
                                                         tempArray, to_delete_cell, to_delete_leaf, n, m);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        resetArraysParallelKernel<<< gridSize, blockSize >>>(domainListIndex, domainListKeys, domainListIndices,
                                                             domainListLevels, lowestDomainListIndices, lowestDomainListIndex,
                                                             lowestDomainListKeys, sortedLowestDomainListKeys, lowestDomainListLevels,
                                                             tempArray, to_delete_cell, to_delete_leaf, n, m);
    }
    return elapsedTime;
}

float KernelsWrapper::computeBoundingBox(int *mutex, float *x, float *y, float *z, float *minX, float *maxX,
                                         float *minY, float *maxY, float *minZ, float *maxZ, int n, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        computeBoundingBoxKernel<<< gridSize, blockSize, 6*sizeof(float)*blockSizeInt >>>(mutex, x, y, z, minX, maxX, minY, maxY, minZ, maxZ, n, blockSizeInt);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        computeBoundingBoxKernel<<< gridSize, blockSize, 6*sizeof(float)*blockSizeInt >>>(mutex, x, y, z, minX, maxX, minY, maxY, minZ, maxZ, n, blockSizeInt);
    }
    return elapsedTime;
}

float KernelsWrapper::buildDomainTree(int *domainListIndex, unsigned long *domainListKeys, int *domainListLevels,
                                      int *domainListIndices, float *x, float *y, float *z, float *mass, float *minX,
                                      float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int *count,
                                      int *start, int *child, int *index, int n, int m, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        buildDomainTreeKernel<<< 1, 1 >>>(domainListIndex, domainListKeys, domainListLevels, domainListIndices, x, y, z,
                                      mass, minX, maxX, minY, maxY, minZ, maxZ, count, start, child, index, n, m);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        buildDomainTreeKernel<<< 1, 1 >>>(domainListIndex, domainListKeys, domainListLevels, domainListIndices, x, y, z,
                                          mass, minX, maxX, minY, maxY, minZ, maxZ, count, start, child, index, n, m);
    }
    return elapsedTime;
}

float KernelsWrapper::treeInfo(float *x, float *y, float *z, float *mass, int *count, int *start, int *child,
                               int *index, float *minX, float *maxX, float *minY, float *maxY, float *minZ,
                               float *maxZ, int n, int m, int *procCounter, SubDomainKeyTree *s, int *sortArray,
                               int *sortArrayOut, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        treeInfoKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                        minX, maxX, minY, maxY, minZ, maxZ, n, m, procCounter, s, sortArray,
                                        sortArrayOut);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        treeInfoKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                                  minX, maxX, minY, maxY, minZ, maxZ, n, m, procCounter, s, sortArray,
                                                  sortArrayOut);
    }
    return elapsedTime;
}

float KernelsWrapper::domainListInfo(float *x, float *y, float *z, float *mass, int *child, int *index, int n,
                                           int *domainListIndices, int *domainListIndex,
                                           int *domainListLevels, int *lowestDomainListIndices,
                                           int *lowestDomainListIndex, SubDomainKeyTree *s, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        domainListInfoKernel<<< gridSize, blockSize >>>(x, y, z, mass, child, index, n, domainListIndices, domainListIndex, domainListLevels,
                             lowestDomainListIndices, lowestDomainListIndex, s);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        domainListInfoKernel<<<gridSize, blockSize>>>(x, y, z, mass, child, index, n, domainListIndices, domainListIndex, domainListLevels,
                             lowestDomainListIndices, lowestDomainListIndex, s);
    }
    return elapsedTime;
}

float KernelsWrapper::particlesPerProcess(float *x, float *y, float *z, float *mass, int *count, int *start, int *child,
                                          int *index, float *minX, float *maxX, float *minY, float *maxY, float *minZ,
                                          float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                                          int *procCounterTemp, int curveType, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        particlesPerProcessKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                                   minX, maxX, minY, maxY, minZ, maxZ, n, m, s, procCounter,
                                                   procCounterTemp, curveType);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        particlesPerProcessKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                                             minX, maxX, minY, maxY, minZ, maxZ, n, m, s, procCounter,
                                                             procCounterTemp, curveType);
    }
    return elapsedTime;
}

float KernelsWrapper::markParticlesProcess(float *x, float *y, float *z, float *mass, int *count, int *start, int *child,
                                           int *index, float *minX, float *maxX, float *minY, float *maxY, float *minZ,
                                           float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                                           int *procCounterTemp, int *sortArray, int curveType, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        markParticlesProcessKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                                              minX, maxX, minY, maxY, minZ, maxZ, n, m, s,
                                                              procCounter, procCounterTemp, sortArray,
                                                              curveType);
        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        markParticlesProcessKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                                              minX, maxX, minY, maxY, minZ, maxZ, n, m, s,
                                                              procCounter, procCounterTemp, sortArray,
                                                              curveType);
    }
    return elapsedTime;
}

float KernelsWrapper::copyArray(float *targetArray, float *sourceArray, int n, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        copyArrayKernel<<<gridSize, blockSize>>>(targetArray, sourceArray, n);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        copyArrayKernel<<<gridSize, blockSize>>>(targetArray, sourceArray, n);
    }
    return elapsedTime;
}

float KernelsWrapper::resetFloatArray(float *array, float value, int n, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        resetFloatArrayKernel<<<gridSize, blockSize>>>(array, value, n);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        resetFloatArrayKernel<<<gridSize, blockSize>>>(array, value, n);
    }
    return elapsedTime;
}

float KernelsWrapper::debug(float *x, float *y, float *z, float *mass, int *count, int *start, int *child,
                                    int *index, float *minX, float *maxX, float *minY, float *maxY, float *minZ,
                                    float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter, float *tempArray,
                                    int *sortArray, int *sortArrayOut, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        debugKernel<<< 1, 1/*gridSize, blockSize*/ >>>(x, y, z, mass, count, start, child, index,
                                                   minX, maxX, minY, maxY, minZ, maxZ, n, m, s, procCounter,
                                                   tempArray, sortArray, sortArrayOut);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        debugKernel<<< 1, 1/*gridSize, blockSize*/ >>>(x, y, z, mass, count, start, child, index,
                                                               minX, maxX, minY, maxY, minZ, maxZ, n, m, s, procCounter,
                                                               tempArray, sortArray, sortArrayOut);
    }
    return elapsedTime;
}

float KernelsWrapper::buildTree(float *x, float *y, float *z, float *mass, int *count, int *start, int *child,
                                int *index, float *minX, float *maxX, float *minY, float *maxY, float *minZ,
                                float *maxZ, int n, int m, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        buildTreeKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                minX, maxX, minY, maxY, minZ, maxZ, n, m);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        buildTreeKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                                   minX, maxX, minY, maxY, minZ, maxZ, n, m);
    }
    return elapsedTime;
}

float KernelsWrapper::getParticleKey(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                                     float *minZ, float *maxZ, unsigned long *key, int maxLevel, int n,
                                     SubDomainKeyTree *s, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        getParticleKeyKernel<<< gridSize, blockSize >>>(x, y, z, minX, maxX, minY, maxY, minZ, maxZ, key, 21, n, s);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        getParticleKeyKernel<<< gridSize, blockSize >>>(x, y, z, minX, maxX, minY, maxY, minZ, maxZ, key, 21, n, s);
    }
    return elapsedTime;
}

float KernelsWrapper::traverseIterative(float *x, float *y, float *z, float *mass, int *child, int n, int m,
                                        SubDomainKeyTree *s, int maxLevel, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        traverseIterativeKernel<<< 1, 1 >>>(x, y, z, mass, child, n, m, s, maxLevel);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        traverseIterativeKernel<<< 1, 1 >>>(x, y, z, mass, child, n, m, s, maxLevel);
    }
    return elapsedTime;
}

float KernelsWrapper::createDomainList(SubDomainKeyTree *s, int maxLevel, unsigned long *domainListKeys, int *levels,
                                       int *index, int curveType, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        createDomainListKernel<<<1, 1>>>(s, maxLevel, domainListKeys, levels, index, curveType);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        createDomainListKernel<<<1, 1>>>(s, maxLevel, domainListKeys, levels, index, curveType);
    }
    return elapsedTime;
}

float KernelsWrapper::centreOfMass(float *x, float *y, float *z, float *mass, int *index, int n, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        centreOfMassKernel<<< gridSize, blockSize >>>(x, y, z, mass, index, n);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        centreOfMassKernel<<< gridSize, blockSize >>>(x, y, z, mass, index, n);
    }
    return elapsedTime;
}

float KernelsWrapper::sort(int *count, int *start, int *sorted, int *child, int *index, int n, int m, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        sortKernel<<< gridSize, blockSize>>>(count, start, sorted, child, index, n, m);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        sortKernel<<< gridSize, blockSize>>>(count, start, sorted, child, index, n, m);
    }
    return elapsedTime;
}

float KernelsWrapper::computeForces(float *x, float *y, float *z, float *vx, float *vy, float *vz, float *ax, float *ay,
                                    float *az, float *mass, int *sorted, int *child, float *minX, float *maxX, float *minY, float *maxY,
                                    float *minZ, float *maxZ, int n,
                                    int m, float g, SubDomainKeyTree *s, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        computeForcesKernel<<<gridSize, blockSize, (sizeof(float)+sizeof(int))*stackSize*blockSizeInt/warp>>>(x, y, z, vx, vy, vz, ax, ay, az,
                mass, sorted, child, minX, maxX, minY, maxY, minZ, maxZ, n, m, g, blockSizeInt, warp, stackSize, s);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        computeForcesKernel<<<gridSize, blockSize, (sizeof(float)+sizeof(int))*stackSize*blockSizeInt/warp>>>(x, y, z, vx, vy, vz, ax, ay, az,
                                                     mass, sorted, child, minX, maxX, minY, maxY, minZ, maxZ, n, m, g, blockSizeInt, warp, stackSize, s);
    }
    return elapsedTime;
}

float KernelsWrapper::update(float *x, float *y, float *z, float *vx, float *vy, float *vz, float *ax, float *ay,
                             float *az, int n, float dt, float d, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        updateKernel<<< gridSize, blockSize >>>(x, y, z, vx, vy, vz, ax, ay, az, n, dt, d);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        updateKernel<<< gridSize, blockSize >>>(x, y, z, vx, vy, vz, ax, ay, az, n, dt, d);
    }
    return elapsedTime;
}

float KernelsWrapper::lowestDomainListNodes(int *domainListIndices, int *domainListIndex, unsigned long *domainListKeys,
                                            int *lowestDomainListIndices, int *lowestDomainListIndex,
                                            unsigned long *lowestDomainListKeys, int *domainListLevels, int *lowestDomainListLevels,
                                            float *x, float *y, float *z,
                                            float *mass, int *count, int *start, int *child, int n, int m,
                                            int *procCounter, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        lowestDomainListNodesKernel<<<gridSize, blockSize>>>(domainListIndices, domainListIndex, domainListKeys,
                                                             lowestDomainListIndices, lowestDomainListIndex,
                                                             lowestDomainListKeys, domainListLevels, lowestDomainListLevels,
                                                             x, y, z, mass, count, start, child,
                                                             n, m, procCounter);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        lowestDomainListNodesKernel<<<gridSize, blockSize>>>(domainListIndices, domainListIndex, domainListKeys,
                                                             lowestDomainListIndices, lowestDomainListIndex,
                                                             lowestDomainListKeys, domainListLevels, lowestDomainListLevels,
                                                             x, y, z, mass, count, start, child,
                                                             n, m, procCounter);
    }
    return elapsedTime;
}

float KernelsWrapper::prepareLowestDomainExchange(float *entry, float *mass, float *tempArray,
                                                  int *lowestDomainListIndices, int *lowestDomainListIndex,
                                                  unsigned long *lowestDomainListKeys, int *counter, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        prepareLowestDomainExchangeKernel<<< gridSize, blockSize >>> (entry, mass, tempArray, lowestDomainListIndices,
                                                                      lowestDomainListIndex, lowestDomainListKeys,
                                                                      counter);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        prepareLowestDomainExchangeKernel<<< gridSize, blockSize >>> (entry, mass, tempArray, lowestDomainListIndices,
                                                                      lowestDomainListIndex, lowestDomainListKeys,
                                                                      counter);
    }
    return elapsedTime;
}

float KernelsWrapper::prepareLowestDomainExchangeMass(float *mass, float *tempArray, int *lowestDomainListIndices,
                                                      int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                                      int *counter, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        prepareLowestDomainExchangeMassKernel<<< gridSize, blockSize >>> (mass, tempArray, lowestDomainListIndices,
                                                                          lowestDomainListIndex, lowestDomainListKeys,
                                                                          counter);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        prepareLowestDomainExchangeMassKernel<<< gridSize, blockSize >>> (mass, tempArray, lowestDomainListIndices,
                                                                          lowestDomainListIndex, lowestDomainListKeys,
                                                                          counter);
    }
    return elapsedTime;
}

float KernelsWrapper::updateLowestDomainListNodes(float *tempArray, float *entry, int *lowestDomainListIndices,
                                                  int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                                  unsigned long *sortedLowestDomainListKeys, int *counter,
                                                  bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        updateLowestDomainListNodesKernel<<< gridSize, blockSize>>>(tempArray, entry, lowestDomainListIndices,
                                                lowestDomainListIndex, lowestDomainListKeys, sortedLowestDomainListKeys,
                                                counter);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        updateLowestDomainListNodesKernel<<< gridSize, blockSize>>>(tempArray, entry, lowestDomainListIndices,
                                                                    lowestDomainListIndex, lowestDomainListKeys,
                                                                    sortedLowestDomainListKeys,
                                                                    counter);
    }
    return elapsedTime;
}

float KernelsWrapper::compLowestDomainListNodes(float *x, float *y, float *z, float *mass, int *lowestDomainListIndices,
                                                int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                                unsigned long *sortedLowestDomainListKeys, int *counter, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        compLowestDomainListNodesKernel<<< gridSize, blockSize >>>(x, y, z, mass, lowestDomainListIndices,
                                                                    lowestDomainListIndex, lowestDomainListKeys,
                                                                    sortedLowestDomainListKeys, counter);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        compLowestDomainListNodesKernel<<< gridSize, blockSize >>>(x, y, z, mass, lowestDomainListIndices,
                                                                   lowestDomainListIndex, lowestDomainListKeys,
                                                                   sortedLowestDomainListKeys, counter);
    }
    return elapsedTime;
}

float KernelsWrapper::zeroDomainListNodes(int *domainListIndex, int *domainListIndices, int *lowestDomainListIndex,
                                          int *lowestDomainListIndices, float *x, float *y, float *z, float *mass,
                                          bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        zeroDomainListNodesKernel<<< gridSize, blockSize >>>(domainListIndex, domainListIndices, lowestDomainListIndex,
                                                             lowestDomainListIndices, x, y, z, mass);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        zeroDomainListNodesKernel<<< gridSize, blockSize >>>(domainListIndex, domainListIndices, lowestDomainListIndex,
                                                             lowestDomainListIndices, x, y, z, mass);
    }
    return elapsedTime;
}

float KernelsWrapper::compLocalPseudoParticlesPar(float *x, float *y, float *z, float *mass, int *index, int n,
                                                  int *domainListIndices, int *domainListIndex,
                                                  int *lowestDomainListIndices, int *lowestDomainListIndex,
                                                  bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        compLocalPseudoParticlesParKernel<<<gridSize, blockSize>>>(x, y, z, mass, index, n, domainListIndices,
                                                                   domainListIndex, lowestDomainListIndices,
                                                                   lowestDomainListIndex);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        compLocalPseudoParticlesParKernel<<<gridSize, blockSize>>>(x, y, z, mass, index, n, domainListIndices,
                                                                   domainListIndex, lowestDomainListIndices,
                                                                   lowestDomainListIndex);
    }
    return elapsedTime;
}

float KernelsWrapper::compDomainListPseudoParticlesPar(float *x, float *y, float *z, float *mass, int *child,
                                                       int *index, int n, int *domainListIndices, int *domainListIndex,
                                                       int *domainListLevels, int *lowestDomainListIndices,
                                                       int *lowestDomainListIndex, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        compDomainListPseudoParticlesParKernel<<<1, 256>>>(x, y, z, mass, child, index, n, domainListIndices,
                                                                domainListIndex, domainListLevels,
                                                                lowestDomainListIndices, lowestDomainListIndex);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        compDomainListPseudoParticlesParKernel<<<gridSize, 1>>>(x, y, z, mass, child, index, n, domainListIndices,
                                                                domainListIndex, domainListLevels,
                                                                lowestDomainListIndices, lowestDomainListIndex);
    }
    return elapsedTime;
}

float KernelsWrapper::collectSendIndices(int *sendIndices, float *entry, float *tempArray, int *domainListCounter,
                                         int sendCount, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        collectSendIndicesKernel<<< gridSize, blockSize >>>(sendIndices, entry, tempArray, domainListCounter,
                                                            sendCount);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        collectSendIndicesKernel<<< gridSize, blockSize >>>(sendIndices, entry, tempArray, domainListCounter,
                                                            sendCount);
    }
    return elapsedTime;
}

float KernelsWrapper::symbolicForce(int relevantIndex, float *x, float *y, float *z, float *mass, float *minX,
                                    float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int *child,
                                    int *domainListIndex, unsigned long *domainListKeys, int *domainListIndices,
                                    int *domainListLevels, int *domainListCounter, int *sendIndices, int *index,
                                    int *particleCounter,
                                    SubDomainKeyTree *s, int n, int m, float diam, float theta, int *mutex,
                                    int *relevantDomainListIndices, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        symbolicForceKernel<<< gridSize, blockSize >>>(relevantIndex, x, y, z, mass, minX, maxX, minY, maxY, minZ, maxZ, child,
                                                     domainListIndex, domainListKeys, domainListIndices, domainListLevels,
                                                     domainListCounter, sendIndices, index, particleCounter, s, n, m,
                                                     diam, theta, mutex, relevantDomainListIndices);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        symbolicForceKernel<<< gridSize, blockSize >>>(relevantIndex, x, y, z, mass, minX, maxX, minY, maxY, minZ, maxZ, child,
                                                     domainListIndex, domainListKeys, domainListIndices, domainListLevels,
                                                     domainListCounter, sendIndices, index, particleCounter, s, n, m,
                                                     diam, theta, mutex, relevantDomainListIndices);
    }
    return elapsedTime;
}

float KernelsWrapper::compTheta(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int *domainListIndex, int *domainListCounter,
                                unsigned long *domainListKeys, int *domainListIndices, int *domainListLevels,
                                int *relevantDomainListIndices, SubDomainKeyTree *s, int curveType, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        compThetaKernel<<<gridSize, blockSize>>>(x, y, z, minX, maxX, minY, maxY, minZ, maxZ, domainListIndex,
                                                 domainListCounter, domainListKeys, domainListIndices, domainListLevels,
                                                 relevantDomainListIndices, s, curveType);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        compThetaKernel<<<gridSize, blockSize>>>(x, y, z, minX, maxX, minY, maxY, minZ, maxZ, domainListIndex,
                                                 domainListCounter, domainListKeys, domainListIndices, domainListLevels,
                                                 relevantDomainListIndices, s, curveType);
    }
    return elapsedTime;
}

float KernelsWrapper::insertReceivedParticles(float *x, float *y, float *z, float *mass, int *count, int *start,
                                              int *child, int *index, float *minX, float *maxX, float *minY,
                                              float *maxY, float *minZ, float *maxZ, int *to_delete_leaf,
                                              int *domainListIndices, int *domainListIndex,
                                              int *lowestDomainListIndices, int *lowestDomainListIndex,
                                              int n, int m, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        insertReceivedParticlesKernel<<<gridSize, blockSize>>>(x, y, z, mass, count, start, child, index, minX, maxX,
                                                               minY, maxY, minZ, maxZ, to_delete_leaf,
                                                               domainListIndices, domainListIndex,
                                                               lowestDomainListIndices, lowestDomainListIndex, n, m);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        insertReceivedParticlesKernel<<<gridSize, blockSize>>>(x, y, z, mass, count, start, child, index, minX, maxX,
                                                               minY, maxY, minZ, maxZ, to_delete_leaf,
                                                               domainListIndices, domainListIndex,
                                                               lowestDomainListIndices, lowestDomainListIndex, n, m);
    }
    return elapsedTime;
}


float KernelsWrapper::centreOfMassReceivedParticles(float *x, float *y, float *z, float *mass, int *startIndex, int *endIndex,
                                         int n, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        centreOfMassReceivedParticlesKernel<<<gridSize, blockSize>>>(x, y, z, mass, startIndex, endIndex, n);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        centreOfMassReceivedParticlesKernel<<<gridSize, blockSize>>>(x, y, z, mass, startIndex, endIndex, n);
    }
    return elapsedTime;

}


float KernelsWrapper::repairTree(float *x, float *y, float *z, float *vx, float *vy, float *vz, float *ax, float *ay,
                                 float *az, float *mass, int *count, int *start, int *child, int *index, float *minX,
                                 float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int *to_delete_cell,
                                 int *to_delete_leaf, int *domainListIndices, int n, int m, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        repairTreeKernel<<<gridSize, blockSize>>>(x, y, z, vx, vy, vz, ax, ay, az, mass, count, start, child, index,
                                                  minX, maxX, minY, maxY, minZ, maxZ, to_delete_cell, to_delete_leaf,
                                                  domainListIndices, n, m);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        repairTreeKernel<<<gridSize, blockSize>>>(x, y, z, vx, vy, vz, ax, ay, az, mass, count, start, child, index,
                                                  minX, maxX, minY, maxY, minZ, maxZ, to_delete_cell, to_delete_leaf,
                                                  domainListIndices, n, m);
    }
    return elapsedTime;
}

float KernelsWrapper::findDuplicates(float *array, float *array_2, int length, SubDomainKeyTree *s,
                                     int *duplicateCounter, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        findDuplicatesKernel<<<gridSize, blockSize>>>(array, array_2, length, s, duplicateCounter);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        findDuplicatesKernel<<<gridSize, blockSize>>>(array, array_2, length, s, duplicateCounter);
    }
    return elapsedTime;
}

float KernelsWrapper::markDuplicates(int *indices, float *x, float *y, float *z, float *mass, SubDomainKeyTree *s,
                                     int *counter, int length, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        markDuplicatesKernel<<<gridSize, blockSize>>>(indices, x, y, z, mass, s, counter, length);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        markDuplicatesKernel<<<gridSize, blockSize>>>(indices, x, y, z, mass, s, counter, length);
    }
    return elapsedTime;
}

float KernelsWrapper::removeDuplicates(int *indices, int *removedDuplicatesIndices, int *counter, int length,
                                       bool timing) {
    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        removeDuplicatesKernel<<<gridSize, blockSize>>>(indices, removedDuplicatesIndices, counter, length);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        removeDuplicatesKernel<<<gridSize, blockSize>>>(indices, removedDuplicatesIndices, counter, length);
    }
    return elapsedTime;
}

float KernelsWrapper::createKeyHistRanges(int bins, unsigned long *keyHistRanges, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        createKeyHistRangesKernel<<<gridSize, blockSize>>>(bins, keyHistRanges);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    } else {
        createKeyHistRangesKernel<<<gridSize, blockSize>>>(bins, keyHistRanges);
    }
    return elapsedTime;

}

float KernelsWrapper::keyHistCounter(unsigned long *keyHistRanges, int *keyHistCounts, int bins, int n,
                     float *x, float *y, float *z, float *mass, int *count, int *start,
                     int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                     float *minZ, float *maxZ, SubDomainKeyTree *s, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        keyHistCounterKernel<<<gridSize, blockSize>>>(keyHistRanges, keyHistCounts, bins, n, x, y, z, mass, count,
                                                      start, child, index, minX, maxX, minY, maxY, minZ, maxZ, s);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    } else {
        keyHistCounterKernel<<<gridSize, blockSize>>>(keyHistRanges, keyHistCounts, bins, n, x, y, z, mass, count,
                                                      start, child, index, minX, maxX, minY, maxY, minZ, maxZ, s);
    }
    return elapsedTime;
}

float KernelsWrapper::calculateNewRange(unsigned long *keyHistRanges, int *keyHistCounts, int bins, int n,
                        float *x, float *y, float *z, float *mass, int *count, int *start,
                        int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                        float *minZ, float *maxZ, SubDomainKeyTree *s, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        calculateNewRangeKernel<<<gridSize, blockSize>>>(keyHistRanges, keyHistCounts, bins, n, x, y, z, mass, count,
                                                        start, child, index, minX, maxX, minY, maxY, minZ, maxZ, s);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        calculateNewRangeKernel<<<gridSize, blockSize>>>(keyHistRanges, keyHistCounts, bins, n, x, y, z, mass, count,
                                                         start, child, index, minX, maxX, minY, maxY, minZ, maxZ, s);
    }
    return elapsedTime;

}


float KernelsWrapper::fixedRadiusNN(int *interactions, int *numberOfInteractions, float *x, float *y, float *z, int *child, float *minX, float *maxX,
                     float *minY, float *maxY, float *minZ, float *maxZ, float sml,
                     int numParticlesLocal, int numParticles, int numNodes, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //Kernel call
        fixedRadiusNNKernel<<<gridSize, blockSize>>>(interactions, numberOfInteractions, x, y, z, child, minX, maxX, minY, maxY, minZ, maxZ,
                                                     sml, numParticlesLocal, numParticles, numNodes);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        //Kernel call
        fixedRadiusNNKernel<<<gridSize, blockSize>>>(interactions, numberOfInteractions, x, y, z, child, minX, maxX, minY, maxY, minZ, maxZ,
                                                     sml, numParticlesLocal, numParticles, numNodes);

    }
    return elapsedTime;

}


float KernelsWrapper::sphDebug(int *interactions, int *numberOfInteractions, float *x, float *y, float *z, int *child, float *minX, float *maxX,
               float *minY, float *maxY, float *minZ, float *maxZ, float sml,
               int numParticlesLocal, int numParticles, int numNodes, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //Kernel call
        sphDebugKernel<<<gridSize, blockSize>>>(interactions, numberOfInteractions, x, y, z, child, minX, maxX, minY, maxY, minZ, maxZ,
                                                     sml, numParticlesLocal, numParticles, numNodes);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        //Kernel call
        sphDebugKernel<<<gridSize, blockSize>>>(interactions, numberOfInteractions, x, y, z, child, minX, maxX, minY, maxY, minZ, maxZ,
                                                     sml, numParticlesLocal, numParticles, numNodes);

    }
    return elapsedTime;

}


/*float KernelsWrapper::sphParticles2Send(int numParticlesLocal, int numParticles, int numNodes, float radius,
                                        float *x, float *y, float *z,
                                        float *minX, float *maxX, float *minY, float *maxY, float *minZ, float *maxZ,
                        SubDomainKeyTree *s, int *domainListIndex, unsigned long *domainListKeys,
                        int *domainListIndices, int *domainListLevels,
                        int *lowestDomainListIndices, int *lowestDomainListIndex,
                        unsigned long *lowestDomainListKeys, int *lowestDomainListLevels, float sml, int maxLevel, int curveType,
                        int *toSend, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //Kernel call
        sphParticles2SendKernel<<<gridSize, blockSize>>>(numParticlesLocal, numParticles, numNodes, radius,
                                x, y, z, minX, maxX, minY, maxY, minZ, maxZ, s, domainListIndex, domainListKeys,
                                domainListIndices, domainListLevels, lowestDomainListIndices, lowestDomainListIndex,
                                lowestDomainListKeys, lowestDomainListLevels,
                                sml, maxLevel, curveType, toSend);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        //Kernel call
        sphParticles2SendKernel<<<gridSize, blockSize>>>(numParticlesLocal, numParticles, numNodes, radius,
                                                         x, y, z, minX, maxX, minY, maxY, minZ, maxZ, s, domainListIndex, domainListKeys,
                                                         domainListIndices, domainListLevels, lowestDomainListIndices, lowestDomainListIndex,
                                                         lowestDomainListKeys, lowestDomainListLevels,
                                                         sml, maxLevel, curveType, toSend);
    }
    return elapsedTime;

}*/

float KernelsWrapper::sphParticles2Send(int numParticlesLocal, int numParticles, int numNodes, float radius,
                                        float *x, float *y, float *z,
                                        float *minX, float *maxX, float *minY, float *maxY, float *minZ, float *maxZ,
                                        SubDomainKeyTree *s, int *domainListIndex, unsigned long *domainListKeys,
                                        int *domainListIndices, int *domainListLevels,
                                        int *lowestDomainListIndices, int *lowestDomainListIndex,
                                        unsigned long *lowestDomainListKeys, int *lowestDomainListLevels, float sml, int maxLevel, int curveType,
                                        int *toSend, int *sendCount, int *alreadyInserted, int insertOffset, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //Kernel call
        sphParticles2SendKernel<<<gridSize, blockSize>>>(numParticlesLocal, numParticles, numNodes, radius,
                                                         x, y, z, minX, maxX, minY, maxY, minZ, maxZ, s, domainListIndex, domainListKeys,
                                                         domainListIndices, domainListLevels, lowestDomainListIndices, lowestDomainListIndex,
                                                         lowestDomainListKeys, lowestDomainListLevels,
                                                         sml, maxLevel, curveType, toSend, sendCount,
                                                         alreadyInserted, insertOffset);

        gpuErrorcheck( cudaPeekAtLastError() );
        gpuErrorcheck( cudaDeviceSynchronize() );

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        //Kernel call
        sphParticles2SendKernel<<<gridSize, blockSize>>>(numParticlesLocal, numParticles, numNodes, radius,
                                                         x, y, z, minX, maxX, minY, maxY, minZ, maxZ, s, domainListIndex, domainListKeys,
                                                         domainListIndices, domainListLevels, lowestDomainListIndices, lowestDomainListIndex,
                                                         lowestDomainListKeys, lowestDomainListLevels,
                                                         sml, maxLevel, curveType, toSend, sendCount,
                                                         alreadyInserted, insertOffset);
    }
    return elapsedTime;

}

float KernelsWrapper::collectSendIndicesSPH(int *toSend, int *toSendCollected, int count, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //Kernel call
        collectSendIndicesSPHKernel<<<gridSize, blockSize>>>(toSend, toSendCollected, count);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        //Kernel call
        collectSendIndicesSPHKernel<<<gridSize, blockSize>>>(toSend, toSendCollected, count);
    }
    return elapsedTime;

}

float KernelsWrapper::collectSendEntriesSPH(float *entry, float *toSend, int *sendIndices, int *sendCount, int totalSendCount,
                            int insertOffset, SubDomainKeyTree *s, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t;
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        gpuErrorcheck( cudaPeekAtLastError() );
        gpuErrorcheck( cudaDeviceSynchronize() );

        //Kernel call
        collectSendEntriesSPHKernel<<<gridSize, blockSize>>>(entry, toSend, sendIndices, sendCount, totalSendCount,
                                                             insertOffset, s);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        //Kernel call
        collectSendEntriesSPHKernel<<<gridSize, blockSize>>>(entry, toSend, sendIndices, sendCount, totalSendCount,
                                                             insertOffset, s);

        gpuErrorcheck( cudaPeekAtLastError() );
        gpuErrorcheck( cudaDeviceSynchronize() );

    }
    return elapsedTime;

}
