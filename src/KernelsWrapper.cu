/**
 * Wrapping CUDA Kernel functions.
 */

#include "../include/KernelsWrapper.cuh"

/*
dim3 gridSize  = 1024; //2048; //1024; //512;
dim3 blockSize = 256; //256;
 */

KernelsWrapper::KernelsWrapper() {
    gridSize = 0;
    blockSize = 0;
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

float KernelsWrapper::resetArrays(int *mutex, float *x, float *y, float *z, float *mass, int *count,
                          int *start, int *sorted, int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                          float *minZ, float *maxZ, int n, int m, int *procCounter, int *procCounterTemp, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
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
    }
    return elapsedTime;

}

void KernelsWrapper::resetArraysParallel(int *domainListIndex, unsigned long *domainListKeys,
                                         unsigned long *domainListIndices, int *domainListLevels,
                                         float *tempArray, int n, int m) {
    resetArraysParallelKernel<<< gridSize, blockSize >>>(domainListIndex, domainListKeys, domainListIndices,
                                                         domainListLevels, tempArray, n, m);
}

float KernelsWrapper::computeBoundingBox(int *mutex, float *x, float *y, float *z, float *minX,
                                 float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int n, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
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

void KernelsWrapper::buildDomainTree(int *domainListIndex, unsigned long *domainListKeys, int *domainListLevels,
                     int *count, int *start, int *child, int *index, int n, int m) {

    buildDomainTreeKernel<<< 1, 1 >>>(domainListIndex, domainListKeys, domainListLevels, count, start, child, index, n, m);

}

void KernelsWrapper::treeInfo(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int n, int m, int *procCounter) {

    treeInfoKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                        minX, maxX, minY, maxY, minZ, maxZ, n, m, procCounter);

}

void KernelsWrapper::particlesPerProcess(float *x, float *y, float *z, float *mass, int *count, int *start,
                                   int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                   float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                                   int *procCounterTemp) {

    particlesPerProcessKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                                   minX, maxX, minY, maxY, minZ, maxZ, n, m, s, procCounter,
                                                   procCounterTemp);

}

void KernelsWrapper::sortParticlesProc(float *x, float *y, float *z, float *mass, int *count, int *start,
                       int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                       float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                       int *procCounterTemp, int *sortArray) {

    sortParticlesProcKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                                         minX, maxX, minY, maxY, minZ, maxZ, n, m, s,
                                                         procCounter, procCounterTemp, sortArray);

}

void KernelsWrapper::sendParticles(float *x, float *y, float *z, float *mass, int *count, int *start,
                   int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                   float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                   float *tempArray, int *sortArray, int *sortArrayOut) {

    float elapsedTime = 0.f;
    cudaEvent_t start_t, stop_t; // used for timing
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);
    cudaEventRecord(start_t, 0);

    sendParticlesKernel<<< 1, 1/*gridSize, blockSize*/ >>>(x, y, z, mass, count, start, child, index,
                                                   minX, maxX, minY, maxY, minZ, maxZ, n, m, s, procCounter,
                                                   tempArray, sortArray, sortArrayOut);

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
    cudaEventDestroy(start_t);
    cudaEventDestroy(stop_t);

    printf("Elapsed time for sorting: %f\n", elapsedTime);

}

float KernelsWrapper::buildTree(float *x, float *y, float *z, float *mass, int *count, int *start,
                        int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                        float *minZ, float *maxZ, int n, int m, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
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

void KernelsWrapper::getParticleKey(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                    float *minZ, float *maxZ, unsigned long *key, int maxLevel, int n, SubDomainKeyTree *s) {

    getParticleKeyKernel<<< gridSize, blockSize >>>(x, y, z, minX, maxX, minY, maxY, minZ, maxZ, 0UL, 21, n, s);

}

void KernelsWrapper::traverseIterative(float *x, float *y, float *z, float *mass, int *child, int n, int m,
                       SubDomainKeyTree *s, int maxLevel) {

    traverseIterativeKernel<<< 1, 1 >>>(x, y, z, mass, child, n, m, s, maxLevel);

}

void KernelsWrapper::createDomainList(float *x, float *y, float *z, float *mass, int *child, int n,
                                      SubDomainKeyTree *s, int maxLevel) {
    //createDomainListKernel<<< gridSize, blockSize >>>(x, y, z, mass, child, n, s, maxLevel);
};

/*void KernelsWrapper::createDomainList(float *x, float *y, float *z, float *mass, float *minX, float *maxX,
                                            float *minY, float *maxY, float *minZ, float *maxZ, int *child, int n,
                                            SubDomainKeyTree *s, int maxLevel) {
    createDomainListKernel<<< 1, 1 >>>(x, y, z, mass, minX, maxX, minY, maxY, minZ, maxZ,
                                                      child, n, s, maxLevel);
};*/

void KernelsWrapper::createDomainList(SubDomainKeyTree *s, int maxLevel, unsigned long *domainListKeys, int *levels,
                                      int *index) {
    createDomainListKernel<<<1, 1>>>(s, maxLevel, domainListKeys, levels, index);
}

float KernelsWrapper::centreOfMass(float *x, float *y, float *z, float *mass, int *index, int n, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
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

float KernelsWrapper::sort(int *count, int *start, int *sorted, int *child, int *index, int n, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        sortKernel<<< gridSize, blockSize>>>(count, start, sorted, child, index, n);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        sortKernel<<< gridSize, blockSize>>>(count, start, sorted, child, index, n);
    }
    return elapsedTime;

}

float KernelsWrapper::computeForces(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                            float *ax, float *ay, float *az, float *mass, int *sorted, int *child,
                            float *minX, float *maxX, int n, float g, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        computeForcesKernel<<<gridSize, blockSize, (sizeof(float)+sizeof(int))*stackSize*blockSizeInt/warp>>>(x, y, z, vx, vy, vz, ax, ay, az,
                mass, sorted, child, minX, maxX, n, g, blockSizeInt, warp, stackSize);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        computeForcesKernel<<<gridSize, blockSize, (sizeof(float)+sizeof(int))*stackSize*blockSizeInt/warp>>>(x, y, z, vx, vy, vz, ax, ay, az,
                                                     mass, sorted, child, minX, maxX, n, g, blockSizeInt, warp, stackSize);
    }
    return elapsedTime;

}

float KernelsWrapper::update(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                    float *ax, float *ay, float *az, int n, float dt, float d, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
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

