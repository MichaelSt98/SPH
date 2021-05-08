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
                          float *minZ, float *maxZ, int n, int m, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        resetArraysKernel<<< gridSize, blockSize >>>(mutex, x, y, z, mass, count, start, sorted, child, index,
                minX, maxX, minY, maxY, minZ, maxZ, n, m);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        resetArraysKernel<<< gridSize, blockSize >>>(mutex, x, y, z, mass, count, start, sorted, child, index,
                                                     minX, maxX, minY, maxY, minZ, maxZ, n, m);
    }
    return elapsedTime;

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
                    float *minZ, float *maxZ, unsigned long *key, int maxLevel, int n) {

    getParticleKeyKernel<<< gridSize, blockSize >>>(x, y, z, minX, maxX, minY, maxY, minZ, maxZ, 0UL, 21, n);

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

