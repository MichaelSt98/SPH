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
                                         int *domainListIndices, int *domainListLevels,
                                         int *lowestDomainListIndices, int *lowestDomainListIndex,
                                         unsigned long *lowestDomainListKeys, unsigned long *sortedLowestDomainListKeys,
                                         float *tempArray, int *to_delete_cell, int *to_delete_leaf, int n, int m) {

    resetArraysParallelKernel<<< gridSize, blockSize >>>(domainListIndex, domainListKeys, domainListIndices,
                                                         domainListLevels, lowestDomainListIndices, lowestDomainListIndex,
                                                         lowestDomainListKeys, sortedLowestDomainListKeys,
                                                         tempArray, to_delete_cell, to_delete_leaf, n, m);
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

float KernelsWrapper::buildDomainTree(int *domainListIndex, unsigned long *domainListKeys, int *domainListLevels,
                                      int *domainListIndices, float *x, float *y, float *z, float *mass, float *minX,
                                      float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int *count,
                                      int *start, int *child, int *index, int n, int m, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

    buildDomainTreeKernel<<< 1, 1 >>>(domainListIndex, domainListKeys, domainListLevels, domainListIndices, x, y, z,
                                      mass, minX, maxX, minY, maxY, minZ, maxZ,count, start, child, index, n, m);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        buildDomainTreeKernel<<< 1, 1 >>>(domainListIndex, domainListKeys, domainListLevels, domainListIndices, x, y, z,
                                          mass, minX, maxX, minY, maxY, minZ, maxZ,count, start, child, index, n, m);
    }
    return elapsedTime;

}

float KernelsWrapper::treeInfo(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int n, int m, int *procCounter, SubDomainKeyTree *s,
                                int *sortArray, int *sortArrayOut, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
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

float KernelsWrapper::particlesPerProcess(float *x, float *y, float *z, float *mass, int *count, int *start,
                                   int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                   float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                                   int *procCounterTemp, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

    particlesPerProcessKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                                   minX, maxX, minY, maxY, minZ, maxZ, n, m, s, procCounter,
                                                   procCounterTemp);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        particlesPerProcessKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                                             minX, maxX, minY, maxY, minZ, maxZ, n, m, s, procCounter,
                                                             procCounterTemp);
    }
    return elapsedTime;

}

float KernelsWrapper::sortParticlesProc(float *x, float *y, float *z, float *mass, int *count, int *start,
                       int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                       float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                       int *procCounterTemp, int *sortArray, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        sortParticlesProcKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                                         minX, maxX, minY, maxY, minZ, maxZ, n, m, s,
                                                         procCounter, procCounterTemp, sortArray);
        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        sortParticlesProcKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                                           minX, maxX, minY, maxY, minZ, maxZ, n, m, s,
                                                           procCounter, procCounterTemp, sortArray);
    }
    return elapsedTime;
}

float KernelsWrapper::copyArray(float *targetArray, float *sourceArray, int n, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
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
        cudaEvent_t start_t, stop_t; // used for timing
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

float KernelsWrapper::reorderArray(float *array, float *tempArray, SubDomainKeyTree *s,
                  int *procCounter, int *receiveLengths, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //reorderArrayKernel<<<gridSize, blockSize>>>(array, tempArray, s, procCounter, receiveLengths);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        //reorderArrayKernel<<<gridSize, blockSize>>>(array, tempArray, s, procCounter, receiveLengths);
    }
    return elapsedTime;
}

float KernelsWrapper::sendParticles(float *x, float *y, float *z, float *mass, int *count, int *start,
                   int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                   float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                   float *tempArray, int *sortArray, int *sortArrayOut, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
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
    }
    else {
        sendParticlesKernel<<< 1, 1/*gridSize, blockSize*/ >>>(x, y, z, mass, count, start, child, index,
                                                               minX, maxX, minY, maxY, minZ, maxZ, n, m, s, procCounter,
                                                               tempArray, sortArray, sortArrayOut);
    }

    //printf("Elapsed time for sorting: %f\n", elapsedTime);
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

float KernelsWrapper::getParticleKey(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                    float *minZ, float *maxZ, unsigned long *key, int maxLevel, int n, SubDomainKeyTree *s, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        getParticleKeyKernel<<< gridSize, blockSize >>>(x, y, z, minX, maxX, minY, maxY, minZ, maxZ, 0UL, 21, n, s);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        getParticleKeyKernel<<< gridSize, blockSize >>>(x, y, z, minX, maxX, minY, maxY, minZ, maxZ, 0UL, 21, n, s);
    }
    return elapsedTime;

}

float KernelsWrapper::traverseIterative(float *x, float *y, float *z, float *mass, int *child, int n, int m,
                       SubDomainKeyTree *s, int maxLevel, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
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

float KernelsWrapper::createDomainList(SubDomainKeyTree *s, int maxLevel, unsigned long *domainListKeys, int *levels,
                                      int *index, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        createDomainListKernel<<<1, 1>>>(s, maxLevel, domainListKeys, levels, index);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        createDomainListKernel<<<1, 1>>>(s, maxLevel, domainListKeys, levels, index);
    }
    return elapsedTime;
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


float KernelsWrapper::lowestDomainListNodes(int *domainListIndices, int *domainListIndex,
                                 unsigned long *domainListKeys,
                                 int *lowestDomainListIndices, int *lowestDomainListIndex,
                                 unsigned long *lowestDomainListKeys,
                                 float *x, float *y, float *z, float *mass, int *count, int *start,
                                 int *child, int n, int m, int *procCounter, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //kernel call
        lowestDomainListNodesKernel<<<gridSize, blockSize>>>(domainListIndices, domainListIndex, domainListKeys,
                                                             lowestDomainListIndices,
                                                             lowestDomainListIndex, lowestDomainListKeys, x, y, z,
                                                             mass, count, start, child,
                                                             n, m, procCounter);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        lowestDomainListNodesKernel<<<gridSize, blockSize>>>(domainListIndices, domainListIndex, domainListKeys,
                                                             lowestDomainListIndices,
                                                             lowestDomainListIndex, lowestDomainListKeys, x, y, z,
                                                             mass, count, start, child,
                                                             n, m, procCounter);
    }
    return elapsedTime;

}

float KernelsWrapper::prepareLowestDomainExchange(float *entry, float *mass, float *tempArray, int *lowestDomainListIndices,
                                        int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                        int *counter, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //kernel call
        prepareLowestDomainExchangeKernel<<< /*1, 1*/gridSize, blockSize >>> (entry, mass, tempArray, lowestDomainListIndices, lowestDomainListIndex,
                                            lowestDomainListKeys, counter);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        prepareLowestDomainExchangeKernel<<< /*1, 1*/ gridSize, blockSize >>> (entry, mass, tempArray, lowestDomainListIndices, lowestDomainListIndex,
                                                                lowestDomainListKeys, counter);
    }
    return elapsedTime;

}

float KernelsWrapper::prepareLowestDomainExchangeMass(float *mass, float *tempArray, int *lowestDomainListIndices,
                                            int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                            int *counter, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //kernel call
        prepareLowestDomainExchangeMassKernel<<< /*1, 1*/ gridSize, blockSize >>> (mass, tempArray, lowestDomainListIndices, lowestDomainListIndex,
                                                                      lowestDomainListKeys, counter);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        prepareLowestDomainExchangeMassKernel<<< /*1, 1*/ gridSize, blockSize >>> (mass, tempArray, lowestDomainListIndices, lowestDomainListIndex,
                                                                          lowestDomainListKeys, counter);
    }
    return elapsedTime;

}

float KernelsWrapper::updateLowestDomainListNodes(float *tempArray, float *entry, int *lowestDomainListIndices,
                                        int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                        unsigned long *sortedLowestDomainListKeys, int *counter,
                                        bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //kernel call
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
                                      unsigned long *sortedLowestDomainListKeys, int *counter,
                                      bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //kernel call
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

float KernelsWrapper::zeroDomainListNodes(int *domainListIndex, int *domainListIndices,
                                          int *lowestDomainListIndex, int *lowestDomainListIndices,
                                          float *x, float *y, float *z,
                                          float *mass, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //kernel call
        zeroDomainListNodesKernel<<< gridSize, blockSize >>>(domainListIndex, domainListIndices, lowestDomainListIndex,
                                                             lowestDomainListIndices, x, y, z, mass);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        //kernel call
        zeroDomainListNodesKernel<<< gridSize, blockSize >>>(domainListIndex, domainListIndices, lowestDomainListIndex,
                                                             lowestDomainListIndices, x, y, z, mass);
    }
    return elapsedTime;

}


float KernelsWrapper::compLocalPseudoParticlesPar(float *x, float *y, float *z, float *mass, int *index, int n,
                                        int *domainListIndices, int *domainListIndex,
                                                  int *lowestDomainListIndices, int *lowestDomainListIndex, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //kernel call
        compLocalPseudoParticlesParKernel<<<gridSize, blockSize>>>(x, y, z, mass, index, n, domainListIndices, domainListIndex,
                                                                   lowestDomainListIndices, lowestDomainListIndex);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        //kernel call
        compLocalPseudoParticlesParKernel<<<gridSize, blockSize>>>(x, y, z, mass, index, n, domainListIndices, domainListIndex,
                                                                   lowestDomainListIndices, lowestDomainListIndex);
    }
    return elapsedTime;

}

float KernelsWrapper::compDomainListPseudoParticlesPar(float *x, float *y, float *z, float *mass, int *child, int *index, int n,
                                             int *domainListIndices, int *domainListIndex,
                                             int *domainListLevels, int *lowestDomainListIndices,
                                             int *lowestDomainListIndex, bool timing) {
    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //kernel call
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
        //kernel call
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
        cudaEvent_t start_t, stop_t; // used for timing
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

float KernelsWrapper::symbolicForce(int relevantIndex, float *x, float *y, float *z, float *minX, float *maxX, float *minY,
                                    float *maxY, float *minZ, float *maxZ, int *child, int *domainListIndex,
                                    unsigned long *domainListKeys, int *domainListIndices, int *domainListLevels,
                                    int *domainListCounter, int *sendIndices, int *index, int *particleCounter,
                                    SubDomainKeyTree *s, int n, int m, float diam, float theta, int *mutex, bool timing) {

    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //kernel call
        symbolicForceKernel<<<1, 1/*gridSize, blockSize*/>>>(relevantIndex, x, y, z, minX, maxX, minY, maxY, minZ, maxZ, child,
                                                     domainListIndex, domainListKeys, domainListIndices, domainListLevels,
                                                     domainListCounter, sendIndices, index, particleCounter, s, n, m,
                                                     diam, theta, mutex);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        //kernel call
        symbolicForceKernel<<<1, 1/*gridSize, blockSize*/>>>(relevantIndex, x, y, z, minX, maxX, minY, maxY, minZ, maxZ, child,
                                                     domainListIndex, domainListKeys, domainListIndices, domainListLevels,
                                                     domainListCounter, sendIndices, index, particleCounter, s, n, m,
                                                     diam, theta, mutex);
    }
    return elapsedTime;

}

float KernelsWrapper::compTheta(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
               float *minZ, float *maxZ, int *domainListIndex, int *domainListCounter,
               unsigned long *domainListKeys, int *domainListIndices, int *domainListLevels,
               int *relevantDomainListIndices, SubDomainKeyTree *s, bool timing) {


    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //kernel call
        compThetaKernel<<<gridSize, blockSize>>>(x, y, z, minX, maxX, minY, maxY, minZ, maxZ, domainListIndex,
                                                 domainListCounter, domainListKeys, domainListIndices, domainListLevels,
                                                 relevantDomainListIndices, s);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        //kernel call
        compThetaKernel<<<gridSize, blockSize>>>(x, y, z, minX, maxX, minY, maxY, minZ, maxZ, domainListIndex,
                                                 domainListCounter, domainListKeys, domainListIndices, domainListLevels,
                                                 relevantDomainListIndices, s);
    }
    return elapsedTime;

}

float KernelsWrapper::insertReceivedParticles(float *x, float *y, float *z, float *mass, int *count, int *start,
                             int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                             float *minZ, float *maxZ, int *to_delete_leaf, int n, int m, bool timing) {


    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //kernel call
        insertReceivedParticlesKernel<<<gridSize, blockSize>>>(x, y, z, mass, count, start, child, index, minX, maxX,
                                                               minY, maxY, minZ, maxZ, to_delete_leaf, n, m);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        //kernel call
        insertReceivedParticlesKernel<<<gridSize, blockSize>>>(x, y, z, mass, count, start, child, index, minX, maxX,
                                                               minY, maxY, minZ, maxZ, to_delete_leaf, n, m);
    }
    return elapsedTime;

}


float KernelsWrapper::repairTree(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                float *ax, float *ay, float *az, float *mass, int *count, int *start,
                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                float *minZ, float *maxZ, int *to_delete_cell, int *to_delete_leaf,
                int *domainListIndices, int n, int m, bool timing) {


    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //kernel call
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
        //kernel call
        repairTreeKernel<<<gridSize, blockSize>>>(x, y, z, vx, vy, vz, ax, ay, az, mass, count, start, child, index,
                                                  minX, maxX, minY, maxY, minZ, maxZ, to_delete_cell, to_delete_leaf,
                                                  domainListIndices, n, m);
    }
    return elapsedTime;

}

float KernelsWrapper::findDuplicates(float *array, int length, SubDomainKeyTree *s, int *duplicateCounter, bool timing) {
    float elapsedTime = 0.f;
    if (timing) {
        cudaEvent_t start_t, stop_t; // used for timing
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
        cudaEventRecord(start_t, 0);

        //kernel call
        findDuplicatesKernel<<<gridSize, blockSize>>>(array, length, s, duplicateCounter);

        cudaEventRecord(stop_t, 0);
        cudaEventSynchronize(stop_t);
        cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }
    else {
        //kernel call
        findDuplicatesKernel<<<gridSize, blockSize>>>(array, length, s, duplicateCounter);
    }
    return elapsedTime;
}
