/**
 * CUDA Kernel functions.
 *
 * Notes:
 *
 * * use `-1` as *null pointer*
 * * last-level cell and then attempts to lock the appropriate child pointer (an array index) by writing an
otherwise unused value (−2) to it using an atomic operation
 */

#include "../include/Kernels.cuh"

//__device__ const int   blockSize = 256; //256;
//extern __shared__ float buffer[];
//__device__ const int   warp = 32;
//__device__ const int   stackSize = 64;
__device__ const float eps_squared = 0.0025;
__device__ const float theta = 1.5; //0.5;


__global__ void resetArraysKernel(int *mutex, float *x, float *y, float *z, float *mass, int *count, int *start,
                                  int *sorted, int *child, int *index, float *minX, float *maxX,
                                  float *minY, float *maxY, float *minZ, float *maxZ, int n, int m,
                                  int *procCounter, int *procCounterTemp) {

    int bodyIndex = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;

    // reset quadtree arrays
    while(bodyIndex + offset < m) {
        #pragma unroll 8
        for (int i=0; i<8; i++) {
            child[(bodyIndex + offset)*8 + i] = -1;
        }
        if (bodyIndex + offset < n) {
            count[bodyIndex + offset] = 1;
        }
        else {
            x[bodyIndex + offset] = 0;
            y[bodyIndex + offset] = 0;
            z[bodyIndex + offset] = 0;

            mass[bodyIndex + offset] = 0;
            count[bodyIndex + offset] = 0;
        }
        start[bodyIndex + offset] = -1;
        sorted[bodyIndex + offset] = 0;
        offset += stride;
    }
    // reset quadtree pointers
    if (bodyIndex == 0) {
        *mutex = 0;
        *index = n;
        *minX = 0;
        *maxX = 0;
        *minY = 0;
        *maxY = 0;
        *minZ = 0;
        *maxZ = 0;
        procCounter[0] = 0;
        procCounter[1] = 0;
        procCounterTemp[0] = 0;
        procCounterTemp[1] = 0;
    }
}

__global__ void resetArraysParallelKernel(int *domainListIndex, unsigned long *domainListKeys,
                                          int *domainListIndices, int *domainListLevels,
                                          int *lowestDomainListIndices, int *lowestDomainListIndex,
                                          unsigned long *lowestDomainListKeys, unsigned long *sortedLowestDomainListKeys,
                                          float *tempArray, int *to_delete_cell, int *to_delete_leaf, int n, int m) {

    int bodyIndex = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;

    while ((bodyIndex + offset) < n) {
        tempArray[bodyIndex + offset] = 0;

        if ((bodyIndex + offset) < DOMAIN_LIST_SIZE) {
            domainListLevels[bodyIndex + offset] = -1;
            domainListKeys[bodyIndex + offset] = KEY_MAX;
            domainListIndices[bodyIndex + offset] = -1;
            lowestDomainListIndices[bodyIndex + offset] = -1;
            lowestDomainListKeys[bodyIndex + offset] = KEY_MAX;
            sortedLowestDomainListKeys[bodyIndex + offset] = KEY_MAX;
            offset += stride;
        }

        offset += stride;
    }
    if (bodyIndex == 0) {
        *domainListIndex = 0;
        *lowestDomainListIndex = 0;
        to_delete_cell[0] = -1;
        to_delete_cell[1] = -1;
        to_delete_leaf[0] = -1;
        to_delete_leaf[1] = -1;
    }
}

// Kernel 1: computes bounding box around all bodies
__global__ void computeBoundingBoxKernel(int *mutex, float *x, float *y, float *z, float *minX, float *maxX,
                                         float *minY, float *maxY, float *minZ, float *maxZ, int n, int blockSize)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    // initialize local min/max
    float x_min = x[index];
    float x_max = x[index];
    float y_min = y[index];
    float y_max = y[index];
    float z_min = z[index];
    float z_max = z[index];

    extern __shared__ float buffer[];

    float* x_min_buffer = (float*)buffer;
    float* x_max_buffer = (float*)&x_min_buffer[blockSize];
    float* y_min_buffer = (float*)&x_max_buffer[blockSize];
    float* y_max_buffer = (float*)&y_min_buffer[blockSize];
    float* z_min_buffer = (float*)&y_max_buffer[blockSize];
    float* z_max_buffer = (float*)&z_min_buffer[blockSize];

    int offset = stride;

    // find (local) min/max
    while (index + offset < n) {

        x_min = fminf(x_min, x[index + offset]);
        x_max = fmaxf(x_max, x[index + offset]);
        y_min = fminf(y_min, y[index + offset]);
        y_max = fmaxf(y_max, y[index + offset]);
        z_min = fminf(z_min, z[index + offset]);
        z_max = fmaxf(z_max, z[index + offset]);

        offset += stride;
    }

    // save value in corresponding buffer
    x_min_buffer[threadIdx.x] = x_min;
    x_max_buffer[threadIdx.x] = x_max;
    y_min_buffer[threadIdx.x] = y_min;
    y_max_buffer[threadIdx.x] = y_max;
    z_min_buffer[threadIdx.x] = z_min;
    z_max_buffer[threadIdx.x] = z_max;

    // synchronize threads / wait for unfinished threads
    __syncthreads();

    int i = blockDim.x/2; // assuming blockDim.x is a power of 2!

    // reduction within block
    while (i != 0) {
        if (threadIdx.x < i) {
            x_min_buffer[threadIdx.x] = fminf(x_min_buffer[threadIdx.x], x_min_buffer[threadIdx.x + i]);
            x_max_buffer[threadIdx.x] = fmaxf(x_max_buffer[threadIdx.x], x_max_buffer[threadIdx.x + i]);
            y_min_buffer[threadIdx.x] = fminf(y_min_buffer[threadIdx.x], y_min_buffer[threadIdx.x + i]);
            y_max_buffer[threadIdx.x] = fmaxf(y_max_buffer[threadIdx.x], y_max_buffer[threadIdx.x + i]);
            z_min_buffer[threadIdx.x] = fminf(z_min_buffer[threadIdx.x], z_min_buffer[threadIdx.x + i]);
            z_max_buffer[threadIdx.x] = fmaxf(z_max_buffer[threadIdx.x], z_max_buffer[threadIdx.x + i]);
        }
        __syncthreads();
        i /= 2;
    }

    // combining the results and generate the root cell
    if (threadIdx.x == 0) {
        while (atomicCAS(mutex, 0 ,1) != 0); // lock

        *minX = fminf(*minX, x_min_buffer[0]);
        *maxX = fmaxf(*maxX, x_max_buffer[0]);
        *minY = fminf(*minY, y_min_buffer[0]);
        *maxY = fmaxf(*maxY, y_max_buffer[0]);
        *minZ = fminf(*minZ, z_min_buffer[0]);
        *maxZ = fmaxf(*maxZ, z_max_buffer[0]);

        atomicExch(mutex, 0); // unlock
    }
}


__global__ void particlesPerProcessKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                    int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                    float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s,
                                    int *procCounter, int *procCounterTemp) {

    //go over domain list (only the ones inherited by own process) and count particles (using count array)
    //BUT: for now use this approach!
    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int offset = 0;

    unsigned long key;
    int proc;

    while ((bodyIndex + offset) < n) {

        key = getParticleKeyPerParticle(x[bodyIndex + offset], y[bodyIndex + offset], z[bodyIndex + offset],
                                        minX, maxX, minY, maxY, minZ, maxZ, 21);

        proc = key2proc(key, s);

        atomicAdd(&procCounter[proc], 1);

        offset += stride;
    }
}

__global__ void sortParticlesProcKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                          int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                          float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s,
                                          int *procCounter, int *procCounterTemp, int *sortArray) {

    //go over domain list (only the ones inherited by own process) and count particles (using count array)
    //BUT: for now use this approach!
    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int offset = 0;

    unsigned long key;
    int proc;
    int counter;

    while ((bodyIndex + offset) < n) {

        key = getParticleKeyPerParticle(x[bodyIndex + offset], y[bodyIndex + offset], z[bodyIndex + offset],
                                        minX, maxX, minY, maxY, minZ, maxZ, 21);
        proc = key2proc(key, s);

        counter = atomicAdd(&procCounterTemp[proc], 1);

        if (proc > 0) {
            sortArray[bodyIndex + offset] = procCounter[proc-1] + counter;
        }
        else {
            sortArray[bodyIndex + offset] = counter;
        }

        // should work as well
        //sortArray[bodyIndex + offset] = proc;

        offset += stride;

    }
}

__global__ void copyArrayKernel(float *targetArray, float *sourceArray, int n) {
    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int offset = 0;

    while ((bodyIndex + offset) < n) {
        targetArray[bodyIndex + offset] = sourceArray[bodyIndex + offset];

        offset += stride;
    }
}

__global__ void resetFloatArrayKernel(float *array, float value, int n) {
    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int offset = 0;

    while ((bodyIndex + offset) < n) {
        array[bodyIndex + offset] = value;

        offset += stride;
    }
}

//TODO: deletable, but used as print-out/debug kernel
__global__ void sendParticlesKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                                float *tempArray, int *sortArray, int *sortArrayOut) {

    for (int i=0; i<8; i++) {
        printf("child[%i] = %i\n", i, child[i]);
        for (int k=0; k<8; k++) {
            printf("\tchild[8*child[%i] + %i] = %i\n", i, k, child[8*child[i] + k]);
        }
    }
}

// Kernel 2: hierarchically subdivides the root cells
__global__ void buildTreeKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int n, int m) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    //note: -1 used as "null pointer"
    //note: -2 used to lock a child (pointer)

    int offset;
    bool newBody = true;

    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float min_z;
    float max_z;

    int childPath;
    int temp;
    int tempTemp;

    offset = 0;

    while ((bodyIndex + offset) < n) {

        if (newBody) {

            newBody = false;

            min_x = *minX;
            max_x = *maxX;
            min_y = *minY;
            max_y = *maxY;
            min_z = *minZ;
            max_z = *maxZ;

            temp = 0;
            childPath = 0;

            // find insertion point for body
            if (x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                childPath += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
            if (y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                childPath += 2;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }
            if (z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {  // z direction
                childPath += 4;
                max_z = 0.5 * (min_z + max_z);
            }
            else {
                min_z = 0.5 * (min_z + max_z);
            }
        }

        int childIndex = child[temp*8 + childPath];

        // traverse tree until hitting leaf node
        while (childIndex >= m) { //n

            tempTemp = temp;
            temp = childIndex;

            childPath = 0;

            // find insertion point for body
            if (x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                childPath += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
            if (y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                childPath += 2;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }
            if (z[bodyIndex + offset] < 0.5 * (min_z + max_z)) { // z direction
                childPath += 4;
                max_z = 0.5 * (min_z + max_z);
            }
            else {
                min_z = 0.5 * (min_z + max_z);
            }

            if (mass[bodyIndex + offset] != 0) {
                atomicAdd(&x[temp], mass[bodyIndex + offset] * x[bodyIndex + offset]);
                atomicAdd(&y[temp], mass[bodyIndex + offset] * y[bodyIndex + offset]);
                atomicAdd(&z[temp], mass[bodyIndex + offset] * z[bodyIndex + offset]);
            }

            atomicAdd(&mass[temp], mass[bodyIndex + offset]);
            atomicAdd(&count[temp], 1);

            childIndex = child[8*temp + childPath];
        }

        // if child is not locked
        if (childIndex != -2) {

            int locked = temp * 8 + childPath;

            if (atomicCAS(&child[locked], childIndex, -2) == childIndex) {

                // check whether a body is already stored at the location
                if (childIndex == -1) {
                    //insert body and release lock
                    child[locked] = bodyIndex + offset;
                }
                else {
                    if (childIndex >= n) {
                        printf("ATTENTION!\n");
                    }
                    int patch = 8 * m; //8*n
                    while (childIndex >= 0 && childIndex < n) { //TODO: was n

                        //create a new cell (by atomically requesting the next unused array index)
                        int cell = atomicAdd(index, 1);
                        patch = min(patch, cell);

                        if (patch != cell) {
                            child[8 * temp + childPath] = cell;
                        }

                        // insert old/original particle
                        childPath = 0;
                        if (x[childIndex] < 0.5 * (min_x + max_x)) { childPath += 1; }
                        if (y[childIndex] < 0.5 * (min_y + max_y)) { childPath += 2; }
                        if (z[childIndex] < 0.5 * (min_z + max_z)) { childPath += 4; }

                        x[cell] += mass[childIndex] * x[childIndex];
                        y[cell] += mass[childIndex] * y[childIndex];
                        z[cell] += mass[childIndex] * z[childIndex];

                        mass[cell] += mass[childIndex];
                        count[cell] += count[childIndex];

                        child[8 * cell + childPath] = childIndex;

                        start[cell] = -1;

                        // insert new particle
                        tempTemp = temp;
                        temp = cell;
                        childPath = 0;

                        // find insertion point for body
                        if (x[bodyIndex + offset] < 0.5 * (min_x + max_x)) {
                            childPath += 1;
                            max_x = 0.5 * (min_x + max_x);
                        } else {
                            min_x = 0.5 * (min_x + max_x);
                        }
                        if (y[bodyIndex + offset] < 0.5 * (min_y + max_y)) {
                            childPath += 2;
                            max_y = 0.5 * (min_y + max_y);
                        } else {
                            min_y = 0.5 * (min_y + max_y);
                        }
                        if (z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {
                            childPath += 4;
                            max_z = 0.5 * (min_z + max_z);
                        } else {
                            min_z = 0.5 * (min_z + max_z);
                        }

                        // COM / preparing for calculation of COM
                        if (mass[bodyIndex + offset] != 0) {
                            x[cell] += mass[bodyIndex + offset] * x[bodyIndex + offset];
                            y[cell] += mass[bodyIndex + offset] * y[bodyIndex + offset];
                            z[cell] += mass[bodyIndex + offset] * z[bodyIndex + offset];
                            mass[cell] += mass[bodyIndex + offset];
                        }
                        count[cell] += count[bodyIndex + offset];
                        childIndex = child[8 * temp + childPath];
                    }

                    child[8 * temp + childPath] = bodyIndex + offset;

                    __threadfence();  // written to global memory arrays (child, x, y, mass) thus need to fence
                    child[locked] = patch;
                }
                offset += stride;
                newBody = true;
            }
        }
        __syncthreads();
    }
}

__global__ void buildDomainTreeKernel(int *domainListIndex, unsigned long *domainListKeys, int *domainListLevels,
                                      int *domainListIndices, float *x, float *y, float *z, float *mass, float *minX,
                                      float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int *count,
                                      int *start, int *child, int *index, int n, int m) {

    int domainListCounter = 0;

    //char keyAsChar[21 * 2 + 3];
    int path[21];

    int min_x, max_x, min_y, max_y, min_z, max_z;
    int currentChild;
    int childPath;
    bool insert = true;

    int childIndex;
    int temp;

    for (int i = 0; i < *domainListIndex; i++) {
        //key2Char(domainListKeys[i], 21, keyAsChar);
        //printf("buildDomainTree: domainListLevels[%i] = %i\n", i, domainListLevels[i]);
        //printf("domain: domainListKeys[%i] = %lu = %s (level: %i)\n", i, domainListKeys[i], keyAsChar, domainListLevels[i]);
        childIndex = 0;
        //temp = 0;
        for (int j = 0; j < domainListLevels[i]; j++) {
            path[j] = (int) (domainListKeys[i] >> (21 * 3 - 3 * (j + 1)) & (int)7);
            temp = childIndex;
            childIndex = child[8*childIndex + path[j]];
            if (childIndex < n) {
                if (childIndex == -1 /*&& childIndex < n*/) {
                    // no child at all
                    int cell = atomicAdd(index, 1);
                    child[8 * temp + path[j]] = cell;
                    childIndex = cell;
                    domainListIndices[domainListCounter] = childIndex; //cell;
                    domainListCounter++;
                } else {

                    int cell = atomicAdd(index, 1);
                    child[8 * temp + path[j]] = cell;

                    min_x = *minX;
                    max_x = *maxX;
                    min_y = *minY;
                    max_y = *maxY;
                    min_z = *minZ;
                    max_z = *maxZ;

                    for (int k=0; k<j; k++) {
                        currentChild = path[k];
                        if (currentChild % 1 == 0) {
                            max_x = 0.5 * (min_x + max_x);
                            currentChild -= 1;
                        }
                        else {
                            min_x = 0.5 * (min_x + max_x);
                        }
                        if (currentChild % 2 == 0 && currentChild % 4 != 0) {
                            max_y = 0.5 * (min_y + max_y);
                            currentChild -= 2;
                        }
                        else {
                            min_y = 0.5 * (min_y + max_y);
                        }
                        if (currentChild % 4 == 0) {
                            max_z = 0.5 * (min_z + max_z);
                            currentChild -= 4;
                        }
                        else {
                            min_z = 0.5 * (min_z + max_z);
                        }
                    }
                    // insert old/original particle
                    childPath = 0;
                    if (x[childIndex] < 0.5 * (min_x + max_x)) { childPath += 1; }
                    if (y[childIndex] < 0.5 * (min_y + max_y)) { childPath += 2; }
                    if (z[childIndex] < 0.5 * (min_z + max_z)) { childPath += 4; }

                    child[8 * cell + childPath] = childIndex;

                    childIndex = cell;
                    domainListIndices[domainListCounter] = childIndex; //temp;
                    domainListCounter++;
                }
            }
            else {
                insert = true;
                for (int k=0; k<domainListCounter; k++) {
                    if (childIndex == domainListIndices[k]) {
                        insert = false;
                        break;
                    }
                }
                if (insert) {
                    domainListIndices[domainListCounter] = childIndex; //temp;
                    domainListCounter++;
                }
            }
        }
    }
    //printf("domainListCounter = %i\n", domainListCounter);
}

__global__ void lowestDomainListNodesKernel(int *domainListIndices, int *domainListIndex,
                                      unsigned long *domainListKeys,
                                      int *lowestDomainListIndices, int *lowestDomainListIndex,
                                      unsigned long *lowestDomainListKeys,
                                      float *x, float *y, float *z, float *mass, int *count, int *start,
                                      int *child, int n, int m, int *procCounter) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;
    bool lowestDomainListNode;
    int domainIndex;
    int lowestDomainIndex;
    int childIndex;

    while ((bodyIndex + offset) < *domainListIndex) {
        lowestDomainListNode = true;
        domainIndex = domainListIndices[bodyIndex + offset];
        for (int i=0; i<8; i++) {
            childIndex = child[8 * domainIndex + i];
            if (childIndex != -1) {
                if (childIndex >= n) {
                    for (int k=0; k<*domainListIndex; k++) {
                        if (domainListKeys[bodyIndex + offset] == 0) {
                            //printf("domainIndex = %i  childIndex: %i  domainListIndices: %i\n", domainIndex,
                            //       childIndex, domainListIndices[k]);
                        }
                        if (childIndex == domainListIndices[k]) {
                            //printf("domainIndex = %i  childIndex: %i  domainListIndices: %i\n", domainIndex,
                            //       childIndex, domainListIndices[k]);
                            lowestDomainListNode = false;
                            break;
                        }
                    }
                    if (lowestDomainListNode) {
                        break;
                    }
                }
            }
        }

        if (lowestDomainListNode) {
            lowestDomainIndex = atomicAdd(lowestDomainListIndex, 1);
            lowestDomainListIndices[lowestDomainIndex] = domainIndex;
            lowestDomainListKeys[lowestDomainIndex] = domainListKeys[bodyIndex + offset];
            //printf("Adding lowest domain list node #%i (key = %lu)\n", lowestDomainIndex,
            // lowestDomainListKeys[lowestDomainIndex]);
        }
        offset += stride;
    }

}

//for debugging purposes
__global__ void treeInfoKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int n, int m, int *procCounter, SubDomainKeyTree *s,
                                int *sortArray, int *sortArrayOut) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    unsigned long key;
    int toCheck;
    int toCheckSorted;
    int proc;

    int offset = 0;

    while ((bodyIndex + offset) < procCounter[0]) {

        key = getParticleKeyPerParticle(x[bodyIndex+offset], y[bodyIndex+offset], z[bodyIndex+offset], minX, maxX, minY, maxY, minZ, maxZ, 21);

        proc = key2proc(key, s);
        if (proc != s->rank) {
            printf("ATTENTION: myrank = %i and proc = %i (bodyIndex + offset = %i)\n", s->rank, proc,
                   bodyIndex + offset);
        }

        offset += stride;

        //printf("&sortArrayOut = %p\n", sortArrayOut);

        //for (int i=0; i<10; i++) {
        //    printf("sortArray[%i] = %i | sortArrayOut[%i] = %i\n", i, sortArray[i], i, sortArrayOut[i]);
        //}

        //toCheck = 200000;
        //toCheckSorted = sortArrayOut[sortArray[toCheck]]; //sortArray[toCheck];
        //key = getParticleKeyPerParticle(x[toCheckSorted], y[toCheckSorted], z[toCheckSorted], minX, maxX, minY, maxY, minZ, maxZ, 21);
        //proc = key2proc(key, s);
        //printf("\t[rank %i] x[%i] = %f ,  sortArray = %i,  sortArrayOut = %i,  x[%i] = (%f, %f, %f) m = %f proc = %i\n", s->rank, toCheck, x[toCheck],
        //       sortArray[toCheck], sortArrayOut[sortArray[toCheck]], toCheckSorted, x[toCheckSorted], y[toCheckSorted], z[toCheckSorted], mass[toCheckSorted],
        //       proc);

        //for (int i = 0; i < 8; i++) {
        //    printf("child[%i] = %i  count = %i, mass = %f, x = %f\n", i,
        //           child[i], count[child[i]], mass[child[i]], x[child[i]]);
        //}
        //printf("[rank %i] x[0] = %f ,  x[%i] = %f,   x[%i] = %f \n", s->rank, x[0], n-1, x[n-1], n, x[n]);
        //printf("[rank %i] m[0] = %f ,  m[%i] = %f,   m[%i] = %f \n", s->rank, mass[0], n-1, mass[n-1], n, mass[n]);

        /*for (int i=procCounter[0]+1; i<(procCounter[0] + 1 + 10); i++) {
            toCheck = i; //procCounter[0]+2;
            key = getParticleKeyPerParticle(x[toCheck], y[toCheck], z[toCheck], minX, maxX, minY, maxY, minZ, maxZ, 21);
            proc = key2proc(key, s);
            printf("[rank %i] toCheck = %i  proc = %i\n", s->rank, toCheck, proc);
        }*/
    }

}

__device__ void key2Char(unsigned long key, int maxLevel, char *keyAsChar) {
    int level[21];
    for (int i=0; i<maxLevel; i++) {
        level[i] = (int)(key >> (maxLevel*3 - 3*(i+1)) & (int)7);
    }
    for (int i=0; i<=maxLevel; i++) {
        keyAsChar[2*i] = level[i] + '0';
        keyAsChar[2*i+1] = '|';
    }
    keyAsChar[2*maxLevel+3] = '\0';
}

__device__ const unsigned char DirTable[12][8] =
        { { 8,10, 3, 3, 4, 5, 4, 5}, { 2, 2,11, 9, 4, 5, 4, 5},
          { 7, 6, 7, 6, 8,10, 1, 1}, { 7, 6, 7, 6, 0, 0,11, 9},
          { 0, 8, 1,11, 6, 8, 6,11}, {10, 0, 9, 1,10, 7, 9, 7},
          {10, 4, 9, 4,10, 2, 9, 3}, { 5, 8, 5,11, 2, 8, 3,11},
          { 4, 9, 0, 0, 7, 9, 2, 2}, { 1, 1, 8, 5, 3, 3, 8, 6},
          {11, 5, 0, 0,11, 6, 2, 2}, { 1, 1, 4,10, 3, 3, 7,10} };

__device__ const unsigned char HilbertTable[12][8] = { {0,7,3,4,1,6,2,5}, {4,3,7,0,5,2,6,1}, {6,1,5,2,7,0,4,3},
                                                       {2,5,1,6,3,4,0,7}, {0,1,7,6,3,2,4,5}, {6,7,1,0,5,4,2,3},
                                                       {2,3,5,4,1,0,6,7}, {4,5,3,2,7,6,0,1}, {0,3,1,2,7,4,6,5},
                                                       {2,1,3,0,5,6,4,7}, {4,7,5,6,3,0,2,1}, {6,5,7,4,1,2,0,3} };

__device__ unsigned long Lebesgue2Hilbert(unsigned long lebesgue, int maxLevel) {
    unsigned long hilbert = 0UL;
    int dir = 0;
    for (int lvl=maxLevel; lvl>0; lvl--) {
        unsigned long cell = (lebesgue >> ((lvl-1)*3)) & (unsigned long)((1<<3)-1);
        hilbert = hilbert << 3;
        if (lvl > 0) {
            hilbert += HilbertTable[dir][cell];
        }
        dir = DirTable[dir][cell];
    }
    return hilbert;
}

__device__ unsigned long getParticleKeyPerParticle(float x, float y, float z,
                                                   float *minX, float *maxX, float *minY,
                                                   float *maxY, float *minZ, float *maxZ,
                                                   int maxLevel) {

    int level = 0;
    unsigned long testKey = 0UL;

    int sonBox = 0;
    float min_x = *minX;
    float max_x = *maxX;
    float min_y = *minY;
    float max_y = *maxY;
    float min_z = *minZ;
    float max_z = *maxZ;

    while (level <= maxLevel) {

        sonBox = 0;
        // find insertion point for body
        if (x < 0.5 * (min_x+max_x)) {
            sonBox += 1;
            max_x = 0.5 * (min_x+max_x);
        }
        else { min_x = 0.5 * (min_x+max_x); }

        if (y < 0.5 * (min_y+max_y)) {
            sonBox += 2;
            max_y = 0.5 * (min_y + max_y);
        }
        else { min_y = 0.5 * (min_y + max_y); }

        if (z < 0.5 * (min_z+max_z)) {
            sonBox += 4;
            max_z = 0.5 * (min_z + max_z);
        }
        else { min_z =  0.5 * (min_z + max_z); }

        testKey = testKey | ((unsigned long)sonBox << (unsigned long)(3 * (maxLevel-level-1)));
        level ++;
    }
    return testKey;

}

__global__ void getParticleKeyKernel(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                               float *minZ, float *maxZ, unsigned long *key, int maxLevel, int n, SubDomainKeyTree *s) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;
    unsigned long testKey;

    if (bodyIndex == 0) {
        char rangeAsChar[21 * 2 + 3];
        for (int i=0; i<3; i++) {
            key2Char(s->range[i], 21, rangeAsChar);
            printf("range[%i] = %lu (%s)\n", i, s->range[i], rangeAsChar);
        }
    }

    while (bodyIndex + offset < n) {

        testKey = 0UL;

        testKey = getParticleKeyPerParticle(x[bodyIndex + offset], y[bodyIndex + offset], z[bodyIndex + offset],
                                            minX, maxX, minY, maxY, minZ, maxZ, maxLevel);

        char keyAsChar[21 * 2 + 3];
        unsigned long hilbertTestKey = Lebesgue2Hilbert(testKey, 21);
        int proc = key2proc(testKey, s);
        //key2Char(testKey, 21, keyAsChar);
        key2Char(hilbertTestKey, 21, keyAsChar);
        if ((bodyIndex + offset) % 5000 == 0) {
            //printf("key[%i]: %lu\n", bodyIndex + offset, testKey);
            //for (int proc=0; proc<=s->numProcesses; proc++) {
            //    printf("range[%i] = %lu\n", proc, s->range[proc]);
            //}
            printf("key[%i]: %s  =  %lu (proc = %i)\n", bodyIndex + offset, keyAsChar, testKey, proc);
        }

        offset += stride;
    }
}

__device__ int key2proc(unsigned long k, SubDomainKeyTree *s) {
    for (int proc=0; proc<s->numProcesses; proc++) {
        if (k >= s->range[proc] && k < s->range[proc+1]) {
            return proc;
        }
    }
    //printf("ERROR: key2proc(k=%lu): -1!", k);
    return -1; // error
}

__global__ void traverseIterativeKernel(float *x, float *y, float *z, float *mass, int *child, int n, int m,
                         SubDomainKeyTree *s, int maxLevel) {

    __shared__ int stack[128];
    __shared__ int *stackPtr;
    stackPtr = stack;
    *stackPtr++ = NULL;

    int childIndex;
    int node;
    int particleCounter = 0;

    for (int j=0; j<8; j++) {
        childIndex;
        node = n;
        stack[0] = child[j];
        stackPtr = stack;
        //counter = 0;
        while (node != NULL /*&& counter < 200000*/) {
            //counter++;
            childIndex = *stackPtr;
            for (int i=0; i<8; i++) {
                if (child[8*childIndex + i] == -1) { /*do nothing*/ }
                else {
                    if (child[8*childIndex + i] < n) {
                        particleCounter++;
                    }
                    else {
                        *stackPtr++ = child[8*childIndex + i]; //push
                    }
                }
            }
            node = *--stackPtr; //pop
        }
    }
    printf("Finished traversing iteratively! particleCounter = %i\n", particleCounter);
}

__global__ void createDomainListKernel(SubDomainKeyTree *s, int maxLevel, unsigned long *domainListKeys, int *levels,
                                       int *index) {

    char keyAsChar[21 * 2 + 3];

    unsigned long shiftValue = 1;
    unsigned long toShift = 63;
    unsigned long keyMax = (shiftValue << toShift) - 1; // 1 << 63 not working!
    //key2Char(keyMax, 21, keyAsChar); //printf("keyMax: %lu = %s\n", keyMax, keyAsChar);

    unsigned long key2test = 0UL;

    int level = 0;

    level++;

    while (key2test < keyMax) {
        if (isDomainListNode(key2test & (~0UL << (3 * (maxLevel - level + 1))), maxLevel, level-1, s)) {
            domainListKeys[*index] = key2test;
            levels[*index] = level;
            *index += 1;
            if (isDomainListNode(key2test, maxLevel, level, s)) {
                level++;
            }
            else {
                key2test = key2test + (1UL << 3 * (maxLevel - level));
            }
        } else {
            level--;
            // not necessary... 1 = 1
            //key2test = keyMaxLevel(key2test & (~0UL << (3 * (maxLevel - level))), maxLevel, level, s) + 1 - (1UL << (3 * (maxLevel - level)));
        }
    }
    for (int i=0; i < *index; i++) {
        key2Char(domainListKeys[i], 21, keyAsChar);
    }
}

__device__ bool isDomainListNode(unsigned long key, int maxLevel, int level, SubDomainKeyTree *s) {
    int p1 = key2proc(key, s);
    int p2 = key2proc(key | ~(~0UL << 3*(maxLevel-level)), s);
    if (p1 != p2) {
        return true;
    }
    else {
        return false;
    }
}

__device__ unsigned long keyMaxLevel(unsigned long key, int maxLevel, int level, SubDomainKeyTree *s) {
    unsigned long keyMax = key | ~(~0UL << 3*(maxLevel-level));
    return keyMax;
}

__global__ void prepareLowestDomainExchangeKernel(float *entry, float *mass, float *tempArray, int *lowestDomainListIndices,
                                                  int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                                  int *counter) {

    //copy x, y, z, mass of lowest domain list nodes into arrays
    //sorting using cub (not here)
    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;
    int index;
    int lowestDomainIndex;

    while ((bodyIndex + offset) < *lowestDomainListIndex) {
        lowestDomainIndex = lowestDomainListIndices[bodyIndex + offset];
        if (lowestDomainIndex >= 0) {
            tempArray[bodyIndex+offset] = entry[lowestDomainIndex];
        }
        offset += stride;
    }

    //serial solution
    /*for (int i=0; i<*lowestDomainListIndex; i++) {
        tempArray[i] = entry[lowestDomainListIndices[i]];
    }*/
}

//TODO: not tested yet
__global__ void prepareLowestDomainExchangeMassKernel(float *mass, float *tempArray, int *lowestDomainListIndices,
                                                  int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                                  int *counter) {

    //copy x, y, z, mass of lowest domain list nodes into arrays
    //sorting using cub (not here)
    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;
    int index;
    int lowestDomainIndex;

    while ((bodyIndex + offset) < *lowestDomainListIndex) {
        lowestDomainIndex = lowestDomainListIndices[bodyIndex + offset];
        if (lowestDomainIndex >= 0) {
            tempArray[bodyIndex + offset] = mass[lowestDomainIndex];
        }
        offset += stride;
    }

    //serial solution
    /*for (int i=0; i<*lowestDomainListIndex; i++) {
        tempArray[i] = mass[lowestDomainListIndices[i]];
    }*/

}

//TODO: problem since not deterministic? keys are not unique
// at least the domain list nodes in general, but the lowest domain list nodes as well?
__global__ void updateLowestDomainListNodesKernel(float *tempArray, float *entry, int *lowestDomainListIndices,
                                                  int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                                  unsigned long *sortedLowestDomainListKeys, int *counter) {

    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;
    int originalIndex = -1;

    while ((bodyIndex + offset) < *lowestDomainListIndex) {
        for (int i=0; i<*lowestDomainListIndex; i++) {
            if (sortedLowestDomainListKeys[bodyIndex + offset] == lowestDomainListKeys[i]) {
                originalIndex = i;
                //break;
            }
        }

        if (originalIndex == -1) {
            printf("ATTENTION: originalIndex = -1!\n");
        }

        entry[lowestDomainListIndices[originalIndex]] = tempArray[bodyIndex + offset];

        offset += stride;
    }

}

__global__ void compLowestDomainListNodesKernel(float *x, float *y, float *z, float *mass, int *lowestDomainListIndices,
                                                int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                                unsigned long *sortedLowestDomainListKeys, int *counter) {

    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;
    int lowestDomainIndex;

    while ((bodyIndex + offset) < *lowestDomainListIndex-1) {

        lowestDomainIndex = lowestDomainListIndices[bodyIndex + offset];

        if (mass[lowestDomainIndex] != 0) {
            x[lowestDomainIndex] /= mass[lowestDomainIndex];
            y[lowestDomainIndex] /= mass[lowestDomainIndex];
            z[lowestDomainIndex] /= mass[lowestDomainIndex];
        }

        //printf("lowestDomainIndex = %i x = (%f, %f, %f) m = %f (key: %lu)\n", lowestDomainIndex, x[lowestDomainIndex],
          //     y[lowestDomainIndex], z[lowestDomainIndex], mass[lowestDomainIndex], lowestDomainListKeys[bodyIndex + offset]);

        offset += stride;
    }
}

__global__ void zeroDomainListNodesKernel(int *domainListIndex, int *domainListIndices,
                                          int *lowestDomainListIndex, int *lowestDomainListIndices,
                                          float *x, float *y, float *z, float *mass) {

    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;
    int domainIndex;
    bool zero;

    while ((bodyIndex + offset) < *domainListIndex) {
        zero = true;
        domainIndex = domainListIndices[bodyIndex + offset];
        for (int i=0; i<*lowestDomainListIndex-1; i++) {
            if (domainIndex = lowestDomainListIndices[i]) {
                zero = false;
            }
        }

        if (zero) {
            x[domainIndex] = 0.f;
            y[domainIndex] = 0.f;
            z[domainIndex] = 0.f;

            mass[domainIndex] = 0.f;
        }

        offset += stride;
    }
}

//TODO: lowest domain list nodes or domain list nodes?
__global__ void compLocalPseudoParticlesParKernel(float *x, float *y, float *z, float *mass, int *index, int n,
                                                  int *domainListIndices, int *domainListIndex,
                                                  int *lowestDomainListIndices, int *lowestDomainListIndex) {
    //equivalent to centreOfMassKernel !?
    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;
    bool domainList;
    //note: most of it already done within buildTreeKernel

    bodyIndex += n;

    while (bodyIndex + offset < *index) {
        domainList = false;

        for (int i=0; i<*domainListIndex; i++) {
            if ((bodyIndex + offset) == domainListIndices[i]) {
                domainList = true; // hence do not insert
                //for (int j=0; j<*lowestDomainListIndex; j++) {
                //    if ((bodyIndex + offset) == lowestDomainListIndices[j]) {
                //        domainList = false;
                //        break;
                //    }
                //}
                break;
            }
        }

        if (mass[bodyIndex + offset] != 0 && !domainList) {
            x[bodyIndex + offset] /= mass[bodyIndex + offset];
            y[bodyIndex + offset] /= mass[bodyIndex + offset];
            z[bodyIndex + offset] /= mass[bodyIndex + offset];
        }

        offset += stride;
    }
}

__global__ void compDomainListPseudoParticlesParKernel(float *x, float *y, float *z, float *mass, int *child, int *index, int n,
                                                       int *domainListIndices, int *domainListIndex,
                                                       int *domainListLevels, int *lowestDomainListIndices,
                                                       int *lowestDomainListIndex) {

    //calculate position (center of mass) and mass for domain list nodes
    //Problem: start with "deepest" nodes
    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset;
    int domainIndex;
    int level = 21; // max level
    bool compute;

    while (level >= 0) {
        offset = 0;
        compute = true;
        while ((bodyIndex + offset) < *domainListIndex) {
            compute = true;
            domainIndex = domainListIndices[bodyIndex + offset];
            for (int i=0; i<*lowestDomainListIndex-1; i++) {
                if (domainIndex == lowestDomainListIndices[i]) {
                    compute = false;
                }
            }
            if (compute && domainListLevels[bodyIndex + offset] == level) {
                // do the calculation
                for (int i=0; i<8; i++) {
                    x[domainIndex] += x[child[8*domainIndex + i]] * mass[child[8*domainIndex + i]];
                    y[domainIndex] += y[child[8*domainIndex + i]] * mass[child[8*domainIndex + i]];
                    z[domainIndex] += z[child[8*domainIndex + i]] * mass[child[8*domainIndex + i]];
                    mass[domainIndex] += mass[child[8*domainIndex + i]];
                }

                if (mass[domainIndex] != 0) {
                    x[domainIndex] /= mass[domainIndex];
                    y[domainIndex] /= mass[domainIndex];
                    z[domainIndex] /= mass[domainIndex];
                }

                //printf("domain node: key = %lu x = (%f, %f, %f) m = %f\n", domainListIndices[bodyIndex + offset],
                  //     x[domainIndex], y[domainIndex], z[domainIndex], mass[domainIndex]);
            }
            offset += stride;
        }
        __syncthreads();
        level--;
    }
}

// Kernel 3: computes the COM for each cell
__global__ void centreOfMassKernel(float *x, float *y, float *z, float *mass, int *index, int n)
{
    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;

    //note: most of it already done within buildTreeKernel
    bodyIndex += n;

    while (bodyIndex + offset < *index) {

        if (mass[bodyIndex + offset] == 0) {
            printf("centreOfMassKernel: mass = 0 (%i)!\n", bodyIndex + offset);
        }

        if (mass != 0) {
            x[bodyIndex + offset] /= mass[bodyIndex + offset];
            y[bodyIndex + offset] /= mass[bodyIndex + offset];
            z[bodyIndex + offset] /= mass[bodyIndex + offset];
        }

        offset += stride;
    }
}


// Kernel 4: sorts the bodies
__global__ void sortKernel(int *count, int *start, int *sorted, int *child, int *index, int n)
{
    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    if (bodyIndex == 0) {
        int sumParticles = 0;
        for (int i=0; i<8; i++) {
            sumParticles += count[child[i]];
        }
    }

    int s = 0;
    if (threadIdx.x == 0) {
        
        for (int i=0; i<8; i++){
            
            int node = child[i];
            // not a leaf node
            if (node >= n) {
                start[node] = s;
                s += count[node];
            }
            // leaf node
            else if (node >= 0) {
                sorted[s] = node;
                s++;
            }
        }
    }
    int cell = n + bodyIndex;
    int ind = *index;

    int counter = 0;
    while ((cell + offset) < ind /*&& counter < 100000*/) {
        counter++;
        
        s = start[cell + offset];

        if (s >= 0) {

            for (int i=0; i<8; i++) {
                int node = child[8*(cell+offset) + i];
                // not a leaf node
                if (node >= n) {
                    start[node] = s;
                    s += count[node];
                }
                // leaf node
                else if (node >= 0) {
                    sorted[s] = node;
                    s++;
                }
            }
            offset += stride;
        }
    }
}


// Kernel 5: computes the (gravitational) forces
__global__ void computeForcesKernel(float* x, float *y, float *z, float *vx, float *vy, float *vz, 
                                    float *ax, float *ay, float *az, float *mass,
                                    int *sorted, int *child, float *minX, float *maxX, int n, float g, int blockSize,
                                    int warp, int stackSize)
{
    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;

    //__shared__ float depth[stackSize * blockSize/warp];
    // stack controlled by one thread per warp
    //__shared__ int   stack[stackSize * blockSize/warp];
    extern __shared__ float buffer[];

    float* depth = (float*)buffer;
    float* stack = (float*)&depth[stackSize* blockSize/warp];

    float radius = 0.5*(*maxX - (*minX));

    // in case that one of the first 8 children are a leaf
    int jj = -1;
    for (int i=0; i<8; i++) {
        if (child[i] != -1) {
            jj++;
        }
    }

    int counter = threadIdx.x % warp;
    int stackStartIndex = stackSize*(threadIdx.x / warp);
    
    while (bodyIndex + offset < n) {
        
        int sortedIndex = sorted[bodyIndex + offset];

        float pos_x = x[sortedIndex];
        float pos_y = y[sortedIndex];
        float pos_z = z[sortedIndex];
        
        float acc_x = 0.0;
        float acc_y = 0.0;
        float acc_z = 0.0;

        // initialize stack
        int top = jj + stackStartIndex;
        
        if (counter == 0) {
            
            int temp = 0;
            
            for (int i=0; i<8; i++) {
                // if child is not locked
                if (child[i] != -1) {
                    stack[stackStartIndex + temp] = child[i];
                    depth[stackStartIndex + temp] = radius*radius/theta;
                    temp++;
                }
            }
        }
        __syncthreads();

        // while stack is not empty / more nodes to visit
        while (top >= stackStartIndex) {
            
            int node = stack[top];
            float dp = 0.25*depth[top]; // float dp = depth[top];
            
            for (int i=0; i<8; i++) {
                
                int ch = child[8*node + i];
                //__threadfence();

                if (ch >= 0) {
                    
                    float dx = x[ch] - pos_x;
                    float dy = y[ch] - pos_y;
                    float dz = z[ch] - pos_z;
                    
                    float r = dx*dx + dy*dy + dz*dz + eps_squared;

                    //unsigned activeMask = __activemask();

                    //if (ch < n /*is leaf node*/ || !__any_sync(activeMask, dp > r)) {
                    if (ch < n /*is leaf node*/ || __all_sync(__activemask(), dp <= r)) {

                        // calculate interaction force contribution
                        r = rsqrt(r);
                        float f = mass[ch] * r * r * r;

                        acc_x += f*dx;
                        acc_y += f*dy;
                        acc_z += f*dz;
                    }
                    else {
                        // if first thread in warp: push node's children onto iteration stack
                        if (counter == 0) {
                            stack[top] = ch;
                            depth[top] = dp; // depth[top] = 0.25*dp;
                        }
                        top++; // descend to next tree level
                        //__threadfence();
                    }
                }
                else { /*top = max(stackStartIndex, top-1); */}
            }
            top--;
        }
        // update body data
        ax[sortedIndex] = acc_x;
        ay[sortedIndex] = acc_y;
        az[sortedIndex] = acc_z;

        offset += stride;

        __syncthreads();
    }
}

__device__ float smallestDistance(float* x, float *y, float *z, int node1, int node2) {
    float dx;
    if (x[node1] < x[node2]) {
        dx = x[node2] - x[node1];
    }
    else if (x[node1] > x[node2]) {
        dx = x[node1] - x[node2];
    }
    else {
        dx = 0.f;
    }

    float dy;
    if (y[node1] < y[node2]) {
        dy = y[node2] - y[node1];
    }
    else if (y[node1] > y[node2]) {
        dy = y[node1] - y[node2];
    }
    else {
        dy = 0.f;
    }

    float dz;
    if (z[node1] < z[node2]) {
        dz = z[node2] - z[node1];
    }
    else if (z[node1] > z[node2]) {
        dz = z[node1] - z[node2];
    }
    else {
        dz = 0.f;
    }

    return sqrtf(dx*dx + dy*dy + dz*dz);
}

__global__ void collectSendIndicesKernel(int *sendIndices, float *entry, float *tempArray, int *domainListCounter,
                                   int sendCount) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;
    int insertIndex;

    while ((bodyIndex + offset) < sendCount) {
        tempArray[bodyIndex + offset] = entry[sendIndices[bodyIndex + offset]];
        offset += stride;
    }
}

//ATTENTION: causes duplicate entries, which need to be removed afterwards
__global__ void symbolicForceKernel(int relevantIndex, float *x, float *y, float *z, float *minX, float *maxX, float *minY,
                                    float *maxY, float *minZ, float *maxZ, int *child, int *domainListIndex,
                              unsigned long *domainListKeys, int *domainListIndices, int *domainListLevels,
                              int *domainListCounter, int *sendIndices, int *index, int *particleCounter,
                              SubDomainKeyTree *s, int n, int m, float diam, float theta, int *mutex) {


    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;
    float r;
    int insertIndex;
    bool insert;
    int level;
    int childIndex;
    bool redo = false;

    while ((bodyIndex + offset) < *index) {
        insert = true;
        redo = false;

        if ((bodyIndex + offset) != relevantIndex && ((bodyIndex + offset) < particleCounter[s->rank] || (bodyIndex + offset) > n)) {
            r = smallestDistance(x, y, z, bodyIndex + offset, relevantIndex); //relevantIndex, bodyIndex + offset);
            level = getTreeLevel(bodyIndex + offset, child, x, y, z, minX, maxX, minY, maxY, minZ, maxZ);

            if ((powf(0.5, level) * diam) >= (theta * r) && level>=0) {
                //TODO: insert cell itself or children?

                /// inserting cell itself
                //check whether node is a domain list node
                for (int i=0; i<*domainListIndex; i++) {
                    if ((bodyIndex + offset) == domainListIndices[i]) {
                        insert = false;
                        break;
                        //printf("domain list nodes do not need to be sent!\n");
                    }
                }
                if (insert) {
                    //add to indices to be sent
                    insertIndex = atomicAdd(domainListCounter, 1);
                    sendIndices[insertIndex] = bodyIndex + offset;
                }
                else {

                }

                /// inserting children
                /*for (int i=0; i<8; i++) {
                    childIndex = child[8*(bodyIndex + offset) + i];
                    //check whether node is already within the indices to be sent
                    for (int i = 0; i < *domainListCounter; i++) {
                        if (childIndex == sendIndices[i]) {
                            insert = false;
                            //printf("already saved to be sent!\n");
                        }
                    }
                    //check whether node is a domain list node
                    for (int i = 0; i < *domainListIndex; i++) {
                        if (childIndex == domainListIndices[i]) {
                            insert = false;
                            //printf("domain list nodes do not need to be sent!\n");
                        }
                    }
                    if (insert && childIndex != -1) {
                        //add to indices to be sent
                        insertIndex = atomicAdd(domainListCounter, 1);
                        sendIndices[insertIndex] = childIndex;
                    }
                }*/
            }
        }
        else {
            //no particle to examine...
        }
        offset += stride;
    }
}

//reset domainListCounter after compTheta!!!
__global__ void compThetaKernel(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                          float *minZ, float *maxZ, int *domainListIndex, int *domainListCounter,
                          unsigned long *domainListKeys, int *domainListIndices, int *domainListLevels,
                          int *relevantDomainListIndices, SubDomainKeyTree *s) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;
    int bodyIndex = 0;
    unsigned long key;
    int domainIndex;

    //"loop" over domain list nodes
    while ((index + offset) < *domainListIndex) {

        bodyIndex = domainListIndices[index + offset];

        //calculate key
        key = getParticleKeyPerParticle(x[bodyIndex], y[bodyIndex], z[bodyIndex], minX, maxX, minY, maxY,
                                        minZ, maxZ, 21);

        //if domain list node belongs to other process: add to relevant domain list indices
        if (key2proc(key, s) != s->rank) {
            domainIndex = atomicAdd(domainListCounter, 1);
            relevantDomainListIndices[domainIndex] = bodyIndex;
            //printf("relevant domain list index: %i\n", bodyIndex);
        }
        offset += stride;
    }

}


// Kernel 6: updates the bodies
__global__ void updateKernel(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                             float *ax, float *ay, float *az, int n, float dt, float d) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    while (bodyIndex + offset < n) {

        vx[bodyIndex + offset] += dt * ax[bodyIndex + offset];
        vy[bodyIndex + offset] += dt * ay[bodyIndex + offset];
        vz[bodyIndex + offset] += dt * az[bodyIndex + offset];

        x[bodyIndex + offset] += d * dt * vx[bodyIndex + offset];
        y[bodyIndex + offset] += d * dt * vy[bodyIndex + offset];
        z[bodyIndex + offset] += d * dt * vz[bodyIndex + offset];

        offset += stride;
    }
}


//TODO: only update/calculate COM for not domain list nodes?!
__global__ void insertReceivedParticlesKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int *to_delete_leaf, int n, int m) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    //note: -1 used as "null pointer"
    //note: -2 used to lock a child (pointer)

    int offset;
    bool newBody = true;

    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float min_z;
    float max_z;

    int childPath;
    int temp;

    offset = 0;

    bodyIndex += to_delete_leaf[0]; //TODO: would be possible but shouldn't change something

    //if ((bodyIndex + offset) % 10000 == 0) {
    //    printf("index = %i x = (%f, %f, %f)\n", bodyIndex + offset, x[bodyIndex + offset], y[bodyIndex + offset], z[bodyIndex + offset]);
    //}

    while ((bodyIndex + offset) < to_delete_leaf[1] && (bodyIndex + offset) > to_delete_leaf[0]) {

        //debugging
        if ((bodyIndex + offset) % 100 == 0) {
            printf("index = %i x = (%f, %f, %f) (index = %i) to_delete_leaf = (%i, %i)\n", bodyIndex + offset, x[bodyIndex + offset], y[bodyIndex + offset], z[bodyIndex + offset], *index, to_delete_leaf[0], to_delete_leaf[1]);
            //printf("index = %i x = (%f, %f, %f) (index = %i) to_delete_leaf = (%i, %i)\n", bodyIndex + offset - 10000, x[bodyIndex + offset - 10000], y[bodyIndex + offset-10000], z[bodyIndex + offset-10000], *index, to_delete_leaf[0], to_delete_leaf[1]);
        }

        for (int i=to_delete_leaf[0]; i<to_delete_leaf[1]; i++) {
            if (i != (bodyIndex + offset)) {
                if (x[i] == x[bodyIndex + offset]) {
                    //printf("ATTENTION: x[%i] = (%f, %f, %f) vs. x[%i] = (%f, %f, %f)\n", i, x[i], y[i], z[i],
                    //       bodyIndex + offset,  x[bodyIndex + offset], y[bodyIndex + offset], z[bodyIndex + offset]);
                }
            }
        }

        //debugging
        //offset += stride;

        if (newBody) {

            newBody = false;

            min_x = *minX;
            max_x = *maxX;
            min_y = *minY;
            max_y = *maxY;
            min_z = *minZ;
            max_z = *maxZ;

            temp = 0;
            childPath = 0;

            // find insertion point for body
            if (x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                childPath += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
            if (y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                childPath += 2;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }
            if (z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {  // z direction
                childPath += 4;
                max_z = 0.5 * (min_z + max_z);
            }
            else {
                min_z = 0.5 * (min_z + max_z);
            }
        }

        int childIndex = child[temp*8 + childPath];

        // traverse tree until hitting leaf node
        while (childIndex >= m /*&& childIndex < (8*m)*/) { //n //TODO: check

            temp = childIndex;

            childPath = 0;

            // find insertion point for body
            if (x[bodyIndex + offset] < 0.5 * (min_x + max_x)) { // x direction
                childPath += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
            if (y[bodyIndex + offset] < 0.5 * (min_y + max_y)) { // y direction
                childPath += 2;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }
            if (z[bodyIndex + offset] < 0.5 * (min_z + max_z)) { // z direction
                childPath += 4;
                max_z = 0.5 * (min_z + max_z);
            }
            else {
                min_z = 0.5 * (min_z + max_z);
            }

            if (mass[bodyIndex + offset] != 0) {
                atomicAdd(&x[temp], mass[bodyIndex + offset] * x[bodyIndex + offset]);
                atomicAdd(&y[temp], mass[bodyIndex + offset] * y[bodyIndex + offset]);
                atomicAdd(&z[temp], mass[bodyIndex + offset] * z[bodyIndex + offset]);
            }

            atomicAdd(&mass[temp], mass[bodyIndex + offset]);
            //atomicAdd(&count[temp], 1);

            childIndex = child[8*temp + childPath];

            //if ((bodyIndex+offset) % 100 == 0) {
            //    printf("bodyIndex + offset = %i -> temp = %i\n", bodyIndex + offset, temp);
            //}
        }

        // if child is not locked
        if (childIndex != -2) {

            int locked = temp * 8 + childPath;

            //lock
            if (atomicCAS(&child[locked], childIndex, -2) == childIndex) {

                // check whether a body is already stored at the location
                if (childIndex == -1) {
                    //insert body and release lock
                    child[locked] = bodyIndex + offset;
                }
                else {
                    int patch = 8 * m; //8*n
                    while (childIndex >= 0 && childIndex < m) {//TODO: was n

                        //debug
                        if (x[childIndex] == x[bodyIndex + offset]) {
                            printf("ATTENTION (shouldn't happen...): x[%i] = (%f, %f, %f) vs. x[%i] = (%f, %f, %f) | to_delete_leaf = (%i, %i)\n",
                                   childIndex, x[childIndex], y[childIndex], z[childIndex], bodyIndex + offset,  x[bodyIndex + offset],
                                   y[bodyIndex + offset], z[bodyIndex + offset], to_delete_leaf[0], to_delete_leaf[1]);
                        }

                        //create a new cell (by atomically requesting the next unused array index)
                        int cell = atomicAdd(index, 1);

                        patch = min(patch, cell);

                        if (patch != cell) {
                            child[8 * temp + childPath] = cell;
                        }

                        // insert old/original particle
                        childPath = 0;
                        if (x[childIndex] < 0.5 * (min_x + max_x)) { childPath += 1; }
                        if (y[childIndex] < 0.5 * (min_y + max_y)) { childPath += 2; }
                        if (z[childIndex] < 0.5 * (min_z + max_z)) { childPath += 4; }

                        x[cell] += mass[childIndex] * x[childIndex];
                        y[cell] += mass[childIndex] * y[childIndex];
                        z[cell] += mass[childIndex] * z[childIndex];

                        mass[cell] += mass[childIndex];
                        count[cell] += count[childIndex];

                        child[8 * cell + childPath] = childIndex;

                        start[cell] = -1; //TODO: needed?

                        // insert new particle
                        temp = cell;
                        childPath = 0;

                        // find insertion point for body
                        if (x[bodyIndex + offset] < 0.5 * (min_x + max_x)) {
                            childPath += 1;
                            max_x = 0.5 * (min_x + max_x);
                        } else {
                            min_x = 0.5 * (min_x + max_x);
                        }
                        if (y[bodyIndex + offset] < 0.5 * (min_y + max_y)) {
                            childPath += 2;
                            max_y = 0.5 * (min_y + max_y);
                        } else {
                            min_y = 0.5 * (min_y + max_y);
                        }
                        if (z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {
                            childPath += 4;
                            max_z = 0.5 * (min_z + max_z);
                        } else {
                            min_z = 0.5 * (min_z + max_z);
                        }

                        // COM / preparing for calculation of COM
                        if (mass[bodyIndex + offset] != 0) {
                            x[cell] += mass[bodyIndex + offset] * x[bodyIndex + offset];
                            y[cell] += mass[bodyIndex + offset] * y[bodyIndex + offset];
                            z[cell] += mass[bodyIndex + offset] * z[bodyIndex + offset];
                            mass[cell] += mass[bodyIndex + offset];
                        }
                        count[cell] += count[bodyIndex + offset];
                        childIndex = child[8 * temp + childPath];
                    }

                    child[8 * temp + childPath] = bodyIndex + offset;

                    __threadfence();  // written to global memory arrays (child, x, y, mass) thus need to fence
                    child[locked] = patch;
                }
                offset += stride;
                newBody = true;
            }
            else {

            }
        }
        else {

        }
        __syncthreads();
    }
}

//TODO: not tested yet!
__global__ void repairTreeKernel(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                           float *ax, float *ay, float *az, float *mass, int *count, int *start,
                           int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                           float *minZ, float *maxZ, int *to_delete_cell, int *to_delete_leaf,
                           int *domainListIndices, int n, int m) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    //delete inserted leaves
    while ((bodyIndex + offset) >= to_delete_leaf[0] && (bodyIndex + offset) < to_delete_leaf[1]) {
        for (int i=0; i<8; i++) {
            child[(bodyIndex + offset)*8 + i] = -1;
        }
        count[bodyIndex + offset] = 1;
        x[bodyIndex + offset] = 0;
        y[bodyIndex + offset] = 0;
        z[bodyIndex + offset] = 0;
        vx[bodyIndex + offset] = 0;
        vy[bodyIndex + offset] = 0;
        vz[bodyIndex + offset] = 0;
        ax[bodyIndex + offset] = 0;
        ay[bodyIndex + offset] = 0;
        az[bodyIndex + offset] = 0;
        mass[bodyIndex + offset] = 0;
        start[bodyIndex + offset] = -1;
        //sorted[bodyIndex + offset] = 0; //TODO: needed?

        offset += stride;
    }

    offset = 0;

    //delete inserted cells
    while ((bodyIndex + offset) >= to_delete_cell[0] && (bodyIndex + offset) < to_delete_cell[1]) {
        for (int i=0; i<8; i++) {
            child[(bodyIndex + offset)*8 + i] = -1;
        }
        count[bodyIndex + offset] = 0;
        x[bodyIndex + offset] = 0;
        y[bodyIndex + offset] = 0;
        z[bodyIndex + offset] = 0;
        vx[bodyIndex + offset] = 0;
        vy[bodyIndex + offset] = 0;
        vz[bodyIndex + offset] = 0;
        ax[bodyIndex + offset] = 0;
        ay[bodyIndex + offset] = 0;
        az[bodyIndex + offset] = 0;
        mass[bodyIndex + offset] = 0;
        start[bodyIndex + offset] = -1;
        //sorted[bodyIndex + offset] = 0;

        offset += stride;
    }

}


__device__ int getTreeLevel(int index, int *child, float *x, float *y, float *z, float *minX, float *maxX, float *minY,
                            float *maxY, float *minZ, float *maxZ) {

    unsigned long key = getParticleKeyPerParticle(x[index], y[index], z[index], minX, maxX, minY, maxY, minZ, maxZ, 21);
    //int proc = key2proc(key, s);
    int level = 0; //TODO: initialize with 0 or 1?
    int childIndex;

    int path[21];
    for (int i=0; i<21; i++) {
        path[i] = (int) (key >> (21*3 - 3 * (i + 1)) & (int)7);
    }

    childIndex = child[path[0]];

    //TODO: where to put level++?
    for (int i=1; i<21; i++) {
        level++;
        if (childIndex == index) {
            return level;
        }
        childIndex = child[8*childIndex + path[i]];
        //level++;
    }

    printf("ATTENTION: level = -1 (index = %i)\n", index);
    return -1;

}

__global__ void findDuplicatesKernel(float *array, float *array_2, int length, SubDomainKeyTree *s, int *duplicateCounter) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    while ((bodyIndex + offset) < length) {

        for (int i=0; i<length; i++) {
            if (i != (bodyIndex + offset)) {
                if (array[bodyIndex + offset] == array[i] && array_2[bodyIndex + offset] == array_2[i]) {
                    duplicateCounter[i] += 1;
                    printf("duplicate! (%i vs. %i) (x = %f, y = %f)\n", i, bodyIndex + offset, array[i], array_2[i]);
                }
            }
        }

        offset += stride;
    }

}

__global__ void markDuplicatesKernel(int *indices, float *x, float *y, float *z,
                                       float *mass, SubDomainKeyTree *s, int *counter, int length) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;
    int maxIndex;

    //remark: check only x, but in principle check all
    while ((bodyIndex + offset) < length) {
        if (indices[bodyIndex + offset] != -1) {
            for (int i = 0; i < length; i++) {
                if (i != (bodyIndex + offset)) {
                    if (x[indices[bodyIndex + offset]] == x[indices[i]] || indices[bodyIndex + offset] == indices[i]) {
                        maxIndex = max(bodyIndex + offset, i);
                        indices[maxIndex] = -1;
                        atomicAdd(counter, 1);
                    }
                }

            }
        }
        __syncthreads();
        offset += stride;
    }
}

__global__ void removeDuplicatesKernel(int *indices, int *removedDuplicatesIndices, int *counter, int length) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    int indexToInsert;

    while ((bodyIndex + offset) < length) {

        if (indices[bodyIndex + offset] != -1) {
            indexToInsert = atomicAdd(counter, 1);
            removedDuplicatesIndices[indexToInsert] = indices[bodyIndex + offset];
        }

        offset += stride;
    }
}