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
__device__ const float theta = 1.5; //0.5; //1.5; //0.5;

__global__ void MyKernel(int *array, int arrayCount)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arrayCount)
    {
        array[idx] *= array[idx];
    }
}

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
                                          int *lowestDomainListLevels,
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
            lowestDomainListLevels[bodyIndex + offset] = -1;
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

// (currently) not needed: mass, count, start, child, index, (counter)
__global__ void particlesPerProcessKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                    int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                    float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s,
                                    int *procCounter, int *procCounterTemp, int curveType) {

    //go over domain list (only the ones inherited by own process) and count particles (using count array)
    //BUT: for now use this approach!
    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int offset = 0;

    unsigned long key;
    int proc;

    while ((bodyIndex + offset) < n) {

        // calculate particle key from particle's position
        key = getParticleKeyPerParticle(x[bodyIndex + offset], y[bodyIndex + offset], z[bodyIndex + offset],
                                        minX, maxX, minY, maxY, minZ, maxZ, 21);
        // get corresponding process
        proc = key2proc(key, s, curveType);

        // increment corresponding counter
        atomicAdd(&procCounter[proc], 1);

        offset += stride;
    }
}

// (currently) not needed: mass, count, start, child, index, (counter)
__global__ void markParticlesProcessKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                           int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                           float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s,
                                           int *procCounter, int *procCounterTemp, int *sortArray, int curveType) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int offset = 0;

    unsigned long key;
    int proc;
    int counter;

    while ((bodyIndex + offset) < n) {

        // calculate particle key from particle's position
        key = getParticleKeyPerParticle(x[bodyIndex + offset], y[bodyIndex + offset], z[bodyIndex + offset],
                                        minX, maxX, minY, maxY, minZ, maxZ, 21);
        // get corresponding process
        proc = key2proc(key, s, curveType);

        /*// increment corresponding counter
        counter = atomicAdd(&procCounterTemp[proc], 1)
        if (proc > 0) {
            sortArray[bodyIndex + offset] = procCounter[proc-1] + counter;
        }
        else {
            sortArray[bodyIndex + offset] = counter;
        }*/

        // mark particle with corresponding process
        sortArray[bodyIndex + offset] = proc;

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

//TODO: use template function
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
__global__ void debugKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int n, int m, SubDomainKeyTree *s, int *procCounter,
                                float *tempArray, int *sortArray, int *sortArrayOut) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int offset = n - 100;

    while ((bodyIndex + offset) < n + 100) {
        printf("x[%i > (%i) = %i] = (%f, %f, %f)\n", bodyIndex+offset, n, (bodyIndex + offset) > n, x[bodyIndex + offset], x[bodyIndex+offset], z[bodyIndex+offset]);
        offset += stride;
    }
    /*for (int i=0; i<8; i++) {
        printf("child[%i] = %i\n", i, child[i]);
        for (int k=0; k<8; k++) {
            printf("\tchild[8*child[%i] + %i] = %i\n", i, k, child[8*child[i] + k]);
        }
    }*/
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

            // copy bounding box
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
                    while (childIndex >= 0 && childIndex < n) { // was n

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

// (currently) not needed: start
// idea: assign already existing domain list nodes and add missing ones
__global__ void buildDomainTreeKernel(int *domainListIndex, unsigned long *domainListKeys, int *domainListLevels,
                                      int *domainListIndices, float *x, float *y, float *z, float *mass, float *minX,
                                      float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int *count,
                                      int *start, int *child, int *index, int n, int m) {

    int domainListCounter = 0;

    //char keyAsChar[21 * 2 + 3];
    int path[21];

    float min_x, max_x, min_y, max_y, min_z, max_z;
    int currentChild;
    int childPath;
    bool insert = true;

    int childIndex;
    int temp;

    // loop over domain list indices (over the keys found/generated by createDomainListKernel)
    for (int i = 0; i < *domainListIndex; i++) {
        //key2Char(domainListKeys[i], 21, keyAsChar);
        //printf("buildDomainTree: domainListLevels[%i] = %i\n", i, domainListLevels[i]);
        //printf("domain: domainListKeys[%i] = %lu = %s (level: %i)\n", i, domainListKeys[i], keyAsChar, domainListLevels[i]);
        childIndex = 0;
        //temp = 0;
        // iterate through levels (of corresponding domainListIndex)
        for (int j = 0; j < domainListLevels[i]; j++) {
            path[j] = (int) (domainListKeys[i] >> (21 * 3 - 3 * (j + 1)) & (int)7);
            temp = childIndex;
            childIndex = child[8*childIndex + path[j]];
            if (childIndex < n) {
                if (childIndex == -1 /*&& childIndex < n*/) {
                    // no child at all here, thus add node
                    int cell = atomicAdd(index, 1);
                    child[8 * temp + path[j]] = cell;
                    childIndex = cell;
                    domainListIndices[domainListCounter] = childIndex; //cell;
                    domainListCounter++;
                } else {
                    // child is a leaf, thus add node in between
                    int cell = atomicAdd(index, 1);
                    child[8 * /*childIndex*/temp + path[j]] = cell;

                    //printf("\tchild[8*%i + %i] = %i\n", temp, path[j], cell);

                    min_x = *minX;
                    max_x = *maxX;
                    min_y = *minY;
                    max_y = *maxY;
                    min_z = *minZ;
                    max_z = *maxZ;

                    for (int k=0; k<=j; k++) {

                        currentChild = path[k];

                        //printf("adding path[%i] = %i (j = %i)\n", k, path[k], j);
                        if (currentChild % 2 != 0) {
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
                        if (currentChild == 4) {
                            max_z = 0.5 * (min_z + max_z);
                            currentChild -= 4;
                        }
                        else {
                            min_z = 0.5 * (min_z + max_z);
                        }
                        //printf("\t\t currentChild[%i] = %i   %i\n", k, currentChild, path[k]);
                    }
                    // insert old/original particle
                    childPath = 0; //(int) (domainListKeys[i] >> (21 * 3 - 3 * ((j+1) + 1)) & (int)7); //0; //currentChild; //0;
                    if (x[childIndex] < 0.5 * (min_x + max_x)) {
                        childPath += 1;
                        //max_x = 0.5 * (min_x + max_x);
                    }
                    //else {
                    //    min_x = 0.5 * (min_x + max_x);
                    //}
                    if (y[childIndex] < 0.5 * (min_y + max_y)) {
                        childPath += 2;
                        //max_y = 0.5 * (min_y + max_y);
                    }
                    //else {
                    //    min_y = 0.5 * (min_y + max_y);
                    //}
                    if (z[childIndex] < 0.5 * (min_z + max_z)) {
                        childPath += 4;
                        //max_z = 0.5 * (min_z + max_z);
                    }
                    //else {
                    //    min_z = 0.5 * (min_z + max_z);
                    //}

                    x[cell] += mass[childIndex] * x[childIndex];
                    y[cell] += mass[childIndex] * y[childIndex];
                    z[cell] += mass[childIndex] * z[childIndex];
                    mass[cell] += mass[childIndex];

                    //printf("path = %i\n", (int) (domainListKeys[i] >> (21 * 3 - 3 * ((j+1) + 1)) & (int)7));
                    //printf("j = %i, domainListLevels[%i] = %i\n", j, i, domainListLevels[i]);
                    printf("adding node in between for index %i  cell = %i (childPath = %i,  j = %i)! x = (%f, %f, %f)\n",
                           childIndex, cell, childPath, j, x[childIndex], y[childIndex], z[childIndex]);
                    //for (int l=0; l<=j; l++) {
                    //    printf("\tpath[%i] = %i\n", l, path[l]);
                    //}

                    child[8 * cell + childPath] = childIndex;
                    //printf("child[8 * %i + %i] = %i\n", cell, childPath, childIndex);

                    childIndex = cell;
                    domainListIndices[domainListCounter] = childIndex; //temp;
                    domainListCounter++;
                }
            }
            else {
                insert = true;
                // check whether node already marked as domain list node
                for (int k=0; k<domainListCounter; k++) {
                    if (childIndex == domainListIndices[k]) {
                        insert = false;
                        break;
                    }
                }
                if (insert) {
                    // mark/save node as domain list node
                    domainListIndices[domainListCounter] = childIndex; //temp;
                    domainListCounter++;
                }
            }
        }
    }
    //printf("domainListCounter = %i\n", domainListCounter);
}

// extract lowest domain list nodes from domain list nodes
// lowest domain list node = domain list node with children not being domain list nodes!
__global__ void lowestDomainListNodesKernel(int *domainListIndices, int *domainListIndex,
                                      unsigned long *domainListKeys,
                                      int *lowestDomainListIndices, int *lowestDomainListIndex,
                                      unsigned long *lowestDomainListKeys,
                                      int *domainListLevels, int *lowestDomainListLevels,
                                      float *x, float *y, float *z, float *mass, int *count, int *start,
                                      int *child, int n, int m, int *procCounter) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    bool lowestDomainListNode;
    int domainIndex;
    int lowestDomainIndex;
    int childIndex;

    // check all domain list nodes
    while ((bodyIndex + offset) < *domainListIndex) {
        lowestDomainListNode = true;
        // get domain list index of current domain list node
        domainIndex = domainListIndices[bodyIndex + offset];
        // check all children
        for (int i=0; i<8; i++) {
            childIndex = child[8 * domainIndex + i];
            // check whether child exists
            if (childIndex != -1) {
                // check whether child is a node
                if (childIndex >= n) {
                    // check if this node is a domain list node
                    for (int k=0; k<*domainListIndex; k++) {
                        if (childIndex == domainListIndices[k]) {
                            //printf("domainIndex = %i  childIndex: %i  domainListIndices: %i\n", domainIndex,
                            //       childIndex, domainListIndices[k]);
                            lowestDomainListNode = false;
                            break;
                        }
                    }
                    // one child being a domain list node is sufficient for not being a lowest domain list node
                    if (!lowestDomainListNode) {
                        break;
                    }
                }
            }
        }

        if (lowestDomainListNode) {
            // increment lowest domain list counter/index
            lowestDomainIndex = atomicAdd(lowestDomainListIndex, 1);
            // add/save index of lowest domain list node
            lowestDomainListIndices[lowestDomainIndex] = domainIndex;
            // add/save key of lowest domain list node
            lowestDomainListKeys[lowestDomainIndex] = domainListKeys[bodyIndex + offset];
            // add/save level of lowest domain list node
            lowestDomainListLevels[lowestDomainIndex] = domainListLevels[bodyIndex + offset];
            // debugging
            //printf("Adding lowest domain list node #%i (key = %lu)\n", lowestDomainIndex,
            //  lowestDomainListKeys[lowestDomainIndex]);
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

    // ---- check whether particles exist that do not belong to this process
    /*while ((bodyIndex + offset) < procCounter[0]) {

        key = getParticleKeyPerParticle(x[bodyIndex + offset], y[bodyIndex + offset], z[bodyIndex + offset], minX, maxX,
                                        minY, maxY, minZ, maxZ, 21);

        proc = key2proc(key, s);
        if (proc != s->rank) {
            printf("ATTENTION: myrank = %i and proc = %i (bodyIndex + offset = %i)\n", s->rank, proc,
                   bodyIndex + offset);
        }
        offset += stride;
    }*/
    // ----------------------------------------------------------------------

    while ((bodyIndex + offset) < 8) {
        printf("rank[%i] count[%i] = %i\n", s->rank, bodyIndex+offset, count[child[bodyIndex+offset]]);
        offset += stride;
    }

    /*// ---- general information about particles ....
    while ((bodyIndex + offset) < n) {
        if ((bodyIndex + offset) % 100000 == 0) {
            printf("particle[%i]: x = (%f, %f, %f)  m = %f\n", bodyIndex+offset, x[bodyIndex+offset],
                   y[bodyIndex+offset], z[bodyIndex+offset], mass[bodyIndex+offset]);
        }
        offset += stride;
    }
    // ----------------------------------------------------------------------*/

}

__global__ void domainListInfoKernel(float *x, float *y, float *z, float *mass, int *child, int *index, int n,
                               int *domainListIndices, int *domainListIndex,
                               int *domainListLevels, int *lowestDomainListIndices,
                               int *lowestDomainListIndex, SubDomainKeyTree *s) {

    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;

    int proc;

    //while ((bodyIndex + offset) < *domainListIndex) {
        /*printf("[rank %i] domainListIndices[%i] = %i  x = (%f, %f, %f) m = %f\n", s->rank, bodyIndex + offset,
               domainListIndices[bodyIndex + offset], x[domainListIndices[bodyIndex + offset]],
               y[domainListIndices[bodyIndex + offset]], z[domainListIndices[bodyIndex + offset]],
               mass[domainListIndices[bodyIndex + offset]]);*/

        /*if (mass[domainListIndices[bodyIndex + offset]] == 0.f) {
            for (int i=0; i<8; i++) {
                printf("[rank %i] domainListIndices[%i] child[%i] = %i\n", s->rank, bodyIndex + offset, i,
                       child[8*domainListIndices[bodyIndex + offset] + i]);
            }
        }*/

        //offset += stride;
    //}

    while ((bodyIndex + offset) < *lowestDomainListIndex) {
        printf("[rank %i] lowestDomainListIndices[%i] = %i x = (%f, %f, %f) m = %f\n", s->rank, bodyIndex + offset,
               lowestDomainListIndices[bodyIndex + offset], x[lowestDomainListIndices[bodyIndex + offset]],
               y[lowestDomainListIndices[bodyIndex + offset]], z[lowestDomainListIndices[bodyIndex + offset]],
               mass[lowestDomainListIndices[bodyIndex + offset]]);

        offset += stride;
    }


}

// convert key (unsigned long) to more readable level-wise (and separated) string/char-array
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

/*// table needed to convert from Lebesgue to Hilbert keys
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
                                                       {2,1,3,0,5,6,4,7}, {4,7,5,6,3,0,2,1}, {6,5,7,4,1,2,0,3} };*/

// convert Lebesgue key to Hilbert key
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

// calculate particle key (Lebesgue) per particle based on position (resulting in a overdetermined key)
__device__ unsigned long getParticleKeyPerParticle(float x, float y, float z,
                                                   float *minX, float *maxX, float *minY,
                                                   float *maxY, float *minZ, float *maxZ,
                                                   int maxLevel) {

    int level = 0;
    unsigned long particleKey = 0UL;

    int sonBox = 0;
    float min_x = *minX;
    float max_x = *maxX;
    float min_y = *minY;
    float max_y = *maxY;
    float min_z = *minZ;
    float max_z = *maxZ;

    // calculate path to the particle's position assuming an octree with above bounding boxes
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

        particleKey = particleKey | ((unsigned long)sonBox << (unsigned long)(3 * (maxLevel-level-1)));
        level ++;
    }
    //TODO: Hilbert change
    return particleKey;
    //return Lebesgue2Hilbert(particleKey, 21);
}

// only for testing...
// calculating the key for all particles
__global__ void getParticleKeyKernel(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                               float *minZ, float *maxZ, unsigned long *key, int maxLevel, int n, SubDomainKeyTree *s) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    unsigned long particleKey;
    unsigned long hilbertParticleKey;
    //char keyAsChar[21 * 2 + 3];

    /*//debugging
    if (bodyIndex == 0) {
        char rangeAsChar[21 * 2 + 3];
        for (int i=0; i<(s->numProcesses + 1); i++) {
            key2Char(s->range[i], 21, rangeAsChar);
            printf("range[%i] = %lu (%s)\n", i, s->range[i], rangeAsChar);
        }
    }
    //end: debugging*/

    while (bodyIndex + offset < n) {

        particleKey = 0UL;

        particleKey = getParticleKeyPerParticle(x[bodyIndex + offset], y[bodyIndex + offset], z[bodyIndex + offset],
                                            minX, maxX, minY, maxY, minZ, maxZ, maxLevel);

        //char keyAsChar[21 * 2 + 3];
        hilbertParticleKey = Lebesgue2Hilbert(particleKey, 21);
        key[bodyIndex + offset] = particleKey; //hilbertParticleKey;
        //int proc = key2proc(particleKey, s);
        //key2Char(testKey, 21, keyAsChar);
        //key2Char(hilbertParticleKey, 21, keyAsChar);
        //if ((bodyIndex + offset) % 5000 == 0) {
            //printf("key[%i]: %lu\n", bodyIndex + offset, testKey);
            //for (int proc=0; proc<=s->numProcesses; proc++) {
            //    printf("range[%i] = %lu\n", proc, s->range[proc]);
            //}
            //printf("key[%i]: %s  =  %lu (proc = %i)\n", bodyIndex + offset, keyAsChar, particleKey, proc);
        //}

        offset += stride;
    }
}

// get the corresponding process of a key (using the range within the SubDomainKeyTree)
__device__ int key2proc(unsigned long k, SubDomainKeyTree *s, int curveType) {

    if (curveType == 0) {
        for (int proc=0; proc<s->numProcesses; proc++) {
            if (k >= s->range[proc] && k < s->range[proc+1]) {
                return proc;
            }
        }
    }
    else {
        unsigned long hilbert = Lebesgue2Hilbert(k, 21);
        for (int proc = 0; proc < s->numProcesses; proc++) {
            if (hilbert >= s->range[proc] && hilbert < s->range[proc + 1]) {
                return proc;
            }
        }
    }
    //printf("ERROR: key2proc(k=%lu): -1!", k);
    return -1; // error
}

// Traversing the tree iteratively using an explicit stack
// not used (yet)
__global__ void traverseIterativeKernel(float *x, float *y, float *z, float *mass, int *child, int n, int m,
                         SubDomainKeyTree *s, int maxLevel) {

    // starting traversing with the child[0, ..., 7] representing the first level of the tree

    // explicit stack using shared memory
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

// get the domain list keys (and levels) resulting from ranges (within the SubDomainKeyTree)
// domain list nodes = common coarse tree for all processes
__global__ void createDomainListKernel(SubDomainKeyTree *s, int maxLevel, unsigned long *domainListKeys, int *levels,
                                       int *index, int curveType) {

    char keyAsChar[21 * 2 + 3];

    // workaround for fixing bug... in principle: unsigned long keyMax = (1 << 63) - 1;
    unsigned long shiftValue = 1;
    unsigned long toShift = 63;
    unsigned long keyMax = (shiftValue << toShift) - 1; // 1 << 63 not working!
    //key2Char(keyMax, 21, keyAsChar); //printf("keyMax: %lu = %s\n", keyMax, keyAsChar);

    unsigned long key2test = 0UL;

    int level = 0;

    level++;

    // in principle: traversing a (non-existent) octree by walking the 1D spacefilling curve (keys of the tree nodes)
    while (key2test < keyMax) {
        if (isDomainListNode(key2test & (~0UL << (3 * (maxLevel - level + 1))), maxLevel, level-1, s, curveType)) {
            // add domain list key
            domainListKeys[*index] = key2test;
            // add domain list level
            levels[*index] = level;
            *index += 1;
            if (isDomainListNode(key2test, maxLevel, level, s, curveType)) {
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
    //for (int i=0; i < *index; i++) {
    //    key2Char(domainListKeys[i], 21, keyAsChar);
    //}
}

// check whether node is a domain list node
__device__ bool isDomainListNode(unsigned long key, int maxLevel, int level, SubDomainKeyTree *s, int curveType) {
    int p1 = key2proc(key, s, curveType);
    int p2 = key2proc(key | ~(~0UL << 3*(maxLevel-level)), s, curveType);
    if (p1 != p2) {
        return true;
    }
    else {
        return false;
    }
}

// get the maximal key of a key regarding a specific level
__device__ unsigned long keyMaxLevel(unsigned long key, int maxLevel, int level, SubDomainKeyTree *s) {
    unsigned long keyMax = key | ~(~0UL << 3*(maxLevel-level));
    return keyMax;
}

__global__ void prepareLowestDomainExchangeKernel(float *entry, float *mass, float *tempArray, int *lowestDomainListIndices,
                                                  int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                                  int *counter) {

    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;

    int index;
    int lowestDomainIndex;

    //copy x, y, z, mass of lowest domain list nodes into arrays
    //sorting using cub (not here)
    while ((bodyIndex + offset) < *lowestDomainListIndex) {
        //if (bodyIndex + offset == 0) {
        //    printf("lowestDomainListIndex = %i\n", *lowestDomainListIndex);
        //}
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

//TODO: it is not necessary to calculate the moment (x_i * m), thus just use prepareLowestDomainExchangeKernel?
__global__ void prepareLowestDomainExchangeMassKernel(float *mass, float *tempArray, int *lowestDomainListIndices,
                                                  int *lowestDomainListIndex, unsigned long *lowestDomainListKeys,
                                                  int *counter) {

    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;
    int index;
    int lowestDomainIndex;

    //copy x, y, z, mass of lowest domain list nodes into arrays
    //sorting using cub (not here)
    while ((bodyIndex + offset) < *lowestDomainListIndex) {
        lowestDomainIndex = lowestDomainListIndices[bodyIndex + offset];
        if (lowestDomainIndex >= 0) {
            tempArray[bodyIndex + offset] = mass[lowestDomainIndex];
            printf("lowestDomainListIndex[%i]: mass = %f\n", bodyIndex+offset, tempArray[bodyIndex + offset]);
        }
        offset += stride;
    }
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
            printf("ATTENTION: originalIndex = -1 (index = %i)!\n", sortedLowestDomainListKeys[bodyIndex + offset]);
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

    while ((bodyIndex + offset) < *lowestDomainListIndex) {

        lowestDomainIndex = lowestDomainListIndices[bodyIndex + offset];

        if (mass[lowestDomainIndex] != 0) {
            x[lowestDomainIndex] /= mass[lowestDomainIndex];
            y[lowestDomainIndex] /= mass[lowestDomainIndex];
            z[lowestDomainIndex] /= mass[lowestDomainIndex];
        }

        // debugging
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

//TODO: check functionality
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

    // go from max level to level=0
    while (level >= 0) {
        offset = 0;
        compute = true;
        while ((bodyIndex + offset) < *domainListIndex) {
            compute = true;
            domainIndex = domainListIndices[bodyIndex + offset];
            for (int i=0; i<*lowestDomainListIndex; i++) {
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

                // debugging
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
__global__ void sortKernel(int *count, int *start, int *sorted, int *child, int *index, int n, int m)
{
    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    if (bodyIndex == 0) {
        int sumParticles = 0;
        for (int i=0; i<8; i++) {
            sumParticles += count[child[i]];
        }
        printf("sumParticles = %i\n", sumParticles);
    }

    int s = 0;
    if (threadIdx.x == 0) {
        
        for (int i=0; i<8; i++){
            
            int node = child[i];
            // not a leaf node
            if (node >= m) { //n
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
    int cell = m + bodyIndex;
    int ind = *index;

    //int counter = 0; // for debugging purposes or rather to achieve the kernel to be finished
    while ((cell + offset) < ind /*&& counter < 100000*/) {
        //counter++;

        //if (counter > 99998) {
            //printf("cell + offset = %i\n", cell+offset);
        //}
        
        s = start[cell + offset];

        if (s >= 0) {

            for (int i=0; i<8; i++) {
                int node = child[8*(cell+offset) + i];
                // not a leaf node
                if (node >= m) { //m
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
                                    int *sorted, int *child, float *minX, float *maxX, float *minY, float *maxY,
                                    float *minZ, float *maxZ, int n, int m,
                                    float g, int blockSize, int warp, int stackSize, SubDomainKeyTree *s)
{
    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;

    //debug
    unsigned long key;

    //__shared__ float depth[stackSize * blockSize/warp];
    // stack controlled by one thread per warp
    //__shared__ int   stack[stackSize * blockSize/warp];
    extern __shared__ float buffer[];

    float* depth = (float*)buffer;
    float* stack = (float*)&depth[stackSize* blockSize/warp];

    float x_radius = 0.5*(*maxX - (*minX));
    float y_radius = 0.5*(*maxY - (*minY));
    float z_radius = 0.5*(*maxZ - (*minZ));

    float radius_max = fmaxf(x_radius, y_radius);
    float radius = fmaxf(radius_max, z_radius);

    // in case that one of the first 8 children are a leaf
    int jj = -1;
    for (int i=0; i<8; i++) {
        if (child[i] != -1) {
            jj++;
        }
    }

    int counter = threadIdx.x % warp;
    int stackStartIndex = stackSize*(threadIdx.x / warp);
    
    while ((bodyIndex + offset) < m) {
        
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
            //debug
            //if (node > n && node < m) {
            //    printf("PARALLEL FORCE! (node = %i x = (%f, %f, %f) m = %f)\n", node, x[node], y[node], z[node],
            //        mass[node]);
            //}
            //end: debug
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

                        /*//debug
                        key = getParticleKeyPerParticle(x[ch], y[ch], z[ch], minX, maxX, minY, maxY,
                                                        minZ, maxZ, 21);
                        if (key2proc(key, s) != s->rank) {
                            printf("Parallel force! child = %i x = (%f, %f, %f) mass = %f\n", ch, x[ch], y[ch], z[ch], mass[ch]);
                        }
                        //end: debug*/

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

// calculating the smallest distance between two nodes
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

//copy non-contiguous array elements into another array contiguously (in order to send them via MPI)
// e.g.: [5, 6, 3, 6, 6, 8] -> relevant indices = [1, 5] -> [6, 8]
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
__global__ void symbolicForceKernel(int relevantIndex, float *x, float *y, float *z, float *mass, float *minX, float *maxX, float *minY,
                                    float *maxY, float *minZ, float *maxZ, int *child, int *domainListIndex,
                              unsigned long *domainListKeys, int *domainListIndices, int *domainListLevels,
                              int *domainListCounter, int *sendIndices, int *index, int *particleCounter,
                              SubDomainKeyTree *s, int n, int m, float diam, float theta_, int *mutex,
                                    int *relevantDomainListIndices) {


    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;
    float r;
    int insertIndex;
    bool insert;
    int level;
    int childIndex;
    //bool redo = false;

    while ((bodyIndex + offset) < *index) {

        //if ((bodyIndex + offset) == 0) {
        //    printf("relevantIndex: %i\n", relevantDomainListIndices[relevantIndex]);
        //}

        insert = true;
        //redo = false;

        for (int i=0; i<*domainListIndex; i++) {
            if ((bodyIndex + offset) == domainListIndices[i]) {
                insert = false;
                break;
            }
        }

        //if (mass[relevantDomainListIndices[relevantIndex]] == 0) {
        //    insert = false;
        //}

        // TODO: CHANGED: relevantIndex -> relevantDomainListIndices[relevantIndex]
        if (insert && (bodyIndex + offset) != relevantDomainListIndices[relevantIndex] && ((bodyIndex + offset) < particleCounter[s->rank] || (bodyIndex + offset) > n)) {

            //r = smallestDistance(x, y, z, bodyIndex + offset, relevantDomainListIndices[relevantIndex]); //relevantIndex, bodyIndex + offset);
            r = smallestDistance(x, y, z, relevantDomainListIndices[relevantIndex], bodyIndex + offset);

            //calculate tree level by determining the particle's key and traversing the tree until hitting that particle
            level = getTreeLevel(bodyIndex + offset, child, x, y, z, minX, maxX, minY, maxY, minZ, maxZ);

            if ((powf(0.5, level) * diam) >= (theta_ * r) && level >= 0) {
                //TODO: insert cell itself or children?

                /// inserting cell itself
                /*//check whether node is a domain list node
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

                }*/

                /// inserting children
                for (int i=0; i<8; i++) {
                    childIndex = child[8*(bodyIndex + offset) + i];
                    //check whether node is already within the indices to be sent
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
                }
            }
        }
        else {
            //no particle to examine...
        }
        offset += stride;
    }
}

//reset domainListCounter after compTheta!
__global__ void compThetaKernel(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                          float *minZ, float *maxZ, int *domainListIndex, int *domainListCounter,
                          unsigned long *domainListKeys, int *domainListIndices, int *domainListLevels,
                          int *relevantDomainListIndices, SubDomainKeyTree *s, int curveType) {

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
        if (key2proc(key, s, curveType) != s->rank) {
            domainIndex = atomicAdd(domainListCounter, 1);
            relevantDomainListIndices[domainIndex] = bodyIndex;
            //printf("relevant domain list index: %i\n", bodyIndex);
        }
        offset += stride;
    }
}


// Kernel 6: updates the bodies/particles
__global__ void updateKernel(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                             float *ax, float *ay, float *az, int n, float dt, float d) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    while (bodyIndex + offset < n) {

        // calculating/updating the velocities
        vx[bodyIndex + offset] += dt * ax[bodyIndex + offset];
        vy[bodyIndex + offset] += dt * ay[bodyIndex + offset];
        vz[bodyIndex + offset] += dt * az[bodyIndex + offset];

        // calculating/updating the positions
        x[bodyIndex + offset] += d * dt * vx[bodyIndex + offset];
        y[bodyIndex + offset] += d * dt * vy[bodyIndex + offset];
        z[bodyIndex + offset] += d * dt * vz[bodyIndex + offset];

        offset += stride;
    }
}


//TODO: only update/calculate COM for not domain list nodes?!
__global__ void insertReceivedParticlesKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int *to_delete_leaf, int *domainListIndices,
                                int *domainListIndex, int *lowestDomainListIndices, int *lowestDomainListIndex,
                                int n, int m) {

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

    bool isDomainList = false;

    offset = 0;

    bodyIndex += to_delete_leaf[0];

    //if ((bodyIndex + offset) % 10000 == 0) {
    //    printf("index = %i x = (%f, %f, %f)\n", bodyIndex + offset, x[bodyIndex + offset], y[bodyIndex + offset], z[bodyIndex + offset]);
    //}

    while ((bodyIndex + offset) < to_delete_leaf[1] && (bodyIndex + offset) > to_delete_leaf[0]) {


        //if ((bodyIndex + offset) % 100 == 0) {
        //if (mass[bodyIndex+offset] > 200.f) {
        //    printf("insert particle %i: x = (%f, %f, %f) m = %f\n", bodyIndex+offset, x[bodyIndex+offset],
        //           y[bodyIndex+offset], z[bodyIndex+offset], mass[bodyIndex+offset]);
        //}

        /*//debugging
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
        //end: debugging*/
        //debugging
        //offset += stride;

        if (newBody) {

            newBody = false;
            isDomainList = false;

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
        while (childIndex >= m /*&& childIndex < (8*m)*/) { //formerly n

            isDomainList = false;

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

            for (int i=0; i<*domainListIndex; i++) {
                if (temp == domainListIndices[i]) {
                    isDomainList = true;
                    break;
                }
            }

            //TODO: !!!
            if (/*true*/ !isDomainList) {
                if (mass[bodyIndex + offset] != 0) {
                    atomicAdd(&x[temp], mass[bodyIndex + offset] * x[bodyIndex + offset]);
                    atomicAdd(&y[temp], mass[bodyIndex + offset] * y[bodyIndex + offset]);
                    atomicAdd(&z[temp], mass[bodyIndex + offset] * z[bodyIndex + offset]);
                }
                atomicAdd(&mass[temp], mass[bodyIndex + offset]);
                //atomicAdd(&count[temp], 1); // do not count, since particles are just temporarily saved on this process
            }
            atomicAdd(&count[temp], 1); // do not count, since particles are just temporarily saved on this process
            childIndex = child[8*temp + childPath];
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
                    while (childIndex >= 0 && childIndex < n) {

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
                        // do not count, since particles are just temporarily saved on this process
                        count[cell] += count[childIndex];

                        child[8 * cell + childPath] = childIndex;

                        start[cell] = -1; //TODO: resetting start needed in insertReceivedParticles()?

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
                        // do not count, since particles are just temporarily saved on this process
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

__global__ void centreOfMassReceivedParticlesKernel(float *x, float *y, float *z, float *mass, int *startIndex, int *endIndex, int n)
{
    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;

    //note: most of it already done within buildTreeKernel
    bodyIndex += *startIndex;

    while ((bodyIndex + offset) < *endIndex) {

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

// probably not needed, since tree is built (newly) for every iteration (step)
__global__ void repairTreeKernel(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                           float *ax, float *ay, float *az, float *mass, int *count, int *start,
                           int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                           float *minZ, float *maxZ, int *to_delete_cell, int *to_delete_leaf,
                           int *domainListIndices, int n, int m) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    offset = to_delete_leaf[0];
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
        //sorted[bodyIndex + offset] = 0;

        offset += stride;
    }

    offset = to_delete_cell[0]; //0;
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
    int level = 0; //TODO: initialize level with 0 or 1 for getTreeLevel()?
    int childIndex;

    int path[21];
    for (int i=0; i<21; i++) {
        path[i] = (int) (key >> (21*3 - 3 * (i + 1)) & (int)7);
    }

    childIndex = 0;//child[path[0]];

    //TODO: where to put level++ for getTreeLevel()?
    for (int i=0; i<21; i++) {
        //level++;
        //childIndex = child[8*childIndex + path[i]];
        if (childIndex == index) {
            return level;
        }
        childIndex = child[8*childIndex + path[i]];
        level++;
        //childIndex = child[8*childIndex + path[i]];
        //level++;
    }

    childIndex = 0; //child[path[0]];
    printf("ATTENTION: level = -1 (index = %i x = (%f, %f, %f))\n", index, x[index], y[index], z[index]);
    //printf("\tlevel = -1  childIndex = %i  path[%i] = %i\n", childIndex, 0, path[0]);
    /*for (int i=0; i<21; i++) {
        childIndex = child[8*childIndex + path[i]];
        printf("\tlevel = -1  childIndex = %i  path[%i] = %i\n", childIndex, i, path[i]);
        //for (int ii=0; ii<21; ii++) {
        //    printf("\t\t child[8*childIndex + %i] = %i\n", ii, child[8*childIndex + ii]);
        //}
    }*/
    return -1;

}

// for debugging purposes
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

// mark duplicates within an array (with -1)
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
                    if (indices[i] != -1 && (x[indices[bodyIndex + offset]] == x[indices[i]] || indices[bodyIndex + offset] == indices[i])) {
                        maxIndex = max(bodyIndex + offset, i);
                        // mark larger index with -1 (thus a duplicate)
                        indices[maxIndex] = -1;
                        atomicAdd(counter, 1);
                    }
                }

            }
        }
        //__syncthreads();
        offset += stride;
    }
}

// remove previously marked duplicates or rather copy non-duplicates into another array
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

__global__ void getParticleCount(int *child, int *count, int *particleCount) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    while ((bodyIndex + offset) < 8) {

        //particleCount += count[child[bodyIndex + offset]];
        atomicAdd(particleCount, count[child[bodyIndex + offset]]);

        offset += stride;
    }
}

__global__ void createKeyHistRangesKernel(int bins, unsigned long *keyHistRanges) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    unsigned long max_key = 1UL << 63;

    while ((bodyIndex + offset) < bins) {

        keyHistRanges[bodyIndex + offset] = (bodyIndex + offset) * (max_key/bins);
        //printf("keyHistRanges[%i] = %lu\n", bodyIndex + offset, keyHistRanges[bodyIndex + offset]);

        if ((bodyIndex + offset) == (bins - 1)) {
            keyHistRanges[bins-1] = KEY_MAX;
        }
        offset += stride;
    }
}

__global__ void keyHistCounterKernel(unsigned long *keyHistRanges, int *keyHistCounts, int bins, int n,
                        float *x, float *y, float *z, float *mass, int *count, int *start,
                        int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                        float *minZ, float *maxZ, SubDomainKeyTree *s, int curveType) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    unsigned long key;

    while ((bodyIndex + offset) < n) {

        key = getParticleKeyPerParticle(x[bodyIndex + offset], y[bodyIndex + offset], z[bodyIndex + offset],
                                        minX, maxX, minY, maxY, minZ, maxZ, 21);

        if (curveType == 0) {
            for (int i=0; i<(bins); i++) {
                if (key >= keyHistRanges[i] && key < keyHistRanges[i+1]) {
                    //keyHistCounts[i] += 1;
                    atomicAdd(&keyHistCounts[i], 1);
                    break;
                }
            }
        }
        else {
            //TODO: Hilbert change
            unsigned long hilbert = Lebesgue2Hilbert(key, 21);

            for (int i = 0; i < (bins); i++) {
                if (hilbert >= keyHistRanges[i] && hilbert < keyHistRanges[i + 1]) {
                    //keyHistCounts[i] += 1;
                    atomicAdd(&keyHistCounts[i], 1);
                    break;
                }
            }
        }

        offset += stride;
    }
}

//TODO: rename index
__global__ void calculateNewRangeKernel(unsigned long *keyHistRanges, int *keyHistCounts, int bins, int n,
                                  float *x, float *y, float *z, float *mass, int *count, int *start,
                                  int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                  float *minZ, float *maxZ, SubDomainKeyTree *s) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    int sum;
    unsigned long newRange;

    while ((bodyIndex + offset) < (bins-1)) {

        sum = 0;
        for (int i=0; i<(bodyIndex+offset); i++) {
            sum += keyHistCounts[i];
        }

        for (int i=1; i<s->numProcesses; i++) {
            if ((sum + keyHistCounts[bodyIndex + offset]) >= (i*n) && sum < (i*n)) {
                printf("[rank %i] new range: %lu\n", s->rank, keyHistRanges[bodyIndex + offset]);
                s->range[i] = keyHistRanges[bodyIndex + offset];
            }
        }


        //printf("[rank %i] keyHistCounts[%i] = %i\n", s->rank, bodyIndex+offset, keyHistCounts[bodyIndex+offset]);
        atomicAdd(index, keyHistCounts[bodyIndex+offset]);
        offset += stride;
    }

}

/* MILUPHCUDA: search interaction partners for each particle
__global__ void nearNeighbourSearch(int *interactions)
{
    register int i, inc, nodeIndex, depth, childNumber, child;
    register double x, interactionDistance, dx, r, d;
    register double y, dy;
    register int currentNodeIndex[MAXDEPTH];
    register int currentChildNumber[MAXDEPTH];
    register int numberOfInteractions;
    register double z, dz;

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

        x = p.x[i];
        y = p.y[i];
		z = p.z[i];

        double sml;
        double smlj;
        // start at root
        depth = 0;
        currentNodeIndex[depth] = numNodes - 1;
        currentChildNumber[depth] = 0;
        numberOfInteractions = 0;
        r = radius * 0.5; // because we start with root children
        sml = p.h[i];
        p.noi[i] = 0;
        interactionDistance = (r + sml);

        do {

            childNumber = currentChildNumber[depth];
            nodeIndex = currentNodeIndex[depth];

            while (childNumber < numChildren) {

                child = childList[childListIndex(nodeIndex, childNumber)];
                childNumber++;
                if (child != EMPTY && child != i) {
                    dx = x - p.x[child];
                    dy = y - p.y[child];
					dz = z - p.z[child];

                    if (child < numParticles) {

                        d = dx*dx + dy*dy + dz*dz;

                        smlj = p.h[child];

                        if (d < sml*sml && d < smlj*smlj) {
                            interactions[i * MAX_NUM_INTERACTIONS + numberOfInteractions] = child;
                            numberOfInteractions++;
                        }
                    } else if (fabs(dx) < interactionDistance && fabs(dy) < interactionDistance
                                        && fabs(dz) < interactionDistance) {
                        // put child on stack
                        currentChildNumber[depth] = childNumber;
                        currentNodeIndex[depth] = nodeIndex;
                        depth++;
                        r *= 0.5;
                        interactionDistance = (r + sml);
                        if (depth >= MAXDEPTH) {
                            printf("Error, maxdepth reached!");
                            assert(depth < MAXDEPTH);
                        }
                        childNumber = 0;
                        nodeIndex = child;
                    }
                }
            }

            depth--;
            r *= 2.0;
            interactionDistance = (r + sml);

        } while (depth >= 0);

        if (numberOfInteractions >= MAX_NUM_INTERACTIONS) {
            //printf("ERROR: Maximum number of interactions exceeded: %d / %d\n", numberOfInteractions, MAX_NUM_INTERACTIONS);


			//for (child = 0; child < MAX_NUM_INTERACTIONS; child++) {
			//	printf("(thread %d): %d - %d\n", threadIdx.x, i, interactions[i*MAX_NUM_INTERACTIONS+child]);
			//}
        }
        p.noi[i] = numberOfInteractions;
    }
}*/

__global__ void fixedRadiusNNKernel(int *interactions, int *numberOfInteractions, float *x, float *y, float *z, int *child, float *minX, float *maxX,
                              float *minY, float *maxY, float *minZ, float *maxZ, float sml,
                              int numParticlesLocal, int numParticles, int numNodes) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    int childNumber, nodeIndex, depth, childIndex;
    float dx, dy, dz, d;
    float x_radius, y_radius, z_radius, r_temp, r, interactionDistance;

    int noOfInteractions;

    int currentNodeIndex[MAXDEPTH];
    int currentChildNumber[MAXDEPTH];

    while ((bodyIndex + offset) < numParticlesLocal) {

        // resetting
        for (int i=0; i<MAX_NUM_INTERACTIONS; i++) {
            interactions[(bodyIndex + offset) * MAX_NUM_INTERACTIONS + i] = -1;
        }
        //numberOfInteractions[bodyIndex + offset] = 0;
        // end: resetting

        depth = 0;
        currentNodeIndex[depth] = 0; //numNodes - 1;
        currentChildNumber[depth] = 0;
        noOfInteractions = 0;

        x_radius = 0.5*(*maxX - (*minX));
        y_radius = 0.5*(*maxY - (*minY));
        z_radius = 0.5*(*maxZ - (*minZ));

        r_temp = fmaxf(x_radius, y_radius);
        r = fmaxf(r_temp, z_radius); //TODO: (0.5 * r) or (1.0 * r)

        interactionDistance = (r + sml);

        do {
            childNumber = currentChildNumber[depth];
            nodeIndex = currentNodeIndex[depth];

            while (childNumber < 8) {
                childIndex = child[8*nodeIndex + childNumber];
                childNumber++;

                if (childIndex != -1 && childIndex != (bodyIndex + offset)) {
                    dx = x[bodyIndex + offset] - x[childIndex];
                    dy = y[bodyIndex + offset] - y[childIndex];
                    dz = z[bodyIndex + offset] - z[childIndex];

                    // its a leaf
                    if (childIndex < numParticles) {
                        d = dx*dx + dy*dy + dz*dz;

                        if ((bodyIndex + offset) % 1000 == 0) {
                            //printf("sph: index = %i, d = %i\n", bodyIndex+offset, d);
                        }

                        if (d < (sml * sml)) {
                            //printf("Adding interaction partner!\n");
                            interactions[(bodyIndex + offset) * MAX_NUM_INTERACTIONS +
                                         noOfInteractions] = childIndex;
                            noOfInteractions++;
                        }
                    }
                    else if (fabs(dx) < interactionDistance &&
                             fabs(dy) < interactionDistance &&
                             fabs(dz) < interactionDistance) {
                            // put child on stack
                            currentChildNumber[depth] = childNumber;
                            currentNodeIndex[depth] = nodeIndex;
                            depth++;
                            r *= 0.5;
                            interactionDistance = (r + sml);

                            if (depth > MAXDEPTH) {
                                printf("ERROR: maximal depth reached! MAXDEPTH = %i\n", MAXDEPTH);
                                assert(depth < MAXDEPTH);
                            }
                            childNumber = 0;
                            nodeIndex = childIndex;
                    }
                }
            }

            depth--;
            r *= 2.0;
            interactionDistance = (r + sml);

        } while(depth >= 0);

        numberOfInteractions[bodyIndex + offset] = noOfInteractions;
        offset += stride;
    }
}

__global__ void sphDebugKernel(int *interactions, int *numberOfInteractions, float *x, float *y, float *z, int *child, float *minX, float *maxX,
                               float *minY, float *maxY, float *minZ, float *maxZ, float sml,
                               int numParticlesLocal, int numParticles, int numNodes) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    while ((bodyIndex + offset) < numParticlesLocal) {

        if ((bodyIndex + offset) % 1000 == 0) {
            printf("index = %i  number of interactions: %i  (interactions[0] = %i interactions[1] = %i)\n", bodyIndex+offset,
                   numberOfInteractions[bodyIndex + offset],
                   interactions[(bodyIndex + offset) * MAX_NUM_INTERACTIONS + 0],
                   interactions[(bodyIndex + offset) * MAX_NUM_INTERACTIONS + 1]);
        }

        offset += stride;
    }

}

__global__ void sphParticles2SendKernel(int numParticlesLocal, int numParticles, int numNodes, float radius,
                                        float *x, float *y, float *z,
                                        float *minX, float *maxX, float *minY, float *maxY, float *minZ, float *maxZ,
                                        SubDomainKeyTree *s, int *domainListIndex, unsigned long *domainListKeys,
                                        int *domainListIndices, int *domainListLevels,
                                        int *lowestDomainListIndices, int *lowestDomainListIndex,
                                        unsigned long *lowestDomainListKeys, int *lowestDomainListLevels,
                                        float sml, int maxLevel, int curveType,
                                        int *toSend, int *sendCount, int *alreadyInserted, int insertOffset) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    int insertIndex;
    int insertIndexOffset;

    int proc, currentChild;
    float dx, dy, dz, d;
    float min_x, max_x, min_y, max_y, min_z, max_z;

    //int alreadyInserted[10];

    while ((bodyIndex + offset) < numParticlesLocal) {

        if ((bodyIndex + offset) == 0) {
            printf("sphParticles2SendKernel: insertOffset = %i\n", insertOffset);
        }

        //toSend[bodyIndex + offset] = -1;

        for (int i=0; i<s->numProcesses; i++) {
            alreadyInserted[i] = 0;
        }

        // loop over (lowest?) domain list nodes
        for (int i=0; i<*lowestDomainListIndex; i++) {

            min_x = *minX;
            max_x = *maxX;
            min_y = *minY;
            max_y = *maxY;
            min_z = *minZ;
            max_z = *maxZ;

            proc = key2proc(lowestDomainListKeys[i], s, curveType);
            // check if (lowest?) domain list node belongs to other process

            if (proc != s->rank && alreadyInserted[proc] != 1) {

                int path[21];
                for (int j=0; j<=lowestDomainListLevels[i]; j++) { //TODO: "<" or "<="
                    path[j] = (int)(lowestDomainListKeys[i] >> (21 * 3 - 3 * (j + 1)) & (int) 7);
                }

                for (int j=0; j<=lowestDomainListLevels[i]; j++) {

                    currentChild = path[j];

                    if (currentChild % 2 != 0) {
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
                    if (currentChild == 4) {
                        max_z = 0.5 * (min_z + max_z);
                        currentChild -= 4;
                    }
                    else {
                        min_z = 0.5 * (min_z + max_z);
                    }
                }

                // x-direction
                if (x[bodyIndex + offset] < min_x) {
                    // outside
                    dx = x[bodyIndex + offset] - min_x;
                }
                else if (x[bodyIndex + offset] > max_x) {
                    // outside
                    dx = x[bodyIndex + offset] - max_x;
                }
                else {
                    // in between: do nothing
                    dx = 0;
                }
                // y-direction
                if (y[bodyIndex + offset] < min_y) {
                    // outside
                    dy = y[bodyIndex + offset] - min_y;
                }
                else if (y[bodyIndex + offset] > max_y) {
                    // outside
                    dy = y[bodyIndex + offset] - max_y;
                }
                else {
                    // in between: do nothing
                    dy = 0;
                }
                // z-direction
                if (z[bodyIndex + offset] < min_z) {
                    // outside
                    dz = z[bodyIndex + offset] - min_z;
                }
                else if (z[bodyIndex + offset] > max_z) {
                    // outside
                    dz = z[bodyIndex + offset] - max_z;
                }
                else {
                    // in between: do nothing
                    dz = 0;
                }

                d = dx*dx + dy*dy + dz*dz;

                if (d < (sml * sml)) {

                    insertIndex = atomicAdd(&sendCount[proc], 1);
                    if (insertIndex > 100000) {
                        printf("Attention!!! insertIndex: %i\n", insertIndex);
                    }
                    insertIndexOffset = insertOffset * proc; //0;
                    toSend[insertIndexOffset + insertIndex] = bodyIndex+offset;
                    //toSend[proc][insertIndex] = bodyIndex+offset;
                    /*if (insertIndex % 100 == 0) {
                        printf("[rank %i] Inserting %i into : %i + %i  toSend[%i] = %i\n", s->rank, bodyIndex+offset,
                               (insertOffset * proc), insertIndex, (insertOffset * proc) + insertIndex,
                               toSend[(insertOffset * proc) + insertIndex]);
                    }*/
                    alreadyInserted[proc] = 1;
                    //break;
                }
                else {
                    // else: do nothing
                }
            }
        }

        //__threadfence();
        offset += stride;
    }
}

__global__ void collectSendIndicesSPHKernel(int *toSend, int *toSendCollected, int count) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    while ((bodyIndex + offset) < count) {
        toSendCollected[bodyIndex + offset] = toSend[bodyIndex + offset];
        offset += stride;
    }
}

__global__ void collectSendEntriesSPHKernel(float *entry, float *toSend, int *sendIndices, int *sendCount,
                                            int totalSendCount, int insertOffset, SubDomainKeyTree *s) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    int proc;
    if (s->rank == 0) {
        proc = 1;
    }
    else {
        proc = 0;
    }

    if ((bodyIndex + offset) == 0) {
        printf("[rank %i] sendCount(%i, %i)\n", s->rank, sendCount[0], sendCount[1]);
    }

    bodyIndex += proc*insertOffset;

    while ((bodyIndex + offset) < totalSendCount) {
        toSend[bodyIndex + offset] = entry[sendIndices[bodyIndex + offset]];
        offset += stride;
    }

    /*while ((bodyIndex + offset) < (proc*insertOffset + sendCount[proc])) {

        //if ((bodyIndex + offset) % 100 == 0) {
        //    printf("[rank %i] toSend[%i] = %i sendCount(%i, %i)\n", s->rank, (bodyIndex + offset),
        //           sendIndices[(bodyIndex + offset)], sendCount[0], sendCount[1]);
        //}

        if (sendIndices[(bodyIndex + offset)] == -1) {
            printf("[rank %i] ATTENTION: toSend[%i] = %i sendCount(%i, %i)\n", s->rank, (bodyIndex + offset),
                   sendIndices[(bodyIndex + offset)], sendCount[0], sendCount[1]);
        }

        offset += stride;
    }*/

    /*while ((bodyIndex + offset) < totalSendCount) {
        printf("[rank %i] toSend[%i] = %i sendCount(%i, %i)\n", s->rank, (bodyIndex + offset),
                   sendIndices[(bodyIndex + offset)], sendCount[0], sendCount[1]);
        offset += stride;
    }*/
}
