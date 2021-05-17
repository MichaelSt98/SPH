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
                                  float *minY, float *maxY, float *minZ, float *maxZ, int n, int m)
{
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

    // initialize block min/max buffer
    /*__shared__ float x_min_buffer[blockSize];
    __shared__ float x_max_buffer[blockSize];
    __shared__ float y_min_buffer[blockSize];
    __shared__ float y_max_buffer[blockSize];
    __shared__ float z_min_buffer[blockSize];
    __shared__ float z_max_buffer[blockSize];*/

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

            temp = 0; //bodyIndex + offset; //0; //m
            childPath = 0;

            // find insertion point for body
            // x direction
            if (x[bodyIndex + offset] < 0.5 * (min_x + max_x)) {
                childPath += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
            // y direction
            if (y[bodyIndex + offset] < 0.5 * (min_y + max_y)) {
                childPath += 2;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }
            // z direction
            if (z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {
                childPath += 4;
                max_z = 0.5 * (min_z + max_z);
            }
            else {
                min_z = 0.5 * (min_z + max_z);
            }
        }

        int childIndex = child[temp*8 + childPath];

        // traverse tree until hitting leaf node
        while (childIndex >= n) {

            //printf("Im here!");

            temp = childIndex;
            childPath = 0;

            // find insertion point for body
            // x direction
            if (x[bodyIndex + offset] < 0.5 * (min_x + max_x)) {
                childPath += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
            // y direction
            if (y[bodyIndex + offset] < 0.5 * (min_y + max_y)) {
                childPath += 2;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }

            // z direction
            if (z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {
                childPath += 4;
                max_z = 0.5 * (min_z + max_z);
            }
            else {
                min_z = 0.5 * (min_z + max_z);
            }

            atomicAdd(&x[temp], mass[bodyIndex + offset] * x[bodyIndex + offset]);
            atomicAdd(&y[temp], mass[bodyIndex + offset] * y[bodyIndex + offset]);
            atomicAdd(&z[temp], mass[bodyIndex + offset] * z[bodyIndex + offset]);

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
                    int patch = 8*n; //8*n //4*n //-1
                    while (childIndex >= 0 && childIndex < n) {

                        //create a new cell (by atomically requesting the next unused array index)
                        int cell = atomicAdd(index, 1);
                        patch = min(patch, cell);

                        if (patch != cell) {
                            child[8*temp + childPath] = cell;
                        }

                        // insert old/original particle
                        childPath = 0;
                        if(x[childIndex] < 0.5 * (min_x+max_x)) {
                            childPath += 1;
                        }

                        if (y[childIndex] < 0.5 * (min_y+max_y)) {
                            childPath += 2;
                        }

                        if (z[childIndex] < 0.5 * (min_z+max_z)) {
                            childPath += 4;
                        }

                        x[cell] += mass[childIndex] * x[childIndex];
                        y[cell] += mass[childIndex] * y[childIndex];
                        z[cell] += mass[childIndex] * z[childIndex];

                        mass[cell] += mass[childIndex];
                        count[cell] += count[childIndex];

                        child[8*cell + childPath] = childIndex;

                        start[cell] = -1;

                        // insert new particle
                        temp = cell;
                        childPath = 0;

                        // find insertion point for body
                        if (x[bodyIndex + offset] < 0.5 * (min_x+max_x)) {
                            childPath += 1;
                            max_x = 0.5 * (min_x+max_x);
                        }
                        else {
                            min_x = 0.5 * (min_x+max_x);
                        }
                        if (y[bodyIndex + offset] < 0.5 * (min_y+max_y)) {
                            childPath += 2;
                            max_y = 0.5 * (min_y + max_y);
                        }
                        else {
                            min_y = 0.5 * (min_y + max_y);
                        }
                        if (z[bodyIndex + offset] < 0.5 * (min_z+max_z)) {
                            childPath += 4;
                            max_z = 0.5 * (min_z + max_z);
                        }
                        else {
                            min_z =  0.5 * (min_z + max_z);
                        }

                        //if (cell >= m) {
                        //    printf("cell index to large!\ncell: %d (> %d)\n", cell, m);
                        //}

                        //printf("cell: %d \n", cell);
                        //printf("bodyIndex + offset: %d \n", bodyIndex + offset);

                        // COM / preparing for calculation of COM
                        x[cell] += mass[bodyIndex + offset] * x[bodyIndex + offset];
                        y[cell] += mass[bodyIndex + offset] * y[bodyIndex + offset];
                        z[cell] += mass[bodyIndex + offset] * z[bodyIndex + offset];
                        mass[cell] += mass[bodyIndex + offset];
                        count[cell] += count[bodyIndex + offset];
                        childIndex = child[8*temp + childPath];

                    }

                    child[8*temp + childPath] = bodyIndex + offset;

                    __threadfence();  // written to global memory arrays (child, x, y, mass) thus need to fence
                    child[locked] = patch;
                }
                //__threadfence();

                offset += stride;
                newBody = true;
            }
        }
        __syncthreads();
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

        sonBox = 0; //TODO: needed?

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

    int sonBox = 0;
    float min_x = *minX;
    float max_x = *maxX;
    float min_y = *minY;
    float max_y = *maxY;
    float min_z = *minZ;
    float max_z = *maxZ;

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
        if ((bodyIndex + offset) % 5000 == 0 /*|| proc == 1*/) {
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
    for (int i=0; i<s->numProcesses; i++) {
        if (k >= s->range[i] && k < s->range[i+1]) {
            return i;
        }
    }
    //printf("ERROR: key2proc(k=%lu): -1!", k);
    return -1; // error
}

/*void createDomainList(TreeNode *t, int level, keytype k, SubDomainKeyTree *s) {
    t->node = domainList;
    int p1 = key2proc(k, s);
    int p2 = key2proc(k | ~(~0L << DIM*(maxlevel-level)),s);
    if (p1 != p2) {
        for (int i = 0; i < POWDIM; i++) {
            t->son[i] = (TreeNode *) calloc(1, sizeof(TreeNode));
            createDomainList(t->son[i], level + 1,  (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), s);
        }
    }
}*/

__global__ void traverseIterativeKernel(float *x, float *y, float *z, float *mass, int *child, int n, int m,
                         SubDomainKeyTree *s, int maxLevel) {

    __shared__ int stack[128];
    __shared__ int *stackPtr;
    stackPtr = stack;
    *stackPtr++ = NULL;

    int childIndex = n;
    int node = childIndex;
    stack[0] = childIndex;

    int popped = 0;
    int pushed = 0;
    int particleCounter = 0;

    int bodyIndex = threadIdx.x + blockDim.x*blockIdx.x;

    if (bodyIndex == 0) {

        while (node != NULL) {
            node = *--stackPtr; // pop
            popped++;

            childIndex = 8*node;

            for (int i = 0; i < 8; i++) {
                if (child[childIndex + i] != -1 && childIndex > 0) {
                    *stackPtr++ = child[childIndex + i];
                    //printf("push stack\n");
                    pushed++;
                }
                else {
                    if (childIndex < 8*n) {
                        particleCounter++;
                    }
                }
            }
            /*for (int i=0; i < 8; i++) {
                if (child[childIndex + i] != -1 && childIndex > 0) {
                    if (child[childIndex + i] >= n) {
                    //if (childIndex >= n) {
                        *stackPtr++ = child[childIndex + i];
                        pushed++;
                    }
                    else {
                        particleCounter++;
                    }
                }
            }*/
            //printf("pushed - popped = %i\n", pushed - popped);

        }
        printf("Finished! popped = %i   pushed = %i   particleCounter = %i\n", popped, pushed, particleCounter);
    }
}

/*__global__ void createDomainListKernel(float *x, float *y, float *z, float *mass, int *child, int n,
                                       SubDomainKeyTree *s, int maxLevel) {


}*/

/*__global__ void createDomainListKernel(float *x, float *y, float *z, float *mass, float *minX, float *maxX,
                                       float *minY, float *maxY, float *minZ, float *maxZ, int *child, int n,
                                       SubDomainKeyTree *s, int maxLevel) {

    int bodyIndex = threadIdx.x + blockDim.x*blockIdx.x;

    unsigned long testKey = 0UL;
    unsigned long currentKey = 0UL;

    unsigned long stack_parentKey[128];
    int stack_level[128];
    unsigned long *stackPtr_parentKey = stack_parentKey;
    *stackPtr_parentKey++ = NULL;
    int *stackPtr_level = stack_level;
    *stackPtr_level++ = NULL;

    unsigned long domainListKeys[1024];
    int levels[1024];

    int currentIndex = 0;
    int lastIndex = 0;

    if (bodyIndex == 0) {

        int level = 0;

        stack_parentKey[0] = testKey;
        stack_level[0] = level;

        do {
            testKey = *--stackPtr_parentKey;
            printf("testKey\n");
        } while (testKey != NULL);

//        while (stackPtr_parentKey != NULL) {
//            printf("tesKey");
//            testKey = NULL;
//            stackPtr_parentKey = NULL;
//        }

//        if (isDomainListNode(testKey, maxLevel, level, s)) {
//            printf("testKey: %lu is domain list!\n", testKey);
//
//            level++;
//            for (int son=0; son<8; son++) {
//                testKey = currentKey | ((unsigned long)son << (unsigned long)(3 * (maxLevel-level-1)));
//                if (isDomainListNode(testKey, maxLevel, level, s)) {
//                    printf("son[%i]: testKey: %lu is domain list!\n", son, testKey);
//                }
//                else {
//                    printf("son[%i]: testKey: %lu is NOT domain list!\n", son, testKey);
//                }
//            }
//        }
//        else {
//            printf("testKey: %lu is NOT domain list!\n", testKey);
//        }

//        while (level <= maxLevel) {
//
//            for (int son=0; son<8; son++) {
//                testKey = testKey | ((unsigned long)son << (unsigned long)(3 * (maxLevel-level-1)));
//            }
//
//            level++;
//
//        }
    }

}*/

__global__ void createDomainListKernel(float *x, float *y, float *z, float *mass, float *minX, float *maxX,
                                       float *minY, float *maxY, float *minZ, float *maxZ, int *child, int n,
                                       SubDomainKeyTree *s, int maxLevel) {

    int bodyIndex = threadIdx.x + blockDim.x * blockIdx.x;

    if (bodyIndex == 0) {

        unsigned long domainListKeys[256];
        int levels[256];
        int index = 0;

        unsigned long shiftValue = 1;
        unsigned long toShift = 63;
        unsigned long keyMax = shiftValue << toShift; // 1 << 63 not working!
        unsigned long key2test = 0UL;

        int level = 0;
        int counter = 0;

        printf("Collecting domain list nodes/keys... (keyMax = %lu)\n", keyMax);
        while (key2test < keyMax /*&& counter < 10*/) {
            counter++;
            printf("key2test = %lu\n", key2test);
            if (isDomainListNode(key2test, maxLevel, level, s)) {
                domainListKeys[index] = key2test;
                levels[index] = level;
                index++;
                level++;
                key2test = key2test + (1 << 3 * (maxLevel - level));
            } else {
                key2test = keyMaxLevel(key2test, maxLevel, level, s) + 1;
                level--;
            }
        }
        printf("Finished! #domain list nodes: %i\n", index);
        for (int i=0; i < index; i++) {
            printf("domainListKeys[%i] = %lu (level: %i)\n", i, domainListKeys[i], levels[i]);
        }
    }
}

__device__ bool isDomainListNode(unsigned long key, int maxLevel, int level, SubDomainKeyTree *s) {
    int p1 = key2proc(key, s);
    int p2 = key2proc(key | ~(~0L << 3*(maxLevel-level)), s);
    if (p1 != p2) {
        return true;
    }
    else {
        return false;
    }
}

__device__ unsigned long keyMaxLevel(unsigned long key, int maxLevel, int level, SubDomainKeyTree *s) {
    unsigned long keyMax = key | ~(~0L << 3*(maxLevel-level));
    return keyMax;
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

        x[bodyIndex + offset] /= mass[bodyIndex + offset];
        y[bodyIndex + offset] /= mass[bodyIndex + offset];
        z[bodyIndex + offset] /= mass[bodyIndex + offset];

        offset += stride;
    }
}


// Kernel 4: sorts the bodies
__global__ void sortKernel(int *count, int *start, int *sorted, int *child, int *index, int n)
{
    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    int s = 0;
    if (threadIdx.x == 0) {
        
        for(int i=0; i<8; i++){
            
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
    
    while ((cell + offset) < ind) {
        
        s = start[cell + offset];

        if (s >= 0) {

            for(int i=0; i<8; i++) {
                
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

                        // calculate intraction force contribution
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


// Kernel 6: updates the bodies
__global__ void updateKernel(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                             float *ax, float *ay, float *az, int n, float dt, float d) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    while (bodyIndex + offset < n) {

        vx[bodyIndex + offset] += dt * ax[bodyIndex + offset]; //*0.5f
        vy[bodyIndex + offset] += dt * ay[bodyIndex + offset]; //*0.5f
        vz[bodyIndex + offset] += dt * az[bodyIndex + offset]; //*0.5f

        x[bodyIndex + offset] += d * dt * vx[bodyIndex + offset];
        y[bodyIndex + offset] += d * dt * vy[bodyIndex + offset];
        z[bodyIndex + offset] += d * dt * vz[bodyIndex + offset];

        offset += stride;
    }
}