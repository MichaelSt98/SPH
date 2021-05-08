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

            temp = 0; //m
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

                // check whether body is already stored at the location
                if (childIndex == -1) {
                    //insert body and release lock
                    child[locked] = bodyIndex + offset;
                }
                else {
                    int patch = 8*n; //4*n //-1
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

/*void SubDomain::getKeyIteratively() {
    KeyType helperKey { 0 };
    ParticleList pList;
    //gatherParticles(pList);
    root.getParticleList(pList);
    for (int i=0; i<pList.size(); i++) {
        //helperKey = 0;
        root.particle2Key(helperKey, pList[i]);
        Logger(INFO) << "iteratively key[" << i << "] = " << helperKey;
    }

}*/

/*void TreeNode::particle2Key(KeyType &key, Particle &p) {
    int level = 0;
    int sonBox;
    Domain domain{ box };
    while (level <= key.maxLevel) {
        sonBox = getSonBox(p, domain);
        //key = key | (KeyType{ sonBox } << (DIM * (key.maxLevel-level-1)));
        level ++;
    }
}*/

__device__ void key2Char(unsigned long key, int maxLevel, char *keyAsChar) {
    int level[21];
    for (int i=0; i<maxLevel; i++) {
        level[i] = (int)(key >> (maxLevel*3 - 3*(i+1)) & (int)7);
    }
    for (int i=0; i<maxLevel; i++) {
        keyAsChar[i] = level[i] + '0';
    }
    level[maxLevel] = '\0';
}

__global__ void getParticleKeyKernel(float *x, float *y, float *z, float *minX, float *maxX, float *minY, float *maxY,
                               float *minZ, float *maxZ, unsigned long *key, int maxLevel, int n) {

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

    while (bodyIndex + offset < n) {

        int level = 0;
        testKey = 0UL;
        while (level <= maxLevel) {

            // find insertion point for body
            if (x[bodyIndex + offset] < 0.5 * (min_x+max_x)) {
                sonBox += 1;
                max_x = 0.5 * (min_x+max_x);
            }
            else { min_x = 0.5 * (min_x+max_x); }
            if (y[bodyIndex + offset] < 0.5 * (min_y+max_y)) {
                sonBox += 2;
                max_y = 0.5 * (min_y + max_y);
            }
            else { min_y = 0.5 * (min_y + max_y); }
            if (z[bodyIndex + offset] < 0.5 * (min_z+max_z)) {
                sonBox += 4;
                max_z = 0.5 * (min_z + max_z);
            }
            else { min_z =  0.5 * (min_z + max_z); }

            //*key = *key; //| ((unsigned long)sonBox << (unsigned long)(3 * (maxLevel-level-1)));
            testKey = testKey | ((unsigned long)sonBox << (unsigned long)(3 * (maxLevel-level-1)));
            level ++;
        }

        char keyAsChar[22];
        key2Char(testKey, 21, keyAsChar);
        if ((bodyIndex + offset) % 1000 == 0) {
            //printf("key[%i]: %lu\n", bodyIndex + offset, testKey);
            printf("key[%i]: %s\n", bodyIndex + offset, keyAsChar);
        }

        offset += stride;
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