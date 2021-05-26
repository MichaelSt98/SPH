#include "../include/BarnesHut.cuh"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) getchar();
    }
}

BarnesHut::BarnesHut(const SimulationParameters p) {

    parameters = p;
    KernelHandler = KernelsWrapper(p);
    step = 0;
    numParticles = p.numberOfParticles; //NUM_BODIES;
    numNodes = 2 * numParticles + 12000; //+ 12000; //2 * numParticles + 12000;

    timeKernels = p.timeKernels; //true;

    // allocate host data
    h_min_x = new float;
    h_max_x = new float;
    h_min_y = new float;
    h_max_y = new float;
    h_min_y = new float;
    h_max_y = new float;

    h_mass = new float[numNodes];

    h_domainListIndices = new unsigned long[DOMAIN_LIST_SIZE];
    h_domainListKeys = new unsigned long[DOMAIN_LIST_SIZE];
    h_domainListLevels = new int[DOMAIN_LIST_SIZE];
    h_domainListIndex = new int;
    for (int i=0; i<DOMAIN_LIST_SIZE; i++) {
        h_domainListIndices[i] = KEY_MAX;
        h_domainListKeys[i] = KEY_MAX;
        h_domainListLevels[i] = -1;
    }

    h_x = new float[numNodes];
    h_y = new float[numNodes];
    h_z = new float[numNodes];

    h_vx = new float[numNodes];
    h_vy = new float[numNodes];
    h_vz = new float[numNodes];

    h_ax = new float[numNodes];
    h_ay = new float[numNodes];
    h_az = new float[numNodes];

    h_child = new int[8*numNodes];
    
    h_start = new int[numNodes];
    h_sorted = new int[numNodes];
    h_count = new int[numNodes];
    h_output = new float[2*numNodes];

    time_resetArrays = new float[parameters.iterations];
    time_computeBoundingBox = new float[parameters.iterations];
    time_buildTree = new float[parameters.iterations];
    time_centreOfMass = new float[parameters.iterations];
    time_sort = new float[parameters.iterations];
    time_computeForces = new float[parameters.iterations];
    time_update = new float[parameters.iterations];
    time_copyDeviceToHost = new float[parameters.iterations];
    time_all = new float [parameters.iterations];


    h_subDomainHandler = new SubDomainKeyTree();
    h_subDomainHandler->rank = 0;
    h_subDomainHandler->range = new unsigned long[3];
    h_subDomainHandler->range[0] = 0;
    h_subDomainHandler->range[1] = 4611686018427387904UL;// + 3872UL;
    h_subDomainHandler->range[2] = KEY_MAX;
    h_subDomainHandler->numProcesses = 2;

    h_procCounter = new int[h_subDomainHandler->numProcesses];

    // allocate device data
    gpuErrorcheck(cudaMalloc((void**)&d_min_x, sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_max_x, sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_min_y, sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_max_y, sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_min_z, sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_max_z, sizeof(float)));

    gpuErrorcheck(cudaMemset(d_min_x, 0, sizeof(float)));
    gpuErrorcheck(cudaMemset(d_max_x, 0, sizeof(float)));
    gpuErrorcheck(cudaMemset(d_min_y, 0, sizeof(float)));
    gpuErrorcheck(cudaMemset(d_max_y, 0, sizeof(float)));
    gpuErrorcheck(cudaMemset(d_min_z, 0, sizeof(float)));
    gpuErrorcheck(cudaMemset(d_max_z, 0, sizeof(float)));

    gpuErrorcheck(cudaMalloc((void**)&d_mass, numNodes*sizeof(float)));

    gpuErrorcheck(cudaMalloc((void**)&d_domainListIndices, DOMAIN_LIST_SIZE*sizeof(unsigned long)));
    gpuErrorcheck(cudaMalloc((void**)&d_domainListKeys, DOMAIN_LIST_SIZE*sizeof(unsigned long)));
    gpuErrorcheck(cudaMalloc((void**)&d_domainListLevels, DOMAIN_LIST_SIZE*sizeof(int)));
    gpuErrorcheck(cudaMalloc((void**)&d_domainListIndex, sizeof(int)));

    gpuErrorcheck(cudaMalloc((void**)&d_tempArray, numParticles*sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_sortArray, numParticles*sizeof(int)));
    gpuErrorcheck(cudaMalloc((void**)&d_sortArrayOut, numParticles*sizeof(int)));

    gpuErrorcheck(cudaMalloc((void**)&d_procCounter, h_subDomainHandler->numProcesses*sizeof(int)));
    gpuErrorcheck(cudaMalloc((void**)&d_procCounterTemp, h_subDomainHandler->numProcesses*sizeof(int)));

    gpuErrorcheck(cudaMalloc((void**)&d_x, numNodes*sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_y, numNodes*sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_z, numNodes*sizeof(float)));

    gpuErrorcheck(cudaMalloc((void**)&d_vx, numNodes*sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_vy, numNodes*sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_vz, numNodes*sizeof(float)));

    gpuErrorcheck(cudaMalloc((void**)&d_ax, numNodes*sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_ay, numNodes*sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_az, numNodes*sizeof(float)));

    gpuErrorcheck(cudaMalloc((void**)&d_index, sizeof(int)));
    gpuErrorcheck(cudaMalloc((void**)&d_child, 8*numNodes*sizeof(int)));
    gpuErrorcheck(cudaMalloc((void**)&d_start, numNodes*sizeof(int)));
    gpuErrorcheck(cudaMalloc((void**)&d_sorted, numNodes*sizeof(int)));
    gpuErrorcheck(cudaMalloc((void**)&d_count, numNodes*sizeof(int)));
    gpuErrorcheck(cudaMalloc((void**)&d_mutex, sizeof(int)));

    //gpuErrorcheck(cudaMalloc((void**)&d_subDomainHandler, sizeof(SubDomainKeyTree)));
    gpuErrorcheck(cudaMalloc((void**)&d_subDomainHandler, sizeof(SubDomainKeyTree)));
    int size = 2 * sizeof(int) + 3 * sizeof(unsigned long);
    gpuErrorcheck(cudaMalloc((void**)&d_range, size));
    //gpuErrorcheck(cudaMemset(d_subDomainHandler->rank, 0, sizeof(int)));
    //gpuErrorcheck(cudaMemset(d_subDomainHandler->range, {0, KEY_MAX/2, KEY_MAX}, 3*sizeof(unsigned long)));
    //gpuErrorcheck(cudaMemset(d_subDomainHandler->numProcesses, 2, sizeof(int)));


    gpuErrorcheck(cudaMemset(d_start, -1, numNodes*sizeof(int)));
    gpuErrorcheck(cudaMemset(d_sorted, 0, numNodes*sizeof(int)));

    int memSize = sizeof(float) * 2 * numParticles;

    gpuErrorcheck(cudaMalloc((void**)&d_output, 2*numNodes*sizeof(float)));

    //plummerModel(h_mass, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_ax, h_ay, h_az, numParticles);
    diskModel(h_mass, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_ax, h_ay, h_az, numParticles);


    // copy data to GPU device

    cudaMemcpy(d_subDomainHandler, h_subDomainHandler, sizeof(SubDomainKeyTree), cudaMemcpyHostToDevice);
    cudaMemcpy(d_range, h_subDomainHandler->range, size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_subDomainHandler->range), &d_range, sizeof(unsigned long*), cudaMemcpyHostToDevice);

    //cudaMemcpy(d_subDomainHandler, h_subDomainHandler, sizeof(*h_subDomainHandler), cudaMemcpyHostToDevice);
    //cudaMemcpy(&d_subDomainHandler->rank, &h_subDomainHandler->rank, sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(&d_subDomainHandler->numProcesses, &h_subDomainHandler->numProcesses, sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(&d_subDomainHandler->range, &h_subDomainHandler->range, 3*sizeof(unsigned long), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mass, h_mass, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_domainListIndices, h_domainListIndices, DOMAIN_LIST_SIZE*sizeof(unsigned long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_domainListKeys, h_domainListKeys, DOMAIN_LIST_SIZE*sizeof(unsigned long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_domainListLevels, h_domainListLevels, DOMAIN_LIST_SIZE*sizeof(int), cudaMemcpyHostToDevice);
    gpuErrorcheck(cudaMemset(d_domainListIndex, 0, sizeof(int)));

    cudaMemcpy(d_x, h_x, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ax, h_ax, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ay, h_ay, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_az, h_az, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);

}

BarnesHut::~BarnesHut() {
    delete h_min_x;
    delete h_max_x;
    delete h_min_y;
    delete h_max_y;
    delete h_min_z;
    delete h_max_z;

    delete [] h_subDomainHandler->range;
    delete h_subDomainHandler;

    delete [] h_mass;

    delete [] h_x;
    delete [] h_y;
    delete [] h_z;

    delete [] h_vx;
    delete [] h_vy;
    delete [] h_vz;

    delete [] h_ax;
    delete [] h_ay;
    delete [] h_az;

    delete [] h_child;
    delete [] h_start;
    delete [] h_sorted;
    delete [] h_count;
    delete [] h_output;

    delete [] time_resetArrays;
    delete [] time_computeBoundingBox;
    delete [] time_buildTree;
    delete [] time_centreOfMass;
    delete [] time_sort;
    delete [] time_computeForces;
    delete [] time_update;
    delete [] time_copyDeviceToHost;
    delete [] time_all;

    gpuErrorcheck(cudaFree(d_min_x));
    gpuErrorcheck(cudaFree(d_max_x));
    gpuErrorcheck(cudaFree(d_min_y));
    gpuErrorcheck(cudaFree(d_max_y));
    gpuErrorcheck(cudaFree(d_min_z));
    gpuErrorcheck(cudaFree(d_max_z));

    gpuErrorcheck(cudaFree(d_mass));

    gpuErrorcheck(cudaFree(d_subDomainHandler->range));
    gpuErrorcheck(cudaFree(d_subDomainHandler));
    gpuErrorcheck(cudaFree(d_range));

    gpuErrorcheck(cudaFree(d_x));
    gpuErrorcheck(cudaFree(d_y));
    gpuErrorcheck(cudaFree(d_z));

    gpuErrorcheck(cudaFree(d_vx));
    gpuErrorcheck(cudaFree(d_vy));
    gpuErrorcheck(cudaFree(d_vz));

    gpuErrorcheck(cudaFree(d_ax));
    gpuErrorcheck(cudaFree(d_ay));
    gpuErrorcheck(cudaFree(d_az));

    gpuErrorcheck(cudaFree(d_index));
    gpuErrorcheck(cudaFree(d_child));
    gpuErrorcheck(cudaFree(d_start));
    gpuErrorcheck(cudaFree(d_sorted));
    gpuErrorcheck(cudaFree(d_count));

    gpuErrorcheck(cudaFree(d_mutex));

    gpuErrorcheck(cudaFree(d_output));

    cudaDeviceSynchronize();
}

void BarnesHut::update(int step)
{

    float elapsedTime;
    cudaEventCreate(&start_global);
    cudaEventCreate(&stop_global);
    cudaEventRecord(start_global, 0);

    float elapsedTimeKernel;

    elapsedTimeKernel = KernelHandler.resetArrays(d_mutex, d_x, d_y, d_z, d_mass, d_count, d_start, d_sorted, d_child, d_index,
                        d_min_x, d_max_x, d_min_y, d_max_y, d_min_z, d_max_z, numParticles, numNodes,
                        d_procCounter, d_procCounterTemp,timeKernels);

    KernelHandler.resetArraysParallel(d_domainListIndex, d_domainListKeys, d_domainListIndices,
                                      d_domainListLevels, d_tempArray, numParticles, numNodes);

    time_resetArrays[step] = elapsedTimeKernel;
    if (timeKernels) {
        Logger(TIME) << "\tReset arrays: " << elapsedTimeKernel << " ms";
    }

    elapsedTimeKernel = KernelHandler.computeBoundingBox(d_mutex, d_x, d_y, d_z, d_min_x, d_max_x, d_min_y, d_max_y,
                               d_min_z, d_max_z, numParticles, timeKernels);

    time_computeBoundingBox[step] = elapsedTimeKernel;
    if (timeKernels) {
        Logger(TIME) << "\tBounding box: " << elapsedTimeKernel << " ms";
    }

    KernelHandler.particlesPerProcess(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
                                      d_min_z, d_max_z, numParticles, numNodes, d_subDomainHandler, d_procCounter, d_procCounterTemp);

    KernelHandler.sortParticlesProc(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
                                      d_min_z, d_max_z, numParticles, numNodes, d_subDomainHandler, d_procCounter, d_procCounterTemp,
                                      d_sortArray);

    KernelHandler.sendParticles(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
                                d_min_z, d_max_z, numParticles, numNodes, d_subDomainHandler, d_procCounter, d_tempArray,
                                d_sortArray, d_sortArrayOut);

    KernelHandler.createDomainList(d_subDomainHandler, 21, d_domainListKeys, d_domainListLevels, d_domainListIndex);
    //KernelHandler.buildDomainTree(d_domainListIndex, d_domainListKeys, d_domainListLevels, d_count, d_start, d_child,
      //                            d_index, numParticles, numNodes);

    elapsedTimeKernel = KernelHandler.buildTree(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
                      d_min_z, d_max_z, numParticles, numNodes, timeKernels);

    KernelHandler.buildDomainTree(d_domainListIndex, d_domainListKeys, d_domainListLevels, d_count, d_start, d_child,
                                d_index, numParticles, numNodes);

    //KernelHandler.buildDomainTree(d_domainListIndex, d_domainListKeys, d_domainListLevels, d_count, d_start, d_child,
      //                            d_index, numParticles, numNodes);

    KernelHandler.treeInfo(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
                           d_min_z, d_max_z, numParticles, numNodes, d_procCounter);

    //KernelHandler.getParticleKey(d_x, d_y, d_z, d_min_x, d_max_x, d_min_y, d_max_y,
                         //      d_min_z, d_max_z, 0UL, 21, numParticles, d_subDomainHandler);


    //KernelHandler.traverseIterative(d_x, d_y, d_z, d_mass, d_child, numParticles, numNodes, d_subDomainHandler, 21);
    //KernelHandler.createDomainList(d_x, d_y, d_z, d_mass, d_child, numParticles, d_subDomainHandler, 21);

    //KernelHandler.createDomainList(d_x, d_y, d_z, d_mass, d_min_x, d_max_x,
      //                                  d_min_y, d_max_y, d_min_z, d_max_z, d_child, numParticles,
        //                                d_subDomainHandler, 21);

    //KernelHandler.createDomainList(d_subDomainHandler, 21, d_domainListIndices, d_domainListLevels, d_domainListIndex);


    time_buildTree[step] = elapsedTimeKernel;
    if (timeKernels) {
        Logger(TIME) << "\tBuilding tree: " << elapsedTimeKernel << " ms";
    }

    elapsedTimeKernel = KernelHandler.centreOfMass(d_x, d_y, d_z, d_mass, d_index, numParticles, timeKernels);

    time_centreOfMass[step] = elapsedTimeKernel;
    if (timeKernels) {
        Logger(TIME) << "\tCenter of mass: " << elapsedTimeKernel << " ms";
    }

    elapsedTimeKernel = KernelHandler.sort(d_count, d_start, d_sorted, d_child, d_index, numParticles, timeKernels);
    //elapsedTimeKernel = 0;

    time_sort[step] = elapsedTimeKernel;
    if (timeKernels) {
        Logger(TIME) << "\tSort particles: " << elapsedTimeKernel << " ms";
    }

    elapsedTimeKernel = KernelHandler.computeForces(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_mass, d_sorted, d_child,
                          d_min_x, d_max_x, numParticles, parameters.gravity, timeKernels);

    time_computeForces[step] = elapsedTimeKernel;
    if (timeKernels) {
        Logger(TIME) << "\tCompute forces: " << elapsedTimeKernel << " ms";
    }

    elapsedTimeKernel = KernelHandler.update(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, numParticles,
                   parameters.timestep, parameters.dampening, timeKernels);


    time_update[step] = elapsedTimeKernel;
    if (timeKernels) {
        Logger(TIME) << "\tUpdate particles: " << elapsedTimeKernel << " ms";
    }

    // copy from device to host
    cudaEvent_t start_t, stop_t; // used for timing
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);
    cudaEventRecord(start_t, 0);

    cudaMemcpy(h_x, d_x, 2*numParticles*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, 2*numParticles*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, 2*numParticles*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vx, d_vx, 2*numParticles*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vy, d_vy, 2*numParticles*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vz, d_vz, 2*numParticles*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaEventElapsedTime(&elapsedTimeKernel, start_t, stop_t);
    cudaEventDestroy(start_t);
    cudaEventDestroy(stop_t);
    // END copy from device to host

    cudaDeviceSynchronize();

    time_copyDeviceToHost[step] = elapsedTimeKernel;
    if (timeKernels) {
        Logger(TIME) << "\tCopying to host: " << elapsedTimeKernel << " ms";
    }

    //std::cout << "x[0]: " << h_x[0] << std::endl;
    //std::cout << "v[0]: " << h_vx[0] << std::endl;


    cudaEventRecord(stop_global, 0);
    cudaEventSynchronize(stop_global);
    cudaEventElapsedTime(&elapsedTime, start_global, stop_global);
    cudaEventDestroy(start_global);
    cudaEventDestroy(stop_global);

    time_all[step] = elapsedTime;
    Logger(TIME) << "Elapsed time for step " << step << " : " << elapsedTime << " ms";

    step++;
}


void BarnesHut::plummerModel(float *mass, float *x, float* y, float *z,
                                    float *x_vel, float *y_vel, float *z_vel,
                                    float *x_acc, float *y_acc, float *z_acc, int n)
{
    float a = 1.0;
    float pi = 3.14159265;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0, 1.0);
    std::uniform_real_distribution<float> distribution2(0, 0.1);
    std::uniform_real_distribution<float> distribution_phi(0.0, 2 * pi);
    std::uniform_real_distribution<float> distribution_theta(-1.0, 1.0);

    // loop through all particles
    for (int i = 0; i < n; i++){
        float phi = distribution_phi(generator);
        float theta = acos(distribution_theta(generator));
        float r = a / sqrt(pow(distribution(generator), -0.666666) - 1);

        // set mass and position of particle
        mass[i] = 1.0;
        x[i] = r*cos(phi);
        y[i] = r*sin(phi);
        if (i%2==0) {
            z[i] = i*0.001;
        }
        else {
            z[i] = i*-0.001;
        }

        // set velocity of particle
        float s = 0.0;
        float t = 0.1;
        while(t > s*s*pow(1.0 - s*s, 3.5)){
            s = distribution(generator);
            t = distribution2(generator);
        }
        float v = 100*s*sqrt(2)*pow(1.0 + r*r, -0.25);
        phi = distribution_phi(generator);
        theta = acos(distribution_theta(generator));
        x_vel[i] = v*cos(phi);
        y_vel[i] = v*sin(phi);
        z_vel[i] = 0.0;

        // set acceleration to zero
        x_acc[i] = 0.0;
        y_acc[i] = 0.0;
        z_acc[i] = 0.0;

    }

}

void BarnesHut::diskModel(float *mass, float *x, float* y, float* z, float *x_vel, float *y_vel, float *z_vel,
                                 float *x_acc, float *y_acc, float *z_acc, int n)
{
    float a = 1.0;
    float pi = 3.14159265;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(1.5, 12.0);
    std::uniform_real_distribution<float> distribution_theta(0.0, 2 * pi);

    float solarMass = 100000;

    // loop through all particles
    for (int i = 0; i < n; i++) {

        float theta = distribution_theta(generator);
        float r = distribution(generator);

        // set mass and position of particle
        if (i==0) {
            mass[i] = solarMass; //100000;
            x[i] = 0;
            y[i] = 0;
            z[i] = 0;
        }
        else {
            mass[i] = 2*solarMass/numParticles;
            x[i] = r*cos(theta);
            y[i] = r*sin(theta);

            if (i%2 == 0) {
                z[i] = i*1e-7;
            }
            else {
                z[i] = i*-1e-7;
            }
        }


        // set velocity of particle
        float rotation = 1;  // 1: clockwise   -1: counter-clockwise
        float v = sqrt(solarMass / (r));

        if (i == 0) {
            x_vel[0] = 0.0;
            y_vel[0] = 0.0;
            z_vel[0] = 0.0;
        }
        else{
            x_vel[i] = rotation*v*sin(theta);
            y_vel[i] = -rotation*v*cos(theta);
            z_vel[i] = 0.0;
        }

        // set acceleration to zero
        x_acc[i] = 0.0;
        y_acc[i] = 0.0;
        z_acc[i] = 0.0;
    }

}

float BarnesHut::getSystemSize() {

    float x_max = 0;
    float y_max = 0;
    float z_max = 0;

    for (int i = 0; i < numParticles; i++) {
        if (abs(h_x[i]) > x_max) {
            x_max = abs(h_x[i]);
        }
        if (abs(h_y[i]) > y_max) {
            y_max = abs(h_y[i]);
        }
        if (abs(h_z[i]) > z_max) {
            z_max = abs(h_z[i]);
        }
    }

    //std::cout << "system size x_max: " << x_max << std::endl;
    //std::cout << "system size y_max: " << y_max << std::endl;
    //std::cout << "system size z_max: " << z_max << std::endl;

    float systemSize = x_max;
    if (y_max > systemSize) {
        systemSize = y_max;
    }
    if (z_max > systemSize) {
        systemSize = z_max;
    }

    return systemSize;

}




