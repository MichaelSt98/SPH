#include "../include/BarnesHut.cuh"

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

BarnesHut::BarnesHut(const SimulationParameters p) {

    parameters = p;
    KernelHandler = KernelsWrapper(p);
    step = 0;

    h_subDomainHandler = new SubDomainKeyTree();
    /*h_subDomainHandler->rank =*/
    MPI_Comm_rank(MPI_COMM_WORLD, &h_subDomainHandler->rank); //0;
    h_subDomainHandler->range = new unsigned long[3];
    h_subDomainHandler->range[0] = 0;
    h_subDomainHandler->range[1] = 2305843009213693952UL; //4611686018427387904UL;// + 3872UL;
    h_subDomainHandler->range[2] = KEY_MAX;
    /*h_subDomainHandler->numProcesses =*/
    MPI_Comm_size(MPI_COMM_WORLD, &h_subDomainHandler->numProcesses); //2;

    numParticles = p.numberOfParticles; //NUM_BODIES;
    numNodes = 2 * numParticles + 12000; //+ 12000; //2 * numParticles + 12000;
    numParticlesLocal = numParticles/h_subDomainHandler->numProcesses;

    Logger(DEBUG) << "numParticles: " << numParticles << "  numParticlesLocal: "
                        << numParticlesLocal << "  numNodes:" << numNodes;

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
    //h_output = new float[2*numNodes];

    time_resetArrays = new float[parameters.iterations];
    time_computeBoundingBox = new float[parameters.iterations];
    time_buildTree = new float[parameters.iterations];
    time_centreOfMass = new float[parameters.iterations];
    time_sort = new float[parameters.iterations];
    time_computeForces = new float[parameters.iterations];
    time_update = new float[parameters.iterations];
    time_copyDeviceToHost = new float[parameters.iterations];
    time_all = new float [parameters.iterations];

    printf("rank: %i  numProcesses: %i\n", h_subDomainHandler->rank, h_subDomainHandler->numProcesses);

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

    gpuErrorcheck(cudaMalloc((void**)&d_tempArray, 2*numParticles*sizeof(float)));
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

    //gpuErrorcheck(cudaMalloc((void**)&d_output, 2*numNodes*sizeof(float)));

    //plummerModel(h_mass, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_ax, h_ay, h_az, numParticles);
    diskModel(h_mass, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_ax, h_ay, h_az, numParticlesLocal); //numParticles);


    // copy data to GPU device

    cudaMemcpy(d_subDomainHandler, h_subDomainHandler, sizeof(SubDomainKeyTree), cudaMemcpyHostToDevice);
    cudaMemcpy(d_range, h_subDomainHandler->range, size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_subDomainHandler->range), &d_range, sizeof(unsigned long*), cudaMemcpyHostToDevice);

    //cudaMemcpy(d_subDomainHandler, h_subDomainHandler, sizeof(*h_subDomainHandler), cudaMemcpyHostToDevice);
    //cudaMemcpy(&d_subDomainHandler->rank, &h_subDomainHandler->rank, sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(&d_subDomainHandler->numProcesses, &h_subDomainHandler->numProcesses, sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(&d_subDomainHandler->range, &h_subDomainHandler->range, 3*sizeof(unsigned long), cudaMemcpyHostToDevice);

    //cudaMemcpy(d_mass, h_mass, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass, numParticlesLocal*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_domainListIndices, h_domainListIndices, DOMAIN_LIST_SIZE*sizeof(unsigned long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_domainListKeys, h_domainListKeys, DOMAIN_LIST_SIZE*sizeof(unsigned long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_domainListLevels, h_domainListLevels, DOMAIN_LIST_SIZE*sizeof(int), cudaMemcpyHostToDevice);
    gpuErrorcheck(cudaMemset(d_domainListIndex, 0, sizeof(int)));

    /*cudaMemcpy(d_x, h_x, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ax, h_ax, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ay, h_ay, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_az, h_az, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);*/

    cudaMemcpy(d_x,  h_x,  numParticlesLocal*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,  h_y,  numParticlesLocal*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z,  h_z,  numParticlesLocal*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, numParticlesLocal*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, numParticlesLocal*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, numParticlesLocal*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ax, h_ax, numParticlesLocal*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ay, h_ay, numParticlesLocal*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_az, h_az, numParticlesLocal*sizeof(float), cudaMemcpyHostToDevice);

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
    //delete [] h_output;

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

    //gpuErrorcheck(cudaFree(d_output));

    cudaDeviceSynchronize();
}

void BarnesHut::update(int step)
{

    int device;
    cudaGetDevice(&device);
    Logger(INFO) << "&d_sortArrayOut = " << d_sortArrayOut << " on device: " << device;

    /*RESETTING ARRAYS*************************************************************************/
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
    /*resetting arrays*************************************************************************/

    /*COMPUTE BOUNDING BOX*********************************************************************/
    elapsedTimeKernel = KernelHandler.computeBoundingBox(d_mutex, d_x, d_y, d_z, d_min_x, d_max_x, d_min_y, d_max_y,
                               d_min_z, d_max_z, numParticles, timeKernels);

    globalizeBoundingBox();

    time_computeBoundingBox[step] = elapsedTimeKernel;
    if (timeKernels) {
        Logger(TIME) << "\tBounding box: " << elapsedTimeKernel << " ms";
    }
    /*compute bounding box*********************************************************************/

    /*COMPUTE BOUNDING BOX*********************************************************************/

    KernelHandler.particlesPerProcess(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
                                      d_min_z, d_max_z, numParticlesLocal, numNodes, d_subDomainHandler, d_procCounter, d_procCounterTemp);

    KernelHandler.sortParticlesProc(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
                                      d_min_z, d_max_z, numParticlesLocal, numNodes, d_subDomainHandler, d_procCounter, d_procCounterTemp,
                                      d_sortArray);

    //KernelHandler.sendParticles(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
    //                            d_min_z, d_max_z, numParticles, numNodes, d_subDomainHandler, d_procCounter, d_tempArray,
    //                            d_sortArray, d_sortArrayOut);

    //KernelHandler.treeInfo(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
      //                     d_min_z, d_max_z, numParticlesLocal, numNodes, d_procCounter, d_subDomainHandler);

//#if TESTING
    Logger(INFO) << "RADIX SORT";

    float elapsedTimeSorting = 0.f;
    cudaEvent_t start_t_sorting, stop_t_sorting; // used for timing
    cudaEventCreate(&start_t_sorting);
    cudaEventCreate(&stop_t_sorting);
    cudaEventRecord(start_t_sorting, 0);

    // position
    sortArrayRadix(d_x, d_tempArray, d_sortArray, d_sortArrayOut, numParticlesLocal);
    KernelHandler.copyArray(d_x, d_tempArray, numParticlesLocal);
    sortArrayRadix(d_y, d_tempArray, d_sortArray, d_sortArrayOut, numParticlesLocal);
    KernelHandler.copyArray(d_y, d_tempArray, numParticlesLocal);
    sortArrayRadix(d_z, d_tempArray, d_sortArray, d_sortArrayOut, numParticlesLocal);
    KernelHandler.copyArray(d_z, d_tempArray, numParticlesLocal);

    // velocity
    sortArrayRadix(d_vx, d_tempArray, d_sortArray, d_sortArrayOut, numParticlesLocal);
    KernelHandler.copyArray(d_vx, d_tempArray, numParticlesLocal);
    sortArrayRadix(d_vy, d_tempArray, d_sortArray, d_sortArrayOut, numParticlesLocal);
    KernelHandler.copyArray(d_vy, d_tempArray, numParticlesLocal);
    sortArrayRadix(d_vz, d_tempArray, d_sortArray, d_sortArrayOut, numParticlesLocal);
    KernelHandler.copyArray(d_vz, d_tempArray, numParticlesLocal);

    // acceleration
    sortArrayRadix(d_ax, d_tempArray, d_sortArray, d_sortArrayOut, numParticlesLocal);
    KernelHandler.copyArray(d_ax, d_tempArray, numParticlesLocal);
    sortArrayRadix(d_ay, d_tempArray, d_sortArray, d_sortArrayOut, numParticlesLocal);
    KernelHandler.copyArray(d_ay, d_tempArray, numParticlesLocal);
    sortArrayRadix(d_az, d_tempArray, d_sortArray, d_sortArrayOut, numParticlesLocal);
    KernelHandler.copyArray(d_az, d_tempArray, numParticlesLocal);

    // mass
    sortArrayRadix(d_mass, d_tempArray, d_sortArray, d_sortArrayOut, numParticlesLocal);
    KernelHandler.copyArray(d_mass, d_tempArray, numParticlesLocal);

    cudaEventRecord(stop_t_sorting, 0);
    cudaEventSynchronize(stop_t_sorting);
    cudaEventElapsedTime(&elapsedTimeSorting, start_t_sorting, stop_t_sorting);
    cudaEventDestroy(start_t_sorting);
    cudaEventDestroy(stop_t_sorting);

    Logger(TIME) << "\tSorting for process: " << elapsedTimeSorting << "ms";

    //cudaMemcpy(h_x, d_x, 2*numParticles*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_procCounter, d_procCounter, h_subDomainHandler->numProcesses*sizeof(int), cudaMemcpyDeviceToHost);

    for (int proc=0; proc<h_subDomainHandler->numProcesses; proc++) {
        printf("[rank %i] HOST: procCounter[%i] = %i\n", h_subDomainHandler->rank, proc, h_procCounter[proc]);
    }


    float elapsedTimeSending = 0.f;
    cudaEvent_t start_t_sending, stop_t_sending; // used for timing
    cudaEventCreate(&start_t_sending);
    cudaEventCreate(&stop_t_sending);
    cudaEventRecord(start_t_sending, 0);


    //send particles
    /*------------------------------------------------------------------------------------------------------------*/
    int *sendLengths;
    sendLengths = new int[h_subDomainHandler->numProcesses];
    sendLengths[h_subDomainHandler->rank] = 0;
    int *receiveLengths;
    receiveLengths = new int[h_subDomainHandler->numProcesses];
    receiveLengths[h_subDomainHandler->rank] = 0;

    for (int proc=0; proc < h_subDomainHandler->numProcesses; proc++) {
        if (proc != h_subDomainHandler->rank) {
            sendLengths[proc] = h_procCounter[proc];
        }
    }

    int reqCounter = 0;
    MPI_Request reqMessageLengths[h_subDomainHandler->numProcesses-1];
    MPI_Status statMessageLengths[h_subDomainHandler->numProcesses-1];

    //send plistLengthSend and receive plistLengthReceive
    for (int proc=0; proc < h_subDomainHandler->numProcesses; proc++) {
        if (proc != h_subDomainHandler->rank) {
            MPI_Isend(&sendLengths[proc], 1, MPI_INT, proc, 17, MPI_COMM_WORLD, &reqMessageLengths[reqCounter]);
            MPI_Recv(&receiveLengths[proc], 1, MPI_INT, proc, 17, MPI_COMM_WORLD, &statMessageLengths[reqCounter]);
            reqCounter++;
        }
    }
    MPI_Waitall(h_subDomainHandler->numProcesses-1, reqMessageLengths, statMessageLengths);

    for (int proc=0; proc < h_subDomainHandler->numProcesses; proc++) {
        printf("[rank %i] reveiceLengths[%i] = %i  sendLengths[%i] = %i\n", h_subDomainHandler->rank,
               proc, receiveLengths[proc], proc, sendLengths[proc]);
    }

    KernelHandler.treeInfo(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
                           d_min_z, d_max_z, numParticlesLocal, numNodes, d_procCounter, d_subDomainHandler, d_sortArray,
                           d_sortArrayOut);


    /*-------------------------------------------------------*/

    //send and receive particles
    //d_x;
    //d_tempArray;

#if CUDA_AWARE_MPI_TESTING
    // ------------------CUDA aware MPI Testing ----------------------------------------------------------------------
    MPI_Request reqTest[h_subDomainHandler->numProcesses - 1];
    MPI_Status statTest[h_subDomainHandler->numProcesses - 1];

    reqCounter = 0;

    for (int proc=0; proc < h_subDomainHandler->numProcesses; proc++) {
        if (proc != h_subDomainHandler->rank) {
            MPI_Isend(d_sortArrayOut, 10, MPI_INT, proc, 0, MPI_COMM_WORLD, &reqTest[reqCounter]);
            MPI_Recv(d_sortArray, 10, MPI_INT, proc, 0, MPI_COMM_WORLD, &statTest[reqCounter]);
            reqCounter++;
        }
    }

    MPI_Waitall(h_subDomainHandler->numProcesses-1, reqTest, statTest);
    // ------------------CUDA aware MPI Testing ----------------------------------------------------------------------
#endif

    Logger(INFO) << "&d_sortArrayOut = " << d_sortArrayOut;

    /*MPI_Request reqParticles[h_subDomainHandler->numProcesses - 1];
    MPI_Status statParticles[h_subDomainHandler->numProcesses - 1];

    reqCounter = 0;
    //int sendOffset = 0;
    int receiveOffset = 0;

    for (int proc=0; proc < h_subDomainHandler->numProcesses; proc++) {
        if (proc != h_subDomainHandler->rank) {
            if (proc == 0) {
                MPI_Isend(&d_x[0], sendLengths[proc], MPI_FLOAT, proc, 17,
                          MPI_COMM_WORLD, &reqParticles[reqCounter]);
            }
            else {
                MPI_Isend(&d_x[h_procCounter[proc-1]], sendLengths[proc], MPI_FLOAT, proc, 17,
                          MPI_COMM_WORLD, &reqParticles[reqCounter]);
            }
            MPI_Recv(&d_tempArray[0] + receiveOffset, receiveLengths[proc], MPI_FLOAT, proc, 17,
                     MPI_COMM_WORLD, &statParticles[reqCounter]);
            receiveOffset += receiveLengths[proc];
            reqCounter++;
        }
    }

    MPI_Waitall(h_subDomainHandler->numProcesses-1, reqParticles, statParticles);*/

    sendParticlesEntry(sendLengths, receiveLengths, d_x);
    sendParticlesEntry(sendLengths, receiveLengths, d_y);
    sendParticlesEntry(sendLengths, receiveLengths, d_z);

    sendParticlesEntry(sendLengths, receiveLengths, d_vx);
    sendParticlesEntry(sendLengths, receiveLengths, d_vz);
    sendParticlesEntry(sendLengths, receiveLengths, d_vy);

    sendParticlesEntry(sendLengths, receiveLengths, d_ax);
    sendParticlesEntry(sendLengths, receiveLengths, d_ay);
    sendParticlesEntry(sendLengths, receiveLengths, d_az);

    numParticlesLocal = sendParticlesEntry(sendLengths, receiveLengths, d_mass);



//#if TESTING

    // delete sent entries and copy received entries
    /*if (h_subDomainHandler->rank != 0) {
        KernelHandler.copyArray(d_x, &d_x[h_procCounter[h_subDomainHandler->rank - 1]], h_procCounter[h_subDomainHandler->rank]); //float *targetArray, float *sourceArray, int n)
    }

    KernelHandler.resetFloatArray(d_x, 0, h_procCounter[h_subDomainHandler->rank]); //resetFloatArrayKernel(float *array, float value, int n)
    KernelHandler.copyArray(&d_x[h_procCounter[h_subDomainHandler->rank]], d_tempArray, receiveOffset);

    printf("FINISHED!!!\n");
    */
    delete[] sendLengths;
    delete[] receiveLengths;
    /*------------------------------------------------------------------------------------------------------------*/

    cudaEventRecord(stop_t_sending, 0);
    cudaEventSynchronize(stop_t_sending);
    cudaEventElapsedTime(&elapsedTimeSending, start_t_sending, stop_t_sending);
    cudaEventDestroy(start_t_sending);
    cudaEventDestroy(stop_t_sending);

    Logger(TIME) << "\tSending particles: " << elapsedTimeSending <<  "ms";

    //KernelHandler.sendParticles(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
      //                          d_min_z, d_max_z, numParticles, numNodes, d_subDomainHandler, d_procCounter, d_tempArray, d_sortArray, d_sortArrayOut);


//#endif

    KernelHandler.sendParticles(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
                                d_min_z, d_max_z, numParticlesLocal, numNodes, d_subDomainHandler, d_procCounter, d_tempArray,
                                d_sortArray, d_sortArrayOut);

    KernelHandler.treeInfo(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
                           d_min_z, d_max_z, numParticlesLocal, numNodes, d_procCounter, d_subDomainHandler, d_sortArray,
                           d_sortArrayOut);

    /*BUILDING TREE*************************************************************************/
    KernelHandler.createDomainList(d_subDomainHandler, 21, d_domainListKeys, d_domainListLevels, d_domainListIndex);
    //KernelHandler.buildDomainTree(d_domainListIndex, d_domainListKeys, d_domainListLevels, d_count, d_start, d_child,
      //                            d_index, numParticles, numNodes);

    elapsedTimeKernel = KernelHandler.buildTree(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
                      d_min_z, d_max_z, numParticlesLocal, numParticles, timeKernels); //numParticles -> numParticlesLocal

    KernelHandler.buildDomainTree(d_domainListIndex, d_domainListKeys, d_domainListLevels, d_count, d_start, d_child,
                                d_index, numParticlesLocal, numNodes); //TODO: numParticlesLocal or numParticles?

    //KernelHandler.buildDomainTree(d_domainListIndex, d_domainListKeys, d_domainListLevels, d_count, d_start, d_child,
      //                            d_index, numParticles, numNodes);

    //KernelHandler.treeInfo(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
      //                     d_min_z, d_max_z, numParticlesLocal, numNodes, d_procCounter, d_subDomainHandler, d_sortArray,
      //                     d_sortArrayOut);

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
    /*building tree*************************************************************************/

    /*CENTER OF MASS************************************************************************/
    elapsedTimeKernel = KernelHandler.centreOfMass(d_x, d_y, d_z, d_mass, d_index, numParticles, timeKernels);

    time_centreOfMass[step] = elapsedTimeKernel;
    if (timeKernels) {
        Logger(TIME) << "\tCenter of mass: " << elapsedTimeKernel << " ms";
    }
    /*center of mass************************************************************************/

    /*SORTING*******************************************************************************/
    elapsedTimeKernel = KernelHandler.sort(d_count, d_start, d_sorted, d_child, d_index, numParticles, timeKernels);
    //elapsedTimeKernel = 0;

    time_sort[step] = elapsedTimeKernel;
    if (timeKernels) {
        Logger(TIME) << "\tSort particles: " << elapsedTimeKernel << " ms";
    }
    /*sorting*******************************************************************************/

    /*COMPUTING FORCES**********************************************************************/
    elapsedTimeKernel = KernelHandler.computeForces(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_mass, d_sorted, d_child,
                          d_min_x, d_max_x, numParticlesLocal, parameters.gravity, timeKernels); //TODO: numParticlesLocal or numParticles?

    time_computeForces[step] = elapsedTimeKernel;
    if (timeKernels) {
        Logger(TIME) << "\tCompute forces: " << elapsedTimeKernel << " ms";
    }
    /*computing forces**********************************************************************/

    /*UPDATING******************************************************************************/
    elapsedTimeKernel = KernelHandler.update(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, numParticlesLocal,
                   parameters.timestep, parameters.dampening, timeKernels); //TODO: numParticlesLocal or numParticles?


    time_update[step] = elapsedTimeKernel;
    if (timeKernels) {
        Logger(TIME) << "\tUpdate particles: " << elapsedTimeKernel << " ms";
    }
    /*updating******************************************************************************/

    /*COPYING TO HOST***********************************************************************/
    cudaEvent_t start_t, stop_t; // used for timing
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);
    cudaEventRecord(start_t, 0);

    cudaMemcpy(h_x, d_x, numNodes*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, numNodes*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, numNodes*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vx, d_vx, numNodes*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vy, d_vy, numNodes*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vz, d_vz, numNodes*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaEventElapsedTime(&elapsedTimeKernel, start_t, stop_t);
    cudaEventDestroy(start_t);
    cudaEventDestroy(stop_t);
    /*copying to host***********************************************************************/

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
    Logger(INFO) << "-----------------------------------------------------------------------------------------";

    float *xAll, *yAll, *zAll;
    gatherParticles(xAll, yAll, zAll);

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
        if (h_subDomainHandler->rank == 0) {
            if (i == 0) {
                mass[i] = solarMass; //100000;
                x[i] = 0;
                y[i] = 0;
                z[i] = 0;
            } else {
                mass[i] = 2 * solarMass / numParticles;
                x[i] = r * cos(theta);
                y[i] = r * sin(theta);

                if (i % 2 == 0) {
                    z[i] = i * 1e-7;
                } else {
                    z[i] = i * -1e-7;
                }
            }
        }
        else {
            mass[i] = 2 * solarMass / numParticles;
            x[i] = (r + h_subDomainHandler->rank * 1.1e-1) * cos(theta);
            y[i] = (r + h_subDomainHandler->rank * 1.3e-1) * sin(theta);

            if (i % 2 == 0) {
                z[i] = i * 1e-7 * h_subDomainHandler->rank;
            } else {
                z[i] = i * -1e-7 * h_subDomainHandler->rank;
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

    float systemSize = x_max;
    if (y_max > systemSize) {
        systemSize = y_max;
    }
    if (z_max > systemSize) {
        systemSize = z_max;
    }

    float globalSystemSize;
    MPI_Allreduce(&systemSize, &globalSystemSize, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    //return systemSize;
    return globalSystemSize;

}

void BarnesHut::globalizeBoundingBox() {
    // X MIN
    MPI_Allreduce(MPI_IN_PLACE, d_min_x, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    // X MAX
    MPI_Allreduce(MPI_IN_PLACE, d_max_x, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    // Y MIN
    MPI_Allreduce(MPI_IN_PLACE, d_min_y, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    // Y MAX
    MPI_Allreduce(MPI_IN_PLACE, d_max_y, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    // Z MIN
    MPI_Allreduce(MPI_IN_PLACE, d_min_z, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    // Z MAX
    MPI_Allreduce(MPI_IN_PLACE, d_max_z, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
}

void BarnesHut::sortArrayRadix(float *arrayToSort, float *tempArray, int *keyIn, int *keyOut, int n) {
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    gpuErrorcheck(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    keyIn, keyOut, arrayToSort, tempArray, n));
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    gpuErrorcheck(cudaMalloc((void**)&d_temp_storage, temp_storage_bytes));

    // Run sorting operation
    gpuErrorcheck(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    keyIn, keyOut, arrayToSort, tempArray, n));

    cudaFree(d_temp_storage);
}
//alternatively using Thrust:
/*thrust::device_vector<int>  indices(N);
thrust::sequence(indices.begin(),indices.end());
thrust::sort_by_key(keys.begin(),keys.end(),indices.begin());

thrust::device_vector<int> temp(N);
thrust::device_vector<int> *sorted = &temp;
thrust::device_vector<int> *pa_01 = &a_01;
thrust::device_vector<int> *pa_02 = &a_02;
...
thrust::device_vector<int> *pa_20 = &a_20;

thrust::gather(indices.begin(), indices.end(), *pa_01, *sorted);
pa_01 = sorted; sorted = &a_01;
thrust::gather(indices.begin(), indices.end(), *pa_02, *sorted);
pa_02 = sorted; sorted = &a_02;
...
thrust::gather(indices.begin(), indices.end(), *pa_20, *sorted);
pa_20 = sorted; sorted = &a_20;*/


void BarnesHut::gatherParticles(float *xAll, float *yAll, float *zAll) {

    //calculate amount of particles for own process

    //gather these information
    //MPI_Gather(&pLength, 1, MPI_INT, pArrayReceiveLength, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //calculate total receive length and allocate memory

    //collect information
    //MPI_Gatherv(pArray, pLength, mpiParticle, pArrayAll, pArrayReceiveLength,
                //pArrayDisplacements, mpiParticle, 0, MPI_COMM_WORLD);
}

int BarnesHut::sendParticlesEntry(int *sendLengths, int *receiveLengths, float *entry) {
    MPI_Request reqParticles[h_subDomainHandler->numProcesses - 1];
    MPI_Status statParticles[h_subDomainHandler->numProcesses - 1];

    int reqCounter = 0;
    int receiveOffset = 0;

    for (int proc=0; proc < h_subDomainHandler->numProcesses; proc++) {
        if (proc != h_subDomainHandler->rank) {
            if (proc == 0) {
                MPI_Isend(&entry[0], sendLengths[proc], MPI_FLOAT, proc, 17,
                          MPI_COMM_WORLD, &reqParticles[reqCounter]);
            }
            else {
                MPI_Isend(&entry[h_procCounter[proc-1]], sendLengths[proc], MPI_FLOAT, proc, 17,
                          MPI_COMM_WORLD, &reqParticles[reqCounter]);
            }
            MPI_Recv(&d_tempArray[0] + receiveOffset, receiveLengths[proc], MPI_FLOAT, proc, 17,
                     MPI_COMM_WORLD, &statParticles[reqCounter]);
            receiveOffset += receiveLengths[proc];
            reqCounter++;
        }
    }

    MPI_Waitall(h_subDomainHandler->numProcesses-1, reqParticles, statParticles);

    int offset = 0;
    for (int i=0; i < h_subDomainHandler->rank; i++) {
        offset += h_procCounter[h_subDomainHandler->rank];
    }


    if (h_subDomainHandler->rank != 0) {
        KernelHandler.copyArray(&entry[0], &entry[offset - h_procCounter[h_subDomainHandler->rank]] /*&entry[h_procCounter[h_subDomainHandler->rank - 1]]*/, h_procCounter[h_subDomainHandler->rank]); //float *targetArray, float *sourceArray, int n)
    }

    KernelHandler.resetFloatArray(&entry[h_procCounter[h_subDomainHandler->rank]], 0, numParticles-h_procCounter[h_subDomainHandler->rank]); //resetFloatArrayKernel(float *array, float value, int n)
    KernelHandler.copyArray(&entry[h_procCounter[h_subDomainHandler->rank]], d_tempArray, receiveOffset);

    Logger(INFO) << "New local particle amount: " << receiveOffset + h_procCounter[h_subDomainHandler->rank]
                        << "  (receiveOffset = " << receiveOffset << ", procCounter = "
                        << h_procCounter[h_subDomainHandler->rank] << ")";

    return receiveOffset + h_procCounter[h_subDomainHandler->rank];
}




