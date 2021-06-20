#include "../include/Body.h"
#include "../include/Constants.h"
#include "../include/Renderer.h"
#include "../include/Logger.h"
#include "../include/Timer.h"
#include "../include/KernelsWrapper.cuh"
#include "../include/BarnesHut.cuh"
#include "../include/cxxopts.h"

//#include <mpi.h>
#include <fenv.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <random>

#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"

void SetDeviceBeforeInit()
{
    char * localRankStr = NULL;
    int rank = 0, devCount = 2;

    // We extract the local rank initialization using an environment variable
    if ((localRankStr = getenv(ENV_LOCAL_RANK)) != NULL)
    {
        rank = atoi(localRankStr);
    }
    Logger(INFO) << "devCount: " << devCount << " | rank: " << rank
    << " | setting device: " << rank % devCount;
    SafeCudaCall(cudaGetDeviceCount(&devCount));
    SafeCudaCall(cudaSetDevice(rank % devCount));
}

structlog LOGCFG = {};

int main(int argc, char** argv)
{

    //SetDeviceBeforeInit();

    MPI_Init(&argc, &argv);

    int rank;
    int numProcesses;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    printf("Hello World from proc %i out of %i\n", rank, numProcesses);

    if (true/*rank == 0*/) {

        cudaSetDevice(rank);

        int device;
        cudaGetDevice(&device);
        Logger(INFO) << "Set device to " << device;

        cxxopts::Options options("HPC NBody", "Multi-GPU CUDA Barnes-Hut NBody code");

        bool render = false;

        options.add_options()
                ("r,render", "render simulation", cxxopts::value<bool>(render))
                ("i,iterations", "number of iterations", cxxopts::value<int>()->default_value("100"))
                ("n,particles", "number of particles", cxxopts::value<int>()->default_value("524288")) //"524288"
                ("b,blocksize", "block size", cxxopts::value<int>()->default_value("256"))
                ("g,gridsize", "grid size", cxxopts::value<int>()->default_value("1024"))
                ("R,renderinterval", "render interval", cxxopts::value<int>()->default_value("10"))
                ("v,verbosity", "Verbosity level")
                ("h,help", "Show this help");

        auto result = options.parse(argc, argv);

        //render = result["render"].as<bool>();

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }


        /** Initialization */
        SimulationParameters parameters;

        parameters.iterations = result["iterations"].as<int>(); //500;
        parameters.numberOfParticles = result["particles"].as<int>(); //512*256*4;
        parameters.timestep = 0.001;
        parameters.gravity = 1.0;
        parameters.dampening = 1.0;
        parameters.gridSize = result["gridsize"].as<int>(); //1024;
        parameters.blockSize = result["blocksize"].as<int>(); //256;
        parameters.warp = 32;
        parameters.stackSize = 64;
        parameters.renderInterval = result["renderinterval"].as<int>(); //10;
        parameters.timeKernels = true;

        LOGCFG.headers = true;
        /*if (result.count("verbosity")) {
            int count = (int) result.count("verbosity");
            //std::cout << "counter: " << count << std::endl;
            if (count == 1) {
                LOGCFG.level = INFO;
            } else if (count == 2) {
                LOGCFG.level = ERROR;
            } else if (count == 3) {
                LOGCFG.level = WARN;
            } else {
                LOGCFG.level = DEBUG;
            }
        }*/
        LOGCFG.level = DEBUG;
        LOGCFG.myrank = rank;
        //LOGCFG.outputRank = 0;

        //Logger(DEBUG) << "DEBUG output";
        //Logger(WARN) << "WARN output";
        //Logger(ERROR) << "ERROR output";
        //Logger(INFO) << "INFO output";
        //Logger(TIME) << "TIME output";

        char *image = new char[2 * WIDTH * HEIGHT * 3];
        double *hdImage = new double[2* WIDTH * HEIGHT * 3];

        int *processNum = new int[parameters.numberOfParticles];

        Body *bodies = new Body[parameters.numberOfParticles];

        BarnesHut *particles = new BarnesHut(parameters);

        Renderer renderer{parameters.numberOfParticles, WIDTH, HEIGHT, DEPTH, RENDER_SCALE, MAX_VEL_COLOR, MIN_VEL_COLOR,
                          PARTICLE_BRIGHTNESS, PARTICLE_SHARPNESS, DOT_SIZE,
                          2 * particles->getSystemSize(), parameters.renderInterval};


        /** Simulation */
        for (int i = 0; i < parameters.iterations; i++) {

            particles->update(i);

            /**
             * Output
             * * optimize (not yet optimized for code structure)
             * */
            if (render && rank == 0) {
                for (int i_body = 0; i_body < parameters.numberOfParticles; i_body++) {

                    Body *current;
                    current = &bodies[i_body];
                    //current->position.x = particles->h_x[i_body];
                    //current->position.y = particles->h_y[i_body];
                    //current->position.z = particles->h_z[i_body];
                    //current->velocity.x = particles->h_vx[i_body];
                    //current->velocity.y = particles->h_vy[i_body];
                    //current->velocity.z = particles->h_vz[i_body];
                    current->position.x = particles->all_x[i_body];
                    current->position.y = particles->all_y[i_body];
                    current->position.z = particles->all_z[i_body];
                    current->velocity.x = particles->all_vx[i_body];
                    current->velocity.y = particles->all_vy[i_body];
                    current->velocity.z = particles->all_vz[i_body];

                    if (i_body < particles->numParticlesLocal) {
                        processNum[i_body] = 0;
                    }
                    else {
                        processNum[i_body] = 1;
                    }
                }
                if (i % parameters.renderInterval == 0) {
                    renderer.createFrame(image, hdImage, bodies, processNum, numProcesses, i);
                }
            }
        }

        /** Postprocessing */
        float total_time_resetArrays = 0.f;
        float total_time_computeBoundingBox = 0.f;
        float total_time_buildTree = 0.f;
        float total_time_centreOfMass = 0.f;
        float total_time_sort = 0.f;
        float total_time_computeForces = 0.f;
        float total_time_update = 0.f;
        float total_time_copyDeviceToHost = 0.f;
        float total_time_all = 0.f;

        for (int i = 0; i < parameters.iterations; i++) {

            total_time_resetArrays += particles->time_resetArrays[i];
            total_time_computeBoundingBox += particles->time_computeBoundingBox[i];
            total_time_buildTree += particles->time_buildTree[i];
            total_time_centreOfMass += particles->time_centreOfMass[i];
            total_time_sort += particles->time_sort[i];
            total_time_computeForces += particles->time_computeForces[i];
            total_time_update += particles->time_update[i];
            total_time_copyDeviceToHost += particles->time_copyDeviceToHost[i];
            total_time_all += particles->time_all[i];

        }

        Logger(INFO) << "----------------------FINISHED----------------------";
        Logger(INFO) << "";

        Logger(TIME) << "Time to reset arrays: " << total_time_resetArrays << "ms";
        Logger(TIME) << "\tper step: " << total_time_resetArrays / parameters.iterations << "ms";

        Logger(TIME) << "Time to compute bounding boxes: " << total_time_computeBoundingBox << "ms";
        Logger(TIME) << "\tper step: " << total_time_computeBoundingBox / parameters.iterations << "ms";

        Logger(TIME) << "Time to build tree: " << total_time_buildTree << "ms";
        Logger(TIME) << "\tper step: " << total_time_buildTree / parameters.iterations << "ms";

        Logger(TIME) << "Time to compute COM: " << total_time_centreOfMass << "ms";
        Logger(TIME) << "\tper step: " << total_time_centreOfMass / parameters.iterations << "ms";

        Logger(TIME) << "Time to sort: " << total_time_sort << "ms";
        Logger(TIME) << "\tper step: " << total_time_sort / parameters.iterations << "ms";

        Logger(TIME) << "Time to compute forces: " << total_time_computeForces << "ms";
        Logger(TIME) << "\tper step: " << total_time_computeForces / parameters.iterations << "ms";

        Logger(TIME) << "Time to update bodies: " << total_time_update << "ms";
        Logger(TIME) << "\tper step: " << total_time_update / parameters.iterations << "ms";

        Logger(TIME) << "Time to copy from device to host: " << total_time_copyDeviceToHost << "ms";
        Logger(TIME) << "\tper step: " << total_time_copyDeviceToHost / parameters.iterations << "ms";

        Logger(TIME) << "----------------------------------------------";
        Logger(TIME) << "TOTAL TIME: " << total_time_all << "ms";
        Logger(TIME) << "\tper step: " << total_time_all / parameters.iterations << "ms";
        Logger(TIME) << "TOTAL TIME (without copying): " << total_time_all - total_time_copyDeviceToHost << "ms";
        Logger(TIME) << "\tper step: " << (total_time_all - total_time_copyDeviceToHost) / parameters.iterations
                     << "ms";
        Logger(TIME) << "----------------------------------------------";

        Logger(INFO) << "Number of particles: " << parameters.numberOfParticles;
        Logger(INFO) << "Number of iterations: " << parameters.iterations;

        /** Cleaning */
        delete[] image;

    }

    MPI_Finalize();
    return 0;
}
