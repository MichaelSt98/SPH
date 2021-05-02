
#include "../include/Logger.h"
//#include "../include/Keytype.h"
#include "../include/ConfigParser.h"
#include "../include/Particle.h"
//#include "../include/Tree.h"
#include "../include/Domain.h"
#include "../include/SubDomain.h"
#include "../include/Renderer.h"
#include "../include/ParticleDistribution.h"
#include "../include/Kernels.h"

#include <iostream>
#include <climits>
#include <boost/mpi.hpp>
#include <boost/exception/all.hpp>
#include <cxxopts/cxxopts.hpp>

structlog LOGCFG = {};

void timeIntegration(float t, float deltaT, float tEnd, float diam, float smoothingLength, SubDomain &subDomain,
                     Renderer &renderer, char *image, double *hdImage, bool loadBalancing, bool render=true, bool processColoring=false);

void timeIntegration(float t, float deltaT, float tEnd, float diam, float smoothingLength, SubDomain &subDomain,
                     Renderer &renderer, char *image, double *hdImage, bool loadBalancing, bool render, bool processColoring) {
    int step = 0;

    while (t < tEnd) {

        Logger(DEBUG) << " ";
        Logger(DEBUG) << "t = " << t;
        Logger(DEBUG) << "--------------------------";

        /** Load balancing */
        if (loadBalancing) {
            subDomain.newLoadDistribution();
            subDomain.root.clearDomainList();
            subDomain.createDomainList();
            subDomain.sendParticles();
            subDomain.compPseudoParticles();
        }
        /** END: Load balancing */

        // rendering
        if (render && step % renderer.getRenderInterval() == 0)
        {
            Particle *prtcls;
            int *prtN;

            ParticleList pList;
            IntList procList;
            KeyList keyList;
            if (processColoring) {
                subDomain.gatherParticles(pList, procList, keyList);
                prtN = new int[(int)procList.size()];
                //prtN = &procList[0];
                std::copy(procList.begin(), procList.end(), prtN);
            }
            else {
                subDomain.gatherParticles(pList);
            }
            prtcls = new Particle[(int)pList.size()];
            //prtcls = &pList[0];
            std::copy(pList.begin(), pList.end(), prtcls);

            if (subDomain.rank == 0) {
                Logger(INFO) << "Rendering timestep #" << step << ": N = " << pList.size();
                for (int i=0; i<subDomain.numProcesses+1; i++) {
                    Logger(INFO) << "\tRANGES: " << subDomain.range[i];
                }
                subDomain.root.printTreeSummary();
                renderer.setNumParticles((int)pList.size());
                if (processColoring) {
                    if (step % 50 == 0) {
                        subDomain.writeToTextFile(pList, procList, keyList, step);
                    }
                    renderer.createFrame(image, hdImage, prtcls, prtN, subDomain.numProcesses, step, &subDomain.root.box);
                    delete [] prtN;
                }
                else {
                    renderer.createFrame(image, hdImage, prtcls, step, &subDomain.root.box);
                }
                delete [] prtcls;
            }
            //output_tree(root, false);
        }
        ++step;

        t += deltaT; // update timestep

        subDomain.compFParallel(diam);

        subDomain.root.repairTree();

        //! NEW
        ////subDomain.nearNeighbourList(smoothingLength);
        //subDomain.sendParticlesSPH(smoothingLength);
        //subDomain.forcesSPH(smoothingLength);
        //! NEW

        subDomain.root.compX(deltaT);

        subDomain.root.compV(deltaT);

        subDomain.moveParticles();

        subDomain.sendParticles();

        subDomain.compPseudoParticles();

        subDomain.root.printTreeSummary(false);

    }
    Logger(DEBUG) << "t = " << t << ", FINISHED";
}

int main(int argc, char** argv) {

    //setting up MPI
    boost::mpi::environment env{argc, argv};
    boost::mpi::communicator comm;

    /*Logger(ERROR) << "Boost version: " << BOOST_VERSION / 100000     << "."  // major version
                  << BOOST_VERSION / 100 % 1000 << "."  // minor version
                  << BOOST_VERSION % 100;*/

    //settings for Logger
    LOGCFG.headers = true;
    LOGCFG.level = DEBUG;

    cxxopts::Options cmdLineParser("BarnesHut", "A parallel OOP implementation");
    cmdLineParser
            .set_width(100)
            .add_options()
                ("c,config", "Input config file", cxxopts::value<std::vector<std::string>>()
                        ->default_value("config/config.info")
                        /*->implicit_value("config/config.info")*/)
                ("h,help", "Show this help");

    auto opts = cmdLineParser.parse(argc, argv);

    if (opts.count("help")) {
        std::cout << cmdLineParser.help();
        exit(0);
    }

    std::string configFile = opts["config"].as<std::vector<std::string>>()[0];

    Logger(INFO) << "configFile: " << configFile;

    ConfigParser confP;
    try {
        confP = ConfigParser {ConfigParser(configFile.c_str())};
    }
    catch (std::exception const& e) {
        //std::string diag = diagnostic_information(e);
        Logger(ERROR) << e.what();
        exit(0);
    }

    SubDomain subDomainHandler(SubDomain::curveType(confP.getVal<int>("curveType")));//SubDomain::lebesgue);

    LOGCFG.myrank = subDomainHandler.rank;
    LOGCFG.outputRank = confP.getVal<int>("outputRank");

    KeyType k1 {20};
    KeyType k2 {40};
    KeyType k3 {7};

    Logger(INFO) << "k1 = " << k1.toIndex();
    Logger(INFO) << "k2 = " << k2.toIndex();
    Logger(INFO) << "k1 & k2 = " << (k1 & k2).toIndex();
    Logger(INFO) << "k1 | k2 = " << (k1 | k2).toIndex();
    Logger(INFO) << "k1 << 2 = " << (k1 << 2).toIndex();
    Logger(INFO) << "k1.maxLevel = " << k1.maxLevel;
    Logger(INFO) << "k2.maxLevel = " << k2.maxLevel;
    Logger(INFO) << "k3 = " << KeyType{k3};
    //Logger(INFO) << "k3 << (DIM * (k3.maxLevel - 1)) = " << (k3 << (DIM * (k3.maxLevel - 1)));
    //Logger(INFO) << "KEY_MAX = " << KeyType::KEY_MAX;
    //Logger(INFO) << "KEY_MAX = " << KeyType{ KeyType::KEY_MAX };
    //Logger(INFO) << "KEY_MAX.maxLevel = " << KeyType{ KeyType::KEY_MAX }.maxLevel;

    char *image;
    double *hdImage;
    int width = confP.getVal<int>("width");
    int height = confP.getVal<int>("height");
    image = new char[2 * width * height * 3];
    hdImage = new double[2 * width * height * 3];

    Renderer renderer(confP);

    const float systemSize{ confP.getVal<float>("systemSize") };
    Domain domain(systemSize);

    int numParticles = confP.getVal<int>("numParticles");

    Particle rootParticle {{0, 0, 0} };
    TreeNode helperTree { rootParticle , domain };
    subDomainHandler.root = helperTree;
    subDomainHandler.root.node = TreeNode::domainList;

    ParticleDistribution particleDistribution(confP);
    ParticleList particleList;
    ParticleDistribution::type distributionType = (ParticleDistribution::type)confP.getVal<int>("distributionType");
    particleDistribution.initParticles(particleList, distributionType);//ParticleDistribution::disk);


    for (auto it = std::begin(particleList); it != std::end(particleList); ++it) {
        subDomainHandler.root.insert(*it);
    }

    subDomainHandler.createRanges();

    TreeNode root { domain };
    subDomainHandler.root = root;

    subDomainHandler.createDomainList();

    for (auto it = std::begin(particleList); it != std::end(particleList); ++it) {
        subDomainHandler.root.insert(*it);
    }

    float diam = subDomainHandler.root.box.upper[0] - subDomainHandler.root.box.lower[0];
    float deltaT = confP.getVal<float>("timeStep");
    float tEnd = confP.getVal<float>("timeEnd");

    float smoothingLength = confP.getVal<float>("smoothingLength");

    bool loadBalancing = confP.getVal<bool>("loadBalancing");

    bool render = confP.getVal<bool>("render");
    bool processColoring = confP.getVal<bool>("processColoring");

    timeIntegration(0.f, deltaT, tEnd, diam, smoothingLength, subDomainHandler,
                    renderer, image, hdImage, loadBalancing, render, processColoring);

    return 0;
}
