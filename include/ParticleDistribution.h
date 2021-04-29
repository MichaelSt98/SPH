#ifndef OOP_PARTICLEDISTRIBUTION_H
#define OOP_PARTICLEDISTRIBUTION_H

#include "Logger.h"
#include "Particle.h"
#include "ConfigParser.h"
#include "Constants.h"

#include <boost/mpi.hpp>
#include <random>

class ParticleDistribution {
private:
    void initDisk(ParticleList &pList);
    void initCluster(ParticleList &pList);
public:
    enum type
    {
        disk, cluster
    };

    boost::mpi::communicator comm;
    int numParticles;
    float systemSize;
    float initVelocity;
    float mass;

    ParticleDistribution(int numParticles, float systemSize, float initVelocity, float mass);
    ParticleDistribution(ConfigParser &confP);

    void initParticles(ParticleList &pList, type distributionType);
};


#endif //OOP_PARTICLEDISTRIBUTION_H
