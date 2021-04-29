//
// Created by Michael Staneker on 13.04.21.
//

#ifndef OOP_SUBDOMAIN_H
#define OOP_SUBDOMAIN_H

#include "Keytype.h"
#include "Tree.h"
#include "Density.h"
#include "Pressure.h"

#include <boost/mpi.hpp>
#include <boost/version.hpp>
#include <vector>
#include <map>
#include <fstream>
#include <string>

typedef std::map<KeyType, Particle> ParticleMap;

class SubDomain {
public:

    enum curveType {
        lebesgue, hilbert
    };

    boost::mpi::communicator comm;

    int rank;
    int numProcesses;
    KeyType *range;
    TreeNode root;
    curveType curve;

    SubDomain();
    SubDomain(curveType curve);

    std::string getCurveType() const;

    void moveParticles();

    void getParticleKeys(KeyList &keyList, KeyType k=0UL, int level=0);

    int key2proc(KeyType k, int level=-1, bool alreadyConverted=false);

    void createRanges();
    void newLoadDistribution();
    void updateRange(int &n, int &p, int *newDist);

    void createDomainList(TreeNode &t, int level, KeyType k);
    void createDomainList();

    void sendParticles();
    void buildSendList(TreeNode &t, ParticleList *pList, KeyType k, int level);

    void symbolicForce(TreeNode &td, TreeNode &t, float diam, ParticleMap &pMap, KeyType k=0UL, int level=0);
    //void symbolicForce(TreeNode &t, float diam, ParticleList pList, KeyType k=0UL, int level=0);
    void compPseudoParticles();
    void compF(TreeNode &t, float diam, KeyType k=0UL, int level=0);
    void compFParallel(float diam);
    void compTheta(TreeNode &t, ParticleMap *pMap, float diam, KeyType k=0UL, int level=0);

    void gatherKeys(KeyList &keyList, IntList &lengths, KeyList &localKeyList);
    void gatherParticles(ParticleList &pList);
    void gatherParticles(ParticleList &pList, IntList &processList);
    void gatherParticles(ParticleList &pList, IntList &processList, KeyList &keyList);

    void writeToTextFile(ParticleList &pList, IntList &processList, KeyList &keyList, int step=0);

    //void nearNeighbourList(tFloat radius);
    void sendParticlesSPH(tFloat radius);
    void forcesSPH(tFloat radius);
    void findInteractionPartnersOutsideDomain(TreeNode &t, Particle &particle, bool &interactionPartner, int &process,
                                              tFloat radius, KeyType k=0UL, int level=0);

private:
    void updateRangeHilbert(TreeNode &t, int &n, int &p, int *newDist, KeyType k=0UL, int level=0);
};


#endif //OOP_SUBDOMAIN_H
