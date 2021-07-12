//
// Created by Michael Staneker on 13.04.21.
//

#include "../include/SubDomain.h"

SubDomain::SubDomain() {
    rank = comm.rank();
    numProcesses = comm.size();
    range = new KeyType[numProcesses + 1];
    curve = lebesgue;
}

SubDomain::SubDomain(curveType curve) {
    rank = comm.rank();
    numProcesses = comm.size();
    range = new KeyType[numProcesses + 1];
    this->curve = curve;
}

std::string SubDomain::getCurveType() const {
    std::string curveTypeStr = "";
    switch (curve) {
        case 0:  curveTypeStr +=  "Lebesgue "; break;
        case 1:  curveTypeStr +=  "Hilbert  "; break;
        default: curveTypeStr +=  "not valid"; break;
    }
    return curveTypeStr;
}

void SubDomain::moveParticles() {
    root.resetParticleFlags();
    root.moveLeaf(root);
    root.repairTree();
}

void SubDomain::getParticleKeys(KeyList &keyList, KeyType k, int level) {
    switch (curve) {
        case 0: {
            root.getParticleKeys(keyList, k, level);
            break;
        }
        case 1: {
            IntList levelList;
            root.getParticleKeys(keyList, levelList, k, level);
            for (int i = 0; i < keyList.size(); i++) {
                keyList[i] = KeyType::Lebesgue2Hilbert(keyList[i], levelList[i]);
            }
            break;
        }
        default: {
            Logger(ERROR) << "getParticleKeys() not implemented for curve type: " << getCurveType();
        }
    }
}

int SubDomain::key2proc(KeyType k, int level, bool alreadyConverted) {
    if (curve == hilbert && !alreadyConverted) {
        k = KeyType::Lebesgue2Hilbert(k, level);
    }
    for (int proc=0; proc<numProcesses; proc++) {
        if (k >= range[proc] && k < range[proc+1]) {
            return proc;
        }
    }
    Logger(ERROR) << "key2proc(k=" << k << ") = -1!";
    return -1; //error
}

void SubDomain::createRanges() {

    KeyList kList;
    getParticleKeys(kList);

    KeyList globalKeyList;
    IntList particlesOnEachProcess;
    gatherKeys(globalKeyList, particlesOnEachProcess, kList);

    std::sort(globalKeyList.begin(), globalKeyList.end());

    for (int i=0; i<globalKeyList.size(); i++) {
        Logger(ERROR) << "globalKeyList[" << i << "] = " << globalKeyList[i];
    }

    int N = globalKeyList.size();
    const int ppr = (N % numProcesses != 0) ? N/numProcesses+1 : N/numProcesses;

    if (rank == 0) {
        range[0] = 0UL;
        for (int i = 1; i < numProcesses; i++) {
            range[i] = globalKeyList[i * ppr];
            Logger(ERROR) << "range[" << i << "] = " << range[i];
        }
        range[numProcesses] = KeyType{ KeyType::KEY_MAX };
    }

    boost::mpi::broadcast(comm, range, numProcesses+1, 0);
}

void SubDomain::newLoadDistribution() {
    int numParticles = root.getParticleCount();

    int *particleCounts = new int[numProcesses];
    boost::mpi::all_gather(comm, &numParticles, 1, particleCounts);

    int oldDist[numProcesses+1];
    int newDist[numProcesses+1];

    oldDist[0] = 0;
    for (int i=0; i < numProcesses; i++) {
        oldDist[i + 1] = oldDist[i] + particleCounts[i];
    }

    for (int i=0; i <= numProcesses; i++) {
        newDist[i] = (i * oldDist[numProcesses]) / numProcesses;
    }

    for (int i=0; i <= numProcesses; i++) {
        range[i] = 0UL;
    }

    int p = 0;
    int n = oldDist[rank];

    while (n > newDist[p]) {
        p++;
    }
    updateRange(n, p, newDist);

    range[0] = 0UL;
    range[numProcesses] = KeyType{ KeyType::KEY_MAX };

    KeyType sendRange[numProcesses+1];
    std::copy(range, range+numProcesses+1, sendRange);

    //boost::mpi::all_reduce(comm, sendRange, numProcesses+1, range, boost::mpi::maximum<KeyType>());
    boost::mpi::all_reduce(comm, sendRange, numProcesses+1, range, boost::mpi::KeyMaximum<KeyType>());

    /*for (int i=0; i <= numProcesses; i++){
        Logger(DEBUG) << "Load balancing: NEW range[" << i << "] = " << range[i];
    }*/

    delete [] particleCounts;
}

void SubDomain::updateRange(int &n, int &p, int *newDist) {
    switch (curve) {
        case 0: {
            root.updateRange(n, p, range, newDist);
            break;
        }
        case 1: {
            updateRangeHilbert(root, n, p, newDist);
            break;
        }
        default: {
            Logger(ERROR) << "updateRange() not implemented for curve type: " << getCurveType();
        }
    }
}

void SubDomain::updateRangeHilbert(TreeNode &t, int &n, int &p, int *newDist, KeyType k, int level) {
    std::map<KeyType, int> keyMap;
    for (int i=0; i<POWDIM; i++) {
        KeyType hilbert = KeyType::Lebesgue2Hilbert(k | (KeyType{ i } << (DIM*(k.maxLevel-level-1))),
                                                    level+1);
        keyMap[hilbert] = i;
    }
    for (std::map<KeyType, int>::iterator kit=keyMap.begin(); kit!=keyMap.end(); kit++) {
        if (t.son[kit->second] != NULL) {
            updateRangeHilbert(*t.son[kit->second], n, p, newDist,
                               k | KeyType{ kit->second << (DIM * (k.maxLevel - level - 1)) }, level + 1);
        }
    }

    if (t.isLeaf() && !t.isDomainList()) {
        while (n >= newDist[p]) {
            Logger(DEBUG) << "updateRangeHilbert(): Found lebesgue = " << k << ", level = " << level;
            range[p] = KeyType::Lebesgue2Hilbert(k , level); //TODO: check how to ensure subtree in process?
            Logger(DEBUG) << "updateRangeHilbert():       hilbert  = " << KeyType::Lebesgue2Hilbert(k, level);
            p++;
        }
        n++;
    }
}

void SubDomain::createDomainList(TreeNode &t, int level, KeyType k) {
    t.node = TreeNode::domainList;

    //Logger(INFO) << "KeyType k = " << k;

    int proc1;
    int proc2;
    if (curve == hilbert) {
        KeyType hilbert = KeyType::Lebesgue2Hilbert(k, level);
        proc1 = key2proc(hilbert, level, true);
        proc2 = key2proc(hilbert | (KeyType{ KeyType::KEY_MAX } >> (DIM * level + 1)), level, true);
    }
    else {
        proc1 = key2proc(k, level);
        proc2 = key2proc(k | ~(~KeyType(0L) << DIM * (k.maxLevel - level)), level);
    }
    if (proc1 != proc2) {
        for (int i=0; i<POWDIM; i++) {
            if (t.son[i] == NULL) {
                t.son[i] = new TreeNode;
            }
            else if (t.son[i]->isLeaf() && t.son[i]->node == TreeNode::particle) {
                t.son[i]->node = TreeNode::domainList;
                t.insert(t.son[i]->p);
                continue;
            }
            createDomainList(*t.son[i], level + 1,
                             k | (KeyType{ i } << (DIM*(k.maxLevel-level-1))));
        }
    }
}

void SubDomain::createDomainList() {
    createDomainList(root, 0, 0UL);
}

void SubDomain::sendParticles() {
    ParticleList *particleLists;
    particleLists = new ParticleList[numProcesses];

    buildSendList(root, particleLists, 0UL, 0);

    root.repairTree();

    Particle ** pArray = new Particle*[numProcesses];

    int *pLengthSend;
    pLengthSend = new int[numProcesses];
    pLengthSend[rank] = -1;

    int *pLengthReceive;
    pLengthReceive = new int[numProcesses];
    pLengthReceive[rank] = -1;

    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            pLengthSend[proc] = (int)particleLists[proc].size();
            pArray[proc] = new Particle[pLengthSend[proc]];
            pArray[proc] = &particleLists[proc][0];
        }
    }

    std::vector<boost::mpi::request> reqMessageLengths;
    std::vector<boost::mpi::status> statMessageLengths;

    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            reqMessageLengths.push_back(comm.isend(proc, 17, &pLengthSend[proc], 1));
            statMessageLengths.push_back(comm.recv(proc, 17, &pLengthReceive[proc], 1));
        }
    }

    boost::mpi::wait_all(reqMessageLengths.begin(), reqMessageLengths.end());

    int totalReceiveLength = 0;
    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            totalReceiveLength += pLengthReceive[proc];
        }
    }

    Logger(INFO) << "totalReceiveLength = " << totalReceiveLength;

    pArray[rank] = new Particle[totalReceiveLength];

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;

    int offset = 0;
    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            reqParticles.push_back(comm.isend(proc, 17, pArray[proc], pLengthSend[proc]));
            statParticles.push_back(comm.recv(proc, 17, pArray[rank] + offset, pLengthReceive[proc]));
            offset += pLengthReceive[proc];
        }
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

    for (int i=0; i<totalReceiveLength; i++) {
        root.insert(pArray[rank][i]);
    }

    delete [] particleLists;
    delete [] pLengthReceive;
    delete [] pLengthSend;
    //for (int proc=0; proc<numProcesses; proc++) {
        //delete [] pArray[proc];
    //}
    delete [] pArray;

}

void SubDomain::buildSendList(TreeNode &t, ParticleList *pList, KeyType k, int level) {
    int proc;
    if (t.isLeaf() && ((proc = key2proc(k, level)) != rank) && !t.isDomainList()) {
        pList[proc].push_back(t.p);
        t.p.toDelete = true;
    }
    for (int i=0; i<POWDIM; i++) {
        if (t.son[i] != NULL) {
            buildSendList(*t.son[i], pList, k | (KeyType{ i } << (DIM*(k.maxLevel-level-1))),
                          level + 1);
        }
    }
}

void SubDomain::symbolicForce(TreeNode &td, TreeNode &t, float diam, ParticleMap &pMap, KeyType k, int level) {
    if (key2proc(k, level) == rank || t.isDomainList()) {
        if (!t.isDomainList()) {
            bool insert = true;
            for (ParticleMap::iterator pit = pMap.begin(); pit != pMap.end(); pit++) {
                if (pit->second.x == t.p.x) {
                    insert = false;
                }
            }
            if (insert) {
                pMap[k] = t.p;
            }
        }
    }
    float r = td.smallestDistance(t.p);

    if (diam > theta*r) {
        for (int i=0; i<POWDIM; i++) {
            if (t.son[i] != NULL) {
                symbolicForce(td, *t.son[i], 0.5 * diam, pMap,
                              k | (KeyType{ i } << (DIM*(k.maxLevel-level-1))),
                              level + 1);
            }
        }
    }
}

void SubDomain::compPseudoParticles() {

    root.resetDomainList();

    root.compLocalPseudoParticles();

    ParticleList lowestDomainList;
    root.getLowestDomainList(lowestDomainList);

    int dLength = (int)lowestDomainList.size();

    pFloat moments[3*dLength];
    pFloat globalMoments[3*dLength];
    pFloat masses[dLength];
    pFloat globalMasses[dLength];

    for (int i=0; i<dLength; i++) {
        moments[3*i] = lowestDomainList[i].x[0] * lowestDomainList[i].m;
        moments[3*i+1] = lowestDomainList[i].x[1] * lowestDomainList[i].m;
        moments[3*i+2] = lowestDomainList[i].x[2] * lowestDomainList[i].m;
        masses[i] = lowestDomainList[i].m;
    }

    //void all_reduce(const communicator & comm, const T * in_values, int n, T * out_values, Op op);
    boost::mpi::all_reduce(comm, moments, 3*dLength, globalMoments, std::plus<pFloat>());
    boost::mpi::all_reduce(comm, masses, dLength, globalMasses, std::plus<pFloat>());

    int dCounter=0;
    root.updateLowestDomainList(dCounter, globalMasses, globalMoments);

    root.compDomainListPseudoParticles();
}

void SubDomain::compF(TreeNode &t, float diam, KeyType k, int level) {
    for (int i=0; i<POWDIM; i++) {
        if (t.son[i] != NULL) {
            compF(*t.son[i], diam, k | (KeyType{ i } << (DIM*(k.maxLevel-level-1))),
                  level + 1);
        }
        if (t.isLeaf() && key2proc(k, level) == rank && !t.isDomainList()) {
            t.p.F = { 0, 0, 0 };
            root.force(t, diam);
        }
    }
}

void SubDomain::compFParallel(float diam) {

    ParticleMap * pMaps;
    pMaps = new ParticleMap[numProcesses];

    compTheta(root, pMaps, diam);

    Particle ** pArray = new Particle*[numProcesses];

    int *pLengthSend;
    pLengthSend = new int[numProcesses];
    pLengthSend[rank] = -1;

    int *pLengthReceive;
    pLengthReceive = new int[numProcesses];
    pLengthReceive[rank] = -1;

    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            pLengthSend[proc] = (int)pMaps[proc].size();
            pArray[proc] = new Particle[pLengthSend[proc]];
            int counter = 0;
            for (ParticleMap::iterator pit = pMaps[proc].begin(); pit != pMaps[proc].end(); pit++) {
                pArray[proc][counter] = pit->second;
                counter++;
            }
        }
    }

    std::vector<boost::mpi::request> reqMessageLengths;
    std::vector<boost::mpi::status> statMessageLengths;

    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            reqMessageLengths.push_back(comm.isend(proc, 17, &pLengthSend[proc], 1));
            statMessageLengths.push_back(comm.recv(proc, 17, &pLengthReceive[proc], 1));
        }
    }

    boost::mpi::wait_all(reqMessageLengths.begin(), reqMessageLengths.end());

    int totalReceiveLength = 0;
    for (int proc=0; proc<numProcesses; proc++) {
        //Logger(INFO) << "Receive length[" << proc << "]: " << pLengthReceive[proc];
        if (proc != rank) {
            totalReceiveLength += pLengthReceive[proc];
        }
    }

    pArray[rank] = new Particle[totalReceiveLength];

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;

    int offset = 0;
    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            reqParticles.push_back(comm.isend(proc, 17, pArray[proc], pLengthSend[proc]));
            statParticles.push_back(comm.recv(proc, 17, pArray[rank] + offset, pLengthReceive[proc]));
            offset += pLengthReceive[proc];
        }
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

    for (int i=0; i<totalReceiveLength; i++) {
        //Logger(INFO) << "Inserting particle pArray[" << i << "].x : " << pArray[rank][i].x;
        pArray[rank][i].toDelete = true;
        root.insert(pArray[rank][i]);
    }

    delete [] pMaps;
    delete [] pLengthReceive;
    delete [] pLengthSend;
    for (int proc=0; proc<numProcesses; proc++) {
        delete [] pArray[proc];
    }
    delete [] pArray;

    compF(root, diam);
}

void SubDomain::compTheta(TreeNode &t, ParticleMap *pMap, float diam, KeyType k, int level) {
    for (int i=0; i<POWDIM; i++) {
        if (t.son[i] != NULL) {
            compTheta(*t.son[i], pMap, diam, k | KeyType{ i << (DIM * (k.maxLevel - level - 1)) },
                      level + 1);
        }
    }
    int proc;
    if (t.isDomainList() && (proc = key2proc(k, level) != rank)) {
        symbolicForce(t, root, diam, pMap[proc], 0UL, 0);
    }
}

void SubDomain::gatherKeys(KeyList &keyList, IntList &lengths, KeyList &localKeyList) {

    KeyType *kArrayLocal = &localKeyList[0];

    int localLength = (int)localKeyList.size();

    //boost::mpi::gather(comm, &localLength, 1, receiveLengths, 0);
    boost::mpi::all_gather(comm, &localLength, 1, lengths);

    int totalReceiveLength = 0;
    for (auto it = std::begin(lengths); it != std::end(lengths); ++it) {
        //std::cout << "receiveLengths: " << *it << std::endl;
        totalReceiveLength += *it;
    }

    KeyType *kArray;

    if (rank == 0) {
        kArray = new KeyType[totalReceiveLength];
    }

    boost::mpi::gatherv(comm, localKeyList, kArray, lengths, 0);

    if (rank == 0) {
        keyList.assign(kArray, kArray + totalReceiveLength);
        delete [] kArray;
    }
}

void SubDomain::gatherParticles(ParticleList &pList) {

    ParticleList myProcessParticles;
    root.getParticleList(myProcessParticles);

    //Particle *pArrayLocal = &myProcessParticles[0];

    int localLength = (int)myProcessParticles.size();
    IntList receiveLengths;

    //boost::mpi::gather(comm, &localLength, 1, receiveLengths, 0);
    boost::mpi::all_gather(comm, &localLength, 1, receiveLengths);

    int totalReceiveLength = 0;
    for (auto it = std::begin(receiveLengths); it != std::end(receiveLengths); ++it) {
        //std::cout << "receiveLengths: " << *it << std::endl;
        totalReceiveLength += *it;
    }

    Particle *pArray;

    if (rank == 0) {
        pArray = new Particle[totalReceiveLength];
    }

    boost::mpi::gatherv(comm, myProcessParticles, pArray, receiveLengths, 0);

    if (rank == 0) {
        pList.assign(pArray, pArray + totalReceiveLength);
        delete [] pArray;
    }
}

void SubDomain::gatherParticles(ParticleList &pList, IntList &processList) {

    ParticleList myProcessParticles;
    root.getParticleList(myProcessParticles);

    //Particle *pArrayLocal = &myProcessParticles[0];

    int localLength = (int)myProcessParticles.size();
    IntList receiveLengths;

    //boost::mpi::gather(comm, &localLength, 1, receiveLengths, 0);
    boost::mpi::all_gather(comm, &localLength, 1, receiveLengths);

    int totalReceiveLength = 0;
    for (auto it = std::begin(receiveLengths); it != std::end(receiveLengths); ++it) {
        //std::cout << "receiveLengths: " << *it << std::endl;
        totalReceiveLength += *it;
    }

    Particle *pArray;

    if (rank == 0) {
        pArray = new Particle[totalReceiveLength];
    }

    boost::mpi::gatherv(comm, myProcessParticles, pArray, receiveLengths, 0);

    if (rank == 0) {
        pList.assign(pArray, pArray + totalReceiveLength);
        IntList helper;
        for (int proc=0; proc<numProcesses; proc++) {
            helper.assign(receiveLengths[proc], proc);
            processList.insert(processList.end(), helper.begin(), helper.end());
        }
        delete [] pArray;
    }
}

void SubDomain::gatherParticles(ParticleList &pList, IntList &processList, KeyList &keyList) {

    ParticleList myProcessParticles;
    KeyList myProcessKeys;
    root.getParticleList(myProcessParticles, myProcessKeys);

    //Particle *pArrayLocal = &myProcessParticles[0];

    int localLength = (int)myProcessParticles.size();
    IntList receiveLengths;

    //boost::mpi::gather(comm, &localLength, 1, receiveLengths, 0);
    boost::mpi::all_gather(comm, &localLength, 1, receiveLengths);

    int totalReceiveLength = 0;
    for (auto it = std::begin(receiveLengths); it != std::end(receiveLengths); ++it) {
        //std::cout << "receiveLengths: " << *it << std::endl;
        totalReceiveLength += *it;
    }

    Particle *pArray;
    KeyType *kArray;

    if (rank == 0) {
        pArray = new Particle[totalReceiveLength];
        kArray = new KeyType[totalReceiveLength];
    }

    boost::mpi::gatherv(comm, myProcessParticles, pArray, receiveLengths, 0);
    boost::mpi::gatherv(comm, myProcessKeys, kArray, receiveLengths, 0);

    if (rank == 0) {
        pList.assign(pArray, pArray + totalReceiveLength);
        keyList.assign(kArray, kArray + totalReceiveLength);
        IntList helper;
        for (int proc=0; proc<numProcesses; proc++) {
            helper.assign(receiveLengths[proc], proc);
            processList.insert(processList.end(), helper.begin(), helper.end());
        }
        delete [] pArray;
        delete [] kArray;
    }
}

void SubDomain::sendParticlesSPH(tFloat radius) {
    ParticleList particleList;
    root.getParticleList(particleList);

    bool interactionPartner;
    int process;

    ParticleList *p2send;
    p2send = new ParticleList[numProcesses];

    //ParticleList p2send;
    //IntList send2process;
    for (int i=0; i<particleList.size(); i++) {
        interactionPartner = false;
        process = -1;
        findInteractionPartnersOutsideDomain(root, particleList[i], interactionPartner, process, radius);
        if (interactionPartner) {
            p2send[process].push_back(particleList[i]);
            //send2process.push_back(process);
        }
    }
    //Logger(ERROR) << "Particles with missing information: #" << p2send.size() << " (out of " << particleList.size() << ")";
    /*for (int proc=0; proc<numProcesses; proc++) {
        for (int i = 0; i < p2send[proc].size(); i++) {
            Logger(ERROR) << "p2send[" << i << "] to process: " << proc << " = " << p2send[proc][i].x;
        }
    }*/

    Particle ** pArray = new Particle*[numProcesses];

    int *pLengthSend;
    pLengthSend = new int[numProcesses];
    pLengthSend[rank] = -1;

    int *pLengthReceive;
    pLengthReceive = new int[numProcesses];
    pLengthReceive[rank] = -1;

    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            pLengthSend[proc] = (int)p2send[proc].size();
            pArray[proc] = new Particle[pLengthSend[proc]];
            pArray[proc] = &p2send[proc][0];
        }
    }

    std::vector<boost::mpi::request> reqMessageLengths;
    std::vector<boost::mpi::status> statMessageLengths;

    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            reqMessageLengths.push_back(comm.isend(proc, 17, &pLengthSend[proc], 1));
            statMessageLengths.push_back(comm.recv(proc, 17, &pLengthReceive[proc], 1));
        }
    }

    boost::mpi::wait_all(reqMessageLengths.begin(), reqMessageLengths.end());

    int totalReceiveLength = 0;
    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            totalReceiveLength += pLengthReceive[proc];
        }
    }

    pArray[rank] = new Particle[totalReceiveLength];

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;

    int offset = 0;
    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            reqParticles.push_back(comm.isend(proc, 17, pArray[proc], pLengthSend[proc]));
            statParticles.push_back(comm.recv(proc, 17, pArray[rank] + offset, pLengthReceive[proc]));
            offset += pLengthReceive[proc];
        }
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

    Logger(INFO) << "sendParticlesSPH(): totalReceiveLength = " << totalReceiveLength;

    for (int i=0; i<totalReceiveLength; i++) {
        pArray[rank][i].toDelete = true;
        root.insert(pArray[rank][i]);
    }

    ParticleList *interactionLists;
    interactionLists = new ParticleList[particleList.size()];
    ParticlePointerList localParticlePointerList;

    root.nearNeighbourList(radius, localParticlePointerList, interactionLists);

    //Logger(ERROR) << "len(localParticlePointerList) = " << localParticlePointerList.size() << "   len(interactionLists) = " << interactionLists->size();

    /*for (int i=0; i<localParticlePointerList.size(); i++) {
        Logger(INFO) << "localParticlePointerList[" << i << "]: " << localParticlePointerList[i]->x;
    }*/

    for (int i=0; i<localParticlePointerList.size(); i++) {
        Logger(INFO) << "len(interactionLists[" << i << "] = " << interactionLists[i].size();
    }

    Pressure pressureHandler(0.1, 1);

    for (int i=0; i<localParticlePointerList.size(); i++) {
        Density::calculateDensity(*localParticlePointerList[i], interactionLists[i], radius);
        pressureHandler.calculatePressure(*localParticlePointerList[i]);
        /*for (int j=0; j<interactionLists[i].size(); j++) {
            if (localParticlePointerList[i]->x != interactionLists[i][j].x) {
                localParticlePointerList[i]->F -= pow(localParticlePointerList[i]->m, 2) *
                                                  (((localParticlePointerList[i]->p) /
                                                    (pow(localParticlePointerList[i]->rho, 2))) +
                                                   ((interactionLists[i][j].p) / (pow(interactionLists[i][j].rho, 2))));
            }
        }*/
    }

    /*for (int i=0; i<localParticlePointerList.size(); i++) {
        Logger(INFO) << "Particle[" << i << "].rho = " << localParticlePointerList[i]->rho;
        Logger(INFO) << "Particle[" << i << "].p = " << localParticlePointerList[i]->p;
        Logger(INFO) << "Particle[" << i << "].F = " << localParticlePointerList[i]->F;
    }*/

    root.repairTree();

    delete [] p2send;
    delete [] pLengthReceive;
    delete [] pLengthSend;
    //for (int proc=0; proc<numProcesses; proc++) {
    //delete [] pArray[proc];
    //}
    delete [] pArray;
    delete [] interactionLists;
}

void SubDomain::forcesSPH(tFloat radius) {
    ParticleList particleList;
    root.getParticleList(particleList);

    bool interactionPartner;
    int process;

    ParticleList *p2send;
    p2send = new ParticleList[numProcesses];

    //ParticleList p2send;
    //IntList send2process;
    for (int i=0; i<particleList.size(); i++) {
        interactionPartner = false;
        process = -1;
        findInteractionPartnersOutsideDomain(root, particleList[i], interactionPartner, process, radius);
        if (interactionPartner) {
            p2send[process].push_back(particleList[i]);
            //send2process.push_back(process);
        }
    }
    //Logger(ERROR) << "Particles with missing information: #" << p2send.size() << " (out of " << particleList.size() << ")";
    /*for (int proc=0; proc<numProcesses; proc++) {
        for (int i = 0; i < p2send[proc].size(); i++) {
            Logger(ERROR) << "p2send[" << i << "] to process: " << proc << " = " << p2send[proc][i].x;
        }
    }*/

    Particle ** pArray = new Particle*[numProcesses];

    int *pLengthSend;
    pLengthSend = new int[numProcesses];
    pLengthSend[rank] = -1;

    int *pLengthReceive;
    pLengthReceive = new int[numProcesses];
    pLengthReceive[rank] = -1;

    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            pLengthSend[proc] = (int)p2send[proc].size();
            pArray[proc] = new Particle[pLengthSend[proc]];
            pArray[proc] = &p2send[proc][0];
        }
    }

    std::vector<boost::mpi::request> reqMessageLengths;
    std::vector<boost::mpi::status> statMessageLengths;

    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            reqMessageLengths.push_back(comm.isend(proc, 17, &pLengthSend[proc], 1));
            statMessageLengths.push_back(comm.recv(proc, 17, &pLengthReceive[proc], 1));
        }
    }

    boost::mpi::wait_all(reqMessageLengths.begin(), reqMessageLengths.end());

    int totalReceiveLength = 0;
    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            totalReceiveLength += pLengthReceive[proc];
        }
    }

    pArray[rank] = new Particle[totalReceiveLength];

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;

    int offset = 0;
    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            reqParticles.push_back(comm.isend(proc, 17, pArray[proc], pLengthSend[proc]));
            statParticles.push_back(comm.recv(proc, 17, pArray[rank] + offset, pLengthReceive[proc]));
            offset += pLengthReceive[proc];
        }
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

    Logger(INFO) << "sendParticles(): totalReceiveLength = " << totalReceiveLength;

    for (int i=0; i<totalReceiveLength; i++) {
        pArray[rank][i].toDelete = true;
        root.insert(pArray[rank][i]);
    }

    ParticleList *interactionLists;
    interactionLists = new ParticleList[particleList.size()];
    ParticlePointerList localParticlePointerList;

    root.nearNeighbourList(radius, localParticlePointerList, interactionLists);

    //Logger(ERROR) << "len(localParticlePointerList) = " << localParticlePointerList.size() << "   len(interactionLists) = " << interactionLists->size();

    /*for (int i=0; i<localParticlePointerList.size(); i++) {
        Logger(INFO) << "localParticlePointerList[" << i << "]: " << localParticlePointerList[i]->x;
    }*/

    for (int i=0; i<localParticlePointerList.size(); i++) {
        Logger(INFO) << "len(interactionLists[" << i << "] = " << interactionLists[i].size();
    }

    Pressure pressureHandler(0.5, 1);
    Kernels kernels(Kernels::gaussianKernel);
    for (int i=0; i<localParticlePointerList.size(); i++) {
        //Density::calculateDensity(*localParticlePointerList[i], interactionLists[i], radius);
        //pressureHandler.calculatePressure(*localParticlePointerList[i]);
        for (int j=0; j<interactionLists[i].size(); j++) {
            Vector3<float> r = localParticlePointerList[i]->x - interactionLists[i][j].x;
            Vector3<float> gradient;
            kernels.gradKernel(r, radius, gradient);
            if (localParticlePointerList[i]->x != interactionLists[i][j].x) {
                localParticlePointerList[i]->F -= pow(localParticlePointerList[i]->m, 2) *
                                                  (((localParticlePointerList[i]->p) /
                                                    (pow(localParticlePointerList[i]->rho, 2))) +
                                                   ((interactionLists[i][j].p) / (pow(interactionLists[i][j].rho, 2)))) *
                                                   gradient;
            }
        }
    }

    /*for (int i=0; i<localParticlePointerList.size(); i++) {
        Logger(INFO) << "Particle[" << i << "].rho = " << localParticlePointerList[i]->rho;
        Logger(INFO) << "Particle[" << i << "].p = " << localParticlePointerList[i]->p;
        Logger(INFO) << "Particle[" << i << "].F = " << localParticlePointerList[i]->F;
    }*/

    root.repairTree();

    delete [] p2send;
    delete [] pLengthReceive;
    delete [] pLengthSend;
    //for (int proc=0; proc<numProcesses; proc++) {
    //delete [] pArray[proc];
    //}
    delete [] pArray;
    delete [] interactionLists;
}

void SubDomain::findInteractionPartnersOutsideDomain(TreeNode &t, Particle &particle,
                                                     bool &interactionPartner, int &process,
                                                     tFloat radius, KeyType k, int level) {
    if (t.isLeaf() && !t.isDomainList()) {
        if (particle.withinRadius(t.p, radius)) {
            //all information on this process available
        }
    } else {
        for (int i = 0; i < POWDIM; i++) {
            if (t.son[i] != NULL) {
                findInteractionPartnersOutsideDomain(*t.son[i], particle, interactionPartner, process, radius,
                                                     KeyType(k | ((keyInteger) i << (DIM * (k.maxLevel - level - 1)))),
                                                     level + 1);
            } else {
                //son[i]->box.withinRadius(particle.x, radius
                if (t.isLeaf() && t.isDomainList() && key2proc(k, level) != rank && t.box.withinRadius(particle.x, radius)) {
                    interactionPartner = true;
                    process = key2proc(k, level);
                }
            }
        }
    }
}

void SubDomain::writeToTextFile(ParticleList &pList, IntList &processList, KeyList &keyList, int step) {

    std::ofstream textFile;
    std::string fileName = "output/test_" + std::to_string(step) + ".txt";
    textFile.open (fileName.c_str());

    textFile << "x" << ";"
             << "y" << ";"
             << "z" << ";"
             << "process" << ";"
             << "key" << "\n";

    for (int i=0; i<pList.size(); i++) {
        textFile << pList[i].x[0] << ";"
                 << pList[i].x[1] << ";"
                 << pList[i].x[2] << ";"
                 << processList[i] << ";"
                 << keyList[i].key << "\n";
    }

    textFile.close();
}