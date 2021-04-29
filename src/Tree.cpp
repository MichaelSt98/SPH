//
// Created by Michael Staneker on 12.04.21.
//

#include "../include/Tree.h"

const char* getNodeType(int nodeIndex) {
    switch (nodeIndex) {
        case 0:  return "particle      ";
        case 1:  return "pseudoParticle";
        case 2:  return "domainList    ";
        default: return "not valid     ";
    }
}

void TreeNode::resetSons() {
    son[0] = NULL;
    son[1] = NULL;
    son[2] = NULL;
    son[3] = NULL;
    son[4] = NULL;
    son[5] = NULL;
    son[6] = NULL;
    son[7] = NULL;
}

TreeNode::TreeNode() {
    resetSons();
    node = particle;
}

TreeNode::TreeNode(Domain &box) : box { box } {
    resetSons();
    node = particle;
}

TreeNode::TreeNode(Particle &p, Domain &box, nodeType node_) : p { p }, box { box } {
    resetSons();
    node = node_;
}

TreeNode::~TreeNode() {
    //*this = NULL;
}

Node::Node() {

}

Node::Node(Particle p_, TreeNode::nodeType n_) {
    p = p_;
    n = n_;
}

void TreeNode::printTreeSummary(bool detailed, int type) {
    NodeList nList;
    getTreeList(nList);

    int counterParticle = 0;
    int counterPseudoParticle = 0;
    int counterDomainList = 0;

    int counter=0;
    for (auto it = std::begin(nList); it != std::end(nList); ++it) {
        if (detailed) {

            if (type == -1) {
                Logger(INFO) << "Node " << std::setw(5) << std::setfill('0') << counter << " " << *it;
            }
            else {
                if (type == it->n) {
                    Logger(INFO) << "Node " << " " << *it;
                }
            }
        }
        if (it->n == particle) {
            counterParticle++;
        }
        else if (it->n == pseudoParticle) {
            counterPseudoParticle++;
        }
        else if (it->n == domainList) {
            counterDomainList++;
        }
        counter++;
    }

    Logger(INFO) << "----------------------------------------------------------------";
    Logger(INFO) << "counterParticle:       " << counterParticle;
    Logger(INFO) << "counterPseudoParticle: " << counterPseudoParticle;
    Logger(INFO) << "counterDomainList:     " << counterDomainList;
    Logger(INFO) << "Nodes:                 " << counter; //nList.size();
    Logger(INFO) << "----------------------------------------------------------------";
}

std::string TreeNode::getNodeType() const {
    std::string nodeTypeStr = "";
    switch (node) {
        case 0:  nodeTypeStr +=  "particle      "; break;
        case 1:  nodeTypeStr +=  "pseudoParticle"; break;
        case 2:  nodeTypeStr +=  "domainList    "; break;
        default: nodeTypeStr +=  "not valid     "; break;
    }
    return nodeTypeStr;
}

bool TreeNode::isLeaf() {
    for (int i = 0; i < POWDIM; i++) {
        if (son[i] != NULL) {
            return false;
        }
    }
    return true;
}

bool TreeNode::isPseudoParticle() {
    if (node == pseudoParticle) {
        return true;
    }
    return false;
}

bool TreeNode::isDomainList() {
    if (node == domainList) {
        return true;
    }
    return false;
}

bool TreeNode::isLowestDomainList() {
    if (node == domainList) {
        if (isLeaf()) {
            return true;
        }
        else {
            for (int i=0; i<POWDIM; i++) {
                if (son[i] && son[i]->node == domainList) {
                    return false;
                }
            }
            return true;
        }
    }
    return false;
}

tFloat TreeNode::smallestDistance(Particle &particle) {
    //smallest distance from p.x to cell box
    tFloat dx;
    if (particle.x[0] < box.lower[0]) dx = box.lower[0] - particle.x[0];
    else if (particle.x[0] > box.upper[0]) dx = particle.x[0] - box.upper[0];
    else dx = (tFloat)0;

    tFloat dy;
    if (particle.x[1] < box.lower[1]) dy = box.lower[1] - particle.x[1];
    else if (particle.x[1] > box.upper[1]) dy = particle.x[1] - box.upper[1];
    else dy = (tFloat)0;

    tFloat dz;
    if (particle.x[2] < box.lower[2]) dz = box.lower[2] - particle.x[2];
    else if (particle.x[2] > box.upper[2]) dz = particle.x[2] - box.upper[2];
    else dz = (tFloat)0;

    return sqrt(dx*dx + dy*dy + dz*dz);
}

int TreeNode::getSonBox(Particle &particle) {
    int son = 0;
    Vector3<dFloat> center;
    box.getCenter(center);
    for (int d=DIM-1; d>= 0; d--) {
        if (particle.x[d] < center[d]) {
            son = 2 * son;
        }
        else {
            son = 2 * son + 1;
        }
    }
    return son;
}

int TreeNode::getSonBox(Particle &particle, Domain &sonBox) {
    int son = 0;
    Vector3<dFloat> center;
    box.getCenter(center);
    for (int d=DIM-1; d>= 0; d--) {
        if (particle.x[d] < center[d]) {
            son = 2 * son;
            sonBox.lower[d] = box.lower[d];
            sonBox.upper[d] = center[d];
        }
        else {
            son = 2 * son + 1;
            sonBox.lower[d] = center[d];
            sonBox.upper[d] = box.upper[d];
        }
    }
    return son;
}

void TreeNode::insert(Particle &p2insert) {

    Domain sonBox;
    int nSon = getSonBox(p2insert, sonBox);

    if (isDomainList() && son[nSon] != NULL) {
        son[nSon]->box = sonBox;
        son[nSon]->insert(p2insert);
    }
    else {
        if (son[nSon] == NULL) {
            if (isLeaf() && !isDomainList()) {
                Particle p2 = p;
                node = pseudoParticle;
                son[nSon] = new TreeNode(p2insert, sonBox);
                p.toDelete = false;
                insert(p2);
            }
            else {
                son[nSon] = new TreeNode(p2insert, sonBox);
            }
        }
        else {
            son[nSon]->box = sonBox;
            son[nSon]->insert(p2insert);
        }
    }
}

void TreeNode::getTreeList(ParticleList &particleList) {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->getTreeList(particleList);
        }
    }
    particleList.push_back(p);
}

void TreeNode::getTreeList(NodeList &nodeList) {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->getTreeList(nodeList);
        }
    }
    nodeList.push_back(Node(p, node));
}

void TreeNode::getParticleList(ParticleList &particleList) {
    if (isLeaf() && !isDomainList()) {
        particleList.push_back(p);
    }
    else {
        for (int i=0; i<POWDIM; i++) {
            if (son[i] != NULL) {
                son[i]->getParticleList(particleList);
            }
        }
    }
}

void TreeNode::getParticleList(ParticleList &particleList, KeyList &keyList, KeyType k, int level) {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            if (son[i]->isLeaf() && !son[i]->isDomainList()) {
                particleList.push_back(son[i]->p);
                keyList.push_back((k | KeyType((keyInteger) i << (DIM * (k.maxLevel - level - 1)))));
            } else {
                son[i]->getParticleList(particleList, keyList,
                                        (k | KeyType((keyInteger) i << (DIM * (k.maxLevel - level - 1)))),
                                        level + 1);
            }
        }
    }
}

/*void TreeNode::getParticleKeys(KeyList &keyList, KeyType k, int level) {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            if (son[i]->isLeaf()) {
                keyList.push_back((k | KeyType((keyInteger) i << (DIM * (k.maxLevel - level - 1)))));
            } else {
                son[i]->getParticleKeys(keyList, (k | KeyType((keyInteger) i << (DIM * (k.maxLevel - level - 1)))),
                                        level + 1);
            }
        }
    }
}*/

int TreeNode::getParticleCount() {
    ParticleList pList;
    getParticleList(pList);
    return (int)pList.size();
}

void TreeNode::getLowestDomainList(ParticleList &particleList){
    if (isDomainList() && isLowestDomainList()) {
        particleList.push_back(p);
    }
    else {
        for (int i=0; i<POWDIM; i++) {
            if (son[i] != NULL) {
                son[i]->getLowestDomainList(particleList);
            }
        }
    }
}

void TreeNode::force(TreeNode &tl, tFloat diam) {
    tFloat r = 0;
    r = sqrt((p.x - tl.p.x) * (p.x -tl.p.x));
    if ((tl.isLeaf() || (diam < theta * r)) && !tl.isDomainList()) {
        if (r == 0) {
            Logger(WARN) << "Zero radius has been encountered.";
        }
        else {
            tl.p.force(p);
        }
    }
    else {
        for (int i=0; i<POWDIM; i++) {
            force(*son[i], 0.5*diam);
        }
    }
}

void TreeNode::compX(tFloat deltaT) {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->compX(deltaT);
        }
    }
    if (isLeaf() && ! isDomainList()) {
        p.updateX(deltaT);
    }
}

void TreeNode::compV(tFloat deltaT) {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->compV(deltaT);
        }
    }
    if (isLeaf() && ! isDomainList()) {
        p.updateV(deltaT);
    }
}

void TreeNode::compPseudoParticles() {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->compPseudoParticles();
        }
    }
    if (!isLeaf()) {
        if (!isDomainList()) {
            node = pseudoParticle;
        }
        p.m = 0;
        p.x = {0, 0, 0};
        for (int j=0; j<POWDIM; j++) {
            if (son[j] != NULL) {
                p.m += son[j]->p.m;
                p.x += son[j]->p.m * son[j]->p.x;
            }
        }
        p.x = p.x/p.m;
    }
}

void TreeNode::compLocalPseudoParticles() {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->compLocalPseudoParticles();
        }
    }
    if (!isLeaf() && (!isDomainList() || isLowestDomainList())) {
        if (!isDomainList()) {
            node = pseudoParticle;
        }
        p.m = 0;
        p.x = {0, 0, 0};
        for (int j=0; j<POWDIM; j++) {
            if (son[j] != NULL) {
                p.m += son[j]->p.m;
                p.x += son[j]->p.m * son[j]->p.x;
            }
        }
        p.x = p.x/p.m;
    }
}

void TreeNode::compDomainListPseudoParticles() {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->compDomainListPseudoParticles();
        }
    }
    if (isDomainList() && !isLowestDomainList()) {
        p.m = 0;
        p.x = {0, 0, 0};
        for (int j=0; j<POWDIM; j++) {
            if (son[j] != NULL) {
                p.m += son[j]->p.m;
                p.x += son[j]->p.m * son[j]->p.x;
            }
        }
        if (p.m > 0) {
            p.x = p.x/p.m;
        }
    }
}

void TreeNode::clearDomainList() {
    if (isDomainList()) {
        node = pseudoParticle;
        for (int i=0; i<POWDIM; i++) {
            if (son[i] != NULL) {
                son[i]->clearDomainList();
                if (son[i]->isLeaf() && son[i]->isPseudoParticle()) {
                    delete son[i];
                    son[i] = NULL;
                }
            }
        }
    }
}

void TreeNode::resetDomainList() {
    if (isDomainList() && isLowestDomainList()) {
        p.x = {0, 0, 0};
        p.m = 0;
    }
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->resetDomainList();
        }
    }
}

void TreeNode::updateLowestDomainList(int &pCounter, pFloat *masses, pFloat *moments) {
    resetDomainList();
    updateLowestDomainListEntries(pCounter, masses, moments);
    updateLowestDomainListCOM();
}

void TreeNode::updateLowestDomainListEntries(int &pCounter, pFloat *masses, pFloat *moments) {
    if (isDomainList() && isLowestDomainList()) {
        p.x = {moments[pCounter*3], moments[pCounter*3 + 1], moments[pCounter*3 + 2]};
        p.m = masses[pCounter];
        pCounter++;
    }
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->updateLowestDomainListEntries(pCounter, masses, moments);
        }
    }
}

void TreeNode::updateLowestDomainListCOM() {
    if (isDomainList() && isLowestDomainList()) {
        if (p.m > 0) {
            p.x = p.x/p.m;
        }
    }
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->updateLowestDomainListCOM();
        }
    }
}


void TreeNode::resetParticleFlags() {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->resetParticleFlags();
        }
        p.moved = false;
        p.toDelete = false;
    }
}

void TreeNode::moveLeaf(TreeNode &root) {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->moveLeaf(root);
        }
    }
    if (isLeaf() && !isDomainList() && !p.moved) {
        p.moved = true;
        if (!box.withinDomain(p.x)) {
            if (!root.box.withinDomain(p.x)) {
                Logger(INFO) << "Particle left system: " << p;
            }
            else {
                root.insert(p);
            }
            p.toDelete = true;
        }
    }
}

void TreeNode::repairTree() {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->repairTree();
        }
    }

    if (!isLeaf()) {
        int numberOfSons = 0;
        int d;
        for (int i=0; i<POWDIM; i++) {
            if (son[i] != NULL && !son[i]->isDomainList()) {
                if (son[i]->p.toDelete) {
                    delete son[i];
                    son[i] = NULL;
                }
                else {
                    numberOfSons++;
                    d = i;
                }
            }
        }
        if (!isDomainList()) {
            if (numberOfSons == 0) {
                p.toDelete = true;
            }
            else if (numberOfSons == 1) {
                if (!son[d]->isDomainList()) {
                    p = son[d]->p;
                    node = son[d]->node;
                    if (son[d]->isLeaf()) {
                        delete son[d];
                        son[d] = NULL;
                    }
                }
            }
        }
    }
}

void TreeNode::getParticleKeys(KeyList &keyList, KeyType k, int level) {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            if (son[i]->isLeaf() && !son[i]->isDomainList()) {
                keyList.push_back((k | KeyType((keyInteger) i << (DIM * (k.maxLevel - level - 1)))));
            } else {
                son[i]->getParticleKeys(keyList, (k | KeyType((keyInteger) i << (DIM * (k.maxLevel - level - 1)))),
                                level + 1);
            }
        }
    }
}

void TreeNode::getParticleKeys(KeyList &keyList, IntList &levelList, KeyType k, int level) {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            if (son[i]->isLeaf() && !son[i]->isDomainList()) {
                keyList.push_back((k | KeyType((keyInteger) i << (DIM * (k.maxLevel - level - 1)))));
                levelList.push_back(level + 1);
            } else {
                son[i]->getParticleKeys(keyList, levelList, (k | KeyType((keyInteger) i << (DIM * (k.maxLevel - level - 1)))),
                                        level + 1);
            }
        }
    }
}

void TreeNode::updateRange(int &n, int &p, KeyType *range, int *newDist, KeyType k, int level) {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->updateRange(n, p, range, newDist, (k | KeyType((keyInteger) i << (DIM * (k.maxLevel - level - 1)))),
                                level + 1);
        }
    }
    if (isLeaf() && !isDomainList()) {
        while (n >= newDist[p]) {
            range[p] = k;
            //Logger(INFO) << "k = " << k;
            p++;
        }
        n++;
    }
}

/*void TreeNode::nearNeighbourList(tFloat radius) {
    ParticleList particleList;
    getParticleList(particleList);
    IntList interactionPartners;
    int counter;
    for (int i=0; i<particleList.size(); i++) {
        counter = 0;
        for (int j=0; j<particleList.size(); j++) {
            if (i != j) {
                if (particleList[i].withinRadius(particleList[j], radius)) {
                    counter++;
                }
            }
        }
        interactionPartners.push_back(counter);
    }
    for (int i=0; i<interactionPartners.size(); i++) {
        Logger(INFO) << "#interaction partners: " << interactionPartners[i];
    }
}*/

void TreeNode::nearNeighbourList(tFloat radius, ParticleList &localParticleList, ParticleList *interactionLists) {
    ParticleList particleList;
    getParticleList(particleList);

    //ParticleList localParticleList;

    for (int i=0; i<particleList.size(); i++) {
        if (!particleList[i].toDelete) {
            localParticleList.push_back(particleList[i]);
        }
    }

    IntList amountOfInteractionPartners;

    interactionLists = new ParticleList[(int)localParticleList.size()];

    for (int i=0; i<localParticleList.size(); i++) {
        ParticleList interactionPartner;
        findInteractionPartners(localParticleList[i], interactionPartner, radius);
        interactionLists[i] = interactionPartner;
        amountOfInteractionPartners.push_back((int)interactionPartner.size());
    }

    for (int i=0; i<amountOfInteractionPartners.size(); i++) {
        Logger(INFO) << "#interaction partners[" << i << "]: " << amountOfInteractionPartners[i];
    }
}

/*void TreeNode::findInteractionPartners(Particle &particle, ParticleList &particleList, tFloat radius) {
    if (isLeaf() && !isDomainList()) {
        if (particle.withinRadius(p, radius)) {
            particleList.push_back(p);
        }
    }
    else {
        for (int i=0; i<POWDIM; i++) {
            if (son[i] != NULL) {
                son[i]->findInteractionPartners(particle, particleList, radius);
            }
        }
    }
}*/

/*void TreeNode::findInteractionPartners(Particle &particle, ParticleList &particleList, tFloat radius) {
    if (isLeaf() && !isDomainList()) {
        if (box.withinDomain(p.x) && particle.withinRadius(p, radius)) {
            particleList.push_back(p);
        }
    }
    else {
        for (int i=0; i<POWDIM; i++) {
            if (son[i] != NULL) {
                son[i]->findInteractionPartners(particle, particleList, radius);
            }
        }
    }
}*/

void TreeNode::findInteractionPartners(Particle &particle, ParticleList &particleList, tFloat radius) {
    if (isLeaf() && !isDomainList()) {
        if (particle.withinRadius(p, radius)) {
            particleList.push_back(p);
        }
    }
    else {
        for (int i=0; i<POWDIM; i++) {
            if (son[i] != NULL) {
                if (son[i]->box.completelyWithinRadius(particle.x, radius)) {
                    son[i]->getParticleList(particleList);
                }
                else {
                    if (son[i]->box.withinRadius(particle.x, radius)) {
                        son[i]->findInteractionPartners(particle, particleList, radius);
                    }
                }
            }
        }
    }
}

