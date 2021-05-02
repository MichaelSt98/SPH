//
// Created by Michael Staneker on 12.04.21.
//

#ifndef OOP_TREE_H
#define OOP_TREE_H

#include "Particle.h"
#include "Domain.h"
#include "Logger.h"
#include "Keytype.h"
#include "Constants.h"

#include <vector>
#include <string>
#include <climits>
#include <iomanip>
#include <cmath>

class TreeNode;
struct Node;

typedef std::vector<int> IntList;
typedef std::vector<Node> NodeList;
typedef std::vector<KeyType> KeyList;

typedef float tFloat;
//typedef unsigned long keytype;

const tFloat theta = 0.7;

class TreeNode {

public:
    enum nodeType
    {
        particle, pseudoParticle, domainList
    };

    std::string getNodeType() const;

    Particle p;
    Domain box;
    TreeNode *son[POWDIM];
    nodeType node;

    void resetSons();

    TreeNode();
    TreeNode(Domain &box);
    TreeNode(Particle &p, Domain &box, nodeType node_=particle);
    ~TreeNode();

    friend std::ostream &operator << (std::ostream &os, const TreeNode &t);
    void printTreeSummary(bool detailed=false, int type=-1);

    bool isLeaf();
    bool isPseudoParticle();
    bool isDomainList();
    bool isLowestDomainList();

    tFloat smallestDistance(Particle &particle);

    int getSonBox(Particle &particle);
    int getSonBox(Particle &particle, Domain &sonBox);

    void insert(Particle &p2insert);

    void force(TreeNode &tl, tFloat diam);
    void compX(tFloat deltaT);
    void compV(tFloat deltaT);

    void compPseudoParticles();
    void compLocalPseudoParticles();
    void compDomainListPseudoParticles();

    void clearDomainList();
    void resetDomainList();
    void updateLowestDomainList(int &pCounter, pFloat *masses, pFloat *moments);
    void updateLowestDomainListEntries(int &pCounter, pFloat *masses, pFloat *moments);
    void updateLowestDomainListCOM();

    void resetParticleFlags();
    void moveLeaf(TreeNode &root);
    void repairTree();
    //void createDomain();

    void getTreeList(ParticleList &particleList);
    void getTreeList(NodeList &nodeList);
    void getParticleList(ParticleList &particleList);
    void getParticleList(ParticlePointerList &particlePointerList);
    void getParticleList(ParticleList &particleList, KeyList &keyList, KeyType k=0UL, int level=0);
    int getParticleCount();
    void getLowestDomainList(ParticleList &particleList);

    void getParticleKeys(KeyList &keyList, KeyType k=0UL, int level=0);
    void getParticleKeys(KeyList &keyList, IntList &levelList, KeyType k=0UL, int level=0);

    void updateRange(int &n, int &p, KeyType *range, int *newDist, KeyType k=0UL, int level=0);

    void nearNeighbourList(tFloat radius, ParticlePointerList &localParticlePointerList, ParticleList *interactionLists);
    void findInteractionPartners(Particle &particle, ParticleList &particleList, tFloat radius);
};

struct Node {
    Particle p;
    TreeNode::nodeType n;

    Node();
    Node(Particle p_, TreeNode::nodeType n_);

    friend std::ostream &operator << (std::ostream &os, const Node &n);
};

inline std::ostream &operator<<(std::ostream &os, const TreeNode &t)
{
    os << "TreeNode------------------------------------" << std::endl;
    os << "Particle: " << std::endl;
    os << t.p << std::endl;
    os << "Box: " << t.box << std::endl;
    os << "NodeType: " << t.getNodeType() << std::endl;
    os << "Son: [";
    for (int i=0; i<POWDIM; i++) {
        os << ((t.son[i] == NULL) ? "-" : std::to_string(i).c_str());
        if (i<POWDIM-1) {
            os << "|";
        }
    }
    os << "]" << std::endl;
    os << "--------------------------------------------";

    return os;
}

const char* getNodeType(int nodeIndex);

inline std::ostream &operator << (std::ostream &os, const Node &n) {
    os  << "[" << getNodeType(n.n) << "]"
        << " x: " << n.p.x;

    return os;
}


#endif //OOP_TREE_H
