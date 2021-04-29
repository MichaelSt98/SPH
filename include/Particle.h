//
// Created by Michael Staneker on 12.04.21.
//

#ifndef OOP_PARTICLE_H
#define OOP_PARTICLE_H

#include "Vector3.h"

typedef float pFloat;

class Particle;

typedef Vector3<pFloat> pVec;
typedef std::vector<Particle> ParticleList;
typedef std::vector<Particle *> ParticlePointerList;

class Particle {

public:

    pFloat m;
    pFloat rho;
    pFloat p;
    Vector3<pFloat> x;
    Vector3<pFloat> v;
    Vector3<pFloat> F;
    Vector3<pFloat> oldF;

    bool moved;
    bool toDelete;

    Particle();
    Particle(pVec x);
    Particle(pVec x, pFloat m);
    Particle(pVec x, pVec v);
    Particle(pVec x, pVec v, pFloat m);

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & m;
        ar & rho;
        ar & p;
        ar & x;
        ar & v;
        ar & F;
        ar & oldF;
        ar & moved;
        ar & toDelete;
    }

    void force(Particle *j);
    void force(Particle &j);

    void updateX(float deltaT);
    void updateV(float deltaT);

    friend std::ostream &operator << (std::ostream &os, const Particle &p);

    bool withinRadius(Particle &particle, pFloat radius);
};

inline std::ostream &operator<<(std::ostream &os, const Particle &p)
{
    os << "\tm = " << p.m << std::endl;
    os << "\tx = " << p.x << std::endl;
    os << "\tv = " << p.v << std::endl;
    os << "\tF = " << p.v << std::endl;
    os << "\tmoved    = " << (p.moved ? "true " : "false") << std::endl;
    os << "\ttoDelete = " << (p.toDelete ? "true" : "false");
    return os;
}

void force(Particle *i, Particle *j);

void force(Particle &i, Particle &j);


#endif //OOP_PARTICLE_H
