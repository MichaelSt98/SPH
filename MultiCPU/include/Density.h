//
// Created by Michael Staneker on 29.04.21.
//

#ifndef SPH_DENSITY_H
#define SPH_DENSITY_H

#include "Kernels.h"
#include "Particle.h"

class Density {
public:
    static void calculateDensity(Particle &particle, ParticleList &interactionPartners, float smoothingLength);
};


#endif //SPH_DENSITY_H
