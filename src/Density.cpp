#include "../include/Density.h"

void Density::calculateDensity(Particle &particle, ParticleList &interactionPartners, float smoothingLength) {
    particle.rho = 0;
    Vector3<float> r;
    Kernels kernels(Kernels::gaussianKernel);
    //Logger(ERROR) << "len(interactionPartners) = " << interactionPartners.size();
    for (int i=0; i<interactionPartners.size(); i++) {
        r = particle.x - interactionPartners[i].x;
        particle.rho += interactionPartners[i].m * kernels.kernel(r, smoothingLength);
    }
}