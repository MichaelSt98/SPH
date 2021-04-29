#ifndef SPH_KERNELS_H
#define SPH_KERNELS_H

#include "Particle.h"
#include "Logger.h"
#include "Constants.h"

#include <cmath>

typedef float kFloat;
typedef Vector3<kFloat> vec3;

class Kernels {

public:
    enum kernelType {
        gaussianKernel
    };

    kernelType usedKernel;

    Kernels(kernelType useKernel=gaussianKernel);

    kFloat kernel(vec3 &position, kFloat h);
    void gradKernel(vec3 &position, kFloat h, vec3 &gradient);

private:
    kFloat gaussian(vec3 &position, kFloat h);
    void gradGaussian(vec3 &position, kFloat h, vec3 &gradient);

};


#endif //SPH_KERNELS_H
