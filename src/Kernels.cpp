#include "../include/Kernels.h"

Kernels::Kernels(kernelType useKernel) {
    usedKernel = useKernel;
}

kFloat Kernels::kernel(vec3 &position, kFloat h) {
    kFloat w = -1;
    switch(usedKernel) {
        case 0: {
            w = gaussian(position, h);
        }
        default: {
            Logger(ERROR) << "Not implemented!";
        }
    }
    return w;
}

void Kernels::gradKernel(vec3 &position, kFloat h, vec3 &gradient) {
    switch(usedKernel) {
        case 0: {
            gradGaussian(position, h, gradient);
        }
        default: {
            Logger(ERROR) << "Not implemented!";
        }
    }
}

kFloat Kernels::gaussian(vec3 &position, kFloat h) {
    kFloat r = sqrt(position * position);
    kFloat w = (1.0 / pow((h * sqrt(PI)), 3)) * exp( -(r*r) / (h*h));
    return w;
}

void Kernels::gradGaussian(vec3 &position, kFloat h, vec3 &gradient) {
    kFloat r = sqrt(position * position);
    kFloat n = -2 * exp( -(r*r) / (h*h)) / pow(h, 5) / sqrt(pow(PI, 3));

    gradient = n * position;
}
