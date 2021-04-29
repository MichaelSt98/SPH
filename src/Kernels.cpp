#include "../include/Kernels.h"

Kernels::Kernels(kernelType useKernel) {
    usedKernel = useKernel;
}

kFloat Kernels::kernel(vec3 &position, kFloat h) {
    kFloat w = -1;
    switch(usedKernel) {
        case gaussianKernel: {
            w = gaussian(position, h);
            break;
        }
        default: {
            Logger(ERROR) << "Kernel not implemented!";
            break;
        }
    }
    return w;
}

void Kernels::gradKernel(vec3 &position, kFloat h, vec3 &gradient) {
    switch(usedKernel) {
        case gaussianKernel: {
            gradGaussian(position, h, gradient);
            break;
        }
        default: {
            Logger(ERROR) << "gradKernel not implemented!";
            break;
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
    kFloat n = -2 * exp( -(r*r) / (h*h)) / pow(h, 5) / pow(PI, 3/2);

    gradient = n * position;
}
