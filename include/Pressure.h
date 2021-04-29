#ifndef SPH_PRESSURE_H
#define SPH_PRESSURE_H

#include "Particle.h"

#include <cmath>

typedef float prFloat;
typedef Vector3<prFloat> vec3;

class Pressure {
public:
    prFloat k;
    prFloat n;

    Pressure(prFloat k, prFloat n);

    void calculatePressure(Particle &p);
};


#endif //SPH_PRESSURE_H
