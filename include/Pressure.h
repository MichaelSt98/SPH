#ifndef SPH_PRESSURE_H
#define SPH_PRESSURE_H

#include "Vector3.h"

#include <cmath>

typedef float prFloat;
typedef Vector3<prFloat> vec3;

class Pressure {
public:
    prFloat k;
    prFloat n;

    Pressure(prFloat k, prFloat n);

    void pressure(vec3 &density, vec3 &pressure);
};


#endif //SPH_PRESSURE_H
