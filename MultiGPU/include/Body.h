#ifndef NBODY_BODY_H
#define NBODY_BODY_H

#include <cmath>

class Vector3D {

public:
    double x;
    double y;
    double z;

    Vector3D();

    Vector3D(double _x, double _y, double _z);

    double magnitude();

    static double magnitude(double _x, double _y, double _z);
};


class Body {

public:
    double mass;
    Vector3D position;
    Vector3D velocity;
    Vector3D acceleration;

    Body();
    Body(double _mass);
};


#endif //NBODY_BODY_H
