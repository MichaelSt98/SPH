//
// Created by Michael Staneker on 25.01.21.
//

#include "../include/Body.h"

Vector3D::Vector3D() : x { 0.0 }, y { 0.0 }, z { 0.0 } {}

Vector3D::Vector3D(double _x, double _y, double _z) : x { _x }, y { _y }, z { _z } {}

double Vector3D::magnitude() {
    return std::sqrt(x*x + y*y + z*z);
}

double Vector3D::magnitude(double _x, double _y, double _z) {
    return std::sqrt(_x*_x + _y*_y + _z*_z);
}

Body::Body() : mass { 0.0 }, position(), velocity(), acceleration() {}

Body::Body(double _mass) : mass { _mass }, position(), velocity(), acceleration() {}