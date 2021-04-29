#include "../include/Pressure.h"

Pressure::Pressure(prFloat k, prFloat n) {
    this->k = k;
    this->n = n;
}

void Pressure::pressure(vec3 &density, vec3 &pressure) {
    pressure = k * vec3{ pow(density.x, 1+1/n), pow(density.y, 1+1/n), pow(density.z, 1+1/n) };
}