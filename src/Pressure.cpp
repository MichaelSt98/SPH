#include "../include/Pressure.h"

Pressure::Pressure(prFloat k, prFloat n) {
    this->k = k;
    this->n = n;
}

void Pressure::calculatePressure(Particle &p) {
    p.p = k * pow(p.rho, 1+1/n);
}