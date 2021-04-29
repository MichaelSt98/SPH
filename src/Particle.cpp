//
// Created by Michael Staneker on 12.04.21.
//

#include "../include/Particle.h"

Particle::Particle() {
    m = 0;
    moved = false;
    toDelete = false;
}
Particle::Particle(pVec x) : x { x } {
    m = 0;
    moved = false;
    toDelete = false;
}

Particle::Particle(pVec x, pFloat m) : x { x } {
    this->m = m;
    moved = false;
    toDelete = false;
}

Particle::Particle(pVec x, pVec v) : x { x }, v { v }{
    m = 0;
    moved = false;
    toDelete = false;
}

Particle::Particle(pVec x, pVec v, pFloat m) : x { x }, v { v }{
    this->m = m;
    moved = false;
    toDelete = false;
}

void Particle::updateX(float deltaT) {
    pFloat a = deltaT * 0.5 / m;
    x += deltaT * (v + a * F);
    oldF = F;
}

void Particle::updateV(float deltaT) {
    pFloat a = deltaT * 0.5 / m;
    v += a * (F + oldF);
}

void Particle::force(Particle *j) {
    pFloat r = 0;
    r += (j->x - x) * (j->x - x);
    pFloat f = m * j->m /(sqrt(r) * r); // + smoothing);
    F += f * (j->x - x);
}

void Particle::force(Particle &j) {
    pFloat r = 0;
    r += (j.x - x) * (j.x - x);
    pFloat f = m * j.m /(sqrt(r) * r); // + smoothing);
    F += f * (j.x - x);
}

void force(Particle *i, Particle *j) {
    float r = 0;
    r += (j->x - i->x) * (j->x - i->x);
    float f = i->m * j->m /(sqrt(r) * r); // + smoothing);
    i->F += f * (j->x - i->x);
}

void force(Particle &i, Particle &j) {
    float r = 0;
    r += (j.x - i.x) * (j.x - i.x);
    float f = i.m * j.m /(sqrt(r) * r); // + smoothing);
    i.F += f * (j.x - i.x);
}

bool Particle::withinRadius(Particle &particle, pFloat radius) {
    return particle.x.withinRadius(x, radius);
}