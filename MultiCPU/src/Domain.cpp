//
// Created by Michael Staneker on 12.04.21.
//

#include "../include/Domain.h"

Domain::Domain() {

}

Domain::Domain(dFloat lowerX, dFloat lowerY, dFloat lowerZ, dFloat upperX, dFloat upperY, dFloat upperZ) {
    lower = { lowerX, lowerY, lowerZ };
    upper = { upperX, upperY, upperZ };
}

Domain::Domain(dFloat size) : Domain {-size, -size, -size, size, size, size } {

}

Domain::Domain(Vector3<dFloat> lowerVec, Vector3<dFloat> upperVec) : lower{ lowerVec }, upper{ upperVec } {

}

Domain::Domain(Domain &domain) : lower { domain.lower }, upper { domain.upper } {

}

const Domain& Domain::operator=(const Domain& rhs) {
    //std::cout << "rhs.lower = " << rhs.lower << std::endl;
    //std::cout << "lower = " << lower << std::endl;
    lower = rhs.lower;
    upper = rhs.upper;
    return (*this);
}

dFloat Domain::getSystemSizeX() {
    return upper.x - lower.y;
}

dFloat Domain::getSystemSizeY() {
    return upper.y - lower.y;
}

dFloat Domain::getSystemSizeZ() {
    return upper.z - lower.z;
}

void Domain::getSystemSize(Vector3<dFloat> &systemSize) {
    systemSize = upper - lower;
}

dFloat Domain::getCenterX() {
    return 0.5 * (upper.x + lower.y);
}

dFloat Domain::getCenterY() {
    return 0.5 * (upper.y + lower.y);
}

dFloat Domain::getCenterZ() {
    return 0.5 * (upper.z + lower.z);
}

void Domain::getCenter(Vector3<dFloat> &center) {
    center = 0.5 * (upper + lower);
}

dFloat Domain::smallestDistance(Vector3<dFloat> &vec) {
    //smallest distance from p.x to cell box
    dFloat dx;
    if (vec[0] < lower[0]) dx = lower[0] - vec[0];
    else if (vec[0] > upper[0]) dx = vec[0] - upper[0];
    else dx = (dFloat)0;

    dFloat dy;
    if (vec[1] < lower[1]) dy = lower[1] - vec[1];
    else if (vec[1] > upper[1]) dy = vec[1] - upper[1];
    else dy = (dFloat)0;

    dFloat dz;
    if (vec[2] < lower[2]) dz = lower[2] - vec[2];
    else if (vec[2] > upper[2]) dz = vec[2] - upper[2];
    else dz = (dFloat)0;

    return sqrt(dx*dx + dy*dy + dz*dz);
}

void Domain::getCorners(VectorList& vectorList) {
    Vector3<dFloat> center;
    getCenter(center);
    Vector3<dFloat> extent = 0.5 * (upper - lower).absolute();

    vectorList.push_back(Vector3<dFloat>{ center.x + extent.x, center.y + extent.y, center.z + extent.z });
    vectorList.push_back(Vector3<dFloat>{ center.x + extent.x, center.y + extent.y, center.z - extent.z });
    vectorList.push_back(Vector3<dFloat>{ center.x + extent.x, center.y - extent.y, center.z - extent.z });
    vectorList.push_back(Vector3<dFloat>{ center.x - extent.x, center.y - extent.y, center.z - extent.z });
    vectorList.push_back(Vector3<dFloat>{ center.x - extent.x, center.y + extent.y, center.z + extent.z });
    vectorList.push_back(Vector3<dFloat>{ center.x - extent.x, center.y - extent.y, center.z + extent.z });
}

bool Domain::withinDomain(Vector3<dFloat> &vec) {
    return (vec < upper && vec > lower);
}

bool Domain::completelyWithinRadius(Vector3<dFloat> &vec, dFloat radius) {
    VectorList corners;
    getCorners(corners);

    for (int i=0; i<corners.size(); i++) {
        if (!vec.withinRadius(corners[i], radius)) {
            return false;
        }
    }
    return true; //if (vec.withinRadius(, radius))
}

bool Domain::withinRadius(Vector3<dFloat> &vec, dFloat radius) {
    dFloat distance = smallestDistance(vec);
    if (distance < radius) {
        return true;
    }
    else {
        return false;
    }
}