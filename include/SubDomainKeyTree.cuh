//
// Created by Michael Staneker on 09.05.21.
//

#ifndef SPH_SUBDOMAINKEYTREE_CUH
#define SPH_SUBDOMAINKEYTREE_CUH

//#include <climits> // for ulong_max
//#define KEY_MAX ULONG_MAX

struct SubDomainKeyTree {
    int rank;
    int numProcesses;
    unsigned long *range;
};

#endif //SPH_SUBDOMAINKEYTREE_CUH
