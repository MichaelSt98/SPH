//
// Created by Michael Staneker on 18.12.20.
//

#ifndef NBODY_LOGGER_H
#define NBODY_LOGGER_H

#include <iostream>
#include <string>
#include "Color.h"

namespace Color {
    class Modifier {
    public:
        Code code;
        Modifier(Code pCode);
        friend std::ostream& operator<<(std::ostream& os, const Color::Modifier& mod);
    };
}

enum typelog {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

struct structlog {
    bool headers = false;
    typelog level = WARN;
    int myrank = -1; // don't use MPI by default
    int outputRank = -1;
};

extern structlog LOGCFG;

class Logger {
public:
    Logger() {}
    Logger(typelog type);
    ~Logger();

    template<class T> Logger &operator<<(const T &msg) {
        if (msglevel >= LOGCFG.level && (LOGCFG.myrank == LOGCFG.outputRank || LOGCFG.outputRank == -1)) {
            std::cout << msg;
            opened = true;
        }
        return *this;
    }


private:
    bool opened = false;
    typelog msglevel = DEBUG;
    inline std::string getLabel(typelog type);
    inline Color::Modifier getColor(typelog type);
};

#endif //NBODY_LOGGER_H
