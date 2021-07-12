#ifndef OOP_KEYTYPE_H
#define OOP_KEYTYPE_H

#include "Constants.h"
#include "Logger.h"

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <bitset>
#include <string>
#include <boost/dynamic_bitset.hpp>
#include <boost/dynamic_bitset/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/mpi.hpp>

#define M 64

class KeyType;

typedef boost::dynamic_bitset<> keyInteger;

const unsigned char DirTable[12][8] =
        { { 8,10, 3, 3, 4, 5, 4, 5}, { 2, 2,11, 9, 4, 5, 4, 5},
          { 7, 6, 7, 6, 8,10, 1, 1}, { 7, 6, 7, 6, 0, 0,11, 9},
          { 0, 8, 1,11, 6, 8, 6,11}, {10, 0, 9, 1,10, 7, 9, 7},
          {10, 4, 9, 4,10, 2, 9, 3}, { 5, 8, 5,11, 2, 8, 3,11},
          { 4, 9, 0, 0, 7, 9, 2, 2}, { 1, 1, 8, 5, 3, 3, 8, 6},
          {11, 5, 0, 0,11, 6, 2, 2}, { 1, 1, 4,10, 3, 3, 7,10} };

const unsigned char HilbertTable[12][8] = { {0,7,3,4,1,6,2,5}, {4,3,7,0,5,2,6,1}, {6,1,5,2,7,0,4,3},
                                            {2,5,1,6,3,4,0,7}, {0,1,7,6,3,2,4,5}, {6,7,1,0,5,4,2,3},
                                            {2,3,5,4,1,0,6,7}, {4,5,3,2,7,6,0,1}, {0,3,1,2,7,4,6,5},
                                            {2,1,3,0,5,6,4,7}, {4,7,5,6,3,0,2,1}, {6,5,7,4,1,2,0,3} };

class KeyType {

public:

    keyInteger key;
    int maxLevel;

    static const std::string KEY_MAX_STRING;
    static const KeyType KEY_MAX;

    KeyType();
    KeyType(keyInteger key_);
    KeyType(const std::string &s);
    template<typename I>KeyType(I key);
    KeyType(const KeyType &keyType);

    static KeyType Lebesgue2Hilbert(KeyType lebesgue, int level);

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & key;
        ar & maxLevel;
    }

    int getMaxLevel();

    int toIndex();

    friend std::ostream &operator<<(std::ostream &os, const KeyType &key2print);

    friend KeyType operator<<(KeyType key2Shift, std::size_t n);
    friend KeyType operator>>(KeyType key2Shift, std::size_t n);
    friend KeyType operator<<(KeyType key2Shift, KeyType n);
    friend KeyType operator>>(KeyType key2Shift, KeyType n);
    friend KeyType operator|(KeyType lhsKey, KeyType rhsKey);
    friend KeyType operator&(KeyType lhsKey, KeyType rhsKey);
    friend KeyType operator+(KeyType lhsKey, KeyType rhsKey);
    KeyType& operator+=(const KeyType& rhsKey);
    friend bool operator<(KeyType lhsKey, KeyType rhsKey);
    friend bool operator<=(KeyType lhsKey, KeyType rhsKey);
    friend bool operator>(KeyType lhsKey, KeyType rhsKey);
    friend bool operator>=(KeyType lhsKey, KeyType rhsKey);

    KeyType operator~() const;

};

namespace boost { namespace mpi {

        template<typename T>

        struct KeyMaximum {
            typedef KeyType first_argument_type;
            typedef KeyType second_argument_type;
            typedef KeyType result_type;

            const KeyType &operator()(const KeyType &x, const KeyType &y) const {
                return x < y ? y : x;
            }
        };
    }
}

//TODO: not working properly
template<typename I>KeyType::KeyType(I key) {
    this->key = keyInteger { M, (unsigned long)key };//(keyInteger)key;
    maxLevel = getMaxLevel();
}

std::ostream &operator<<(std::ostream &os, const KeyType &key2print);

KeyType operator<<(KeyType key2Shift, std::size_t n);
KeyType operator>>(KeyType key2Shift, std::size_t n);
KeyType operator<<(KeyType key2Shift, KeyType n);
KeyType operator>>(KeyType key2Shift, KeyType n);
KeyType operator|(KeyType lhsKey, KeyType rhsKey);
KeyType operator&(KeyType lhsKey, KeyType rhsKey);
KeyType operator+(KeyType lhsKey, KeyType rhsKey);
//KeyType& operator+=(const KeyType& rhsKey);
bool operator<(KeyType lhsKey, KeyType rhsKey);
bool operator<=(KeyType lhsKey, KeyType rhsKey);
bool operator>(KeyType lhsKey, KeyType rhsKey);
bool operator>=(KeyType lhsKey, KeyType rhsKey);

#endif //OOP_KEYTYPE_H
