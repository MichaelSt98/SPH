#ifndef SPH_LIST_CUH
#define SPH_LIST_CUH

#include <iostream>
#include <assert.h>

#define CHECKARRAYBOUNDS

template<class value_t, class index_t>
class culist {

private:

    const bool external_memory;
    const index_t capacity;

    index_t length, lower, upper;
    value_t * data;

public:

    __host__
    culist(index_t capacity_) : external_memory(false), capacity(capacity_),
                                length(0), lower(0), upper(0) {

        data = new value_t[capacity];
    }

    __device__
    culist(value_t * data_, index_t capacity_) :
            external_memory(true),
            data(data_), capacity(capacity_),
            length(0), lower(0), upper(0) {}

    __host__ __device__
    ~culist() {
        if(!external_memory)
            delete [] data;
    }

    __host__
    void show() {

        std::cout << "[ ";
        for (int m = 0, i = lower; m < length;
             ++m, i = (i+1) == capacity ? 0 : i+1)
            std::cout << data[i] << ", ";
        std::cout << "]" << std::endl;
    }

    __forceinline__ __host__ __device__
    void push_front(value_t value) {

#ifdef CHECKARRAYBOUNDS
        assert (length+1 <= capacity);
#endif

        lower = (lower == 0) ? capacity-1 : lower-1;
        data[lower] = value;
        length += 1;
    }

    __forceinline__ __host__ __device__
    void push_back(value_t value) {

#ifdef CHECKARRAYBOUNDS
        assert (length+1 <= capacity);
#endif

        data[upper] = value;
        upper = (upper == capacity-1) ? 0 : upper+1;
        length += 1;
    }

    __forceinline__ __host__ __device__
    value_t pop_front() {

#ifdef CHECKARRAYBOUNDS
        assert (length > 0);
#endif

        value_t result = data[lower];
        lower = (lower == capacity-1) ? 0 : lower+1;
        length -= 1;

        return result;
    }

    __forceinline__ __host__ __device__
    value_t pop_back() {

#ifdef CHECKARRAYBOUNDS
        assert (length > 0);
#endif

        upper = (upper == 0) ? capacity-1 : upper-1;
        length -= 1;

        return data[upper];
    }

    __forceinline__ __host__ __device__
    value_t& get(index_t i) {

#ifdef CHECKARRAYBOUNDS
        assert (i < length);
#endif

        i += lower;
        i = i > capacity-1 ? i-capacity : i;

        return data[i];
    }

    __forceinline__ __host__ __device__
    value_t& operator[] (index_t i) {return get(i);}
};


#endif //SPH_LIST_CUH
