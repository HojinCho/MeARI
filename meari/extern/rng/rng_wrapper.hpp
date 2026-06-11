/*
 * rng_wrapper.hpp
 * 
 * Written by Hojin Cho on 2025-01-13.
 * 
 * A simple wrapper for the random number generator in C++
 * intended to be read from Cython as cppclass.
 * 
 */

#include <cstdint>
#include <random>
template <typename RandomNumberEngine>
class RandomNumberGeneratorWrapper{
public:
    RandomNumberGeneratorWrapper(uint_fast32_t seed) : rng(seed) {}
    RandomNumberGeneratorWrapper(uint_fast64_t seed, uint_fast64_t seq) : rng(seed, seq) {}
    double gen_normal(double mu, double sig) {
        return std::normal_distribution<double>(mu, sig)(rng);
    }
    void gen_normal_arr(double mu, double sig, double* vec, size_t size) {
        std::normal_distribution<double> dist(mu, sig);
        for (size_t i = 0; i < size; i++) {
            vec[i] = dist(rng);
        }
    }
private:
    RandomNumberEngine rng;
};