// main.h
#ifndef MAIN_H
#define MAIN_H

#include <vector>
#include <random>
#include <cmath>

struct Point {
    double x, y;
};

struct Individual {
    std::vector<int> tour;
    double fitness;

    // Constructor
    Individual(const std::vector<int>& t, double f) : tour(t), fitness(f) {}

    // Comparison operator for sorting
    bool operator<(const Individual& other) const {
        return fitness < other.fitness;
    }
};

extern const bool DEBUG;
extern const int SINGLE_OPT_PARAM;
extern const double THREE_OPT_TIME_LIMIT;
extern const int POP_SIZE;
extern const double SWAP_MUTATION_RATE;
extern const double SINGLE_INSERTION_MUTATION_RATE;
extern const double ONE_SEGMENT_INSERTION_MUTATION_RATE;
extern const double TWO_OPT_MUTATION_RATE;
extern const double TWO_SEGMENT_EXCHANGE_MUTATION_RATE;
extern const int ELITISM_NUM;
extern const double TIME_OUT;
extern int seed;
extern std::mt19937 rng;
extern std::vector<std::vector<double>> distanceMatrix;

double euclideanDistance(const Point& a, const Point& b);
void calDistMatrix(const std::vector<Point>& points, int N);
double tourDistance(const std::vector<int>& tour);
double evaluateFitness(const std::vector<int>& tour);
// void initializePopulation(std::vector<Individual>& population, const std::vector<Point>& points);

#endif // MAIN_H
