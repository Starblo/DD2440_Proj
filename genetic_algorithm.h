#ifndef GENETIC_ALGORITHMS_H
#define GENETIC_ALGORITHMS_H

#include <vector>
#include "main.h"

void twoOptMutation(std::vector<int>& individual);
void swapMutation(std::vector<int>& individual);
void singleInsertionMutation(std::vector<int>& individual);
void oneSegmentInsertionMutation(std::vector<int>& individual);
void singleInsertion(std::vector<int>& individual, int i, int j, int k);
void singleOpt(std::vector<int>& tour);
void threeOpt(std::vector<int>& tour);
void twoSegmentExchangeMutation(std::vector<int>& individual);
void greedy_initialize(int N, const std::vector<Point>& points, std::vector<int>& tour);
void initializePopulation(std::vector<Individual>& population, const std::vector<Point>& points);
double evaluateFitness(const std::vector<int>& tour);

#endif // GENETIC_ALGORITHMS_H
