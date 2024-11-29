#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <chrono>
#include "main.h"
#include "two_opt.h"
#include "genetic_algorithm.h"


using namespace std;

const bool DEBUG = false;

const int SINGLE_OPT_PARAM = 6;
const double THREE_OPT_TIME_LIMIT = 1.8;

const int POP_SIZE = 50;

const double SWAP_MUTATION_RATE = 0.25;
const double SINGLE_INSERTION_MUTATION_RATE = 0.25 + SWAP_MUTATION_RATE;
const double ONE_SEGMENT_INSERTION_MUTATION_RATE = 0.1 + SINGLE_INSERTION_MUTATION_RATE;
const double TWO_OPT_MUTATION_RATE = 0.2 + ONE_SEGMENT_INSERTION_MUTATION_RATE;
const double TWO_SEGMENT_EXCHANGE_MUTATION_RATE = 0.25 + TWO_OPT_MUTATION_RATE;

const int ELITISM_NUM = 5;
const double TIME_OUT = 1.97;

// Random number generator
int seed = 123456;
mt19937 rng(seed);

vector<vector<double> > distanceMatrix;

double euclideanDistance(const Point& a, const Point& b) {
    return hypot(a.x - b.x, a.y - b.y);
}

void calDistMatrix(const vector<Point>& points, int N) {
    distanceMatrix.resize(N, vector<double>(N, 0.0));
    // 计算距离矩阵
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dist = euclideanDistance(points[i], points[j]);
            distanceMatrix[i][j] = dist;
            distanceMatrix[j][i] = dist; // 对称赋值
        }
    }
}

double tourDistance(const vector<int>& tour) {
    double totalDist = 0.0;
    int N = tour.size();
    for (int i = 0; i < N; ++i) {
        int current = tour[i];
        int next = tour[(i + 1) % N]; // 确保路径是循环的
        totalDist += distanceMatrix[current][next];
    }
    return totalDist;
}

int main() {
    int N;
    cin >> N;
    vector<Point> points(N);
    for (int i = 0; i < N; ++i) {
        cin >> points[i].x >> points[i].y;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    calDistMatrix(points, N);

    vector<Individual> population;
    initializePopulation(population, points);

    Individual bestIndividual = population[0];
    auto endTime = std::chrono::high_resolution_clock::now();
    if(DEBUG){
        std::chrono::duration<double> elapsedTime = endTime - startTime;
        cout << "Initialization Time: " << elapsedTime.count() << endl;
    }
    long iter = 0;

    while (true) {
        vector<Individual> newPopulation;

        // Elitism
        sort(population.begin(), population.end());
        if (population[0].fitness < bestIndividual.fitness) {
            bestIndividual = population[0];
        }
        if(DEBUG){
            cout << "Best Individual: " << bestIndividual.fitness << endl;
            cout << "Iteration: " << iter++ << endl;
        }

        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = currentTime - startTime;
        if (elapsedTime.count() > TIME_OUT) {
            break;
        }

        for (int i = 0; i < ELITISM_NUM; ++i) {
            newPopulation.push_back(population[i]);
        }

        // Generate rest of the population
        while (newPopulation.size() < POP_SIZE) {
            // Selection
            // 定义权重，使排名靠前的个体有更高的权重
            vector<double> weights(POP_SIZE);
            for (int i = 0; i < POP_SIZE; ++i) {
                weights[i] = POP_SIZE - i; // 权重与排名成反比，索引越小，权重越大
            }
            std::discrete_distribution<int> dist(weights.begin(), weights.end());
//            std::uniform_int_distribution<int> dist(0, POP_SIZE - 1);
            int randomIndex = dist(rng);
            Individual selectedIndividual = population[randomIndex];

            vector<int> offspringTour;
            offspringTour = selectedIndividual.tour;
            double random_num = uniform_real_distribution<double>(0.0, 1.0)(rng);
            // Mutation
            if (random_num <= SWAP_MUTATION_RATE) {
                swapMutation(offspringTour);
            }
            else if (random_num <= SINGLE_INSERTION_MUTATION_RATE) {
                singleInsertionMutation(offspringTour);
            }
            else if (random_num <= ONE_SEGMENT_INSERTION_MUTATION_RATE) {
                oneSegmentInsertionMutation(offspringTour);
            }

            else if (random_num <= TWO_OPT_MUTATION_RATE) {
                twoOptMutation(offspringTour);
            }
            else {
                twoSegmentExchangeMutation(offspringTour);
            }

            // Evaluate fitness
            double fitness = evaluateFitness(offspringTour);

            newPopulation.push_back(Individual(offspringTour, fitness));

        }

        population = newPopulation;
    }

    // Output the optimized tour
    for (int i = 0; i < N; ++i) {
        cout << bestIndividual.tour[i] << endl;
    }

    if(DEBUG){
        double fitness = evaluateFitness(bestIndividual.tour);
        cout << "Final fitness: " << fitness << endl;
    }

    return 0;
}