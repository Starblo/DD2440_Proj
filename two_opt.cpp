#include "two_opt.h"
#include "main.h"
#include <vector>
#include <algorithm>


extern std::vector<std::vector<double>> distanceMatrix;

void twoOptSwap(std::vector<int>& tour, int i, int k) {
    std::reverse(tour.begin() + i, tour.begin() + k + 1);
}

void twoOpt(std::vector<int>& tour) {
    int N = tour.size();
    bool improvement = true;
    while (improvement) {
        improvement = false;
        for (int i = 1; i < N - 1; ++i) {
            for (int k = i + 1; k < N; ++k) {
                double delta = distanceMatrix[tour[i - 1]][tour[k]] +
                               distanceMatrix[tour[i]][tour[(k + 1) % N]] -
                               distanceMatrix[tour[i - 1]][tour[i]] -
                               distanceMatrix[tour[k]][tour[(k + 1) % N]];
                if (delta < -1e-6) {
                    twoOptSwap(tour, i, k);
                    improvement = true;
                }
            }
        }
    }
}
