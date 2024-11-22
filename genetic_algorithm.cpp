#include "genetic_algorithm.h"
#include "main.h"
#include "two_opt.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
using namespace std;

extern mt19937 rng;
extern vector<vector<double>> distanceMatrix;

double evaluateFitness(const vector<int>& tour) {
    return tourDistance(tour);
}

// Reverse a Random Segment in the route
void twoOptMutation(vector<int>& individual) {
    int N = individual.size();
    if (N < 4) return;

    uniform_int_distribution<int> dist(0, N - 1);
    int i = dist(rng);
    int k = dist(rng);
    while (k == i) {
        k = dist(rng);
    }
    if (i > k) swap(i, k);

    twoOptSwap(individual, i, k);
}

// Swap Two Random Nodes
void swapMutation(vector<int>& individual) {
    int N = individual.size();
    uniform_int_distribution<int> dist(0, N - 1);
    int idx1 = dist(rng);
    int idx2 = dist(rng);
    swap(individual[idx1], individual[idx2]);
}

// Move a Random Node to a Random Position
void singleInsertionMutation(vector<int>& individual) {
    int N = individual.size();
    if (N < 3) return; // Need at least 3 nodes to perform single insertion

    uniform_int_distribution<int> dist(0, N - 1);
    int fromIndex = dist(rng);
    int toIndex = dist(rng);
    while (toIndex == fromIndex) {
        toIndex = dist(rng);
    }

    int point_idx = individual[fromIndex];
    // 从原位置删除节点
    individual.erase(individual.begin() + fromIndex);

    // 如果原位置在目标位置之前，删除节点后，目标位置索引需要减一
    if (fromIndex < toIndex) {
        toIndex -= 1;
    }

    // 在目标位置插入节点
    individual.insert(individual.begin() + toIndex, point_idx);
}

// Move a Random Segment into a Random Position
void oneSegmentInsertionMutation(vector<int>& individual) {
    int N = individual.size();
    if (N < 4) return; // 至少需要4个节点进行片段移动

    //FIXME: not efficient
    vector<int> indices(N);
    for (int i = 0; i < N; ++i) {
        indices[i] = i;
    }
    shuffle(indices.begin(), indices.end(), rng);

    vector<int> selectedIndices(indices.begin(), indices.begin() + 3);
    sort(selectedIndices.begin(), selectedIndices.end());

    int start = selectedIndices[0];
    int end = selectedIndices[1];
    int insertPos = selectedIndices[2];
    double random_num = uniform_real_distribution<double>(0.0, 1.0)(rng);
    if(random_num < 0.5){
        insertPos = selectedIndices[0];
        start = selectedIndices[1];
        end = selectedIndices[2];
    }

    // 提取要移动的片段
    vector<int> segment(individual.begin() + start, individual.begin() + end + 1);

    // 从路径中删除片段
    individual.erase(individual.begin() + start, individual.begin() + end + 1);

    // 调整插入位置
    if (insertPos > start) {
        insertPos -= (end - start + 1);
    }

    // 在新的位置插入片段
    individual.insert(individual.begin() + insertPos, segment.begin(), segment.end());
}

void singleInsertion(vector<int>& individual, int i, int j, int k) {
    vector<int> segment(individual.begin() + i, individual.begin() + j + 1);
    individual.erase(individual.begin() + i, individual.begin() + j + 1);
    if(i < k){
        individual.insert(individual.begin() + k - (j - i + 1), segment.begin(), segment.end());
    }else{
        individual.insert(individual.begin() + k, segment.begin(), segment.end());
    }
}

void singleOpt(vector<int>& tour) {
    int N = tour.size();
    bool improvement = true;
    while(improvement){
        improvement = false;
        for(int i = 0; i < N; ++i){ // 片段起始位置
            for(int j = i; j < i + SINGLE_OPT_PARAM & j < N; ++j){ // 片段结束位置 （包含）
                for(int k = 0; k < N; ++k){ // 插入位置
                    if(i <= k && k <= j + 1 || k == (j + 1) % N) continue;
                    double delta = - distanceMatrix[tour[(i - 1 + N) % N]][tour[i]] - distanceMatrix[tour[j]][tour[(j+1) % N]] - distanceMatrix[tour[k]][tour[(k - 1 + N) % N]]
                            + distanceMatrix[tour[(k - 1 + N) % N]][tour[i]] + distanceMatrix[tour[j]][tour[k]] + distanceMatrix[tour[(i - 1 + N) % N]][tour[(j+1) %N]];
                    if(delta < -1e-6){
                        singleInsertion(tour, i, j, k);
                        improvement = true;
                    }
                }
            }
        }
    }
}

// Function to perform 3-opt optimization on the tour
void threeOpt(vector<int>& tour) {
    int N = tour.size();
    bool improved = true;
    clock_t start_time = clock();

    while (improved) {
        improved = false;
        for (int i = 0; i < N - 2; ++i) {
            for (int j = i + 1; j < N - 1; ++j) {
                // Check for time limit
                if ((double)(clock() - start_time) / CLOCKS_PER_SEC > THREE_OPT_TIME_LIMIT) {
                    return;
                }
                for (int k = j + 1; k < N; ++k) {
                    // Current distances
                    int A = tour[i], B = tour[(i + 1) % N];
                    int C = tour[j], D = tour[(j + 1) % N];
                    int E = tour[k], F = tour[(k + 1) % N];

                    double d0 = distanceMatrix[A][B] + distanceMatrix[C][D] + distanceMatrix[E][F];

                    // Possible 3-opt moves
                    double d1 = distanceMatrix[A][C] + distanceMatrix[B][D] + distanceMatrix[E][F];
                    double d2 = distanceMatrix[A][B] + distanceMatrix[C][E] + distanceMatrix[D][F];
                    double d3 = distanceMatrix[A][D] + distanceMatrix[E][B] + distanceMatrix[C][F];
                    double d4 = distanceMatrix[F][B] + distanceMatrix[C][D] + distanceMatrix[E][A];

                    if (d1 < d0) {
                        reverse(tour.begin() + i + 1, tour.begin() + j + 1);
                        improved = true;
                    } else if (d2 < d0) {
                        reverse(tour.begin() + j + 1, tour.begin() + k + 1);
                        improved = true;
                    } else if (d3 < d0) {
                        vector<int> temp;
                        temp.insert(temp.end(), tour.begin() + j + 1, tour.begin() + k + 1);
                        temp.insert(temp.end(), tour.begin() + i + 1, tour.begin() + j + 1);
                        copy(temp.begin(), temp.end(), tour.begin() + i + 1);
                        improved = true;
                    } else if (d4 < d0) {
                        reverse(tour.begin() + i + 1, tour.begin() + k + 1);
                        improved = true;
                    }
                }
            }
        }
    }
}

// Swap Two Random Segments
void twoSegmentExchangeMutation(vector<int>& individual) {
    int N = individual.size();
    if (N < 4) return; // Need at least 4 nodes to perform two-segment exchange
    //FIXME: not efficient

    vector<int> indices(N);
    for (int i = 0; i < N; ++i) {
        indices[i] = i;
    }
    shuffle(indices.begin(), indices.end(), rng);

    vector<int> selectedIndices(indices.begin(), indices.begin() + 4);
    sort(selectedIndices.begin(), selectedIndices.end());

    int i1 = selectedIndices[0];
    int j1 = selectedIndices[1];
    int i2 = selectedIndices[2];
    int j2 = selectedIndices[3];

    // 确保段不重叠且有足够的长度
    if (j1 < i2 && (j1 - i1 >= 1) && (j2 - i2 >= 1)) {
        // 提取段
        vector<int> segment1(individual.begin() + i1, individual.begin() + j1 + 1);
        vector<int> segment2(individual.begin() + i2, individual.begin() + j2 + 1);

        // 移除原有的段（注意顺序）
        individual.erase(individual.begin() + i2, individual.begin() + j2 + 1);
        individual.erase(individual.begin() + i1, individual.begin() + j1 + 1);

        // 插入交换的段
        individual.insert(individual.begin() + i1, segment2.begin(), segment2.end());
        individual.insert(individual.begin() + i1 + segment2.size(), segment1.begin(), segment1.end());
    }
}

void greedy_initialize(int N, const vector<Point>& points, vector<int>& tour) {
    vector<bool> visited(N, false);
    tour[0] = 0;
    visited[0] = true;

    for (int i = 1; i < N; ++i) {
        int current = tour[i - 1];
        int best = -1;
        double bestDist = numeric_limits<double>::max();

        for (int j = 0; j < N; ++j) {
            if (!visited[j]) {
                double dist = distanceMatrix[current][j];
                if (dist < bestDist) {
                    bestDist = dist;
                    best = j;
                }
            }
        }

        tour[i] = best;
        visited[best] = true;
    }
}

void initializePopulation(vector<Individual>& population, const vector<Point>& points) {
    population.clear();
    int N = points.size();
    vector<int> baseTour(N);
    greedy_initialize(N, points, baseTour);
    if(DEBUG){
        cout << "Before applying OPT fitness:" << evaluateFitness(baseTour) << endl;
    }
    threeOpt(baseTour);
    twoOpt(baseTour);
//    singleOpt(baseTour);
    double baseFitness = evaluateFitness(baseTour);
    if(DEBUG){
        cout << "Base fitness: " << baseFitness << endl;
    }
    population.push_back(Individual(baseTour, baseFitness));
    for (int i = 1; i < POP_SIZE; ++i) {
        vector<int> tour = baseTour;
        shuffle(tour.begin(), tour.end(), rng);
        double fitness = evaluateFitness(tour);
        population.push_back(Individual(tour, fitness));
    }
}
