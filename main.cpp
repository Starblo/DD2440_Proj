#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <chrono>


using namespace std;

const int POP_SIZE = 50;

const double SWAP_MUTATION_RATE = 0.25;
const double SINGLE_INSERTION_MUTATION_RATE = 0.25 + SWAP_MUTATION_RATE;
const double ONE_SEGMENT_INSERTION_MUTATION_RATE = 0.1 + SINGLE_INSERTION_MUTATION_RATE;
const double TWO_OPT_MUTATION_RATE = 0.2 + ONE_SEGMENT_INSERTION_MUTATION_RATE;
const double TWO_SEGMENT_EXCHANGE_MUTATION_RATE = 0.25 + TWO_OPT_MUTATION_RATE;

const int ELITISM_NUM = 5;
const double TIME_OUT = 1.95;

// Random number generator
int seed = 123456;
mt19937 rng(seed);

vector<vector<double>> distanceMatrix;

struct Point {
    double x, y;
};

struct Individual {
    vector<int> tour;
    double fitness;

    // Constructor
    Individual(const vector<int>& t, double f) : tour(t), fitness(f) {}

    // Comparison operator for sorting
    bool operator<(const Individual& other) const {
        return fitness < other.fitness;
    }
};

inline double euclideanDistance(const Point& a, const Point& b) {
    return hypot(a.x - b.x, a.y - b.y);
}

inline void calDistMatrix(const vector<Point>& points, int N) {
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

inline double tourDistance(const vector<int>& tour) {
    double totalDist = 0.0;
    int N = tour.size();
    for (int i = 0; i < N; ++i) {
        int current = tour[i];
        int next = tour[(i + 1) % N]; // 确保路径是循环的
        totalDist += distanceMatrix[current][next];
    }
    return totalDist;
}

inline double evaluateFitness(const vector<Point>& points, const vector<int>& tour) {
    return tourDistance(tour);
}

inline void twoOptSwap(vector<int>& tour, int i, int k) {
    reverse(tour.begin() + i, tour.begin() + k + 1);
}

void twoOpt(const vector<Point>& points, vector<int>& tour) {
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

// Reverse a Random Segment in the route
void twoOptMutation(vector<int>& individual) {
    int N = individual.size();
    if (N < 4) return; // 至少需要4个节点进行2-opt交换

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
    twoOpt(points, baseTour);
    double baseFitness = evaluateFitness(points, baseTour);
//    cout << "Base fitness: " << baseFitness << endl;
    population.push_back(Individual(baseTour, baseFitness));
    for (int i = 0; i < POP_SIZE; ++i) {
        vector<int> tour = baseTour;
        shuffle(tour.begin(), tour.end(), rng);
        double fitness = evaluateFitness(points, tour);
        population.push_back(Individual(tour, fitness));
    }
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
    long iter = 0;

    while (true) {
        vector<Individual> newPopulation;

        // Elitism
        sort(population.begin(), population.end());
        if (population[0].fitness < bestIndividual.fitness) {
            bestIndividual = population[0];
        }
//        cout << "Best Individual: " << bestIndividual.fitness << endl;
//        cout << "Iteration: " << iter++ << endl;

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
            double fitness = evaluateFitness(points, offspringTour);

            newPopulation.push_back(Individual(offspringTour, fitness));

        }

        population = newPopulation;
    }


    // Output the optimized tour
    for (int i = 0; i < N; ++i) {
        cout << bestIndividual.tour[i] << endl;
    }

    double fitness = evaluateFitness(points, bestIndividual.tour);
    cout << "Final fitness: " << fitness << endl;

    return 0;
}