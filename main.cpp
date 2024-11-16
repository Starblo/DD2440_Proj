#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

struct Point {
    double x, y;
};

double euclideanDistance(const Point& a, const Point& b) {
    return hypot(a.x - b.x, a.y - b.y);
}
double tourDistance(const vector<Point>& points, const vector<int>& tour) {
    double totalDist = 0.0;
    int N = tour.size();
    for (int i = 0; i < N; ++i) {
        int current = tour[i];
        int next = tour[(i + 1) % N]; // Ensure the tour is circular
        totalDist += euclideanDistance(points[current], points[next]);
    }
    return totalDist;
}

void twoOptSwap(vector<int>& tour, int i, int k) {
    reverse(tour.begin() + i, tour.begin() + k + 1);
}

void twoOpt(const vector<Point>& points, vector<int>& tour) {
    int N = tour.size();
    bool improvement = true;
    while (improvement) {
        improvement = false;
        for (int i = 1; i < N - 1; ++i) {
            for (int k = i + 1; k < N; ++k) {
                double delta = 
                    euclideanDistance(points[tour[i - 1]], points[tour[k]]) +
                    euclideanDistance(points[tour[i]], points[tour[(k + 1) % N]]) -
                    euclideanDistance(points[tour[i - 1]], points[tour[i]]) -
                    euclideanDistance(points[tour[k]], points[tour[(k + 1) % N]]);
                if (delta < -1e-6) {
                    twoOptSwap(tour, i, k);
                    improvement = true;
                }
            }
        }
    }
}

int main() {
    int N;
    cin >> N;
    vector<Point> points(N);
    for (int i = 0; i < N; ++i) {
        cin >> points[i].x >> points[i].y;
    }

    vector<int> tour(N);
    vector<bool> visited(N, false);
    tour[0] = 0;
    visited[0] = true;
    for (int i = 1; i < N; ++i) {
        int current = tour[i - 1];
        int best = -1;
        double bestDist = numeric_limits<double>::max();

        for (int j = 0; j < N; ++j) {
            if (!visited[j]) {
                double dist = euclideanDistance(points[current], points[j]);
                if (dist < bestDist) {
                    bestDist = dist;
                    best = j;
                }
            }
        }

        tour[i] = best;
        visited[best] = true;
    }

    // Apply 2-opt optimization to improve the tour
    twoOpt(points, tour);

    // Output the optimized tour
    for (int i = 0; i < N; ++i) {
        cout << tour[i] << endl;
    }

    return 0;
}