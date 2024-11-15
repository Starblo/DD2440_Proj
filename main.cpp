#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

struct Point {
    double x, y;
};

int euclideanDistance(const Point& a, const Point& b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
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
        int bestDist = numeric_limits<int>::max();
        
        for (int j = 0; j < N; ++j) {
            if (!visited[j]) {
                int dist = euclideanDistance(points[current], points[j]);
                if (dist < bestDist) {
                    bestDist = dist;
                    best = j;
                }
            }
        }
        
        tour[i] = best;
        visited[best] = true;
    }

    for (int i = 0; i < N; ++i) {
        cout << tour[i] << endl;
    }

    return 0;
}