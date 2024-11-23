#include "christofides.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>

using namespace std;

extern mt19937 rng;
extern vector<vector<double>> distanceMatrix;

void christofidesAlgorithm(vector<int>& tour, const vector<Point>& points) {
    int N = points.size();
    tour.clear();

    // Step 1: Construct a Minimum Spanning Tree (MST) using Prim's Algorithm
    vector<bool> inMST(N, false);
    vector<double> key(N, numeric_limits<double>::max());
    vector<int> parent(N, -1);
    key[0] = 0.0;

    for (int count = 0; count < N - 1; ++count) {
        double minKey = numeric_limits<double>::max();
        int u = -1;

        for (int v = 0; v < N; ++v) {
            if (!inMST[v] && key[v] < minKey) {
                minKey = key[v];
                u = v;
            }
        }

        if (u == -1) break;
        inMST[u] = true;

        for (int v = 0; v < N; ++v) {
            double weight = euclideanDistance(points[u], points[v]);
            if (!inMST[v] && weight < key[v]) {
                key[v] = weight;
                parent[v] = u;
            }
        }
    }

    // Step 2: Add edges of MST to create an Eulerian graph
    vector<vector<int>> mstGraph(N);
    for (int v = 1; v < N; ++v) {
        int u = parent[v];
        mstGraph[u].push_back(v);
        mstGraph[v].push_back(u);
    }

    // Step 3: Find a perfect matching for vertices with odd degree
    vector<int> oddVertices;
    for (int i = 0; i < N; ++i) {
        if (mstGraph[i].size() % 2 != 0) {
            oddVertices.push_back(i);
        }
    }

    vector<bool> matched(N, false);
    for (int i = 0; i < oddVertices.size(); i += 2) {
        int u = oddVertices[i];
        int v = oddVertices[i + 1];
        mstGraph[u].push_back(v);
        mstGraph[v].push_back(u);
    }

    // Step 4: Form an Eulerian circuit using Hierholzer's algorithm
    vector<bool> visitedEdge(N, false);
    vector<int> stack;
    stack.push_back(0);

    while (!stack.empty()) {
        int u = stack.back();
        bool found = false;

        for (int v : mstGraph[u]) {
            if (!visitedEdge[v]) {
                visitedEdge[v] = true;
                stack.push_back(v);
                found = true;
                break;
            }
        }

        if (!found) {
            tour.push_back(u);
            stack.pop_back();
        }
    }

    // Step 5: Shortcut the tour to form a Hamiltonian circuit
    vector<bool> visited(N, false);
    vector<int> hamiltonianTour;
    for (int v : tour) {
        if (!visited[v]) {
            visited[v] = true;
            hamiltonianTour.push_back(v);
        }
    }
    tour = hamiltonianTour;
}
