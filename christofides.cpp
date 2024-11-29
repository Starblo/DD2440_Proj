#include "christofides.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <stack>

using namespace std;

extern mt19937 rng;
extern vector<vector<double>> distanceMatrix;

struct Edge {
    int u, v;
    double weight;
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

class UnionFind {
public:
    UnionFind(int n) : parent(n) {
        for (int i=0; i<n; ++i)
            parent[i] = i;
    }
    int find(int x) {
        if (parent[x]!=x)
            parent[x]=find(parent[x]);
        return parent[x];
    }
    void unite(int x, int y) {
        int fx = find(x);
        int fy = find(y);
        if (fx!=fy)
            parent[fx]=fy;
    }
private:
    vector<int> parent;
};

vector<Edge> kruskal(int n, vector<Edge>& edges) {
    UnionFind uf(n);
    vector<Edge> mst;
    sort(edges.begin(), edges.end());
    for (const Edge& e : edges) {
        if (uf.find(e.u)!=uf.find(e.v)) {
            uf.unite(e.u, e.v);
            mst.push_back(e);
        }
    }
    return mst;
}

void christofidesAlgorithm(vector<int>& tour, const vector<Point>& points) {
    int n = points.size();
    // Step 1: Compute all pairwise distances
    vector<Edge> edges;
    for (int i=0; i<n; ++i) {
        for (int j=i+1; j<n; ++j) {
            edges.push_back({i, j, distanceMatrix[i][j]});
        }
    }
    // Step 2: Compute MST using Kruskal's algorithm
    vector<Edge> mst = kruskal(n, edges);
    // Step 3: Find vertices with odd degree in MST
    vector<int> degrees(n, 0);
    for (const Edge& e : mst) {
        degrees[e.u]++;
        degrees[e.v]++;
    }
    vector<int> odd_vertices;
    for (int i=0; i<n; ++i) {
        if (degrees[i]%2==1)
            odd_vertices.push_back(i);
    }
    // Step 4: Compute minimum weight perfect matching on odd degree vertices
    // We use a greedy approximation here
    vector<Edge> matching;
    vector<bool> matched(n, false);
    vector<Edge> odd_edges;
    // Build list of all edges between odd-degree vertices
    for (size_t i=0; i<odd_vertices.size(); ++i) {
        for (size_t j=i+1; j<odd_vertices.size(); ++j) {
            int u = odd_vertices[i];
            int v = odd_vertices[j];
            odd_edges.push_back({u, v, distanceMatrix[u][v]});
        }
    }
    // Sort edges by weight
    sort(odd_edges.begin(), odd_edges.end());
    // Greedily match vertices
    for (const Edge& e : odd_edges) {
        if (!matched[e.u] && !matched[e.v]) {
            matched[e.u] = matched[e.v] = true;
            matching.push_back(e);
        }
    }
    // Step 5: Combine MST and matching edges to form multigraph
    vector<vector<int>> multigraph(n);
    for (const Edge& e : mst) {
        multigraph[e.u].push_back(e.v);
        multigraph[e.v].push_back(e.u);
    }
    for (const Edge& e : matching) {
        multigraph[e.u].push_back(e.v);
        multigraph[e.v].push_back(e.u);
    }
    // Step 6: Find Eulerian circuit
    vector<int> eulerian_tour;
    stack<int> curr_path;
    vector<int> circuit;
    // Copy multigraph to modify during traversal
    vector<vector<int>> temp_graph = multigraph;
    curr_path.push(0); // Start from vertex 0
    int curr_v = 0;
    while (!curr_path.empty()) {
        if (!temp_graph[curr_v].empty()) {
            curr_path.push(curr_v);
            int next_v = temp_graph[curr_v].back();
            temp_graph[curr_v].pop_back();
            // Remove edge from next_v to curr_v
            auto it = find(temp_graph[next_v].begin(), temp_graph[next_v].end(), curr_v);
            if (it != temp_graph[next_v].end())
                temp_graph[next_v].erase(it);
            curr_v = next_v;
        } else {
            circuit.push_back(curr_v);
            curr_v = curr_path.top();
            curr_path.pop();
        }
    }
    // Step 7: Convert Eulerian circuit to Hamiltonian circuit
    reverse(circuit.begin(), circuit.end());
    vector<bool> visited(n, false);
    tour.clear();
    for (int v : circuit) {
        if (!visited[v]) {
            tour.push_back(v);
            visited[v] = true;
        }
    }
    // Close the tour
    tour.push_back(tour[0]);
}