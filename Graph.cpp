// Graph.cpp
#include "Graph.h"

Graph::Graph(int n) : V(n), adj(n, std::vector<int>(n, 0)) {}

void Graph::addEdge(int u, int v) {
    if (u >= 0 && u < V && v >= 0 && v < V) {
        adj[u][v] = 1;
        adj[v][u] = 1;  // 无向图
    }
}

void Graph::removeEdge(int u, int v) {
    if (u >= 0 && u < V && v >= 0 && v < V) {
        adj[u][v] = 0;
        adj[v][u] = 0;
    }
}

// 新增一个孤立节点：邻接矩阵扩展一行一列
void Graph::addNode() {
    for (auto& row : adj)
        row.push_back(0);
    adj.push_back(std::vector<int>(V + 1, 0));
    ++V;
}

// 删除节点v：移除第v行和第v列
void Graph::removeNode(int v) {
    if (v < 0 || v >= V) return;
    adj.erase(adj.begin() + v);
    for (auto& row : adj)
        row.erase(row.begin() + v);
    --V;
}

// BFS连通性判断
bool Graph::isConnected() const {
    if (V == 0) return true;
    std::vector<bool> visited(V, false);
    std::queue<int> q;
    q.push(0);
    visited[0] = true;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v = 0; v < V; ++v) {
            if (adj[u][v] && !visited[v]) {
                visited[v] = true;
                q.push(v);
            }
        }
    }
    for (bool v : visited)
        if (!v) return false;
    return true;
}

// DFS递归辅助函数
void Graph::dfs(int u, std::vector<bool>& visited) const {
    visited[u] = true;
    for (int v = 0; v < V; ++v) {
        if (adj[u][v] && !visited[v])
            dfs(v, visited);
    }
}

// DFS连通性判断
bool Graph::isConnectedDFS() const {
    if (V == 0) return true;
    std::vector<bool> visited(V, false);
    dfs(0, visited);
    for (bool v : visited)
        if (!v) return false;
    return true;
}

int Graph::getV() const { return V; }

const std::vector<std::vector<int>>& Graph::getAdjMatrix() const { return adj; }