// Graph.h
#pragma once
#include <vector>
#include <queue>

class Graph {
public:
    Graph(int n);
    void addEdge(int u, int v);
    void removeEdge(int u, int v);
    void addNode();                     // 新增一个孤立节点
    void removeNode(int v);             // 删除节点v及其关联边
    bool isConnected() const;           // 基于BFS的连通性判断
    bool isConnectedDFS() const;        // 基于DFS的连通性判断
    int getV() const;
    const std::vector<std::vector<int>>& getAdjMatrix() const;

private:
    int V;
    std::vector<std::vector<int>> adj;
    void dfs(int u, std::vector<bool>& visited) const;  // DFS递归辅助函数
};