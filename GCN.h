// GCN.h
#pragma once
#include "SparseMatrix.h"
#include "Graph.h"
#include <vector>

SparseMatrix normalize_adjacency(const Graph& g);

// 仅前向传播（不保存中间值）
std::vector<std::vector<double>> gcn_forward(const SparseMatrix& A_norm,
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& W);

// 前向传播 + 保存中间值（用于训练）
void gcn_forward_train(const SparseMatrix& A_norm,
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& W,
    std::vector<std::vector<double>>& H,
    std::vector<std::vector<double>>& AX,
    std::vector<std::vector<double>>& Z);

// 反向传播（更新权重）
void gcn_backward(const std::vector<std::vector<double>>& AX,
    const std::vector<std::vector<double>>& Z,
    std::vector<std::vector<double>>& W,
    const std::vector<std::vector<double>>& grad_H,
    double learning_rate);