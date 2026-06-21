// GCN.cpp
#include "GCN.h"
#include "utils.h"   // 用到 relu, matmul
#include <cmath>

SparseMatrix normalize_adjacency(const Graph& g) {
    int n = g.getV();
    const auto& adj = g.getAdjMatrix();
    std::vector<std::vector<int>> A_tilde = adj;
    for (int i = 0; i < n; ++i)
        A_tilde[i][i] = 1;

    std::vector<double> deg(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            deg[i] += A_tilde[i][j];

    std::vector<int> row_ptr(n + 1);
    std::vector<int> col_idx;
    std::vector<double> vals;
    row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A_tilde[i][j] != 0) {
                double val = 1.0 / sqrt(deg[i]) * 1.0 / sqrt(deg[j]);
                col_idx.push_back(j);
                vals.push_back(val);
            }
        }
        row_ptr[i + 1] = static_cast<int>(col_idx.size());
    }

    SparseMatrix A_norm;
    A_norm.set_data(n, n, row_ptr, col_idx, vals);
    return A_norm;
}

std::vector<std::vector<double>> gcn_forward(const SparseMatrix& A_norm,
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& W) {
    std::vector<std::vector<double>> AX = A_norm.multiply(X);
    std::vector<std::vector<double>> AXW = matmul(AX, W);
    relu(AXW);
    return AXW;
}

void gcn_forward_train(const SparseMatrix& A_norm,
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& W,
    std::vector<std::vector<double>>& H,
    std::vector<std::vector<double>>& AX,
    std::vector<std::vector<double>>& Z) {
    AX = A_norm.multiply(X);           // N x F
    Z = matmul(AX, W);                 // N x K (未激活)
    H = Z;                             // 复制
    relu(H);                           // H = ReLU(Z)
}

void gcn_backward(const std::vector<std::vector<double>>& AX,
    const std::vector<std::vector<double>>& Z,
    std::vector<std::vector<double>>& W,
    const std::vector<std::vector<double>>& grad_H,
    double learning_rate) {
    int N = AX.size();
    int F = AX[0].size();
    int K = W[0].size();

    // 1. ReLU反向
    std::vector<std::vector<double>> grad_Z(N, std::vector<double>(K, 0.0));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < K; ++j)
            if (Z[i][j] > 0.0)
                grad_Z[i][j] = grad_H[i][j];

    // 2. 计算 dL/dW = (AX)^T * grad_Z
    std::vector<std::vector<double>> grad_W(F, std::vector<double>(K, 0.0));
    for (int i = 0; i < N; ++i)
        for (int f = 0; f < F; ++f)
            for (int k = 0; k < K; ++k)
                grad_W[f][k] += AX[i][f] * grad_Z[i][k];

    // 3. 梯度下降更新 W
    for (int f = 0; f < F; ++f)
        for (int k = 0; k < K; ++k)
            W[f][k] -= learning_rate * grad_W[f][k];
}