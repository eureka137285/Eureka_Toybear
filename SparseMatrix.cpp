// SparseMatrix.cpp
#include "SparseMatrix.h"
#include <stdexcept>
#include <iomanip>

SparseMatrix::SparseMatrix() : rows(0), cols(0) {}
SparseMatrix::SparseMatrix(int n, int m) : rows(n), cols(m), row_ptr(static_cast<size_t>(n + 1), 0) {}

void SparseMatrix::set_data(int n, int m,
    const std::vector<int>& rp,
    const std::vector<int>& ci,
    const std::vector<double>& vals) {
    rows = n;
    cols = m;
    row_ptr = rp;
    col_idx = ci;
    values = vals;
}

std::vector<std::vector<double>> SparseMatrix::multiply(const std::vector<std::vector<double>>& X) const {
    if (X.empty() || cols != static_cast<int>(X.size()))
        throw std::runtime_error("维度不匹配：SparseMatrix::multiply");
    int k = static_cast<int>(X[0].size());
    std::vector<std::vector<double>> res(rows, std::vector<double>(k, 0.0));
    for (int i = 0; i < rows; ++i) {
        for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            int j = col_idx[idx];
            double val = values[idx];
            for (int c = 0; c < k; ++c) {
                res[i][c] += val * X[j][c];
            }
        }
    }
    return res;
}

void SparseMatrix::print() const {
    std::cout << "稀疏矩阵 " << rows << "x" << cols << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            std::cout << "(" << i << "," << col_idx[idx] << ") = " << values[idx] << "  ";
        }
        std::cout << "\n";
    }
}