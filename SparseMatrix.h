// SparseMatrix.h
#pragma once
#include <vector>
#include <iostream>

class SparseMatrix {
public:
    SparseMatrix();
    SparseMatrix(int n, int m);
    void set_data(int n, int m,
        const std::vector<int>& rp,
        const std::vector<int>& ci,
        const std::vector<double>& vals);
    std::vector<std::vector<double>> multiply(const std::vector<std::vector<double>>& X) const;
    void print() const;

private:
    int rows, cols;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
};