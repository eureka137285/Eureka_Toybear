// utils.cpp
#include "utils.h"
#include <limits>

std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    int m = static_cast<int>(A.size());
    int n = static_cast<int>(A[0].size());
    int p = static_cast<int>(B[0].size());
    std::vector<std::vector<double>> C(m, std::vector<double>(p, 0.0));
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < n; ++k)
            if (A[i][k] != 0.0)
                for (int j = 0; j < p; ++j)
                    C[i][j] += A[i][k] * B[k][j];
    return C;
}

void relu(std::vector<std::vector<double>>& H) {
    for (auto& row : H)
        for (auto& val : row)
            if (val < 0.0) val = 0.0;
}

void print_matrix(const std::vector<std::vector<double>>& M, const std::string& name) {
    std::cout << name << " (" << M.size() << "x" << M[0].size() << "):\n";
    for (const auto& row : M) {
        for (double v : row)
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << v << " ";
        std::cout << "\n";
    }
}

int read_int(const std::string& prompt) {
    int val;
    while (true) {
        std::cout << prompt;
        if (std::cin >> val) break;
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "输入无效，请输入一个整数。\n";
    }
    return val;
}

double read_double(const std::string& prompt) {
    double val;
    while (true) {
        std::cout << prompt;
        if (std::cin >> val) break;
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "输入无效，请输入一个数字。\n";
    }
    return val;
}