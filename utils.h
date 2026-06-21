// utils.h
#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B);
void relu(std::vector<std::vector<double>>& H);
void print_matrix(const std::vector<std::vector<double>>& M, const std::string& name);

int read_int(const std::string& prompt);
double read_double(const std::string& prompt);
