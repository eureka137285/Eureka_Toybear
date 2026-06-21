// main.cpp
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <iomanip>
#include <windows.h>
#include "Graph.h"
#include "SparseMatrix.h"
#include "GCN.h"
#include "utils.h"

using namespace std;

int main() {
    // 设置控制台编码为 UTF-8，解决中文乱码问题
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
   cout << "===== 图卷积神经网络 (GCN) =====\n\n";

    // ===== 1. 输入图数据，带重试 =====
    int N;
    while (true) {
        N = read_int("请输入节点数量 (>=1): ");
        if (N > 0) break;
        cout << "节点数量必须为正数，请重新输入。\n";
    }

    Graph g(N);
    // 暂存用户输入的完整矩阵，用于对称性校验
    vector<vector<int>> input_adj(N, vector<int>(N));
    cout << "请输入邻接矩阵（空格分隔，" << N << "x" << N << " 个整数，0或1）:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            string prompt = "  adj[" + to_string(i) + "][" + to_string(j) + "] = ";
            int val;
            while (true) {
                val = read_int(prompt);
                if (val == 0 || val == 1) break;
                cout << "只允许输入0或1，请重新输入。\n";
            }
            input_adj[i][j] = val;
            // 无向图：仅用上三角构建边，避免重复添加
            if (val == 1 && i < j) {
                g.addEdge(i, j);
            }
        }
    }

    // 对称性校验：对无向图，邻接矩阵必须对称
    bool symmetric = true;
    for (int i = 0; i < N && symmetric; ++i)
        for (int j = i + 1; j < N; ++j)
            if (input_adj[i][j] != input_adj[j][i])
                symmetric = false;
    if (!symmetric)
        cout << "\n[警告] 检测到邻接矩阵不对称，已以上三角元素为准构建图。\n";

    // 连通性：同时展示 BFS 与 DFS 两种方法
    cout << "\n图连通性检查: BFS -> " << (g.isConnected() ? "连通" : "不连通")
         << ", DFS -> " << (g.isConnectedDFS() ? "连通" : "不连通") << "\n";

    // 演示图基本操作：删除边、删除节点、新增节点
    cout << "\n--- 图基本操作演示 ---\n";

    // 删除边演示
    if (N >= 2) {
        int eu, ev;
        eu = read_int("删除边操作：请输入第一个节点编号: ");
        ev = read_int("删除边操作：请输入第二个节点编号: ");
        g.removeEdge(eu, ev);
        cout << "删除边 (" << eu << "," << ev << ") 后, BFS连通性: "
             << (g.isConnected() ? "连通" : "不连通") << "\n";
    }

    // 删除节点演示
    int dv = read_int("删除节点操作：请输入要删除的节点编号: ");
    g.removeNode(dv);
    cout << "删除节点 " << dv << " 后，当前节点数: " << g.getV()
         << ", BFS连通性: " << (g.isConnected() ? "连通" : "不连通") << "\n";

    // 新增节点演示
    g.addNode();
    cout << "新增一个孤立节点后，当前节点数: " << g.getV()
         << ", BFS连通性: " << (g.isConnected() ? "连通" : "不连通") << "\n";

    // 更新 N 为当前节点数，后续特征矩阵维度以新 N 为准
    N = g.getV();

    // ===== 2. 输入节点特征矩阵 X，维度带重试 =====
    int F;
    while (true) {
        F = read_int("\n请输入输入特征维度 F (>=1): ");
        if (F > 0) break;
        cout << "特征维度必须为正数，请重新输入。\n";
    }
    cout << "请输入节点特征矩阵 X (" << N << " x " << F << ")，按行输入：\n";
    vector<vector<double>> X(N, vector<double>(F));
    for (int i = 0; i < N; ++i) {
        cout << "第" << i << " 行: ";
        for (int j = 0; j < F; ++j) {
            X[i][j] = read_double("");
        }
    }
    print_matrix(X, "特征矩阵 X");

    // ===== 3. 权重矩阵 W，维度带重试 =====
    cout << "\n权重矩阵 W (大小 " << F << " x K):\n";
    int choice;
    while (true) {
        choice = read_int("选择: 1 - 手动输入, 2 - 随机初始化（默认范围 [-0.5,0.5]）: ");
        if (choice == 1 || choice == 2) break;
        cout << "无效选择，请重新输入 1 或 2。\n";
    }

    int K;
    while (true) {
        K = read_int("请输入输出特征维度 K (>=1): ");
        if (K > 0) break;
        cout << "输出维度必须为正数，请重新输入。\n";
    }

    vector<vector<double>> W(F, vector<double>(K));
    if (choice == 1) {
        cout << "请输入权重矩阵 W (" << F << " x " << K << ")，按行输入：\n";
        for (int i = 0; i < F; ++i) {
            cout << "第" << i << " 行: ";
            for (int j = 0; j < K; ++j) {
                W[i][j] = read_double("");
            }
        }
    }
    else {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dist(-0.5, 0.5);
        for (int i = 0; i < F; ++i)
            for (int j = 0; j < K; ++j)
                W[i][j] = dist(gen);
        cout << "随机初始化的权重矩阵 W:\n";
    }
    print_matrix(W, "权重矩阵 W");

    // ===== 4. 归一化邻接矩阵 =====
    SparseMatrix A_norm = normalize_adjacency(g);
    cout << "\n归一化邻接矩阵（稀疏格式）:\n";
    A_norm.print();


    // ===== 5. 模式选择与执行（可循环，方便反复实验） =====
    while (true) {
        int mode;
        while (true) {
            mode = read_int("\n请选择操作模式:\n  1 - 前向传播（仅计算输出特征）\n  2 - 反向传播（更新权重）\n请输入选择 (1 或 2): ");
            if (mode == 1 || mode == 2) break;
            cout << "无效选择，请重新输入 1 或 2。\n";
        }

        if (mode == 1) {
            // ---- 仅前向传播 ----
            vector<vector<double>> H = gcn_forward(A_norm, X, W);
            print_matrix(H, "GCN 输出节点特征 H'");
        }
        else {
            // ---- 反向传播 ----
            // 当前反向传播实现基于 sigmoid + 二分类交叉熵，仅支持 K=1
            int K_local = static_cast<int>(W[0].size());
            if (K_local != 1) {
                cout << "\n[提示] 反向传播目前仅支持 K=1（二分类），已自动将 K 设为 1。\n";
                K_local = 1;
                vector<vector<double>> W_new(F, vector<double>(1));
                for (int i = 0; i < F; ++i)
                    W_new[i][0] = W[i][0];
                W = W_new;
            }

            cout << "\n--- 反向传播模式设置 ---\n";
            int target_node;
            while (true) {
                target_node = read_int("请输入目标节点编号 (0 ~ " + to_string(N - 1) + "): ");
                if (target_node >= 0 && target_node < N) break;
                cout << "节点编号无效，必须在 0 到 " << N - 1 << " 之间，请重新输入。\n";
            }

            int label;
            while (true) {
                label = read_int("请输入目标节点的真实标签 (0 或 1): ");
                if (label == 0 || label == 1) break;
                cout << "标签只能为 0 或 1，请重新输入。\n";
            }

            double lr;
            while (true) {
                lr = read_double("请输入学习率 (例如 0.1): ");
                if (lr > 0.0) break;
                cout << "学习率必须为正数，请重新输入。\n";
            }

            int epochs;
            while (true) {
                epochs = read_int("请输入迭代次数 (>=1): ");
                if (epochs >= 1) break;
                cout << "迭代次数至少为 1，请重新输入。\n";
            }

            vector<vector<double>> H, AX, Z;
            cout << "\n开始训练...\n";
            for (int iter = 0; iter < epochs; ++iter) {
                gcn_forward_train(A_norm, X, W, H, AX, Z);

                double z = Z[target_node][0];               // K=1，取第0列
                double pred = 1.0 / (1.0 + exp(-z));        // sigmoid

                double loss = (label == 1) ? -log(pred) : -log(1.0 - pred);
                double dL_dz = pred - label;                // sigmoid+cross-entropy梯度

                // 仅目标节点的第0维有梯度（K=1）
                vector<vector<double>> grad_H(N, vector<double>(1, 0.0));
                grad_H[target_node][0] = dL_dz;

                gcn_backward(AX, Z, W, grad_H, lr);

                if (iter % 20 == 0 || iter == epochs - 1) {
                    cout << "迭代 " << iter << " / " << epochs
                         << "  损失: " << setprecision(6) << loss
                         << "  预测概率: " << pred << "\n";
                }
            }
            cout << "\n训练完成！\n";
            print_matrix(W, "更新后的权重矩阵 W");
        }

        // 询问是否继续
        int cont;
        while (true) {
            cont = read_int("\n是否继续实验？\n  1 - 继续（重新选择模式）\n  2 - 退出程序\n请输入选择 (1 或 2): ");
            if (cont == 1 || cont == 2) break;
            cout << "无效选择，请重新输入 1 或 2。\n";
        }
        if (cont == 2) break;
    }
    return 0;
}