#include <iostream>
#include <iomanip>
#include <random>

int main() {
    int N = 900; // 节点数量，可以根据需要修改

    // 输出节点数量
    std::cout << N << std::endl;

    // 使用 Mersenne Twister 引擎和均匀分布
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    // 生成 N 个随机坐标，范围在 0 到 100 之间，保留 4 位小数
    for (int i = 0; i < N; ++i) {
        double x = dist(rng);
        double y = dist(rng);

        // 输出坐标，固定小数点，保留 4 位g++ main.cpp -o main_program
        std::cout << std::fixed << std::setprecision(4) << x << " " << y << std::endl;
    }

    return 0;
}
