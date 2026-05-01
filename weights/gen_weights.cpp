// 随机权重生成器：为 ResNet-20/32/56/110 生成随机 .bin 权重文件
// 编译：g++ -std=c++17 -O2 gen_weights.cpp -o gen_weights
// 运行：./gen_weights <weights_dir> <n>
//   n: 3=ResNet20, 5=ResNet32, 9=ResNet56, 18=ResNet110

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static const int NT = 32768;

static std::mt19937 rng(42);
static std::normal_distribution<double> dist(0.0, 0.01);

static void write_vec(const std::string& path, int size) {
    std::ofstream f(path);
    for (int i = 0; i < size; i++)
        f << dist(rng) << "\n";
}

// 每层权重：fh*fw*q 个文件（每个长度 NT）+ 1 个 bn_bias 文件
static void gen_layer(const std::string& dir, const std::string& name,
                      int q, int fh = 3, int fw = 3) {
    for (int i3 = 0; i3 < q; i3++)
        for (int k = 0; k < fh * fw; k++)
            write_vec(dir + "/" + name + "-w" + std::to_string(i3)
                      + "-k" + std::to_string(k) + ".bin", NT);
    write_vec(dir + "/" + name + "-bn_bias.bin", NT);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "用法: ./gen_weights <weights_dir> <n>\n";
        return 1;
    }
    std::string dir = argv[1];
    int n = std::atoi(argv[2]);

    // ConvBN1: q=2
    gen_layer(dir, "conv1", 2);

    // Stage 2: n blocks, 每 block 2 层, q=8
    for (int b = 1; b <= n; b++) {
        gen_layer(dir, "conv2_" + std::to_string(b) + "a", 8);
        gen_layer(dir, "conv2_" + std::to_string(b) + "b", 8);
    }

    // Stage 3 block 1: conv3_1a(q=16), conv3_1b(q=8), conv3_s1(q=16, skip)
    gen_layer(dir, "conv3_1a", 16);
    gen_layer(dir, "conv3_1b", 8);
    gen_layer(dir, "conv3_s1", 16);
    // Stage 3 identity blocks: q=8
    for (int b = 2; b <= n; b++) {
        gen_layer(dir, "conv3_" + std::to_string(b) + "a", 8);
        gen_layer(dir, "conv3_" + std::to_string(b) + "b", 8);
    }

    // Stage 4 block 1: conv4_1a(q=16), conv4_1b(q=8), conv4_s2(q=16, skip)
    gen_layer(dir, "conv4_1a", 16);
    gen_layer(dir, "conv4_1b", 8);
    gen_layer(dir, "conv4_s2", 16);
    // Stage 4 identity blocks: q=8
    for (int b = 2; b <= n; b++) {
        gen_layer(dir, "conv4_" + std::to_string(b) + "a", 8);
        gen_layer(dir, "conv4_" + std::to_string(b) + "b", 8);
    }

    // FC: 64 条对角线，每条长度 NT（有效值在前 10 个槽）
    write_vec(dir + "/fc.bin", 64 * NT);

    std::cout << "权重生成完成，目录: " << dir << "\n";
    return 0;
}
