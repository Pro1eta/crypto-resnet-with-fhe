// 计算 FHE ResNet 推理中每个阶段所需的全部旋转量
// 与 src/ 中的实际算法逻辑完全对齐
// 编译：g++ -std=c++17 -O2 tools/compute_rotations.cpp -o tools/compute_rotations
// 运行：./tools/compute_rotations

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>

static const int NT = 32768;

struct PP {
    int hi, wi, ci, ho, wo, co, ki, ko, ti, to, pi, po, q, s;
    int fh = 3, fw = 3;
};

// 与 packing.cpp sum_slots 完全对齐（Algorithm 1 两阶段）
static void add_sum_slots(std::set<int>& S, int m, int gap) {
    if (m <= 1) return;
    int logm = (int)std::floor(std::log2((double)m));
    // 第一阶段：倍增
    for (int j = 1; j <= logm; j++) {
        int r = (1 << (j-1)) * gap;
        if (r != 0) S.insert(r);
    }
    // 第二阶段：余量修正
    for (int j = 0; j < logm; j++) {
        if ((m >> j) % 2 == 1) {
            int shift = ((m >> (j+1)) << (j+1)) * gap;
            if (shift != 0) S.insert(shift);
        }
    }
}

// 与 conv.cpp mult_par_conv_bn 完全对齐
static void add_conv_rots(std::set<int>& S, const PP& p) {
    // 步骤1：预计算 fh*fw 旋转
    for (int i1 = 0; i1 < p.fh; i1++)
        for (int i2 = 0; i2 < p.fw; i2++) {
            int r = p.ki*p.ki*p.wi*(i1-(p.fh-1)/2) + p.ki*(i2-(p.fw-1)/2);
            if (r != 0) S.insert(r);
        }
    // 步骤2：SumSlots 三次（gap 与 conv.cpp 对齐）
    add_sum_slots(S, p.ki, 1);
    add_sum_slots(S, p.ki, p.ki*p.wi);          // 修正：ki*wi，非 ki²*wi
    add_sum_slots(S, p.ti, p.ki*p.ki*p.hi*p.wi);
    // 步骤3：内循环选择放置旋转
    for (int i3 = 0; i3 < p.q; i3++) {
        int i4_max = std::min(p.pi-1, p.co-1-p.pi*i3);
        for (int i4 = 0; i4 <= i4_max; i4++) {
            int i = p.pi*i3 + i4;
            int r = -(i/(p.ko*p.ko))*(p.ko*p.ko*p.ho*p.wo)
                    + (NT/p.pi)*(i % p.pi)
                    - ((i % (p.ko*p.ko))/p.ko)*(p.ko*p.wo)
                    - (i % p.ko);
            if (r != 0) S.insert(r);
        }
    }
    // 步骤4：log2(po) 次并行累加
    for (int j = 0; j < (int)std::log2(p.po); j++) {
        int r = -(1<<j)*(NT/p.po);
        if (r != 0) S.insert(r);
    }
}

// 与 downsample.cpp downsamp 完全对齐
static void add_downsamp_rots(std::set<int>& S, const PP& p) {
    for (int i1 = 0; i1 < p.ki; i1++) {
        for (int i2 = 0; i2 < p.ti; i2++) {
            int tmp = p.ki*i2 + i1;
            int i3 = (tmp % (2*p.ko)) / 2;
            int i4 = tmp % 2;
            int i5 = tmp / (2*p.ko);
            int r = p.ki*p.ki*p.hi*p.wi*(i2-i5) + p.ki*p.wi*(i1-i3) - p.ki*i4;
            if (r != 0) S.insert(r);
        }
    }
    // 最终对齐旋转（修正：ti/8，非 to/2）
    int fr = -(p.ko*p.ko*p.ho*p.wo*p.ti) / 8;
    if (fr != 0) S.insert(fr);
    // log2(po) 次并行累加
    for (int j = 0; j < (int)std::log2(p.po); j++) {
        int r = -(1<<j)*p.ko*p.ko*p.ho*p.wo*p.to;
        if (r != 0) S.insert(r);
    }
}

// 与 avgpool.cpp avg_pool 完全对齐
static void add_avgpool_rots(std::set<int>& S, const PP& p) {
    for (int j = 0; j < (int)std::log2(p.wi); j++) S.insert((1<<j)*p.ki);
    for (int j = 0; j < (int)std::log2(p.hi); j++) S.insert((1<<j)*p.ki*p.ki*p.wi);
    for (int i1 = 0; i1 < p.ki; i1++)
        for (int i2 = 0; i2 < p.ti; i2++) {
            int i3 = p.ki*i2 + i1;
            int r = p.ki*p.ki*p.hi*p.wi*i2 + p.ki*p.wi*i1 - p.ki*i3;
            if (r != 0) S.insert(r);
        }
}

static void print_vec(const std::string& tag, const std::set<int>& S) {
    std::cout << "// " << tag << " (" << S.size() << " 个)\n{";
    bool first = true;
    for (int v : S) { if (!first) std::cout << ","; std::cout << v; first = false; }
    std::cout << "}\n\n";
}

int main() {
    // 与 resnet.cpp 中 PackParams 完全一致
    PP C1  = {32,32, 3, 32,32,16, 1,1, 3,16, 8,2, 2,1};
    PP C2  = {32,32,16, 32,32,16, 1,1,16,16, 2,2, 8,1};
    PP C3A = {32,32,16, 16,16,32, 1,2,16, 8, 2,4,16,2};
    PP C3  = {16,16,32, 16,16,32, 2,2, 8, 8, 4,4, 8,1};
    PP C4A = {16,16,32,  8, 8,64, 2,4, 8, 4, 4,8,16,2};
    PP C4  = { 8, 8,64,  8, 8,64, 4,4, 4, 4, 8,8, 8,1};
    PP PL  = { 8, 8,64,  1, 1,64, 4,4, 4, 4, 0,0, 0,1};

    // stage1：conv1(C1) + conv2_*a/b(C2) + conv3_1a/conv3_s1(C3A)
    std::set<int> s1;
    add_conv_rots(s1, C1);
    add_conv_rots(s1, C2);
    add_conv_rots(s1, C3A);
    print_vec("stage1", s1);

    // stage3：conv3_1a/conv3_s1(C3A) + conv3_*a/b(C3)
    // 注意：C3A 的卷积旋转在 stage3 密钥加载期间也会用到
    std::set<int> s3;
    add_conv_rots(s3, C3A);
    add_conv_rots(s3, C3);
    print_vec("stage3", s3);

    // stage4：conv4_1a/conv4_s2(C4A) + conv4_*a/b(C4)
    std::set<int> s4;
    add_conv_rots(s4, C4A);
    add_conv_rots(s4, C4);
    print_vec("stage4", s4);

    // downsamp1：PP_CONV3A
    std::set<int> ds1;
    add_downsamp_rots(ds1, C3A);
    print_vec("downsamp1", ds1);

    // downsamp2：PP_CONV4A
    std::set<int> ds2;
    add_downsamp_rots(ds2, C4A);
    print_vec("downsamp2", ds2);

    // avgpool：PP_POOL
    std::set<int> ap;
    add_avgpool_rots(ap, PL);
    print_vec("avgpool", ap);

    // head：FC in_dim=64，对角线 d=1..63
    std::set<int> hd;
    for (int d = 1; d < 64; d++) hd.insert(d);
    print_vec("head", hd);

    return 0;
}
