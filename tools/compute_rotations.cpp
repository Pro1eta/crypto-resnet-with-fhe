// 计算 FHE ResNet 推理中每个阶段所需的全部旋转量
// 编译：g++ -std=c++17 -O2 tools/compute_rotations.cpp -o compute_rotations
// 运行：./compute_rotations

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>

static const int NT = 32768;

struct PP {
    std::string name;
    int hi, wi, ci, ho, wo, co, ki, ko, ti, to, pi, po, q, s, fh, fw;
};

// ---- 精确复现各函数的旋转逻辑 ----

// sum_slots(c, m, gap)：step=1,2,4,...<m，旋转 step*gap
static void add_sum_slots(std::set<int>& S, int m, int gap) {
    for (int step = 1; step < m; step <<= 1)
        if (step * gap != 0) S.insert(step * gap);
}

// mult_par_conv_bn 的旋转
static void add_conv_rots(std::set<int>& S, const PP& p) {
    int fh = p.fh, fw = p.fw;
    // 步骤1：预计算旋转
    for (int i1 = 0; i1 < fh; i1++)
        for (int i2 = 0; i2 < fw; i2++) {
            int r = p.ki*p.ki*p.wi*(i1-(fh-1)/2) + p.ki*(i2-(fw-1)/2);
            if (r != 0) S.insert(r);
        }
    // 步骤2：sum_slots 三次
    add_sum_slots(S, p.ki, 1);
    add_sum_slots(S, p.ki, p.ki*p.ki*p.wi);
    add_sum_slots(S, p.ti, p.ki*p.ki*p.hi*p.wi);
    // 步骤3：内循环旋转（选择张量放置）
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
    // 步骤4：log2(po) 次累加
    for (int j = 0; j < (int)std::log2(p.po); j++) {
        int r = -(1<<j)*(NT/p.po);
        if (r != 0) S.insert(r);
    }
}

// downsamp 的旋转
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
    // final_rot
    int fr = -(p.ko*p.ko*p.ho*p.wo*p.to) / 2;
    if (fr != 0) S.insert(fr);
    // log2(po) 累加
    for (int j = 0; j < (int)std::log2(p.po); j++) {
        int r = -(1<<j)*p.ko*p.ko*p.ho*p.wo*p.to;
        if (r != 0) S.insert(r);
    }
}

// avg_pool 的旋转
static void add_avgpool_rots(std::set<int>& S, const PP& p) {
    // wi 方向
    for (int j = 0; j < (int)std::log2(p.wi); j++) S.insert((1<<j)*p.ki);
    // hi 方向
    for (int j = 0; j < (int)std::log2(p.hi); j++) S.insert((1<<j)*p.ki*p.ki*p.wi);
    // 重排
    for (int i1 = 0; i1 < p.ki; i1++)
        for (int i2 = 0; i2 < p.ti; i2++) {
            int i3 = p.ki*i2 + i1;
            int r = p.ki*p.ki*p.hi*p.wi*i2 + p.ki*p.wi*i1 - p.ki*i3;
            if (r != 0) S.insert(r);
        }
}

// fc_layer 的旋转：对角线 d=1..in_dim-1
static void add_fc_rots(std::set<int>& S, int in_dim) {
    for (int d = 1; d < in_dim; d++) S.insert(d);
}

static void print_set(const std::string& tag, const std::set<int>& S) {
    std::cout << "\n[" << tag << "] " << S.size() << " 个旋转量:\n  ";
    for (int v : S) std::cout << v << " ";
    std::cout << "\n";
}

int main() {
    // 论文 Table 6 参数（与 resnet.cpp 一致）
    PP C1  = {"conv1",      32,32,3,  32,32,16, 1,1, 3,16, 8,2, 2,1,3,3};
    PP C2  = {"conv2",      32,32,16, 32,32,16, 1,1,16,16, 2,2, 8,1,3,3};
    PP C3A = {"conv3A",     32,32,16, 16,16,32, 1,2,16, 8, 2,4,16,2,3,3};
    PP C3  = {"conv3",      16,16,32, 16,16,32, 2,2, 8, 8, 4,4, 8,1,3,3};
    PP C4A = {"conv4A",     16,16,32,  8, 8,64, 2,4, 8, 4, 4,8,16,2,3,3};
    PP C4  = {"conv4",       8, 8,64,  8, 8,64, 4,4, 4, 4, 8,8, 8,1,3,3};
    PP PL  = {"pool",        8, 8,64,  1, 1,64, 4,4, 4, 4, 0,0, 0,1,3,3};

    // stage1：C1, C2, C3A(输入侧 ki=1)
    std::set<int> s1;
    add_conv_rots(s1, C1);
    add_conv_rots(s1, C2);
    add_conv_rots(s1, C3A);  // skip conv 也用 C3A
    print_set("stage1 (C1+C2+C3A)", s1);

    // stage3：C3, C4A(输入侧 ki=2)
    std::set<int> s3;
    add_conv_rots(s3, C3);
    add_conv_rots(s3, C4A);
    print_set("stage3 (C3+C4A)", s3);

    // stage4：C4
    std::set<int> s4;
    add_conv_rots(s4, C4);
    print_set("stage4 (C4)", s4);

    // downsamp1：C3A 的下采样
    std::set<int> ds1;
    add_downsamp_rots(ds1, C3A);
    print_set("downsamp1 (C3A)", ds1);

    // downsamp2：C4A 的下采样
    std::set<int> ds2;
    add_downsamp_rots(ds2, C4A);
    print_set("downsamp2 (C4A)", ds2);

    // avgpool
    std::set<int> ap;
    add_avgpool_rots(ap, PL);
    print_set("avgpool", ap);

    // head (FC)
    std::set<int> hd;
    add_fc_rots(hd, PL.ci);  // ci=64
    print_set("head (FC, in_dim=64)", hd);

    return 0;
}
