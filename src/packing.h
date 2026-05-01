#pragma once
#include "fhe_context.h"

// 论文 Appendix E/F.2：打包参数结构体
struct PackParams {
    int hi, wi, ci;   // 输入张量尺寸
    int ho, wo, co;   // 输出张量尺寸
    int ki, ko;       // 间隙（gap）
    int ti, to;       // 通道组数 ti=ceil(ci/ki²), to=ceil(co/ko²)
    int pi, po;       // 并行副本数
    int q;            // 并行通道组数 q=ceil(co/pi)
    int s;            // 卷积步长
    int fh = 3, fw = 3; // 卷积核尺寸
};

// 计算打包参数（论文 Appendix F.2 公式）
PackParams compute_pack_params(int hi, int wi, int ci, int ho, int wo, int co,
                                int ki, int ko, int s, int nt = 32768);

// Vec 函数：三维张量 → 一维向量（论文 Appendix C）
vector<double> vec_tensor(const vector<double>& A, int h, int w, int c, int nt = 32768);

// MultParPack：将输入向量打包为 parallelly multiplexed 格式
// 权重文件已预处理为此格式，此函数主要用于输入图像打包
Ctxt mult_par_pack(const FHEContext& ctx, const vector<double>& A,
                   const PackParams& pp, int level);

// 生成选择张量 S'(i)（论文 Appendix F.1）
vector<double> gen_select_tensor(int i, const PackParams& pp, int nt = 32768);

// SumSlots 子程序（Algorithm 1）
Ctxt sum_slots(const FHEContext& ctx, const Ctxt& c, int m, int p);
