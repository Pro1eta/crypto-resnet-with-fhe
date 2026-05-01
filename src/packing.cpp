#include "packing.h"
#include <cmath>

PackParams compute_pack_params(int hi, int wi, int ci, int ho, int wo, int co,
                                int ki, int ko, int s, int nt) {
    PackParams pp;
    pp.hi = hi; pp.wi = wi; pp.ci = ci;
    pp.ho = ho; pp.wo = wo; pp.co = co;
    pp.ki = ki; pp.ko = ko; pp.s  = s;
    pp.ti = (ci + ki*ki - 1) / (ki*ki);
    pp.to = (co + ko*ko - 1) / (ko*ko);
    // pi = 2^floor(log2(nt / (ki²·hi·wi·ti)))
    int denom_i = ki*ki * hi*wi * pp.ti;
    pp.pi = (denom_i > 0) ? (1 << (int)floor(log2((double)nt / denom_i))) : 1;
    int denom_o = ko*ko * ho*wo * pp.to;
    pp.po = (denom_o > 0) ? (1 << (int)floor(log2((double)nt / denom_o))) : 1;
    pp.q  = (co + pp.pi - 1) / pp.pi;
    return pp;
}

// 论文 Appendix C：Vec 映射
vector<double> vec_tensor(const vector<double>& A, int h, int w, int c, int nt) {
    vector<double> y(nt, 0.0);
    for (int i = 0; i < h*w*c && i < nt; i++) {
        int ch  = i / (h*w);
        int row = (i % (h*w)) / w;
        int col = i % w;
        y[i] = A[ch*h*w + row*w + col];
    }
    return y;
}

// 生成选择张量 S'(i)（论文 Appendix F.1）
vector<double> gen_select_tensor(int i, const PackParams& pp, int nt) {
    int ko = pp.ko, ho = pp.ho, wo = pp.wo, to = pp.to;
    vector<double> s(nt, 0.0);
    for (int i3 = 0; i3 < ko*ho; i3++) {
        for (int i4 = 0; i4 < ko*wo; i4++) {
            for (int i5 = 0; i5 < to; i5++) {
                int ch_idx = ko*ko*i5 + ko*(i3 % ko) + i4 % ko;
                if (ch_idx == i) {
                    int flat = i5*(ko*ho)*(ko*wo) + i3*(ko*wo) + i4;
                    if (flat < nt) s[flat] = 1.0;
                }
            }
        }
    }
    return s;
}

// Algorithm 1：SumSlots(cta; m, p)
Ctxt sum_slots(const FHEContext& ctx, const Ctxt& c, int m, int gap) {
    if (m <= 1) return c;
    Ctxt res = c;
    for (int step = 1; step < m; step <<= 1)
        res = ctx.add(res, ctx.rot(res, step * gap));
    return res;
}

Ctxt mult_par_pack(const FHEContext& ctx, const vector<double>& A,
                   const PackParams& pp, int level) {
    // 输入图像按 MultParPack 格式打包后加密
    // 实际上是 pi 份 MultPack(A) 的副本拼接
    // 此处简化：直接将 Vec(A) 重复 pi 次填满 nt 个槽
    int nt = ctx.num_slots;
    int block = nt / pp.pi;
    vector<double> packed(nt, 0.0);
    vector<double> base = vec_tensor(A, pp.hi, pp.wi, pp.ci, block);
    for (int j = 0; j < pp.pi; j++) {
        int off = j * block;
        for (int k = 0; k < block && off+k < nt; k++)
            packed[off+k] = base[k];
    }
    return ctx.encrypt(packed, level);
}
