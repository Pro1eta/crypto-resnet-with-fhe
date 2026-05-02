#include "downsample.h"
#include <cmath>

// Algorithm 5：DownSamp
Ctxt downsamp(const FHEContext& ctx, const Ctxt& in, const PackParams& pp) {
    int ki = pp.ki, ko = pp.ko;
    int hi = pp.hi, wi = pp.wi;
    int ti = pp.ti, to = pp.to;
    int nt = ctx.num_slots;
    int po = pp.po;

    vector<double> zeros(nt, 0.0);
    Ctxt out = ctx.encrypt(zeros, in->GetLevel());

    for (int i1 = 0; i1 < ki; i1++) {
        for (int i2 = 0; i2 < ti; i2++) {
            // Algorithm 5 第6-8行
            int tmp = ki*i2 + i1;
            int i3 = (tmp % (2*ko)) / 2;
            int i4 = tmp % 2;
            int i5 = tmp / (2*ko);

            // 生成选择张量 S''(i1,i2)
            vector<double> sel(nt, 0.0);
            for (int r = 0; r < ki*hi; r++) {
                for (int c = 0; c < ki*wi; c++) {
                    if ((r/ki) % 2 == 0 && (c/ki) % 2 == 0
                        && r % ki == i1) {
                        int t = i2;
                        int flat = t*(ki*hi)*(ki*wi) + r*(ki*wi) + c;
                        if (flat < nt) sel[flat] = 1.0;
                    }
                }
            }

            Ctxt cb = ctx.mul(in, ctx.encode(sel, in->GetLevel()));
            // Algorithm 5 第10行：旋转到目标位置（修正：旋转而非加法）
            int rot_amt = ki*ki*hi*wi*(i2 - i5) + ki*wi*(i1 - i3) - ki*i4;
            if (rot_amt != 0)
                cb = ctx.rot(cb, rot_amt);
            out = ctx.add(out, cb);
        }
    }

    // Algorithm 5 第13行：最终对齐旋转，论文公式 -ko²·ho·wo·ti/8
    int final_rot = -(ko*ko*pp.ho*pp.wo*ti) / 8;
    if (final_rot != 0)
        out = ctx.rot(out, final_rot);

    // Algorithm 5 第14-16行：log2(po) 次累加
    for (int j = 0; j < (int)log2(po); j++)
        out = ctx.add(out, ctx.rot(out, -(1 << j) * ko*ko*pp.ho*pp.wo*to));

    return out;
}
