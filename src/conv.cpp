#include "conv.h"
#include "packing.h"
#include <cmath>

// Algorithm 7：MultParConvBN
Ctxt mult_par_conv_bn(const FHEContext& ctx, const Ctxt& in,
                      const PackParams& pp,
                      const vector<Ptxt>& weights,
                      const Ptxt& bn_bias) {
    int fh = pp.fh, fw = pp.fw;
    int ki = pp.ki, ko = pp.ko;
    int wi = pp.wi, hi = pp.hi;
    int ti = pp.ti, q = pp.q;
    int pi = pp.pi, po = pp.po;
    int nt = ctx.num_slots;

    // 步骤1：预计算 fh*fw 个旋转密文（Algorithm 7 第4-8行）
    vector<Ctxt> rots(fh * fw, in);
    for (int i1 = 0; i1 < fh; i1++)
        for (int i2 = 0; i2 < fw; i2++) {
            int r = ki*ki*wi*(i1 - (fh-1)/2) + ki*(i2 - (fw-1)/2);
            rots[i1*fw + i2] = (r == 0) ? in : ctx.rot(in, r);
        }

    // 零密文（不消耗层级）
    vector<double> zeros(nt, 0.0);
    Ctxt out = ctx.encrypt(zeros, in->GetLevel());

    // 步骤2：外循环 q 次（第9-23行）
    for (int i3 = 0; i3 < q; i3++) {
        // 累加 fh*fw 个旋转密文乘权重，首次直接赋值避免加零
        Ctxt cb = ctx.mul(rots[0], weights[i3*fh*fw]);
        for (int i1 = 0; i1 < fh; i1++)
            for (int i2 = 0; i2 < fw; i2++) {
                if (i1 == 0 && i2 == 0) continue;
                cb = ctx.add(cb, ctx.mul(rots[i1*fw + i2], weights[i3*fh*fw + i1*fw + i2]));
            }

        // SumSlots 三次（第16-18行）
        // 第一次：ki 方向（列内），gap=1
        cb = sum_slots(ctx, cb, ki, 1);
        // 第二次：hi 方向（行间），gap=ki²·wi
        cb = sum_slots(ctx, cb, ki, ki*ki*wi);
        // 第三次：ti 方向（通道组），gap=ki²·hi·wi
        cb = sum_slots(ctx, cb, ti, ki*ki*hi*wi);

        // 内循环：选择并放置输出通道（第19-22行，融合 BN 缩放）
        int i4_max = min(pi - 1, pp.co - 1 - pi*i3);
        for (int i4 = 0; i4 <= i4_max; i4++) {
            int i = pi*i3 + i4;
            // Algorithm 7 第21行旋转量
            int r = -(i/(ko*ko))*(ko*ko*pp.ho*pp.wo)
                    + (nt/pi)*(i % pi)
                    - ((i % (ko*ko))/ko)*(ko*pp.wo)
                    - (i % ko);
            auto sel = gen_select_tensor(i, pp, nt);
            Ptxt sel_ptxt = ctx.encode(sel, cb->GetLevel());
            out = ctx.add(out, ctx.mul(ctx.rot(cb, r), sel_ptxt));
        }
    }

    // 步骤3：log2(po) 次旋转累加（第24-26行）
    for (int j = 0; j < (int)log2(po); j++)
        out = ctx.add(out, ctx.rot(out, -(1 << j) * (nt / po)));

    // 步骤4：加 BN 偏置（第27行）
    Ptxt bias_copy = bn_bias;
    out = ctx.cc->EvalAdd(out, bias_copy);

    return out;
}
