#include "conv.h"
#include "packing.h"

// Algorithm 7：MultParConvBN
// weights[i3 * fh*fw + i1*fw + i2] = ParMultWgt(U; i1, i2, i3)，已预处理
Ctxt mult_par_conv_bn(const FHEContext& ctx, const Ctxt& in,
                      const PackParams& pp,
                      const vector<Ptxt>& weights,
                      const Ptxt& bn_bias) {
    int fh = 3, fw = 3;
    int ki = pp.ki, ko = pp.ko;
    int wi = pp.wi, hi = pp.hi;
    int ti = pp.ti, q = pp.q;
    int pi = pp.pi, po = pp.po;
    int nt = ctx.num_slots;

    // 步骤1：预计算 fh*fw 个旋转密文（论文 Algorithm 7 第4-8行）
    vector<Ctxt> rots(fh * fw, in);
    for (int i1 = 0; i1 < fh; i1++) {
        for (int i2 = 0; i2 < fw; i2++) {
            int r = ki*ki*wi*(i1 - (fh-1)/2) + ki*(i2 - (fw-1)/2);
            rots[i1*fw + i2] = (r == 0) ? in : ctx.rot(in, r);
        }
    }

    // 步骤2：外循环 q 次（第9-23行）
    Ctxt out = ctx.cc->EvalMult(in, ctx.encode(vector<double>(nt, 0.0), in->GetLevel()));
    for (int i3 = 0; i3 < q; i3++) {
        // 累加 fh*fw 个旋转密文乘权重
        Ctxt cb = ctx.cc->EvalMult(in, ctx.encode(vector<double>(nt, 0.0), in->GetLevel()));
        for (int i1 = 0; i1 < fh; i1++) {
            for (int i2 = 0; i2 < fw; i2++) {
                int widx = i3*fh*fw + i1*fw + i2;
                cb = ctx.add(cb, ctx.mul(rots[i1*fw + i2], weights[widx]));
            }
        }

        // SumSlots 三次（第16-18行）
        cb = sum_slots(ctx, cb, ki, 1);
        cb = sum_slots(ctx, cb, ki, ki*wi);
        cb = sum_slots(ctx, cb, ti, ki*ki*hi*wi);

        // 内循环：选择并放置输出通道（第19-22行，融合 BN 缩放）
        int i4_max = min(pi - 1, pp.co - 1 - pi*i3);
        for (int i4 = 0; i4 <= i4_max; i4++) {
            int i = pi*i3 + i4;
            int r = -(i/(ko*ko))*(ko*ko*pp.ho*pp.wo)
                    + (nt/pi)*(i % pi)
                    - ((i % (ko*ko))/ko)*(ko*pp.wo)
                    - (i % ko);
            auto sel = gen_select_tensor(i, pp, nt);
            // 选择张量已融合 BN 的 C（ParBNConst(C) * S'(i)）
            Ptxt sel_ptxt = ctx.encode(sel, cb->GetLevel());
            Ctxt piece = ctx.mul(ctx.rot(cb, r), sel_ptxt);
            out = ctx.add(out, piece);
        }
    }

    // 步骤3：log2(po) 次旋转累加（第24-26行）
    for (int j = 0; j < (int)log2(po); j++)
        out = ctx.add(out, ctx.rot(out, -(1 << j) * (nt / po)));

    // 步骤4：减去 BN 偏置（第27行）
    Ptxt bias_copy = bn_bias;
    out = ctx.cc->EvalAdd(out, bias_copy);

    return out;
}
