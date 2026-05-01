#include "avgpool.h"
#include "packing.h"
#include <cmath>

// Algorithm 6：AvgPool
Ctxt avg_pool(const FHEContext& ctx, const Ctxt& in, const PackParams& pp) {
    int ki = pp.ki, hi = pp.hi, wi = pp.wi, ti = pp.ti;
    int nt = ctx.num_slots;

    Ctxt a = in;

    // 对 wi 方向求和（log2(wi) 次旋转累加）
    for (int j = 0; j < (int)log2(wi); j++)
        a = ctx.add(a, ctx.rot(a, (1 << j) * ki));

    // 对 hi 方向求和
    for (int j = 0; j < (int)log2(hi); j++)
        a = ctx.add(a, ctx.rot(a, (1 << j) * ki*ki*wi));

    // 重排：将 ki²·ti 个有效值顺序放置（论文 Appendix G.3）
    Ctxt out = ctx.encrypt(vector<double>(nt, 0.0), a->GetLevel());
    for (int i1 = 0; i1 < ki; i1++) {
        for (int i2 = 0; i2 < ti; i2++) {
            int i3 = ki*i2 + i1;
            // 选择向量 s̄'(i3)：位置 [ki*i3, ki*i3+ki) 处值为 1/(hi*wi)
            vector<double> sel(nt, 0.0);
            for (int j = 0; j < ki && ki*i3+j < nt; j++)
                sel[ki*i3 + j] = 1.0 / (hi * wi);
            int r = ki*ki*hi*wi*i2 + ki*wi*i1 - ki*i3;
            Ctxt piece = ctx.mul(ctx.rot(a, r), ctx.encode(sel, a->GetLevel()));
            out = ctx.add(out, piece);
        }
    }
    return out;
}
