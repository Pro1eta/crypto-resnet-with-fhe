#include "fc.h"
#include "utils.h"

Ctxt fc_layer(const FHEContext& ctx, const Ctxt& in,
              const string& weight_path, int in_dim, int out_dim) {
    auto w = utils::read_bin(weight_path);
    int nt = ctx.num_slots;

    // 对角线方法：将权重矩阵按对角线展开，每条对角线乘一次旋转密文
    // 权重文件已预处理为对角线格式，每段长度为 in_dim
    Ctxt out = ctx.encrypt(vector<double>(nt, 0.0), in->GetLevel());
    for (int d = 0; d < in_dim; d++) {
        vector<double> diag(nt, 0.0);
        for (int j = 0; j < out_dim; j++)
            diag[j] = w[d * out_dim + j];  // 第 d 条对角线
        Ctxt rot_in = (d == 0) ? in : ctx.rot(in, d);
        out = ctx.add(out, ctx.mul(rot_in, ctx.encode(diag, in->GetLevel())));
    }
    return out;
}
