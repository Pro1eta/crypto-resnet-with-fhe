#include "weight_loader.h"
#include "utils.h"

LayerWeights load_conv_weights(const FHEContext& ctx, const string& layer_name,
                                const PackParams& pp, int level,
                                const string& weights_dir) {
    LayerWeights lw;
    int fh = 3, fw = 3;
    int nt = ctx.num_slots;

    // 读取 fh*fw*q 个权重明文
    for (int i3 = 0; i3 < pp.q; i3++) {
        for (int i1 = 0; i1 < fh; i1++) {
            for (int i2 = 0; i2 < fw; i2++) {
                string path = weights_dir + "/" + layer_name
                    + "-w" + to_string(i3)
                    + "-k" + to_string(i1*fw + i2) + ".bin";
                auto v = utils::read_bin(path);
                v.resize(nt, 0.0);
                lw.w.push_back(ctx.encode(v, level, nt));
            }
        }
    }

    // 读取 BN 偏置
    auto bias_v = utils::read_bin(weights_dir + "/" + layer_name + "-bn_bias.bin");
    bias_v.resize(nt, 0.0);
    lw.bn_bias = ctx.encode(bias_v, level, nt);

    return lw;
}
