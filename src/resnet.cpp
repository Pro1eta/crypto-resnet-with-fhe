#include "resnet.h"
#include "conv.h"
#include "downsample.h"
#include "avgpool.h"
#include "activation.h"
#include "fc.h"
#include "weight_loader.h"

// 论文 Table 6 中各层参数
static const PackParams PP_CONV1  = {32,32,3,  32,32,16, 1,1, 3,16, 8,2, 2, 1};
static const PackParams PP_CONV2  = {32,32,16, 32,32,16, 1,1,16,16, 2,2, 8, 1};
static const PackParams PP_CONV3A = {32,32,16, 16,16,32, 1,2,16, 8, 2,4,16, 2};
static const PackParams PP_CONV3  = {16,16,32, 16,16,32, 2,2, 8, 8, 4,4, 8, 1};
static const PackParams PP_CONV4A = {16,16,32,  8, 8,64, 2,4, 8, 4, 4,8,16, 2};
static const PackParams PP_CONV4  = { 8, 8,64,  8, 8,64, 4,4, 4, 4, 8,8, 8, 1};
static const PackParams PP_POOL   = { 8, 8,64,  1, 1,64, 4,4, 4, 4, 0,0, 0, 1};

static Ctxt conv_bn_relu(FHEContext& ctx, const Ctxt& in,
                          const string& name, const PackParams& pp,
                          const string& wd, int level) {
    auto lw = load_conv_weights(ctx, name, pp, level, wd);
    Ctxt out = mult_par_conv_bn(ctx, in, pp, lw.w, lw.bn_bias);
    out = ctx.bootstrap(out);
    return app_relu(ctx, out, 1.0);
}

int resnet_infer(FHEContext& ctx, const Ctxt& input,
                 int n, const string& wd, const string& kd) {
    // Stage 1：ConvBN1 (3→16, 32×32)
    ctx.load_rot_keys(kd, "stage1", 16384);
    Ctxt x = conv_bn_relu(ctx, input, "conv1", PP_CONV1, wd, ctx.circuit_depth - 4);

    // Stage 2：n 个 identity block (16→16, 32×32)
    for (int b = 1; b <= n; b++) {
        Ctxt skip = x;
        x = conv_bn_relu(ctx, x, "conv2_" + to_string(b) + "a", PP_CONV2, wd, x->GetLevel());
        auto lw = load_conv_weights(ctx, "conv2_" + to_string(b) + "b", PP_CONV2, x->GetLevel(), wd);
        x = mult_par_conv_bn(ctx, x, PP_CONV2, lw.w, lw.bn_bias);
        x = ctx.add(x, skip);
        x = ctx.bootstrap(x);
        x = app_relu(ctx, x, 1.0);
    }

    // Stage 3 block 1：downsample (16→32, s=2)
    ctx.clear_rot_keys();
    ctx.load_rot_keys(kd, "stage3", 8192);
    {
        Ctxt skip = x;
        x = ctx.bootstrap(x);
        Ctxt xa = conv_bn_relu(ctx, x, "conv3_1a", PP_CONV3A, wd, x->GetLevel());
        auto lw_b = load_conv_weights(ctx, "conv3_1b", PP_CONV3, xa->GetLevel(), wd);
        xa = mult_par_conv_bn(ctx, xa, PP_CONV3, lw_b.w, lw_b.bn_bias);
        auto lw_s = load_conv_weights(ctx, "conv3_s1", PP_CONV3A, skip->GetLevel(), wd);
        Ctxt xs = mult_par_conv_bn(ctx, skip, PP_CONV3A, lw_s.w, lw_s.bn_bias);
        xs = downsamp(ctx, xs, PP_CONV3A);
        x = ctx.add(xa, xs);
        x = ctx.bootstrap(x);
        x = app_relu(ctx, x, 1.0);
    }
    // Stage 3 identity blocks
    for (int b = 2; b <= n; b++) {
        Ctxt skip = x;
        x = conv_bn_relu(ctx, x, "conv3_" + to_string(b) + "a", PP_CONV3, wd, x->GetLevel());
        auto lw = load_conv_weights(ctx, "conv3_" + to_string(b) + "b", PP_CONV3, x->GetLevel(), wd);
        x = mult_par_conv_bn(ctx, x, PP_CONV3, lw.w, lw.bn_bias);
        x = ctx.add(x, skip);
        x = ctx.bootstrap(x);
        x = app_relu(ctx, x, 1.0);
    }

    // Stage 4 block 1：downsample (32→64, s=2)
    ctx.clear_rot_keys();
    ctx.load_rot_keys(kd, "stage4", 4096);
    {
        Ctxt skip = x;
        x = ctx.bootstrap(x);
        Ctxt xa = conv_bn_relu(ctx, x, "conv4_1a", PP_CONV4A, wd, x->GetLevel());
        auto lw_b = load_conv_weights(ctx, "conv4_1b", PP_CONV4, xa->GetLevel(), wd);
        xa = mult_par_conv_bn(ctx, xa, PP_CONV4, lw_b.w, lw_b.bn_bias);
        auto lw_s = load_conv_weights(ctx, "conv4_s2", PP_CONV4A, skip->GetLevel(), wd);
        Ctxt xs = mult_par_conv_bn(ctx, skip, PP_CONV4A, lw_s.w, lw_s.bn_bias);
        xs = downsamp(ctx, xs, PP_CONV4A);
        x = ctx.add(xa, xs);
        x = ctx.bootstrap(x);
        x = app_relu(ctx, x, 1.0);
    }
    // Stage 4 identity blocks
    for (int b = 2; b <= n; b++) {
        Ctxt skip = x;
        x = conv_bn_relu(ctx, x, "conv4_" + to_string(b) + "a", PP_CONV4, wd, x->GetLevel());
        auto lw = load_conv_weights(ctx, "conv4_" + to_string(b) + "b", PP_CONV4, x->GetLevel(), wd);
        x = mult_par_conv_bn(ctx, x, PP_CONV4, lw.w, lw.bn_bias);
        x = ctx.add(x, skip);
        x = ctx.bootstrap(x);
        x = app_relu(ctx, x, 1.0);
    }

    // Head：AvgPool + FC
    ctx.clear_rot_keys();
    ctx.load_rot_keys(kd, "head", 0);
    x = avg_pool(ctx, x, PP_POOL);
    x = fc_layer(ctx, x, wd + "/fc.bin", 64, 10);

    auto result = ctx.decrypt(x, 10);
    return (int)(max_element(result.begin(), result.end()) - result.begin());
}
