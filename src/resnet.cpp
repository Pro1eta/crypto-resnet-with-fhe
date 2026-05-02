#include "resnet.h"
#include "conv.h"
#include "downsample.h"
#include "avgpool.h"
#include "activation.h"
#include "fc.h"
#include "weight_loader.h"

static const PackParams PP_CONV1  = {32,32,3,  32,32,16, 1,1, 3,16, 8,2, 2, 1};
static const PackParams PP_CONV2  = {32,32,16, 32,32,16, 1,1,16,16, 2,2, 8, 1};
static const PackParams PP_CONV3A = {32,32,16, 16,16,32, 1,2,16, 8, 2,4,16, 2};
static const PackParams PP_CONV3  = {16,16,32, 16,16,32, 2,2, 8, 8, 4,4, 8, 1};
static const PackParams PP_CONV4A = {16,16,32,  8, 8,64, 2,4, 8, 4, 4,8,16, 2};
static const PackParams PP_CONV4  = { 8, 8,64,  8, 8,64, 4,4, 4, 4, 8,8, 8, 1};
static const PackParams PP_POOL   = { 8, 8,64,  1, 1,64, 4,4, 4, 4, 0,0, 0, 1};

// 纯卷积+BN，不含 bootstrap
static Ctxt conv_bn(FHEContext& ctx, const Ctxt& in,
                    const string& name, const PackParams& pp,
                    const string& wd, int level) {
    auto lw = load_conv_weights(ctx, name, pp, level, wd);
    return mult_par_conv_bn(ctx, in, pp, lw.w, lw.bn_bias);
}

// bootstrap + AppReLU
static Ctxt boot_relu(FHEContext& ctx, Ctxt x, int boot_slots) {
    ctx.set_slots(x, boot_slots);
    x = ctx.bootstrap(x);
    return app_relu(ctx, x, 1.0);
}

int resnet_infer(FHEContext& ctx, const Ctxt& input,
                 int n, const string& wd, const string& kd) {
    // ConvBN1 → AppReLU（无 Boot）
    ctx.load_rot_keys(kd, "stage1", 16384);
    Ctxt x = conv_bn(ctx, input, "conv1", PP_CONV1, wd, ctx.circuit_depth - 4);
    x = app_relu(ctx, x, 1.0);

    // Stage2：n 个 block，每 block：ConvBN_a → Boot14 → ReLU → ConvBN_b → (+skip) → Boot14 → ReLU
    for (int b = 1; b <= n; b++) {
        Ctxt skip = x;
        x = conv_bn(ctx, x, "conv2_" + to_string(b) + "a", PP_CONV2, wd, x->GetLevel());
        x = boot_relu(ctx, x, 16384);
        x = conv_bn(ctx, x, "conv2_" + to_string(b) + "b", PP_CONV2, wd, x->GetLevel());
        x = ctx.add(x, skip);
        x = boot_relu(ctx, x, 16384);
    }

    // Stage3 block1：ConvBN3_1a → Boot13 → ReLU → ConvBN3_1b → (+DS1) → Boot13 → ReLU
    ctx.clear_rot_keys();
    ctx.load_rot_keys(kd, "stage3", 8192);
    {
        Ctxt skip = x;
        x = conv_bn(ctx, x, "conv3_1a", PP_CONV3A, wd, x->GetLevel());
        x = boot_relu(ctx, x, 8192);
        x = conv_bn(ctx, x, "conv3_1b", PP_CONV3, wd, x->GetLevel());

        Ctxt xs = conv_bn(ctx, skip, "conv3_s1", PP_CONV3A, wd, skip->GetLevel());
        ctx.clear_rot_keys();
        ctx.load_rot_keys(kd, "downsamp1", 0);
        xs = downsamp(ctx, xs, PP_CONV3A);
        ctx.set_slots(xs, 8192);
        ctx.set_slots(x, 8192);
        ctx.clear_rot_keys();
        ctx.load_rot_keys(kd, "stage3", 8192);

        x = ctx.add(x, xs);
        x = boot_relu(ctx, x, 8192);
    }
    // Stage3 identity blocks
    for (int b = 2; b <= n; b++) {
        Ctxt skip = x;
        x = conv_bn(ctx, x, "conv3_" + to_string(b) + "a", PP_CONV3, wd, x->GetLevel());
        x = boot_relu(ctx, x, 8192);
        x = conv_bn(ctx, x, "conv3_" + to_string(b) + "b", PP_CONV3, wd, x->GetLevel());
        x = ctx.add(x, skip);
        x = boot_relu(ctx, x, 8192);
    }

    // Stage4 block1：ConvBN4_1a → Boot12 → ReLU → ConvBN4_1b → (+DS2) → Boot12 → ReLU
    ctx.clear_rot_keys();
    ctx.load_rot_keys(kd, "stage4", 4096);
    {
        Ctxt skip = x;
        x = conv_bn(ctx, x, "conv4_1a", PP_CONV4A, wd, x->GetLevel());
        x = boot_relu(ctx, x, 4096);
        x = conv_bn(ctx, x, "conv4_1b", PP_CONV4, wd, x->GetLevel());

        Ctxt xs = conv_bn(ctx, skip, "conv4_s2", PP_CONV4A, wd, skip->GetLevel());
        ctx.clear_rot_keys();
        ctx.load_rot_keys(kd, "downsamp2", 0);
        xs = downsamp(ctx, xs, PP_CONV4A);
        ctx.set_slots(xs, 4096);
        ctx.set_slots(x, 4096);
        ctx.clear_rot_keys();
        ctx.load_rot_keys(kd, "stage4", 4096);

        x = ctx.add(x, xs);
        x = boot_relu(ctx, x, 4096);
    }
    // Stage4 identity blocks
    for (int b = 2; b <= n; b++) {
        Ctxt skip = x;
        x = conv_bn(ctx, x, "conv4_" + to_string(b) + "a", PP_CONV4, wd, x->GetLevel());
        x = boot_relu(ctx, x, 4096);
        x = conv_bn(ctx, x, "conv4_" + to_string(b) + "b", PP_CONV4, wd, x->GetLevel());
        x = ctx.add(x, skip);
        x = boot_relu(ctx, x, 4096);
    }

    ctx.clear_rot_keys();
    ctx.load_rot_keys(kd, "avgpool", 0);
    x = avg_pool(ctx, x, PP_POOL);
    ctx.clear_rot_keys();
    ctx.load_rot_keys(kd, "head", 0);
    x = fc_layer(ctx, x, wd + "/fc.bin", PP_POOL.ci, 10);

    auto result = ctx.decrypt(x, 10);
    return (int)(max_element(result.begin(), result.end()) - result.begin());
}
