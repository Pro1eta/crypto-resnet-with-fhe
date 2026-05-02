#include "resnet.h"
#include "packing.h"
#include "utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static vector<double> load_cifar_image(const string& path) {
    int w, h, c;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) { cerr << "无法加载图片: " << path << "\n"; return {}; }
    vector<double> v;
    v.reserve(32 * 32 * 3);
    // 按 R/G/B 三通道分别展开（与参考项目一致）
    for (int ch = 0; ch < 3; ch++)
        for (int i = 0; i < 32*32; i++)
            v.push_back(data[i*3 + ch] / 255.0);
    stbi_image_free(data);
    return v;
}

static void print_usage(const char* prog) {
    cerr << "用法:\n"
         << "  " << prog << " gen_keys  <keys_dir> [n]\n"
         << "  " << prog << " infer     <keys_dir> <weights_dir> <image.png> [n]\n"
         << "n: ResNet 深度参数 (3=ResNet20, 5=ResNet32, 9=ResNet56, 18=ResNet110)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    string mode     = argv[1];
    string keys_dir = argv[2];
    int n = 3;  // 默认 ResNet-20

    FHEContext ctx;

    if (mode == "gen_keys") {
        if (argc >= 4) n = atoi(argv[3]);
        cout << "生成密钥到 " << keys_dir << " ...\n";
        ctx.generate(keys_dir, true);

        // 自举密钥与旋转密钥合并，按 stage 分组序列化
        // Stage 1/2：boot_slots=16384
        ctx.gen_rot_keys(
            {-16384,-14336,-12288,-10240,-8192,-7200,-7168,-6176,-6144,
             -5152,-5120,-4128,-4096,-3104,-3072,-2080,-2048,-1056,-1024,
             -33,-32,-31,-1,1,31,32,33,
             1024,2048,3072,4096,5120,6144,7168,8192,
             9183,9215,9216,10207,10239,10240,11231,11263,11264,
             12255,12287,12288,13279,13311,13312,14303,14335,
             15327,15359,15360,16351,16383,18432,21504},
            16384, keys_dir, "stage1", true);
        // 由 tools/compute_rotations 生成（59 个）
        ctx.clear_rot_keys(); ctx.load(keys_dir);

        // Stage 3：boot_slots=8192
        // 由 tools/compute_rotations 生成（76 个）
        ctx.gen_rot_keys(
            {-16384,-8192,-7200,-7168,-6176,-6144,-5152,-5120,-4128,-4096,
             -3104,-3072,-2080,-2048,-1056,-1024,-66,-64,-62,-33,-32,-31,-2,-1,
             1,2,31,32,33,62,64,66,
             1023,1024,2047,2048,3071,4095,4096,5119,6143,7167,8191,8192,
             9183,9184,9215,10207,10208,10239,11231,11232,11263,
             12255,12256,12287,13279,13280,13311,14303,14304,14335,
             15327,15328,15359,16351,16352,16383,17375,18399,19423,20447,
             21471,22495,23519,24543},
            8192, keys_dir, "stage3", true);
        ctx.clear_rot_keys(); ctx.load(keys_dir);

        // Stage 4：boot_slots=4096
        // 由 tools/compute_rotations 生成（142 个）
        ctx.gen_rot_keys(
            {-16384,-8192,-4096,-3168,-3136,-3104,-3072,-2144,-2112,-2080,-2048,
             -1120,-1088,-1056,-1024,-132,-128,-124,-96,-66,-64,-62,-32,-4,-2,
             1,2,4,32,62,64,66,124,128,132,
             959,1023,1024,1983,2047,2048,3007,3071,4031,4095,4096,
             5023,5054,5055,5087,5118,5119,6047,6078,6079,6111,6142,6143,
             7071,7102,7103,7135,7166,7167,8095,8126,8127,8159,8190,8191,
             9149,9213,10173,10237,11197,11261,12221,12285,
             13214,13216,13246,13278,13280,13310,14238,14240,14270,14302,14304,14334,
             15262,15264,15294,15326,15328,15358,16286,16288,16318,16350,16352,16382,
             17311,17375,18335,18399,19359,19423,20383,20447,
             21405,21406,21437,21469,21470,21501,
             22429,22430,22461,22493,22494,22525,
             23453,23454,23485,23517,23518,23549,
             24477,24478,24509,24541,24542,24573,
             25501,25565,26525,26589,27549,27613,28573,28637},
            4096, keys_dir, "stage4", true);
        ctx.clear_rot_keys(); ctx.load(keys_dir);

        // DownSamp1/2 和 AvgPool/Head 不需要自举，boot_slots=0
        // 由 tools/compute_rotations 生成（18 个）
        ctx.gen_rot_keys(
            {-16384,-8192,-2048,
             1023,2016,3039,3072,4095,5088,6111,6144,
             7167,8160,9183,9216,10239,11232,12255},
            0, keys_dir, "downsamp1", true);
        ctx.clear_rot_keys(); ctx.load(keys_dir);

        // 由 tools/compute_rotations 生成（19 个）
        ctx.gen_rot_keys(
            {-16384,-8192,-4096,-1024,
             30,992,1022,1984,2014,2976,3006,3072,3102,
             4064,4094,5056,5086,6048,6078},
            0, keys_dir, "downsamp2", true);
        ctx.clear_rot_keys(); ctx.load(keys_dir);

        ctx.gen_rot_keys(
            {4,8,16,28,56,84,128,256,512,
             1008,1036,1064,1092,2016,2044,2072,2100,
             3024,3052,3080,3108},
            0, keys_dir, "avgpool", true);
        ctx.clear_rot_keys(); ctx.load(keys_dir);

        {
            vector<int> fc_rots;
            for (int d = 1; d < 64; d++) fc_rots.push_back(d);
            ctx.gen_rot_keys(fc_rots, 0, keys_dir, "head", true);
        }
        ctx.clear_rot_keys();

        cout << "密钥生成完成。\n";
        return 0;
    }

    if (mode == "infer") {
        if (argc < 5) { print_usage(argv[0]); return 1; }
        string weights_dir = argv[3];
        string image_path  = argv[4];
        if (argc >= 6) n = atoi(argv[5]);

        ctx.load(keys_dir);

        auto img = load_cifar_image(image_path);
        if (img.empty()) return 1;

        // 输入图像打包加密（Stage 1 参数：ki=1, 32×32×3）
        PackParams pp_in = compute_pack_params(32,32,3, 32,32,16, 1,1, 1, ctx.num_slots);
        Ctxt input = mult_par_pack(ctx, img, pp_in, ctx.circuit_depth - 10);
        ctx.set_slots(input, 16384);  // RS packing：有效数据 16384 槽

        auto t0 = utils::start_time();
        int label = resnet_infer(ctx, input, n, weights_dir, keys_dir);
        utils::print_duration(t0, "ResNet 推理");

        static const char* CLASSES[] = {
            "飞机","汽车","鸟","猫","鹿","狗","青蛙","马","船","卡车"
        };
        cout << "分类结果: " << CLASSES[label] << " (index=" << label << ")\n";
        return 0;
    }

    print_usage(argv[0]);
    return 1;
}
