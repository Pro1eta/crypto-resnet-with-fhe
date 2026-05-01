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

        // Stage 1 旋转密钥（32×32, ki=1）
        ctx.gen_rot_keys({1,-1,32,-32}, 16384, keys_dir, "stage1", true);
        ctx.clear_rot_keys(); ctx.load(keys_dir);

        // Stage 3 旋转密钥（16×16, ki=2）
        ctx.gen_rot_keys({1,-1,16,-16,2,4,8}, 8192, keys_dir, "stage3", true);
        ctx.clear_rot_keys(); ctx.load(keys_dir);

        // Stage 4 旋转密钥（8×8, ki=4）
        ctx.gen_rot_keys({1,-1,8,-8,2,4}, 4096, keys_dir, "stage4", true);
        ctx.clear_rot_keys(); ctx.load(keys_dir);

        // Head 旋转密钥（FC 对角线方法）
        ctx.gen_rot_keys({1,2,4,8,16,32,64}, 0, keys_dir, "head", true);
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
        PackParams pp_in = compute_pack_params(32,32,3, 32,32,16, 1,1, 1);
        Ctxt input = mult_par_pack(ctx, img, pp_in, ctx.circuit_depth - 4);

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
