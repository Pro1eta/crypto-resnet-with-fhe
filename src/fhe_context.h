#pragma once
#include "utils.h"

// 论文 Section 7.1 参数：N=2^16, nt=2^15, 128-bit 安全
class FHEContext {
public:
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> kp;
    int circuit_depth = 0;
    int num_slots     = 1 << 15;  // nt = 32768

    // 生成上下文与密钥，serialize=true 时序列化到 keys_dir
    void generate(const string& keys_dir, bool serialize = false);
    void load(const string& keys_dir);

    // 生成/加载旋转密钥（按层分组）
    void gen_rot_keys(const vector<int>& rots, int boot_slots,
                      const string& keys_dir, const string& tag, bool serialize = false);
    void load_rot_keys(const string& keys_dir, const string& tag,
                       int boot_slots);
    void clear_rot_keys();

    // 基本同态操作封装
    Ctxt rot(const Ctxt& c, int r) const;
    Ctxt add(const Ctxt& a, const Ctxt& b) const;
    Ctxt mul(const Ctxt& c, const Ptxt& p) const;
    Ctxt bootstrap(const Ctxt& c) const;  // imaginary-removing bootstrapping

    Ptxt encode(const vector<double>& v, int level, int slots = 0) const;
    Ctxt  encrypt(const vector<double>& v, int level) const;
    vector<double> decrypt(const Ctxt& c, int slots = 0) const;

private:
    vector<uint32_t> level_budget_ = {4, 4};
};
