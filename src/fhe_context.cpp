#include "fhe_context.h"
#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;

static const string CTX_FILE  = "/crypto-context.bin";
static const string PUB_FILE  = "/public-key.bin";
static const string SEC_FILE  = "/secret-key.bin";
static const string MULT_FILE = "/mult-keys.bin";

void FHEContext::generate(const string& keys_dir, bool serialize) {
    CCParams<CryptoContextCKKSRNS> p;
    // 论文 Section 7.1：N=2^16, Hamming=192, 51-bit base/special, 46-bit default
    p.SetSecretKeyDist(SPARSE_TERNARY);
    p.SetSecurityLevel(HEStd_128_classic);
    p.SetRingDim(1 << 16);
    p.SetBatchSize(num_slots);
    p.SetNumLargeDigits(3);
    p.SetScalingModSize(46);
    p.SetFirstModSize(51);
    p.SetScalingTechnique(FLEXIBLEAUTO);

    // ConvBN 消耗 2 层，AppReLU(deg=27) 消耗 5 层，合计 7 层
    uint32_t levels_before = 7;
    uint32_t approx_boot_depth = level_budget_[0] + level_budget_[1];  // 4+4=8
    circuit_depth = levels_before +
        FHECKKSRNS::GetBootstrapDepth(approx_boot_depth, level_budget_, SPARSE_TERNARY);
    p.SetMultiplicativeDepth(circuit_depth);

    cc = GenCryptoContext(p);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);
    kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);

    if (!serialize) return;

    ofstream mf(keys_dir + MULT_FILE, ios::binary);
    cc->SerializeEvalMultKey(mf, SerType::BINARY);
    Serial::SerializeToFile(keys_dir + CTX_FILE,  cc,            SerType::BINARY);
    Serial::SerializeToFile(keys_dir + PUB_FILE,  kp.publicKey,  SerType::BINARY);
    Serial::SerializeToFile(keys_dir + SEC_FILE,  kp.secretKey,  SerType::BINARY);
}

void FHEContext::load(const string& keys_dir) {
    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();
    CryptoContextFactory<DCRTPoly>::ReleaseAllContexts();

    Serial::DeserializeFromFile(keys_dir + CTX_FILE, cc, SerType::BINARY);
    Serial::DeserializeFromFile(keys_dir + PUB_FILE, kp.publicKey, SerType::BINARY);
    Serial::DeserializeFromFile(keys_dir + SEC_FILE, kp.secretKey, SerType::BINARY);

    ifstream mf(keys_dir + MULT_FILE, ios::binary);
    cc->DeserializeEvalMultKey(mf, SerType::BINARY);

    uint32_t levels_before = 7;
    uint32_t approx_boot_depth = level_budget_[0] + level_budget_[1];
    circuit_depth = levels_before +
        FHECKKSRNS::GetBootstrapDepth(approx_boot_depth, level_budget_, SPARSE_TERNARY);
}

void FHEContext::gen_rot_keys(const vector<int>& rots,
                               const string& keys_dir, const string& tag,
                               bool serialize) {
    cc->EvalRotateKeyGen(kp.secretKey, rots);
    if (!serialize) return;
    ofstream rf(keys_dir + "/rot-" + tag + ".bin", ios::binary);
    cc->SerializeEvalAutomorphismKey(rf, SerType::BINARY);
}

void FHEContext::load_rot_keys(const string& keys_dir, const string& tag) {
    ifstream rf(keys_dir + "/rot-" + tag + ".bin", ios::binary);
    if (!rf.is_open()) {
        cerr << "无法打开旋转密钥文件: " << keys_dir + "/rot-" + tag + ".bin" << "\n";
        exit(1);
    }
    if (!cc->DeserializeEvalAutomorphismKey(rf, SerType::BINARY)) {
        cerr << "旋转密钥反序列化失败: " << tag << "\n";
        exit(1);
    }
}

// 为所有需要的槽数生成自举密钥（32768/16384/8192/4096）
void FHEContext::gen_boot_keys(const string& keys_dir, bool serialize) {
    for (int slots : {32768, 16384, 8192, 4096}) {
        cc->EvalBootstrapSetup(level_budget_, {0, 0}, slots);
        cc->EvalBootstrapKeyGen(kp.secretKey, slots);
    }
    if (!serialize) return;
    ofstream bf(keys_dir + "/boot-keys.bin", ios::binary);
    cc->SerializeEvalAutomorphismKey(bf, SerType::BINARY);
}

void FHEContext::load_boot_keys(const string& keys_dir) {
    for (int slots : {32768, 16384, 8192, 4096})
        cc->EvalBootstrapSetup(level_budget_, {0, 0}, slots);
    ifstream bf(keys_dir + "/boot-keys.bin", ios::binary);
    if (!bf.is_open()) {
        cerr << "无法打开自举密钥文件: " << keys_dir + "/boot-keys.bin" << "\n";
        exit(1);
    }
    if (!cc->DeserializeEvalAutomorphismKey(bf, SerType::BINARY)) {
        cerr << "自举密钥反序列化失败\n";
        exit(1);
    }
}

void FHEContext::clear_rot_keys() { cc->ClearEvalAutomorphismKeys(); }

Ctxt FHEContext::rot(const Ctxt& c, int r) const {
    if (r == 0) return c;
    return cc->EvalRotate(c, r);
}
Ctxt FHEContext::add(const Ctxt& a, const Ctxt& b) const { return cc->EvalAdd(a, b); }
Ctxt FHEContext::mul(const Ctxt& c, const Ptxt& p) const { return cc->EvalMult(c, p); }

Ctxt FHEContext::bootstrap(const Ctxt& c) const {
    return cc->EvalBootstrap(c);
}

Ptxt FHEContext::encode(const vector<double>& v, int level, int slots) const {
    if (slots == 0) slots = num_slots;
    auto p = cc->MakeCKKSPackedPlaintext(v, 1, level, nullptr, slots);
    p->SetLength(slots);
    return p;
}

Ctxt FHEContext::encrypt(const vector<double>& v, int level) const {
    return cc->Encrypt(kp.publicKey, encode(v, level));
}

vector<double> FHEContext::decrypt(const Ctxt& c, int slots) const {
    if (slots == 0) slots = num_slots;
    Ptxt p;
    cc->Decrypt(kp.secretKey, c, &p);
    p->SetLength(slots);
    return p->GetRealPackedValue();
}
