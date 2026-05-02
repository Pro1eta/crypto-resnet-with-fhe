#pragma once
#include "utils.h"

class FHEContext {
public:
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> kp;
    int circuit_depth = 0;
    int num_slots     = 1 << 15;

    void generate(const string& keys_dir, bool serialize = false);
    void load(const string& keys_dir);

    // boot_slots>0 时同时生成/加载自举密钥
    void gen_rot_keys(const vector<int>& rots, int boot_slots,
                      const string& keys_dir, const string& tag, bool serialize = false);
    void load_rot_keys(const string& keys_dir, const string& tag, int boot_slots);
    void clear_rot_keys();

    Ctxt rot(const Ctxt& c, int r) const;
    Ctxt add(const Ctxt& a, const Ctxt& b) const;
    Ctxt mul(const Ctxt& c, const Ptxt& p) const;
    Ctxt bootstrap(const Ctxt& c) const;
    void set_slots(Ctxt& c, int slots) const {
        c->SetSlots(slots);
        cerr << "[SET_SLOTS] " << slots << "\n";
    }

    Ptxt encode(const vector<double>& v, int level, int slots = 0) const;
    Ctxt  encrypt(const vector<double>& v, int level) const;
    vector<double> decrypt(const Ctxt& c, int slots = 0) const;

private:
    vector<uint32_t> level_budget_ = {4, 4};
};
