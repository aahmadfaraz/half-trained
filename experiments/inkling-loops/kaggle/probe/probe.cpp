// ht-probe: run prompts through the model and save, for the last prompt token, every layer slot's output (l_out),
// the 8 routed experts and their weights, plus the top next-token probabilities for a short greedy continuation.
// usage: ht-probe MODEL LOOPS PROMPT_DIR OUT_DIR N_GEN THREADS
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

struct capture {
    std::regex re{R"(^(l_out|ffn_moe_topk|ffn_moe_weights)-(\d+)$)"};
    bool on = false;
    std::map<std::string, std::vector<float>> f32;
    std::map<std::string, std::vector<int32_t>> i32;
};

static bool on_tensor(struct ggml_tensor * t, bool ask, void * ud) {
    auto * c = (capture *) ud;
    if (!c->on || !std::regex_match(t->name, c->re)) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    const int64_t n0 = t->ne[0], n1 = t->ne[1];
    const size_t es = ggml_type_size(t->type);
    const size_t off = (size_t) (n1 - 1) * t->nb[1];
    std::vector<uint8_t> raw(n0 * es);
    if (t->nb[0] == es) {
        ggml_backend_tensor_get(t, raw.data(), off, n0 * es);
    } else {
        for (int64_t i = 0; i < n0; ++i) {
            ggml_backend_tensor_get(t, raw.data() + i * es, off + i * t->nb[0], es);
        }
    }
    if (t->type == GGML_TYPE_F32) {
        std::vector<float> v(n0);
        memcpy(v.data(), raw.data(), n0 * 4);
        c->f32[t->name] = v;
    } else if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> v(n0);
        memcpy(v.data(), raw.data(), n0 * 4);
        c->i32[t->name] = v;
    }
    return true;
}

static std::string piece(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    return n > 0 ? std::string(buf, n) : std::string();
}

static std::string jesc(const std::string & s) {
    std::ostringstream o;
    for (unsigned char ch : s) {
        if (ch == '"' || ch == '\\') o << '\\' << ch;
        else if (ch == '\n') o << "\\n";
        else if (ch < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", ch); o << b; }
        else o << ch;
    }
    return o.str();
}

int main(int argc, char ** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s MODEL LOOPS PROMPT_DIR OUT_DIR N_GEN THREADS\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1], prompt_dir = argv[3], out_dir = argv[4];
    const int loops = atoi(argv[2]), n_gen = atoi(argv[5]), threads = atoi(argv[6]);
    std::filesystem::create_directories(out_dir);

    llama_backend_init();
    std::vector<llama_model_kv_override> kvo(2);
    memset(kvo.data(), 0, sizeof(llama_model_kv_override) * kvo.size());
    kvo[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    strncpy(kvo[0].key, "inkling2_loop.loop.count", sizeof(kvo[0].key) - 1);
    kvo[0].val_i64 = loops;
    auto mp = llama_model_default_params();
    mp.kv_overrides = kvo.data();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    capture cap;
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 4096; cp.n_ubatch = 4096;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.cb_eval = on_tensor; cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 3; }

    std::vector<std::string> files;
    for (auto & e : std::filesystem::directory_iterator(prompt_dir)) files.push_back(e.path().string());
    std::sort(files.begin(), files.end());

    for (const auto & fp : files) {
        std::ifstream in(fp);
        std::stringstream ss; ss << in.rdbuf();
        const std::string text = ss.str();
        const std::string id = std::filesystem::path(fp).stem().string();

        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<llama_token> toks(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, true);
        if (n < 0) { fprintf(stderr, "%s: tokenize failed\n", id.c_str()); continue; }
        toks.resize(n);

        cap.f32.clear(); cap.i32.clear(); cap.on = true;
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), (int) toks.size()))) { fprintf(stderr, "%s: decode failed\n", id.c_str()); continue; }
        cap.on = false;

        std::ofstream bin(out_dir + "/" + id + ".bin", std::ios::binary);
        std::ostringstream js;
        js << "{\"id\":\"" << jesc(id) << "\",\"loops\":" << loops << ",\"n_prompt\":" << n << ",\"tensors\":[";
        size_t offset = 0; bool first = true;
        for (auto & kv : cap.f32) {
            bin.write((const char *) kv.second.data(), kv.second.size() * 4);
            js << (first ? "" : ",") << "{\"name\":\"" << kv.first << "\",\"type\":\"f32\",\"n\":" << kv.second.size() << ",\"offset\":" << offset << "}";
            offset += kv.second.size() * 4; first = false;
        }
        for (auto & kv : cap.i32) {
            bin.write((const char *) kv.second.data(), kv.second.size() * 4);
            js << (first ? "" : ",") << "{\"name\":\"" << kv.first << "\",\"type\":\"i32\",\"n\":" << kv.second.size() << ",\"offset\":" << offset << "}";
            offset += kv.second.size() * 4; first = false;
        }
        js << "],\"steps\":[";

        std::string gen;
        for (int step = 0; step < n_gen; ++step) {
            const float * lg = llama_get_logits_ith(ctx, -1);
            float mx = -INFINITY;
            for (int i = 0; i < n_vocab; ++i) mx = std::max(mx, lg[i]);
            double z = 0;
            for (int i = 0; i < n_vocab; ++i) z += std::exp((double) lg[i] - mx);
            std::vector<int> idx(n_vocab);
            for (int i = 0; i < n_vocab; ++i) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            js << (step ? "," : "") << "[";
            for (int k = 0; k < 20; ++k) {
                const int t = idx[k];
                js << (k ? "," : "") << "[" << t << ",\"" << jesc(piece(vocab, t)) << "\"," << std::exp((double) lg[t] - mx) / z << "]";
            }
            js << "]";
            llama_token next = idx[0];
            if (llama_vocab_is_eog(vocab, next)) break;
            gen += piece(vocab, next);
            if (llama_decode(ctx, llama_batch_get_one(&next, 1))) break;
        }
        js << "],\"generated\":\"" << jesc(gen) << "\"}";
        std::ofstream(out_dir + "/" + id + ".json") << js.str();
        fprintf(stderr, "loops %d %s: %d prompt tokens, %zu tensors, generated \"%s\"\n", loops, id.c_str(), n, cap.f32.size() + cap.i32.size(), gen.c_str());
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
