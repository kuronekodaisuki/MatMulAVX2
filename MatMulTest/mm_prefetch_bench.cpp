#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ── 設定用 CLI ─────────────────────────────────────────────
// 例: ./a.out --n 4096 --bs 256 --pd 64 --it 3 --mode baseline
// mode: baseline | prefetchA | prefetchB | prefetchAB
struct Args {
    int N = 2048;        // 行列サイズ (N x N)
    int BS = 256;        // ブロッキングサイズ
    int iters = 3;       // 測定繰り返し回数
    int pd = 64;         // プリフェッチ距離（要素数）
    std::string mode = "baseline";
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&](int& dst) { if (i + 1 < argc) dst = std::stoi(argv[++i]); };
        if (s == "--n") next(a.N);
        else if (s == "--bs") next(a.BS);
        else if (s == "--it") next(a.iters);
        else if (s == "--pd") next(a.pd);
        else if (s == "--mode" && i + 1 < argc) a.mode = argv[++i];
    }
    return a;
}

// ── 配列ユーティリティ ───────────────────────────────────
static inline float* aligned_alloc_float(size_t n, size_t align = 64) {
    void* p = nullptr;
#if defined(_MSC_VER)
    p = _aligned_malloc(n * sizeof(float), align);
    if (!p) throw std::bad_alloc();
#else
    if (posix_memalign(&p, align, n * sizeof(float))) throw std::bad_alloc();
#endif
    return (float*)p;
}
static inline void aligned_free(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

// ── 計測ユーティリティ ───────────────────────────────────
struct Timer {
    using clk = std::chrono::high_resolution_clock;
    clk::time_point t0;
    void tic() { t0 = clk::now(); }
    double toc_ms() const {
        auto dt = std::chrono::duration<double, std::milli>(clk::now() - t0);
        return dt.count();
    }
};

// ── プリフェッチヘルパ ───────────────────────────────────
#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH_T0(ptr) __builtin_prefetch((ptr), 0, 3)
#define PREFETCH_T1(ptr) __builtin_prefetch((ptr), 0, 2)
#else
#define PREFETCH_T0(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#define PREFETCH_T1(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T1)
#endif

// ── カーネル: ブロッキング + ループ順序(ikj) ─────────────
// C += A * B  (float, row-major)
void gemm_blocked_baseline(float* C, const float* A, const float* B, int N, int BS) {
    for (int ii = 0; ii < N; ii += BS) {
        for (int kk = 0; kk < N; kk += BS) {
            for (int jj = 0; jj < N; jj += BS) {
                int iimax = std::min(ii + BS, N);
                int kkmax = std::min(kk + BS, N);
                int jjmax = std::min(jj + BS, N);
                for (int i = ii; i < iimax; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        float aik = A[i * (size_t)N + k];
                        float* __restrict c_row = &C[i * (size_t)N + jj];
                        const float* __restrict b_row = &B[k * (size_t)N + jj];
                        for (int j = jj; j < jjmax; ++j) {
                            c_row[j - jj] += aik * b_row[j - jj];
                        }
                    }
                }
            }
        }
    }
}

// A行の先読み（行方向に先に進むパターン）
void gemm_blocked_prefetchA(float* C, const float* A, const float* B, int N, int BS, int pd) {
    for (int ii = 0; ii < N; ii += BS) {
        for (int kk = 0; kk < N; kk += BS) {
            for (int jj = 0; jj < N; jj += BS) {
                int iimax = std::min(ii + BS, N);
                int kkmax = std::min(kk + BS, N);
                int jjmax = std::min(jj + BS, N);
                for (int i = ii; i < iimax; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        // 次の a(i,k+pd) を先読み（範囲内なら）
                        int pk = k + pd;
                        if (pk < kkmax) {
                            PREFETCH_T0(&A[i * (size_t)N + pk]);
                        }
                        float aik = A[i * (size_t)N + k];
                        float* __restrict c_row = &C[i * (size_t)N + jj];
                        const float* __restrict b_row = &B[k * (size_t)N + jj];
                        for (int j = jj; j < jjmax; ++j) {
                            c_row[j - jj] += aik * b_row[j - jj];
                        }
                    }
                }
            }
        }
    }
}

// B行の先読み（列ブロック方向に進むパターン）
void gemm_blocked_prefetchB(float* C, const float* A, const float* B, int N, int BS, int pd) {
    for (int ii = 0; ii < N; ii += BS) {
        for (int kk = 0; kk < N; kk += BS) {
            for (int jj = 0; jj < N; jj += BS) {
                int iimax = std::min(ii + BS, N);
                int kkmax = std::min(kk + BS, N);
                int jjmax = std::min(jj + BS, N);
                for (int i = ii; i < iimax; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        float aik = A[i * (size_t)N + k];
                        float* __restrict c_row = &C[i * (size_t)N + jj];
                        const float* __restrict b_row = &B[k * (size_t)N + jj];
                        // 次の B(k, j+pd) を先読み（範囲内なら）
                        int pj = jj + pd;
                        if (pj < jjmax) {
                            PREFETCH_T1(&b_row[pj - jj]); // T1に置く例
                        }
                        for (int j = jj; j < jjmax; ++j) {
                            c_row[j - jj] += aik * b_row[j - jj];
                        }
                    }
                }
            }
        }
    }
}

// AとBの両方を先読み
void gemm_blocked_prefetchAB(float* C, const float* A, const float* B, int N, int BS, int pd) {
    for (int ii = 0; ii < N; ii += BS) {
        for (int kk = 0; kk < N; kk += BS) {
            for (int jj = 0; jj < N; jj += BS) {
                int iimax = std::min(ii + BS, N);
                int kkmax = std::min(kk + BS, N);
                int jjmax = std::min(jj + BS, N);
                for (int i = ii; i < iimax; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        int pk = k + pd;
                        if (pk < kkmax) {
                            PREFETCH_T0(&A[i * (size_t)N + pk]);
                        }
                        float aik = A[i * (size_t)N + k];
                        float* __restrict c_row = &C[i * (size_t)N + jj];
                        const float* __restrict b_row = &B[k * (size_t)N + jj];
                        int pj = jj + pd;
                        if (pj < jjmax) {
                            PREFETCH_T1(&b_row[pj - jj]);
                        }
                        for (int j = jj; j < jjmax; ++j) {
                            c_row[j - jj] += aik * b_row[j - jj];
                        }
                    }
                }
            }
        }
    }
}

// ── 検算（簡易） ─────────────────────────────────────────
double checksum(const float* x, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += x[i];
    return s;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    const int N = args.N;
    const int BS = args.BS;
    const int iters = args.iters;
    const int pd = std::max(1, args.pd);

    std::cout << "N=" << N << " BS=" << BS << " iters=" << iters << " pd=" << pd << " mode=" << args.mode << "\n";

    const size_t NN = (size_t)N * (size_t)N;
    float* A = aligned_alloc_float(NN);
    float* B = aligned_alloc_float(NN);
    float* C = aligned_alloc_float(NN);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < NN; i++) { A[i] = dist(rng); B[i] = dist(rng); }
    std::fill(C, C + NN, 0.0f);

    // ウォームアップ
    gemm_blocked_baseline(C, A, B, N, std::min(BS, N));
    std::fill(C, C + NN, 0.0f);

    auto run_once = [&](int it)->double {
        Timer t; t.tic();
        if (args.mode == "baseline")           gemm_blocked_baseline(C, A, B, N, BS);
        else if (args.mode == "prefetchA")     gemm_blocked_prefetchA(C, A, B, N, BS, pd);
        else if (args.mode == "prefetchB")     gemm_blocked_prefetchB(C, A, B, N, BS, pd);
        else if (args.mode == "prefetchAB")    gemm_blocked_prefetchAB(C, A, B, N, BS, pd);
        else { std::cerr << "unknown mode\n"; std::exit(1); }
        double ms = t.toc_ms();
        // 1回ごとにチェックサムを印字（最適化での消去防止）
        double cs = checksum(C, NN);
        std::cout << "iter=" << it << "  ms=" << ms << "  checksum=" << cs << "\n";
        // 次回計測に向けてCをクリア（毎回同条件）
        std::fill(C, C + NN, 0.0f);
        return ms;
        };

    double best = 1e100, sum = 0.0;
    for (int it = 0; it < iters; ++it) {
        double ms = run_once(it);
        best = std::min(best, ms);
        sum += ms;
    }
    double avg = sum / iters;
    // 演算回数: 2*N^3 (floatのFMA無し算定)。GFLOP/sを表示。
    double gflops_best = (2.0 * N * (double)N * (double)N) / (best / 1000.0) / 1e9;
    double gflops_avg = (2.0 * N * (double)N * (double)N) / (avg / 1000.0) / 1e9;

    std::cout << "Best: " << best << " ms,  " << gflops_best << " GFLOP/s\n";
    std::cout << "Avg : " << avg << " ms,  " << gflops_avg << " GFLOP/s\n";

    aligned_free(A); aligned_free(B); aligned_free(C);
    return 0;
}