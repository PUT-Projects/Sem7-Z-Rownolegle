// =====================================================================================
//  Projekt PKG (CUDA) – suma okna (2R+1)x(2R+1)
//
//  TAB: N x N (float), OUT: M x M, M = N - 2R
//  OUT[i][j] = sum_{u=-R..R} sum_{v=-R..R} TAB[i+R+u][j+R+v]
//
//  Uzycie:
//    stencil -N=<n> -R=<r> [-BS=16|32] [-mode=cpu|gpu1|gpu2|all] [-reps=<k>]
//            [-data=random|uniform] [-threads=<t>] [-dev=<d>] [-seed=<s>]
//
//    -mode   cpu  – tylko referencja CPU (OpenMP)
//            gpu1 – wariant 1 (efektywny),  gpu2 – wariant 2 (nieefektywny)  [TODO A4/A5]
//            all  – CPU + oba warianty GPU, z porownaniem wynikow
//    -reps   liczba mierzonych powtorzen (po 1 rozgrzewce); czas reprezentatywny = mediana
//    -data   random  – dane niejednorodne, U[0,1), staly seed (domyslnie)
//            uniform – same 1.0f -> kazdy element OUT musi byc dokladnie (2R+1)^2
//
//  Na stdout: opis czytelny dla czlowieka + linie "CSV,..." do zbierania wynikow skryptem.
// =====================================================================================

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "helper_cuda.h"  // checkCudaErrors, _ConvertSMVer2Cores

// -------------------------------------------------------------------------------------
//  Parametry uruchomienia
// -------------------------------------------------------------------------------------
struct Params {
    int N = 1024;
    int R = 4;
    int BS = 32;
    int reps = 10;
    int threads = 0;  // 0 = omp_get_max_threads()
    int dev = 0;
    unsigned seed = 12345u;
    std::string mode = "cpu";
    std::string data = "random";

    int M() const { return N - 2 * R; }  // wymiar tablicy wynikowej
};

static void usage(const char *prog) {
    printf(
        "Uzycie: %s -N=<n> -R=<r> [-BS=16|32] [-mode=cpu|gpu1|gpu2|all] [-reps=<k>]\n"
        "          [-data=random|uniform] [-threads=<t>] [-dev=<d>] [-seed=<s>]\n",
        prog);
}

// Argumenty w postaci -nazwa=wartosc, nazwa dopasowywana dokladnie (z rozroznieniem
// wielkosci liter). Uwaga: helper_string.h porownuje tylko prefiks bez rozrozniania
// wielkosci liter, przez co "-reps" bylby brany za "-R" – dlatego wlasny parser.
static bool parseArgs(int argc, char **argv, Params &p) {
    bool ok = true;
    for (int a = 1; a < argc; ++a) {
        std::string arg = argv[a];
        if (arg == "-h" || arg == "--help" || arg == "-help") return false;
        const size_t eq = arg.find('=');
        if (arg.size() < 2 || arg[0] != '-' || eq == std::string::npos) {
            fprintf(stderr, "Blad: niepoprawny argument '%s' (oczekiwano -nazwa=wartosc)\n", argv[a]);
            ok = false;
            continue;
        }
        const std::string key = arg.substr(arg[1] == '-' ? 2 : 1, eq - (arg[1] == '-' ? 2 : 1));
        const std::string val = arg.substr(eq + 1);
        const int iv = atoi(val.c_str());
        if (key == "N") p.N = iv;
        else if (key == "R") p.R = iv;
        else if (key == "BS") p.BS = iv;
        else if (key == "reps") p.reps = iv;
        else if (key == "threads") p.threads = iv;
        else if (key == "dev") p.dev = iv;
        else if (key == "seed") p.seed = (unsigned)strtoul(val.c_str(), nullptr, 10);
        else if (key == "mode") p.mode = val;
        else if (key == "data") p.data = val;
        else { fprintf(stderr, "Blad: nieznany parametr '%s'\n", key.c_str()); ok = false; }
    }

    if (p.R < 0) { fprintf(stderr, "Blad: R musi byc >= 0\n"); ok = false; }
    if (p.N <= 2 * p.R) { fprintf(stderr, "Blad: wymagane N > 2R (N=%d, R=%d)\n", p.N, p.R); ok = false; }
    if (p.BS != 16 && p.BS != 32) { fprintf(stderr, "Blad: BS musi byc 16 lub 32\n"); ok = false; }
    if (p.reps < 1) { fprintf(stderr, "Blad: reps >= 1\n"); ok = false; }
    if (p.mode != "cpu" && p.mode != "gpu1" && p.mode != "gpu2" && p.mode != "all") {
        fprintf(stderr, "Blad: nieznany mode '%s'\n", p.mode.c_str()); ok = false;
    }
    if (p.data != "random" && p.data != "uniform") {
        fprintf(stderr, "Blad: nieznany data '%s'\n", p.data.c_str()); ok = false;
    }
    return ok;
}

// -------------------------------------------------------------------------------------
//  Miary
// -------------------------------------------------------------------------------------
// Liczba operacji zmiennoprzecinkowych: kazdy z M*M elementow wyniku to suma (2R+1)^2
// skladnikow, czyli (2R+1)^2 - 1 dodawan. Adresowanie i sterowanie nie sa liczone.
static double flopCount(const Params &p) {
    const double M = (double)p.M();
    const double K = (double)(2 * p.R + 1);
    return M * M * (K * K - 1.0);
}

struct TimingStats {
    double med_ms = 0, min_ms = 0, mean_ms = 0;
};

static TimingStats stats(std::vector<double> t_ms) {
    TimingStats s;
    std::sort(t_ms.begin(), t_ms.end());
    const size_t n = t_ms.size();
    s.min_ms = t_ms.front();
    s.med_ms = (n % 2) ? t_ms[n / 2] : 0.5 * (t_ms[n / 2 - 1] + t_ms[n / 2]);
    double sum = 0;
    for (double x : t_ms) sum += x;
    s.mean_ms = sum / n;
    return s;
}

// -------------------------------------------------------------------------------------
//  Dane wejsciowe
// -------------------------------------------------------------------------------------
// Deterministyczny generator (xorshift32) – te same dane na Windows i Linux,
// niezaleznie od implementacji rand().
static void initInput(std::vector<float> &tab, const Params &p) {
    if (p.data == "uniform") {
        std::fill(tab.begin(), tab.end(), 1.0f);
        return;
    }
    uint32_t x = p.seed ? p.seed : 1u;
    for (size_t k = 0; k < tab.size(); ++k) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        tab[k] = (float)(x >> 8) * (1.0f / 16777216.0f);  // U[0,1), 24 bity mantysy
    }
}

// -------------------------------------------------------------------------------------
//  Referencja CPU (OpenMP)
// -------------------------------------------------------------------------------------
// Watki OpenMP dziela miedzy siebie wiersze tablicy wynikowej (schedule static –
// rowna praca na wiersz). Wiersz i jest liczony fragmentami po JB elementow:
//   acc[k] += tab[i+u][jb+k+v]   dla k = 0..JB-1, u,v = 0..2R
// acc[] to zmienna prywatna watku (miesci sie w rejestrach wektorowych SIMD), wiec
// w petli wewnetrznej sa tylko odczyty kolejnych adresow (lokalnosc przestrzennna,
// wektoryzacja), a wiersze tab[i..i+2R] uzywane wielokrotnie pozostaja w cache
// (lokalnosc czasowa). Kolejnosc sumowania pojedynczego elementu jest taka sama
// jak w definicji: u = -R..R (zewnetrzna), v = -R..R (wewnetrzna).
static constexpr int JB = 64;

static void stencilCPU(const float *__restrict tab, float *__restrict out, int N, int R) {
    const int M = N - 2 * R;
    const int K = 2 * R + 1;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        float *o = out + (size_t)i * M;
        for (int jb = 0; jb < M; jb += JB) {
            float acc[JB];
            for (int k = 0; k < JB; ++k) acc[k] = 0.0f;
            if (jb + JB <= M) {  // pelny fragment – stala dlugosc petli, latwa wektoryzacja
                for (int u = 0; u < K; ++u) {
                    const float *row = tab + (size_t)(i + u) * N + jb;
                    for (int v = 0; v < K; ++v)
                        for (int k = 0; k < JB; ++k) acc[k] += row[v + k];
                }
                for (int k = 0; k < JB; ++k) o[jb + k] = acc[k];
            } else {  // koncowka wiersza (M nie jest wielokrotnoscia JB)
                const int n = M - jb;
                for (int u = 0; u < K; ++u) {
                    const float *row = tab + (size_t)(i + u) * N + jb;
                    for (int v = 0; v < K; ++v)
                        for (int k = 0; k < n; ++k) acc[k] += row[v + k];
                }
                for (int k = 0; k < n; ++k) o[jb + k] = acc[k];
            }
        }
    }
}

static TimingStats runCPU(const std::vector<float> &tab, std::vector<float> &out, const Params &p) {
    stencilCPU(tab.data(), out.data(), p.N, p.R);  // rozgrzewka (page faults, watki OpenMP)
    std::vector<double> t(p.reps);
    for (int r = 0; r < p.reps; ++r) {
        const double t0 = omp_get_wtime();
        stencilCPU(tab.data(), out.data(), p.N, p.R);
        t[r] = (omp_get_wtime() - t0) * 1e3;
    }
    return stats(t);
}

// -------------------------------------------------------------------------------------
//  Weryfikacja
// -------------------------------------------------------------------------------------
struct CheckResult {
    double max_abs = 0, max_rel = 0;
    size_t bad = 0, first_bad = (size_t)-1;
    bool ok() const { return bad == 0; }
};

// Tolerancja: suma K^2 wartosci float w roznej kolejnosci rozni sie co najwyzej o ok.
// K^2 * eps * sum|x| (oszacowanie bledu sumowania). Dane >= 0, wiec sum|x| = |wynik|.
static CheckResult compare(const float *ref, const float *got, size_t n, int R) {
    const double K2 = (double)(2 * R + 1) * (2 * R + 1);
    const double tol = K2 * (double)FLT_EPSILON;  // wzgledna
    CheckResult c;
    for (size_t k = 0; k < n; ++k) {
        const double a = ref[k], b = got[k];
        const double ae = std::fabs(a - b);
        const double re = ae / std::max(std::fabs(a), 1.0);
        c.max_abs = std::max(c.max_abs, ae);
        c.max_rel = std::max(c.max_rel, re);
        if (!(re <= tol)) {  // !(<=) lapie tez NaN
            if (c.first_bad == (size_t)-1) c.first_bad = k;
            ++c.bad;
        }
    }
    return c;
}

// Dla danych jednorodnych (1.0f) kazdy wynik musi byc dokladnie (2R+1)^2
// (liczby calkowite <= 2^24 sa reprezentowane w float dokladnie).
static CheckResult checkUniform(const float *got, size_t n, int R) {
    std::vector<float> ref(n, (float)((2 * R + 1) * (2 * R + 1)));
    CheckResult c = compare(ref.data(), got, n, R);
    if (c.max_abs != 0.0 && c.bad == 0) c.bad = 1;  // wymagamy wyniku dokladnego
    return c;
}

// -------------------------------------------------------------------------------------
//  GPU
// -------------------------------------------------------------------------------------
static void printDeviceInfo(int dev) {
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, dev));
    printf("GPU %d: %s, CC %d.%d, %d SM x %d rdzeni, %.0f MB global, shared/blok %zu B, "
           "shared/SM %zu B, max watkow/SM %d\n",
           dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           _ConvertSMVer2Cores(prop.major, prop.minor), prop.totalGlobalMem / 1048576.0,
           prop.sharedMemPerBlock, prop.sharedMemPerMultiprocessor, prop.maxThreadsPerMultiProcessor);
}

// TODO A3-A5: kernel0, wariant 1, wariant 2 + pomiar cudaEvent.
static bool runGPU(int variant, const std::vector<float> &tab, std::vector<float> &out,
                   const Params &p, TimingStats &ts) {
    (void)tab; (void)out; (void)p; (void)ts;
    printf("[gpu%d] jeszcze nie zaimplementowany\n", variant);
    return false;
}

// -------------------------------------------------------------------------------------
//  Raportowanie
// -------------------------------------------------------------------------------------
static void report(const char *device, const char *variant, const Params &p, int threads,
                   const TimingStats &ts, const CheckResult *chk) {
    const double gflops = flopCount(p) / (ts.med_ms * 1e-3) / 1e9;
    printf("[%s/%s] N=%d R=%d BS=%d: t_med=%.4f ms (min %.4f, avg %.4f), %.2f GFLOP/s",
           device, variant, p.N, p.R, p.BS, ts.med_ms, ts.min_ms, ts.mean_ms, gflops);
    if (chk) printf(", max_rel_err=%.3e -> %s", chk->max_rel, chk->ok() ? "OK" : "BLAD");
    printf("\n");
    // CSV,device,variant,BS,N,R,threads,reps,data,t_med_ms,t_min_ms,t_mean_ms,gflops,max_abs_err,max_rel_err,ok
    printf("CSV,%s,%s,%d,%d,%d,%d,%d,%s,%.6f,%.6f,%.6f,%.4f,%.3e,%.3e,%d\n", device, variant, p.BS,
           p.N, p.R, threads, p.reps, p.data.c_str(), ts.med_ms, ts.min_ms, ts.mean_ms, gflops,
           chk ? chk->max_abs : 0.0, chk ? chk->max_rel : 0.0, chk ? (int)chk->ok() : -1);
}

// -------------------------------------------------------------------------------------
int main(int argc, char **argv) {
    Params p;
    if (!parseArgs(argc, argv, p)) {
        usage(argv[0]);
        return 1;
    }
    if (p.threads > 0) omp_set_num_threads(p.threads);
    const int threads = omp_get_max_threads();
    const int M = p.M();

    printf("N=%d R=%d M=N-2R=%d BS=%d mode=%s data=%s reps=%d watki_CPU=%d FLOP=%.4e\n", p.N, p.R,
           M, p.BS, p.mode.c_str(), p.data.c_str(), p.reps, threads, flopCount(p));

    std::vector<float> tab((size_t)p.N * p.N);
    std::vector<float> outCPU((size_t)M * M);
    initInput(tab, p);

    // --- CPU (referencja) ---
    const bool needCPU = (p.mode == "cpu" || p.mode == "all");
    bool allOk = true;
    if (needCPU) {
        TimingStats ts = runCPU(tab, outCPU, p);
        CheckResult chk;
        const CheckResult *pchk = nullptr;
        if (p.data == "uniform") {
            chk = checkUniform(outCPU.data(), outCPU.size(), p.R);
            pchk = &chk;
            allOk &= chk.ok();
        }
        report("cpu", "omp", p, threads, ts, pchk);
    }

    // --- GPU ---
    if (p.mode != "cpu") {
        checkCudaErrors(cudaSetDevice(p.dev));
        printDeviceInfo(p.dev);
        std::vector<float> outGPU((size_t)M * M);
        for (int variant = 1; variant <= 2; ++variant) {
            if (p.mode != "all" && p.mode != (variant == 1 ? "gpu1" : "gpu2")) continue;
            TimingStats ts;
            if (!runGPU(variant, tab, outGPU, p, ts)) { allOk = false; continue; }
            CheckResult chk = needCPU ? compare(outCPU.data(), outGPU.data(), outGPU.size(), p.R)
                                      : (p.data == "uniform" ? checkUniform(outGPU.data(), outGPU.size(), p.R)
                                                             : CheckResult{});
            allOk &= chk.ok();
            report("gpu", variant == 1 ? "v1" : "v2", p, 0, ts,
                   (needCPU || p.data == "uniform") ? &chk : nullptr);
        }
    }

    return allOk ? 0 : 2;
}
