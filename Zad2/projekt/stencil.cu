// =====================================================================================
//  Projekt PKG (CUDA) – suma okna (2R+1)x(2R+1)
//
//  TAB: N x N (float), OUT: M x M, M = N - 2R
//  OUT[i][j] = sum_{u=-R..R} sum_{v=-R..R} TAB[i+R+u][j+R+v]
//
//  Uzycie:
//    stencil -N=<n> -R=<r> [-BS=16|32] [-mode=<lista>] [-reps=<k>]
//            [-data=random|uniform] [-threads=<t>] [-dev=<d>] [-seed=<s>]
//
//    -mode   lista oddzielona przecinkami (domyslnie cpu,gpu1,gpu2):
//            cpu  – referencja CPU (OpenMP)
//            gpu0 – kernel kontrolny: tylko pamiec globalna, bez pamieci wspoldzielonej
//            gpu1 – wariant 1: pamiec globalna laczona, shared bez konfliktow bankow
//            gpu2 – wariant 2: pamiec globalna nielaczona, shared z konfliktami bankow
//            all  = cpu,gpu0,gpu1,gpu2
//            Gdy jest cpu, wyniki GPU sa porownywane z CPU element po elemencie.
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
    bool cpu = false, gpu[3] = {false, false, false};  // gpu[0]=kernel0, gpu[1]=wariant1, gpu[2]=wariant2
    std::string data = "random";

    int M() const { return N - 2 * R; }  // wymiar tablicy wynikowej
    bool anyGpu() const { return gpu[0] || gpu[1] || gpu[2]; }
};

static void usage(const char *prog) {
    printf(
        "Uzycie: %s -N=<n> -R=<r> [-BS=16|32] [-mode=cpu,gpu0,gpu1,gpu2|all] [-reps=<k>]\n"
        "          [-data=random|uniform] [-threads=<t>] [-dev=<d>] [-seed=<s>]\n",
        prog);
}

static bool parseMode(const std::string &list, Params &p) {
    p.cpu = false;
    p.gpu[0] = p.gpu[1] = p.gpu[2] = false;
    size_t start = 0;
    while (start <= list.size()) {
        size_t end = list.find(',', start);
        if (end == std::string::npos) end = list.size();
        const std::string m = list.substr(start, end - start);
        if (m == "cpu") p.cpu = true;
        else if (m == "gpu0") p.gpu[0] = true;
        else if (m == "gpu1") p.gpu[1] = true;
        else if (m == "gpu2") p.gpu[2] = true;
        else if (m == "all") p.cpu = p.gpu[0] = p.gpu[1] = p.gpu[2] = true;
        else { fprintf(stderr, "Blad: nieznany tryb '%s'\n", m.c_str()); return false; }
        start = end + 1;
    }
    return true;
}

// Argumenty w postaci -nazwa=wartosc, nazwa dopasowywana dokladnie (z rozroznieniem
// wielkosci liter). Uwaga: helper_string.h porownuje tylko prefiks bez rozrozniania
// wielkosci liter, przez co "-reps" bylby brany za "-R" – dlatego wlasny parser.
static bool parseArgs(int argc, char **argv, Params &p) {
    bool ok = parseMode("cpu,gpu1,gpu2", p);
    for (int a = 1; a < argc; ++a) {
        std::string arg = argv[a];
        if (arg == "-h" || arg == "--help" || arg == "-help") return false;
        const size_t eq = arg.find('=');
        if (arg.size() < 2 || arg[0] != '-' || eq == std::string::npos) {
            fprintf(stderr, "Blad: niepoprawny argument '%s' (oczekiwano -nazwa=wartosc)\n", argv[a]);
            ok = false;
            continue;
        }
        const size_t k0 = (arg[1] == '-') ? 2 : 1;
        const std::string key = arg.substr(k0, eq - k0);
        const std::string val = arg.substr(eq + 1);
        const int iv = atoi(val.c_str());
        if (key == "N") p.N = iv;
        else if (key == "R") p.R = iv;
        else if (key == "BS") p.BS = iv;
        else if (key == "reps") p.reps = iv;
        else if (key == "threads") p.threads = iv;
        else if (key == "dev") p.dev = iv;
        else if (key == "seed") p.seed = (unsigned)strtoul(val.c_str(), nullptr, 10);
        else if (key == "mode") ok &= parseMode(val, p);
        else if (key == "data") p.data = val;
        else { fprintf(stderr, "Blad: nieznany parametr '%s'\n", key.c_str()); ok = false; }
    }

    if (p.R < 0) { fprintf(stderr, "Blad: R musi byc >= 0\n"); ok = false; }
    if (p.N <= 2 * p.R) { fprintf(stderr, "Blad: wymagane N > 2R (N=%d, R=%d)\n", p.N, p.R); ok = false; }
    if (p.BS != 16 && p.BS != 32) { fprintf(stderr, "Blad: BS musi byc 16 lub 32\n"); ok = false; }
    if (p.reps < 1) { fprintf(stderr, "Blad: reps >= 1\n"); ok = false; }
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
// w petli wewnetrznej sa tylko odczyty kolejnych adresow (lokalnosc przestrzenna,
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
        if (!(re <= tol)) {  // !(<=) lapie tez NaN (np. element, ktorego kernel nie zapisal)
            if (c.first_bad == (size_t)-1) c.first_bad = k;
            ++c.bad;
            c.max_abs = std::isnan(ae) ? INFINITY : std::max(c.max_abs, ae);
            c.max_rel = std::isnan(re) ? INFINITY : std::max(c.max_rel, re);
        } else {
            c.max_abs = std::max(c.max_abs, ae);
            c.max_rel = std::max(c.max_rel, re);
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

// =====================================================================================
//  GPU
// =====================================================================================
//
//  Podzial pracy (wszystkie kernele):
//    blok (bx,by) o wymiarach BS x BS liczy fragment OUT[i0..i0+BS-1][j0..j0+BS-1],
//    i0 = by*BS, j0 = bx*BS; watek (tx,ty) liczy OUT[i0+ty][j0+tx].
//    Siatka: ceil(M/BS) x ceil(M/BS) blokow; watki poza M x M nic nie zapisuja.
//
//  Warianty z pamiecia wspoldzielona: blok potrzebuje fragmentu TAB o wymiarach
//    T x T, T = BS + 2R (wiersze i0..i0+T-1, kolumny j0..j0+T-1), ktory wszystkie
//    BS*BS watkow laduje kolektywnie do pamieci wspoldzielonej (petla o kroku BS*BS),
//    a nastepnie kazdy watek sumuje (2R+1)^2 elementow z pamieci wspoldzielonej.
//    Pamiec wspoldzielona to 32 banki 4-bajtowe: slowo o indeksie a lezy w banku a % 32.
//
//  Limit pamieci wspoldzielonej: 48 KB/blok (GTX 1080 nie pozwala przekroczyc;
//  ten sam limit przyjmujemy na RTX 4060, zeby kod byl identyczny).
static constexpr int SMEM_LIMIT = 48 * 1024;

// Najmniejsze P >= T takie, ze P % 32 == want (want w 0..31).
__host__ __device__ constexpr int padTo(int T, int want) { return T + ((want - T % 32) + 32) % 32; }

// Wariant 1: tile[r*P + c] (wierszami). Przy obliczeniach warp czyta adresy
//   (ty+u)*P + tx+v.
//   BS=32: warp = jeden wiersz bloku (32 kolejne tx) -> 32 kolejne slowa -> 32 rozne
//          banki, bez konfliktow dla dowolnego P, wiec P = T (bez dopelnienia).
//   BS=16: warp = dwa polwiersze (ty i ty+1, po 16 watkow); polwarpy sa przesuniete
//          o P slow. Banki sa rozlaczne tylko gdy P % 32 == 16 -> dopelnienie wiersza.
__host__ __device__ constexpr int pitchV1(int BS, int T) { return BS == 32 ? T : padTo(T, 16); }

// Wariant 2: tileT[c*P + r] (transpozycja – kolumna TAB to wiersz w shared). Przy
//   obliczeniach warp czyta adresy (tx+v)*P + ty+u: kolejne tx -> krok P slow ->
//   bank ((tx+v)*P + ty+u) % 32. Stopien konfliktu = gcd(P, 32):
//   P % 32 == 0 -> wszystkie 32 watki (BS=32) / 16 watkow polwarpa (BS=16) w tym samym banku.
//   Jesli kafel z P = wielokrotnosc 32 nie miesci sie w 48 KB (BS=32, R>32),
//   bierzemy P % 32 == 16 -> 16-krotny konflikt (dwa banki na warp).
__host__ __device__ constexpr int pitchV2(int T) {
    return (T * padTo(T, 0) * 4 <= SMEM_LIMIT) ? padTo(T, 0) : padTo(T, 16);
}

__host__ __device__ constexpr int gcd32(int P) {
    int g = 32;
    while (P % g) g /= 2;
    return g;
}

// -------------------------------------------------------------------------------------
//  Kernel 0 – kontrolny: bez pamieci wspoldzielonej, kazdy watek czyta (2R+1)^2
//  elementow bezposrednio z pamieci globalnej (warp czyta kolejne adresy -> laczone,
//  ale kazdy element TAB jest pobierany wielokrotnie; pomaga tylko cache L1/L2).
// -------------------------------------------------------------------------------------
template <int BS>
__global__ void stencilKernel0(const float *__restrict__ in, float *__restrict__ out, int N, int R) {
    const int M = N - 2 * R;
    const int K = 2 * R + 1;
    const int i = blockIdx.y * BS + threadIdx.y;
    const int j = blockIdx.x * BS + threadIdx.x;
    if (i >= M || j >= M) return;
    float s = 0.0f;
    for (int u = 0; u < K; ++u) {
        const float *row = in + (size_t)(i + u) * N + j;
        for (int v = 0; v < K; ++v) s += row[v];
    }
    out[(size_t)i * M + j] = s;
}

// -------------------------------------------------------------------------------------
//  Wariant 1 – efektywny
//    ladowanie: liniowy indeks watku t = ty*BS + tx przebiega kafel wierszami
//               (r = idx / T, c = idx % T) -> kolejne watki warpa czytaja kolejne
//               adresy TAB (laczenie dostepow, 1 wiersz kafla = ciagly obszar),
//    obliczenia: watek (tx,ty) czyta tile[(ty+u)*P + tx+v] -> kolejne watki warpa,
//               kolejne slowa -> rozne banki (P dobrane jak w pitchV1).
// -------------------------------------------------------------------------------------
template <int BS>
__global__ void stencilKernelV1(const float *__restrict__ in, float *__restrict__ out, int N, int R) {
    extern __shared__ float tile[];
    const int T = BS + 2 * R;
    const int P = pitchV1(BS, T);
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i0 = blockIdx.y * BS, j0 = blockIdx.x * BS;

    for (int idx = ty * BS + tx; idx < T * T; idx += BS * BS) {
        const int r = idx / T, c = idx - r * T;
        const int gr = i0 + r, gc = j0 + c;
        tile[r * P + c] = (gr < N && gc < N) ? in[(size_t)gr * N + gc] : 0.0f;
    }
    __syncthreads();

    const int M = N - 2 * R;
    const int K = 2 * R + 1;
    const int i = i0 + ty, j = j0 + tx;
    if (i < M && j < M) {
        const float *base = tile + ty * P + tx;
        float s = 0.0f;
        for (int u = 0; u < K; ++u)
            for (int v = 0; v < K; ++v) s += base[u * P + v];
        out[(size_t)i * M + j] = s;
    }
}

// -------------------------------------------------------------------------------------
//  Wariant 2 – nieefektywny
//    ladowanie: ten sam kafel, ale indeks liniowy przebiega go kolumnami
//               (r = idx % T, c = idx / T) -> kolejne watki warpa czytaja kolejne
//               WIERSZE TAB (krok N*4 B) -> kazdy watek w innym segmencie pamieci,
//               brak laczenia (do 32 transakcji na instrukcje warpa).
//               Zapis do tileT[c*P + r] jest ciagly (bez konfliktow).
//    obliczenia: watek (tx,ty) czyta tileT[(tx+v)*P + ty+u] -> kolejne watki warpa
//               co P slow -> ten sam bank (konflikt gcd(P,32)-krotny, patrz pitchV2).
//    Kolejnosc sumowania (u zewn., v wewn.) taka sama jak w wariancie 1 i na CPU.
// -------------------------------------------------------------------------------------
template <int BS>
__global__ void stencilKernelV2(const float *__restrict__ in, float *__restrict__ out, int N, int R) {
    extern __shared__ float tileT[];
    const int T = BS + 2 * R;
    const int P = pitchV2(T);
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i0 = blockIdx.y * BS, j0 = blockIdx.x * BS;

    for (int idx = ty * BS + tx; idx < T * T; idx += BS * BS) {
        const int c = idx / T, r = idx - c * T;
        const int gr = i0 + r, gc = j0 + c;
        tileT[c * P + r] = (gr < N && gc < N) ? in[(size_t)gr * N + gc] : 0.0f;
    }
    __syncthreads();

    const int M = N - 2 * R;
    const int K = 2 * R + 1;
    const int i = i0 + ty, j = j0 + tx;
    if (i < M && j < M) {
        const float *base = tileT + tx * P + ty;
        float s = 0.0f;
        for (int u = 0; u < K; ++u)
            for (int v = 0; v < K; ++v) s += base[v * P + u];
        out[(size_t)i * M + j] = s;
    }
}

// -------------------------------------------------------------------------------------
//  Host: konfiguracja i pomiar
// -------------------------------------------------------------------------------------
typedef void (*KernelFn)(const float *, float *, int, int);

struct LaunchCfg {
    KernelFn fn;
    dim3 grid, block;
    size_t smem;  // bajty pamieci wspoldzielonej (dynamicznej)
    int pitch;    // P (slowa) – 0 dla kernela 0
    int conflict; // przewidywany stopien konfliktu bankow przy obliczeniach
};

template <int BS>
static LaunchCfg makeCfgBS(int variant, int N, int R) {
    LaunchCfg c;
    const int M = N - 2 * R;
    const int T = BS + 2 * R;
    c.block = dim3(BS, BS);
    c.grid = dim3((M + BS - 1) / BS, (M + BS - 1) / BS);  // pokrycie calego M x M
    if (variant == 0) {
        c.fn = stencilKernel0<BS>; c.pitch = 0; c.smem = 0; c.conflict = 0;
    } else if (variant == 1) {
        c.fn = stencilKernelV1<BS>; c.pitch = pitchV1(BS, T); c.conflict = 1;
        c.smem = (size_t)T * c.pitch * sizeof(float);
    } else {
        c.fn = stencilKernelV2<BS>; c.pitch = pitchV2(T);
        c.conflict = std::min(gcd32(c.pitch), BS == 16 ? 16 : 32);
        c.smem = (size_t)T * c.pitch * sizeof(float);
    }
    return c;
}

static LaunchCfg makeCfg(int variant, const Params &p) {
    return p.BS == 16 ? makeCfgBS<16>(variant, p.N, p.R) : makeCfgBS<32>(variant, p.N, p.R);
}

static void printDeviceInfo(int dev) {
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, dev));
    printf("GPU %d: %s, CC %d.%d, %d SM x %d rdzeni, %.0f MB global, shared/blok %zu B, "
           "shared/SM %zu B, max watkow/SM %d\n",
           dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           _ConvertSMVer2Cores(prop.major, prop.minor), prop.totalGlobalMem / 1048576.0,
           prop.sharedMemPerBlock, prop.sharedMemPerMultiprocessor, prop.maxThreadsPerMultiProcessor);
}

// Pomiar: zdarzenia CUDA (cudaEvent) zapisane w strumieniu przed i po kernelu – mierza
// czas wykonania na GPU niezaleznie od asynchronicznego uruchamiania. Transfery
// CPU<->GPU sa poza pomiarem. 1 uruchomienie rozgrzewkowe + reps mierzonych.
static bool runGPU(int variant, const float *d_in, float *d_out, std::vector<float> &out,
                   const Params &p, TimingStats &ts, LaunchCfg &cfg) {
    cfg = makeCfg(variant, p);
    if (cfg.smem > SMEM_LIMIT) {
        printf("[gpu%d] pominiety: kafel wymaga %zu B pamieci wspoldzielonej (> %d B), BS=%d R=%d\n",
               variant, cfg.smem, SMEM_LIMIT, p.BS, p.R);
        return false;
    }
    const size_t outBytes = out.size() * sizeof(float);
    // 0xFF -> NaN: element niezapisany przez kernel zostanie wykryty przy porownaniu.
    checkCudaErrors(cudaMemset(d_out, 0xFF, outBytes));

    cfg.fn<<<cfg.grid, cfg.block, cfg.smem>>>(d_in, d_out, p.N, p.R);  // rozgrzewka
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    std::vector<double> t(p.reps);
    for (int r = 0; r < p.reps; ++r) {
        checkCudaErrors(cudaEventRecord(start));
        cfg.fn<<<cfg.grid, cfg.block, cfg.smem>>>(d_in, d_out, p.N, p.R);
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        checkCudaErrors(cudaGetLastError());
        float ms = 0;
        checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
        t[r] = ms;
    }
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaMemcpy(out.data(), d_out, outBytes, cudaMemcpyDeviceToHost));
    ts = stats(t);
    return true;
}

// -------------------------------------------------------------------------------------
//  Raportowanie
// -------------------------------------------------------------------------------------
static void report(const char *device, const char *variant, const Params &p, int threads,
                   const TimingStats &ts, const CheckResult *chk, const LaunchCfg *cfg) {
    const double gflops = flopCount(p) / (ts.med_ms * 1e-3) / 1e9;
    printf("[%s/%s] N=%d R=%d BS=%d: t_med=%.4f ms (min %.4f, avg %.4f), %.2f GFLOP/s", device,
           variant, p.N, p.R, p.BS, ts.med_ms, ts.min_ms, ts.mean_ms, gflops);
    if (cfg)
        printf(", grid=%ux%u, smem=%zu B, P=%d, konflikt=%d-way", cfg->grid.x, cfg->grid.y, cfg->smem,
               cfg->pitch, cfg->conflict);
    if (chk) printf(", max_rel_err=%.3e -> %s", chk->max_rel, chk->ok() ? "OK" : "BLAD");
    printf("\n");
    if (chk && !chk->ok()) {
        const int M = p.M();
        printf("    bledne elementy: %zu, pierwszy: OUT[%zu][%zu]\n", chk->bad, chk->first_bad / M,
               chk->first_bad % M);
    }
    // CSV,device,variant,BS,N,R,threads,reps,data,t_med_ms,t_min_ms,t_mean_ms,gflops,
    //     max_abs_err,max_rel_err,ok,smem_B,pitch,conflict
    printf("CSV,%s,%s,%d,%d,%d,%d,%d,%s,%.6f,%.6f,%.6f,%.4f,%.3e,%.3e,%d,%zu,%d,%d\n", device, variant,
           p.BS, p.N, p.R, threads, p.reps, p.data.c_str(), ts.med_ms, ts.min_ms, ts.mean_ms, gflops,
           chk ? chk->max_abs : 0.0, chk ? chk->max_rel : 0.0, chk ? (int)chk->ok() : -1,
           cfg ? cfg->smem : (size_t)0, cfg ? cfg->pitch : 0, cfg ? cfg->conflict : 0);
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

    printf("N=%d R=%d M=N-2R=%d BS=%d data=%s reps=%d watki_CPU=%d FLOP=%.4e\n", p.N, p.R, M, p.BS,
           p.data.c_str(), p.reps, threads, flopCount(p));

    std::vector<float> tab((size_t)p.N * p.N);
    std::vector<float> outCPU;
    initInput(tab, p);
    bool allOk = true;

    // --- CPU (referencja) ---
    if (p.cpu) {
        outCPU.resize((size_t)M * M);
        TimingStats ts = runCPU(tab, outCPU, p);
        CheckResult chk;
        const CheckResult *pchk = nullptr;
        if (p.data == "uniform") {
            chk = checkUniform(outCPU.data(), outCPU.size(), p.R);
            pchk = &chk;
            allOk &= chk.ok();
        }
        report("cpu", "omp", p, threads, ts, pchk, nullptr);
    }

    // --- GPU ---
    if (p.anyGpu()) {
        checkCudaErrors(cudaSetDevice(p.dev));
        printDeviceInfo(p.dev);
        float *d_in = nullptr, *d_out = nullptr;
        checkCudaErrors(cudaMalloc(&d_in, tab.size() * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_out, (size_t)M * M * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_in, tab.data(), tab.size() * sizeof(float), cudaMemcpyHostToDevice));

        std::vector<float> outGPU((size_t)M * M);
        static const char *names[3] = {"v0", "v1", "v2"};
        for (int variant = 0; variant < 3; ++variant) {
            if (!p.gpu[variant]) continue;
            TimingStats ts;
            LaunchCfg cfg;
            if (!runGPU(variant, d_in, d_out, outGPU, p, ts, cfg)) { allOk = false; continue; }
            CheckResult chk;
            const CheckResult *pchk = nullptr;
            if (p.cpu) { chk = compare(outCPU.data(), outGPU.data(), outGPU.size(), p.R); pchk = &chk; }
            else if (p.data == "uniform") { chk = checkUniform(outGPU.data(), outGPU.size(), p.R); pchk = &chk; }
            if (pchk) allOk &= chk.ok();
            report("gpu", names[variant], p, 0, ts, pchk, &cfg);
        }
        checkCudaErrors(cudaFree(d_in));
        checkCudaErrors(cudaFree(d_out));
    }

    return allOk ? 0 : 2;
}
