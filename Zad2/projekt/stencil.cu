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
//    -reps   liczba mierzonych powtorzen; czas reprezentatywny = mediana
//    -warmup minimalny czas rozgrzewki w ms przed pomiarem (domyslnie 1000, co najmniej
//            1 uruchomienie) – stabilizuje zegar GPU/CPU
//    -data   random  – dane niejednorodne, U[0,1), staly seed (domyslnie)
//            uniform – same 1.0f -> kazdy element OUT musi byc dokladnie (2R+1)^2
//
//  Na stdout: opis czytelny dla czlowieka + linie "CSV,..." do zbierania wynikow skryptem.
//
//  Pliki (kompilowany jest tylko stencil.cu, pozostale sa dolaczane przez #include):
//    common.h    – parametry, miary, dane wejsciowe, weryfikacja
//    cpu_ref.h   – referencja CPU (OpenMP)
//    kernels.cuh – kernele GPU: kernel 0, wariant 1 (efektywny), wariant 2 (nieefektywny)
//    stencil.cu  – konfiguracja uruchomienia kerneli, pomiar czasu (cudaEvent), main
// =====================================================================================
#include <cuda_runtime.h>

#include "helper_cuda.h"  // checkCudaErrors, _ConvertSMVer2Cores (cuda_pliki)

#include "common.h"   // parametry, miary, dane wejsciowe, weryfikacja
#include "cpu_ref.h"  // referencja CPU (OpenMP)
#include "kernels.cuh" // kernele GPU (v0, v1, v2)

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

    // Rozgrzewka przez co najmniej p.warmup_ms: GPU (szczegolnie w laptopie) startuje
    // z niskim zegarem i podnosi go dopiero po ok. 1 s ciaglego obciazenia – bez tego
    // krotkie serie (male N) bylyby mierzone przy nizszym zegarze niz dlugie.
    const double w0 = omp_get_wtime();
    do {
        cfg.fn<<<cfg.grid, cfg.block, cfg.smem>>>(d_in, d_out, p.N, p.R);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    } while ((omp_get_wtime() - w0) * 1e3 < p.warmup_ms);

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
