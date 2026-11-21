// =====================================================================================
//  cpu_ref.h – rownolegly program referencyjny CPU (OpenMP) i pomiar jego czasu.
// =====================================================================================
#pragma once

#include "common.h"

// -------------------------------------------------------------------------------------
//  Referencja CPU (OpenMP)
// -------------------------------------------------------------------------------------
// Watki OpenMP dziela miedzy siebie wiersze tablicy wynikowej, pobierajac je
// dynamicznie po 4 (schedule(dynamic, 4)). Praca na wiersz jest rowna, ale przy
// podziale statycznym (schedule(static)) zdarzalo sie na Windows, ze system
// umieszczal dwa watki na jednym rdzeniu – caly pomiar byl wtedy ~2x wolniejszy,
// bo pozostale watki czekaly na najwolniejszy. Podzial dynamiczny wyrownuje to
// (watek na wolniejszym rdzeniu po prostu pobiera mniej wierszy).
// Wiersz i jest liczony fragmentami po JB elementow:
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
#pragma omp parallel for schedule(dynamic, 4)
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
    // Rozgrzewka (page faults, watki OpenMP, zegar turbo) przez co najmniej p.warmup_ms.
    const double w0 = omp_get_wtime();
    do {
        stencilCPU(tab.data(), out.data(), p.N, p.R);
    } while ((omp_get_wtime() - w0) * 1e3 < p.warmup_ms);
    std::vector<double> t(p.reps);
    for (int r = 0; r < p.reps; ++r) {
        const double t0 = omp_get_wtime();
        stencilCPU(tab.data(), out.data(), p.N, p.R);
        t[r] = (omp_get_wtime() - t0) * 1e3;
    }
    return stats(t);
}
