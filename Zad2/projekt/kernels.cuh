// =====================================================================================
//  kernels.cuh – kernele GPU: kontrolny (kernel 0), wariant 1 (efektywny),
//  wariant 2 (nieefektywny) oraz dobor szerokosci wiersza kafla w pamieci wspoldzielonej.
//  Parametry kazdego kernela: adres TAB i OUT w pamieci globalnej, N, R.
// =====================================================================================
#pragma once

#include <cuda_runtime.h>

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
