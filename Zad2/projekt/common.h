// =====================================================================================
//  common.h – czesc wspolna: parametry uruchomienia, miary (FLOP, statystyki czasu),
//  generowanie danych wejsciowych i weryfikacja wynikow.
// =====================================================================================
#pragma once

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

// -------------------------------------------------------------------------------------
//  Parametry uruchomienia
// -------------------------------------------------------------------------------------
struct Params {
    int N = 1024;
    int R = 4;
    int BS = 32;
    int reps = 10;
    int warmup_ms = 1000;  // minimalny czas rozgrzewki przed pomiarem (CPU i GPU), min. 1 uruchomienie
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
        "          [-warmup=<ms>] [-data=random|uniform] [-threads=<t>] [-dev=<d>] [-seed=<s>]\n",
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
        else if (key == "warmup") p.warmup_ms = iv;
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
    if (p.warmup_ms < 0) { fprintf(stderr, "Blad: warmup >= 0\n"); ok = false; }
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
