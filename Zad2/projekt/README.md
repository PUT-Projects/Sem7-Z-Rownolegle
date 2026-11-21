# Suma okna (2R+1)×(2R+1) na GPU (CUDA) – instrukcja kompilacji i uruchamiania

Projekt PKG 2026/2027 – Mateusz Chlebosz (151817), Wojciech Durczak (161609).

Dla tablicy `TAB` N×N (float) program liczy `OUT` (N−2R)×(N−2R), gdzie każdy element to suma okna (2R+1)×(2R+1):
na CPU (OpenMP) oraz na GPU w dwóch wariantach dostępu do pamięci, i porównuje wyniki element po elemencie.

## Pliki

| plik | zawartość |
|---|---|
| `stencil.cu` | program główny: konfiguracja uruchomienia kerneli, pomiar czasu (`cudaEvent`), porównanie wyników – **jedyny plik do kompilacji** |
| `kernels.cuh` | kernele GPU: `stencilKernelV1` (wariant 1 – efektywny), `stencilKernelV2` (wariant 2 – nieefektywny), `stencilKernel0` (kontrolny, bez pamięci współdzielonej) |
| `cpu_ref.h` | referencyjny program CPU (OpenMP) |
| `common.h` | parametry, dane wejściowe, statystyki czasu, weryfikacja wyników |
| `run_experiments.sh` | skrypt eksperymentów (seria N_stab i eksperyment główny) → `results/*.csv` |
| `analyze_nstab.py`, `analyze_energy.py` | tabele N_stab, energii i CGMA z wyników |
| `figures/*.py` | generowanie rysunków do `images/` |
| `results/`, `images/` | wyniki pomiarów i rysunki użyte w sprawozdaniu |

Kod korzysta z nagłówka `helper_cuda.h` z przykładów NVIDIA (katalog `cuda_pliki/` z materiałów do zajęć) –
ścieżkę do niego podaje się opcją `-I`.

## Kompilacja

Wymagania: CUDA Toolkit z kompilatorem `nvcc`, kompilator C++17 z OpenMP.
Dla GTX 1080 (Pascal, CC 6.1) potrzebna jest CUDA **12.x** (CUDA 13 nie obsługuje `sm_61`).

**Linux – maszyny laboratoryjne `gpu` / `gpu1` (GTX 1080), z katalogu nadrzędnego zawierającego `Makefile`, `cuda_pliki/` i `projekt/`:**

```bash
make SRC=projekt/stencil.cu ARCH=sm_61        # wynik: build/stencil
```

lub bezpośrednio:

```bash
nvcc -std=c++17 -O3 -arch=sm_61 -Xcompiler -fopenmp -Xcompiler -march=native \
     -Icuda_pliki -Iprojekt projekt/stencil.cu -o build/stencil
```

Kompilować na maszynie, na której program będzie uruchamiany (`-march=native` dobiera instrukcje do jej procesora).
Inna karta: zmienić `-arch` (np. `sm_89` dla RTX 40xx, `sm_86` dla RTX 30xx).

**Windows (MSVC + CUDA, np. RTX 4060 – `sm_89`), z wiersza poleceń „x64 Native Tools” lub PowerShell z `cl.exe` w `PATH`:**

```powershell
nvcc -std=c++17 -O3 -gencode=arch=compute_89,code=sm_89 -Xcompiler /openmp -Xcompiler /arch:AVX2 -Xcompiler /utf-8 -I cuda_pliki -I projekt projekt/stencil.cu -o build/stencil.exe
```

(W Git Bash poprzedzić polecenie `MSYS_NO_PATHCONV=1`, inaczej opcje `/openmp` są zamieniane na ścieżki.)

## Uruchamianie

```bash
./build/stencil -N=8192 -R=36 -BS=32 -mode=cpu,gpu1,gpu2 -reps=10
```

| parametr | znaczenie | domyślnie |
|---|---|---|
| `-N=` | rozmiar tablicy wejściowej N×N (wymagane N > 2R) | 1024 |
| `-R=` | promień okna (R ≥ 0) | 4 |
| `-BS=` | rozmiar bloku wątków: 16 (16×16) lub 32 (32×32) | 32 |
| `-mode=` | lista: `cpu`, `gpu0` (kontrolny), `gpu1` (wariant 1), `gpu2` (wariant 2), `all` | `cpu,gpu1,gpu2` |
| `-reps=` | liczba mierzonych powtórzeń (czas reprezentatywny = mediana) | 10 |
| `-warmup=` | minimalny czas rozgrzewki przed pomiarem [ms] | 1000 |
| `-data=` | `random` (dane niejednorodne, stały seed) lub `uniform` (same 1,0 – wynik musi być dokładnie (2R+1)²) | `random` |
| `-threads=` | liczba wątków OpenMP | wszystkie |
| `-dev=` | numer karty (`cudaSetDevice`) | 0 |
| `-seed=` | ziarno generatora danych | 12345 |

Przykładowy wynik:

```
[cpu/omp] N=8192 R=36 BS=32: t_med=1442.0382 ms (min 1436.8087, avg 1477.3481), 243.61 GFLOP/s
[gpu/v1] N=8192 R=36 BS=32: t_med=320.9374 ms (...), 1095.13 GFLOP/s, grid=254x254, smem=43264 B, P=104, konflikt=1-way, max_rel_err=0.000e+00 -> OK
[gpu/v2] N=8192 R=36 BS=32: t_med=4703.6 ms (...), 74.27 GFLOP/s, grid=254x254, smem=46592 B, P=112, konflikt=16-way, max_rel_err=0.000e+00 -> OK
```

Każdy wynik jest też wypisywany jako linia `CSV,...` (zbierana przez skrypt eksperymentów).
Kod wyjścia: 0 – wszystkie wyniki poprawne, 1 – błędne argumenty, 2 – niezgodność wyników GPU z CPU lub pominięty wariant
(np. kafel > 48 KB pamięci współdzielonej: BS=32 dopuszcza R ≤ 39 dla wariantu 1 i R ≤ 38 dla wariantu 2).

**Test poprawności:**

```bash
./build/stencil -N=1031 -R=33 -BS=16 -mode=all -data=random -reps=1
./build/stencil -N=1031 -R=33 -BS=32 -mode=all -data=uniform -reps=1
```

## Eksperymenty, tabele i rysunki

Z katalogu nadrzędnego (tego z `Makefile`), w labie na `gpu`/`gpu1`, lokalnie w Git Bash:

```bash
projekt/run_experiments.sh nstab              # N_stab: wariant 1, BS=32, R=8 i R=36, N=128..16384
projekt/run_experiments.sh main 4096 8192     # CPU, wariant 1, wariant 2 dla BS 16/32, R 8/36
python3 projekt/analyze_nstab.py projekt/results/gtx1080_nstab.csv
python3 projekt/analyze_energy.py             # energia (P_nom · t) i CGMA -> results/energia_cgma.md
python3 projekt/figures/access_patterns.py    # schematy dostępu do pamięci -> images/
python3 projekt/figures/results_plots.py      # wykresy wyników -> images/
```

Zmienne skryptu eksperymentów: `TAG` (nazwa systemu w plikach wyników), `BIN`, `DEV`, `R1`, `R2`, `REPS`, `NSTAB_N`.
Skrypty Python wymagają biblioteki `matplotlib` (`pip install matplotlib`).
