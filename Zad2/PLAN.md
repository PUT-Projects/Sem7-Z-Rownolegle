# Plan pracy – projekt CUDA (PKG), suma okna (2R+1)×(2R+1)

Źródło wymagań: `CUDA_NS_26_27.pdf`, etapy: `gpu_zadania_26_27_.txt`, środowisko: `uzycie_gpu_pr_ns26_27.txt`.
Termin: **9.01.2027** (zip z kodem + pdf sprawozdania, nazwy `NR_INDEKSU1_NR_INDEKSU2.*`), potem ~15 min prezentacji.

## Problem w skrócie
- `TAB` N×N (float) → `OUT` (N−2R)×(N−2R), `OUT[i][j] = Σ_{u,v∈[−R,R]} TAB[i+R+u][j+R+v]`.
- FLOP na element wyjściowy: `(2R+1)² − 1` dodawań → `FLOP = (N−2R)² · ((2R+1)² − 1)`.
- Kernel: parametry = wskaźnik wejścia, wskaźnik wyjścia, N, R. Bloki BS×BS, BS ∈ {16, 32}.
- Grid: `ceil((N−2R)/BS)` × `ceil((N−2R)/BS)`, warunek brzegowy w kernelu (wymiar nie musi być wielokrotnością BS).

## Środowiska
| | Lab (gpu / gpu1) | Lokalnie |
|---|---|---|
| GPU | 2× GTX 1080 (Pascal GP104, CC 6.1, 20 SM, 8 GB GDDR5X, 180 W) | RTX 4060 **Laptop** w Lenovo Legion Slim 5 14 (Ada AD107, CC 8.9, 8 GB GDDR6); `nvidia-smi -q -d POWER`: Max Power Limit **105 W**, default 55 W, aktualnie 50 W |
| CUDA | 12.5, `-arch=sm_61`, przez `Makefile` | 13.5 (sm_89), też 12.8 i 13.0; zadania w `.vscode/tasks.json` |
| CPU | Intel Core i7-8700 (Coffee Lake), 6C/12T, 3.2 GHz base / 4.6 GHz turbo, L3 12 MB, TDP **65 W** (Intel ARK); host `gpu-6` (lscpu z 2026-11-21) | AMD Ryzen 7 7840HS, 8C/16T, TDP 35–54 W (źródło: AMD) |

Uwaga do pomiarów lokalnych: limit mocy GPU zależy od trybu zasilania laptopa (50 W teraz vs max 105 W).
Pomiary robić na zasilaczu w trybie Performance i zapisać `nvidia-smi -q -d POWER` z czasu pomiarów.
Moc nominalna do energii: 105 W (max TGP z nvidia-smi) – „140 W” to prawdopodobnie łączna moc CPU+GPU laptopa.

Shared mem/blok: 48 KB (GTX 1080 – bez opt-in powyżej 48 KB), co ogranicza R przy BS=32 (patrz niżej).

## Decyzje projektowe (do potwierdzenia przy implementacji)
- Jeden program `projekt/stencil.cu`: kod CPU (OpenMP) + 2 kernele (wariant 1 efektywny, wariant 2 nieefektywny),
  szablony po BS (`template<int BS>`), shared jako `extern __shared__` o rozmiarze `(BS+2R)²·4 B`.
- **Wybór R**: R1 < 32 (np. R=4 lub 8) i R2 > 32. Limit 48 KB: `(32+2R)²·4 ≤ 49152` → **R ≤ 39** dla BS=32.
  Proponowane R2 = 36 (kafel 104×104 = 43 KB). Dla BS=16 R=36 → 88² ·4 = 31 KB – OK.
- **Wariant 1 – pamięć globalna łączona**: kolektywne ładowanie kafla (BS+2R)² pętlą o kroku BS×BS, liniowy indeks
  wątku `t = ty·BS+tx` → kolejne wątki warpa czytają kolejne kolumny tego samego wiersza.
- **Wariant 2 – pamięć globalna niełączona**: ten sam kafel, ale mapowanie zamienione (kolejne wątki warpa → kolejne
  **wiersze**, krok N·4 B) → ~32 transakcje na instrukcję warpa zamiast 1–2.
- **Wariant 1 – shared bez konfliktów**: obliczenia `tile[ty+u][tx+v]` – wątki warpa (kolejne tx) → kolejne słowa → różne banki.
- **Wariant 2 – shared z konfliktami**: przestawienie indeksowania (np. `tile[tx+v][ty+u]` / wątek czyta kolumnę)
  z krokiem wiersza dopełnionym do wielokrotności 32 słów → wszystkie wątki warpa w tym samym banku (32-way).
  Uwaga: bez dopełnienia stopień konfliktu zależy od `gcd(BS+2R, 32)` – opisać w sprawozdaniu.
- Dane niejednorodne: losowe float w [0,1) (stały seed) + test jednorodny (same 1.0 → wynik dokładny (2R+1)²).
- Porównanie: błąd względny `|gpu−cpu| / max(|cpu|,1) < 1e−5·(2R+1)²`-ish – dobrać i uzasadnić; raport max błędu.
- Pomiar: `cudaEvent` wokół samego kernela, rozgrzewka + k powtórzeń (np. 10), reprezentatywny = **mediana**
  (podać też min/średnią). CPU: `omp_get_wtime`, te same N, R, też mediana.
- Wyjście programu: linia CSV (`device,variant,BS,N,R,time_ms_med,time_ms_min,gflops,max_err,ok`) → skrypty zbierają do `projekt/results/*.csv`.
- Energia: `E = P_nom · t` (CPU: TDP; GPU: moc znamionowa karty). CGMA liczone na poziomie kodu źródłowego (wzór od BS, R).

## Zadania
### A. Kod (`projekt/`)
- [x] A1. Szkielet `projekt/stencil.cu`: argumenty `-N= -R= -BS= -mode=cpu|gpu1|gpu2|all -reps= -data=random|uniform -threads= -dev= -seed=`,
      dane xorshift32 (deterministyczne), mediana/min/średnia, linie `CSV,...`, `compare()` z tolerancją `(2R+1)²·FLT_EPSILON` (wzgl.),
      `checkUniform()` (wynik musi być dokładnie (2R+1)²), kod wyjścia 0/1 (argumenty)/2 (błąd wyniku lub brak GPU).
      Uwaga: nie używać `helper_string.h` do argumentów – porównuje prefiksy bez wielkości liter (`-reps` łapał się jako `-R`).
- [x] A2. Referencja CPU OpenMP: `parallel for schedule(static)` po wierszach, bloki JB=64 z akumulatorem `acc[64]` (wektoryzacja),
      kolejność sumowania jak w definicji. Lokalnie (7840HS, 16 wątków): ~97 GFLOP/s dla N=2048 R=8 (wersja naiwna: 22).
      Przetestowane na danych jednorodnych: R=0,1,4,7,36,37, M niebędące wielokrotnością 64, błędne argumenty.
      Możliwe ulepszenie: `/arch:AVX2` (lokalnie) / `-march=native` (lab) – na razie host kompilowany z SSE2 (domyślnie).
      Build lokalnie z Git Bash wymaga `MSYS_NO_PATHCONV=1` (inaczej `/openmp` zamieniane na ścieżkę).
- [ ] A3. Kernel0 (naiwny, tylko pamięć globalna) – do sanity-checku.
- [ ] A4. Kernel wariant 1 (coalesced + bez konfliktów), `template<int BS>`.
- [ ] A5. Kernel wariant 2 (non-coalesced + konflikty banków).
- [ ] A6. Weryfikacja: dane jednorodne i niejednorodne, różne N (też N−2R niebędące wielokrotnością BS), R, BS.
- [ ] A7. `compute-sanitizer` memcheck + racecheck (zadania w VS Code).
- [ ] A8. (opcjonalnie) potwierdzenie transakcji/konfliktów profilerem: `ncu` – metryki `l1tex__t_requests/sectors_pipe_lsu_mem_global_op_ld`, `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` (RTX 4060; na GTX 1080 `nvprof` nie wspiera CC 6.1 w CUDA 12.5? – sprawdzić).
- [ ] A9. `README.md` w `projekt/` z instrukcją kompilacji i uruchamiania (wymagane w zip).

### B. Infrastruktura
- [ ] B1. Dopisać upload katalogu `projekt/` do zadania „Remote: upload to polluks” w `.vscode/tasks.json`.
- [ ] B2. Skrypty eksperymentów: `projekt/run_experiments.sh` (lab) i `.ps1` (lokalnie) → CSV.
- [ ] B3. Zebrać specyfikację: `lscpu`, `nvidia-smi -q`, deviceQuery na lab i lokalnie (moc nominalna + źródło).

### C. Eksperymenty (oba GPU: GTX 1080 i RTX 4060)
- [ ] C1. N_stab: wariant 1, BS=32, R1 i R2, N rosnące (np. 128…16384) → 2 tabele (czas, GFLOP/s).
- [ ] C2. Dla N ≥ max(N_stab): CPU f(R,N), GPU wariant 1 f(BS,R,N), GPU wariant 2 f(BS,R,N).
- [ ] C3. Energia CPU vs GPU, CGMA, przyspieszenie GPU/CPU oraz wariant1/wariant2.

### D. Rysunki (`images/`) – KLUCZOWY PUNKT
- [ ] D1. Skrypt Python (matplotlib) generujący schematy dostępu: adresy czytane przez wątki warpa 0..31,
      kolejność dostępów, transakcje 32/128 B – dla wariantu 1 i 2, BS=16/32, R1/R2.
- [ ] D2. Schemat banków shared (nr banku, id wątku) – brak konfliktów vs konflikty, układ kafla w shared.
- [ ] D3. Wykresy wyników (N_stab, porównania wariantów/BS/R, CPU vs GPU).

### E. Sprawozdanie (`sprawozdanie.md` → PDF)
- [ ] E1. Szkielet sekcji (zrobione – uzupełniać).
- [ ] E2. Opis systemów, opis kodu z listingami, rysunki z kodem, konfiguracja uruchomienia, wzory.
- [ ] E3. Tabele wyników, wnioski. Numeracja i podpisy listingów/rysunków/tabel.
- [ ] E4. Eksport do PDF, zip z kodem.

### F. Prezentacja (~15 min)
- [ ] F1. Krótkie slajdy / ściąga do omówienia wariantów i wyników.

## Otwarte pytania
- Autor: Mateusz Chlebosz, 151817. Możliwy drugi członek – w sprawozdaniu zostawione pole TODO; przy oddaniu nazwy plików `151817[_NR2].zip/.pdf`.
- ~~CPU gpu vs gpu1~~ – ustalone: gpu1 = `gpu-6`, gpu = `gpu-5`; przyjmujemy i7-8700 dla obu. Pomiary preferencyjnie na gpu1.
