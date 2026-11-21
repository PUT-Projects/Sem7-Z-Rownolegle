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
      kolejność sumowania jak w definicji. Lokalnie (7840HS, 16 wątków, N=2048 R=8): naiwna 22 → bloki SSE2 97 → **AVX2 138 GFLOP/s**.
      Przetestowane na danych jednorodnych: R=0,1,4,7,36,37, M niebędące wielokrotnością 64, błędne argumenty.
      Host kompilowany z `/arch:AVX2` (tasks.json) i `-march=native` (Makefile; i7-8700 ma AVX2+FMA wg lscpu) –
      w labie budować na gpu/gpu1, nie na polluksie. **Lab jeszcze nieprzetestowany.**
      Build lokalnie z Git Bash wymaga `MSYS_NO_PATHCONV=1` (inaczej `/openmp` zamieniane na ścieżkę).
- [x] A3. `stencilKernel0<BS>` – tylko pamięć globalna (kontrolny, `-mode=gpu0`).
- [x] A4. `stencilKernelV1<BS>`: ładowanie kafla T×T (T=BS+2R) wierszami (`r=idx/T, c=idx%T`) → coalesced;
      obliczenia `tile[(ty+u)*P + tx+v]`; pitch P=T dla BS=32, P≡16 (mod 32) dla BS=16 (dwa półwarpy w rozłącznych bankach).
- [x] A5. `stencilKernelV2<BS>`: ładowanie kolumnami (`c=idx/T, r=idx%T`) → wątki warpa w różnych wierszach (krok N·4 B);
      kafel transponowany `tileT[c*P + r]`, obliczenia `tileT[(tx+v)*P + ty+u]` → krok P między wątkami → konflikt gcd(P,32):
      P≡0 (mod 32) gdy mieści się w 48 KB (32-way dla BS=32, 16-way dla BS=16), inaczej P≡16 → 16-way (BS=32, R>32).
      **Limit: wariant 2 przy BS=32 działa dla R ≤ 38** (R=39 → 49280 B > 48 KB, program go pomija). Wariant 1: R ≤ 39.
- [x] A6. Weryfikacja RTX 4060: BS 16/32 × R {0,1,4,7,16,33,36,37,38} × N (też M niepodzielne przez BS), random i uniform →
      wszystkie warianty błąd 0 (ta sama kolejność sumowania co CPU). Wyjście kernela inicjowane NaN (0xFF) – wykrywa niezapisane elementy.
- [x] A7. `compute-sanitizer` memcheck + racecheck (build `-G`): 0 błędów, 0 hazardów.
- [x] A8. Potwierdzenie `ncu` na RTX 4060 (N=1024; surowe dane: `projekt/results/ncu_rtx4060.csv`; liczniki włączone w panelu NVIDIA).
      Sektory 32 B na żądanie odczytu globalnego (ideał dla 32 floatów = 4) i fale (wavefronts) shared na instrukcję odczytu shared:

      | BS | R | v1 sektory/żądanie | v2 sektory/żądanie | v1 fale/LDS | v2 fale/LDS | przewidywanie v2 |
      |---|---|---|---|---|---|---|
      | 16 | 4  | 3.98 | 23.9 | ~1 (0 konfliktów) | 15.9 | 16-way (T=24<32, część warpa się zawija) |
      | 16 | 36 | 3.99 | 32.0 | ~1 (0.01%) | 15.9 | 16-way |
      | 32 | 4  | 3.98 | 31.8 | ~1 | 31.8 | 32-way |
      | 32 | 36 | 3.99 | 31.9 | ~1 (0 konfliktów) | 15.9 | 16-way (P=112) |

      Wszystkie przewidywania potwierdzone. Ten sam CSV dla GTX 1080 w labie – sprawdzić czy jest `nvprof`/`ncu` dla CC 6.1
      (ncu nie wspiera Pascala od dawna → prawdopodobnie `nvprof --metrics gld_transactions_per_request,shared_load_transactions_per_request`).
- Pomiar próbny RTX 4060 (2026-11-21, zasilacz + tryb Performance, limit GPU 105 W; N=4096, mediana 10 powt.) [GFLOP/s]
  – dane: `projekt/results/rtx4060_perf_probe.csv`, pobór mocy: `..._power.csv`:

  | R | BS | CPU (16 wątków) | v0 | v1 | v2 | v1/v2 |
  |---|---|---|---|---|---|---|
  | 4  | 16 | 120 | 549 | 531 | 85  | 6.2× |
  | 4  | 32 | 153 | 480 | 505 | 56  | 9.0× |
  | 36 | 16 | 155 | 826 | 911 | 117 | 7.8× |
  | 36 | 32 | 144 | 914 | 905 | 118 | 7.7× |

  GPU pobierało maks. 65 W (śr. ~56 W pod obciążeniem), zegar SM ~2475 MHz – **nie dochodzi do limitu 105 W**, więc
  wyniki prawie takie same jak przy limicie 50 W (wcześniej R=36: v0 769–835, v1 879–903). Kernel ograniczony przepustowością
  LDS/L1, nie mocą. Do energii: rozważyć podanie zarówno mocy nominalnej (105 W, wymaganie projektu) jak i zmierzonej (nvidia-smi).
  Rozrzut: CPU min–mediana do ~25%, v2 BS16 R4 dwumodalny (min 11.9 vs med 15.7 ms) – przy eksperymentach więcej powtórzeń.
  Obserwacja: v1 ≈ v0 – na Ada L1 i shared to ta sama pamięć, a v1 robi 1 odczyt shared na 1 FLOP (ograniczenie przepustowości LDS).
  Do rozważenia (opcjonalnie): ponowne użycie danych w rejestrach (np. kilka wyników na wątek / sumy wierszy) – ale zmienia to
  „czystość” porównania wariantów; najpierw przedyskutować.
- [x] A9. `projekt/README.md` – pliki, kompilacja (Linux/make, Windows/nvcc), parametry, testy, eksperymenty, skrypty.

### B. Infrastruktura
- [x] B1. `.vscode/tasks.json`: upload obejmuje `projekt/`; nowe zadania „Remote: experiments on GTX 1080” (make + run_experiments.sh
      z argumentami, domyślnie `nstab`) i „Remote: download GTX 1080 results” (scp `gtx1080_*` → `projekt/results`).
      Z terminala: `ssh gpu1 'cd ~/Zad2 && make SRC=projekt/stencil.cu ARCH=sm_61 && projekt/run_experiments.sh nstab'`.
- [x] B2. `projekt/run_experiments.sh nstab|main|all [N...]` (bash; lokalnie przez Git Bash, w labie na gpu/gpu1).
      Zmienne: TAG (rtx4060/gtx1080 auto), BIN, DEV, R1=8, R2=36, REPS=10, NSTAB_N, OUTDIR. Zapisuje `<tag>_nstab.csv`,
      `<tag>_main.csv`, `<tag>_*_power.csv` (nvidia-smi co 200 ms), `<tag>_sysinfo.txt`.
      `projekt/analyze_nstab.py <csv> [prog%]` – tabele Markdown + N_stab (`<tag>_nstab.md`), tylko stdlib (brak matplotlib lokalnie).
- [x] B2a. **Rozgrzewka na czas** (`-warmup=ms`, domyślnie 1000): GPU w laptopie startuje z 1890 MHz i dochodzi do ~2475 MHz
      dopiero po ~1 s obciążenia → bez tego małe N mierzone przy niższym zegarze (skoki 550→717 i 690→907 GFLOP/s = 2475/1890).
      Dowód zachowany: `results/rtx4060_nstab_bez_rozgrzewki.{csv,md}` + `_power.csv` (zegar SM w czasie) – opisać w sprawozdaniu
      jako element metodyki.
- [ ] B3. Zebrać specyfikację: `lscpu`, `nvidia-smi -q`, deviceQuery na lab i lokalnie (moc nominalna + źródło).

### C. Eksperymenty (oba GPU: GTX 1080 i RTX 4060)
- [ ] C1. N_stab: wariant 1, BS=32, R1=8 i R2=36, N = 128…16384 → 2 tabele (czas, GFLOP/s).
  - [x] RTX 4060 (2026-11-21, 105 W, rozgrzewka 1 s, mediana 10 powt.) – `results/rtx4060_nstab.{csv,md}`.
        Kryterium: N_stab = najmniejsze N, dla którego żadne większe N nie daje > 5% więcej GFLOP/s.
        **R=8: N_stab = 768** (707 GFLOP/s, plateau ~716–730); **R=36: N_stab = 640** (865, plateau ~907).
        Przy progu 2% oba wychodzą 1024. Interpretacja: 24 SM, BS=32 → 1 blok (1024 wątki) na SM naraz (limit 1536 wątków/SM);
        małe N = mniej bloków niż SM / mało fal → niewykorzystane SM i efekt ostatniej fali (R=36: N=128 i 192 ten sam czas – 1 fala).
  - [x] GTX 1080 (gpu1 = gpu-6, GPU 0, 2026-11-21) – `results/gtx1080_nstab.{csv,md}`:
        **R=8: N_stab = 640** (800 GFLOP/s, plateau ~830); **R=36: N_stab = 512** (1045, plateau ~1090).
        20 SM, 2048 wątków/SM → 2 bloki 32×32 na SM naraz (R=36: 2×43 KB ≤ 96 KB shared/SM) → 40 bloków na falę.
        GPU pracuje na limicie mocy: śr. 174 W (maks. 187 W) przy limicie 180 W, zegar SM ~1843 MHz (maks. 1911).
  - **max N_stab (obie karty) = 768** → główny eksperyment (C2) dla N ≥ 768; propozycja `main 4096 8192`.
- Obserwacja GTX 1080 (N=1024, R=36, BS=32): v0 481, **v1 1089**, v2 75 GFLOP/s; CPU i7-8700 (12 wątków, gcc -march=native) 232.
  Na Pascalu v1 jest 2,3× szybszy od v0 (odczyty globalne domyślnie nie są cache'owane w L1, tylko L2), na Adzie v1 ≈ v0.
- Lab: Makefile wybiera CUDA **12.9** (`/usr/local/cuda-12.9`), nie 12.5 – podać w sprawozdaniu. Wszystkie 40 testów poprawności
  (BS 16/32 × 10 konfiguracji × random/uniform) OK na GTX 1080.
- [x] C2. `run_experiments.sh main 4096 8192` na obu kartach (2026-11-21) → `results/{gtx1080,rtx4060}_main.csv` (+ `_power.csv`).
      Wszystkie porównania GPU–CPU OK (błąd 0). GFLOP/s (mediana 10 powt., rozgrzewka 1 s):

      **Wersja ostateczna** (po zmianie CPU na `schedule(dynamic, 4)`, OneDrive wyłączony, oba przebiegi 2026-11-21):

      | karta | N | R | BS | CPU | v1 | v2 | v1/CPU | v1/v2 |
      |---|---|---|---|---|---|---|---|---|
      | GTX 1080 | 4096 | 8  | 16 | 209 | 861  | 75  | 4.1× | 11.4× |
      | GTX 1080 | 4096 | 8  | 32 | 209 | 811  | 38  | 3.9× | 21.5× |
      | GTX 1080 | 4096 | 36 | 16 | 245 | 1005 | 75  | 4.1× | 13.4× |
      | GTX 1080 | 4096 | 36 | 32 | 244 | 1109 | 75  | 4.5× | 14.7× |
      | GTX 1080 | 8192 | 8  | 16 | 201 | 847  | 74  | 4.2× | 11.4× |
      | GTX 1080 | 8192 | 8  | 32 | 201 | 825  | 37  | 4.1× | 22.2× |
      | GTX 1080 | 8192 | 36 | 16 | 244 | 964  | 74  | 4.0× | 13.0× |
      | GTX 1080 | 8192 | 36 | 32 | 244 | 1095 | 75  | 4.5× | 14.7× |
      | RTX 4060 | 4096 | 8  | 16 | 166 | 814  | 115 | 4.9× | 7.1×  |
      | RTX 4060 | 4096 | 8  | 32 | 172 | 713  | 58  | 4.1× | 12.2× |
      | RTX 4060 | 4096 | 36 | 16 | 160 | 911  | 117 | 5.7× | 7.8×  |
      | RTX 4060 | 4096 | 36 | 32 | 152 | 905  | 118 | 5.9× | 7.7×  |
      | RTX 4060 | 8192 | 8  | 16 | 159 | 815  | 115 | 5.1× | 7.1×  |
      | RTX 4060 | 8192 | 8  | 32 | 162 | 716  | 58  | 4.4× | 12.3× |
      | RTX 4060 | 8192 | 36 | 16 | 157 | 912  | 117 | 5.8× | 7.8×  |
      | RTX 4060 | 8192 | 36 | 32 | 158 | 907  | 118 | 5.7× | 7.7×  |

      **CPU na laptopie – `schedule(static)` → `schedule(dynamic, 4)`**: przy static ok. 1/6 uruchomień było dokładnie 2× wolniejsze
      (~75 GFLOP/s zamiast ~150; wszystkie powtórzenia w procesie, także min) – niezależnie od OneDrive i od wcześniejszego obciążenia GPU
      (test: zdarzało się po kilku przebiegach samego CPU). Wyjaśnienie: Windows czasem stawia 2 wątki na jednym rdzeniu, a przy
      podziale statycznym wszyscy czekają na najwolniejszego. Dynamic: 12/12 uruchomień 164–176, brak spadków; w labie bez zmian (195–198).
      **Throttling CPU+GPU (wspólny budżet mocy laptopa) – sprawdzony, brak wpływu**: CPU i GPU nie liczą jednocześnie (w jednym procesie
      kolejno: CPU, potem v1, potem v2). Pomiary osobno po 60 s spoczynku: CPU 163 / 167 (w serii 157 / 166), GPU v1 912.6 / 814,
      v2 114.8 (w serii 912 / 814 / 115) – różnice w granicach rozrzutu.
      Kod podzielony na `common.h`, `cpu_ref.h`, `kernels.cuh`, `stencil.cu` (kompilowany jest tylko stencil.cu); Makefile
      przebudowuje też po zmianie nagłówków w katalogu źródła.

      Poprzednia wersja tabeli (static, z zakłóceniami) zastąpiona – patrz historia git.

      Uwagi do wniosków: v2 przy R=8, BS=32 dwa razy wolniejszy niż przy BS=16 (konflikt 32-way vs 16-way) – na R=36 oba 16-way
      i v2 jednakowe. v1: BS=32 lepszy przy R=36 na GTX 1080 (mniejszy narzut halo: (T/BS)² = 3.3 vs 5.5), przy R=8 BS=16 ≥ BS=32.
      CPU (laptop) ma duży rozrzut; punkt N=4096 R=36 BS=16 dał raz 76 GFLOP/s (zakłócenie w tle – repo w OneDrive synchronizuje
      wyniki), po powtórzeniu 150 – wiersze tego punktu w `rtx4060_main.csv` podmienione na powtórzony pomiar.
      Pobór mocy: GTX 1080 śr. ~172 W (limit 180 W), SM ~1830 MHz; RTX 4060 pod obciążeniem ~37–76 W. **nvidia-smi na laptopie
      czasem zwraca 590.01 W przy zegarze 1890 MHz – to błędny odczyt, odfiltrować (>200 W) przy analizie.**
- [x] C3. `python projekt/analyze_energy.py` → `results/energia_cgma.md` (tabele) + `images/energia_N{4096,8192}.png`.
      Wzory: E = P_nom·t (t = mediana czasu kernela / funkcji CPU), efektywność = GFLOP/s / P_nom [GFLOP/J].
      P_nom: i7-8700 65 W (Intel ARK), Ryzen 7 7840HS 54 W (AMD: TDP 35–54 W, przyjęto górną), GTX 1080 180 W (NVIDIA),
      RTX 4060 Laptop 105 W (NVIDIA: 35–115 W; maks. limit w tym laptopie wg nvidia-smi).
      CGMA (FLOP / dostęp do global, odczyty+zapisy, poziom kodu): kernel 0 = (K²−1)/(K²+1) ≈ 1;
      v1 = v2 = BS²(K²−1)/(T²+BS²): R8/BS16 57.6, R8/BS32 88.6, R36/BS16 170.5, R36/BS32 460.8
      (v2 ma ten sam CGMA – różni się wzorcem dostępu, nie liczbą dostępów; transakcji ~8× więcej).
      Wyniki N=8192: GTX 1080 v1 4.6–6.1 GFLOP/J vs CPU 3.1–3.8 (CPU zużywa 1.4–1.6× więcej energii), v2 0.2–0.4 GFLOP/J
      (9–20× więcej energii niż v1). RTX 4060 v1 6.8–8.7 GFLOP/J vs CPU 2.9–3.0 (CPU 2.3–3.0× więcej), v2 0.6–1.1.
      Uwaga do wniosków: rzeczywisty pobór laptopa (~37–76 W GPU) < 105 W → oszacowanie z P_nom zawyża energię RTX 4060;
      GTX 1080 pracuje na limicie (~172 W ≈ 180 W), więc tam oszacowanie jest bliskie rzeczywistości.

### D. Rysunki (`images/`) – KLUCZOWY PUNKT
Skrypty w `projekt/figures/` (matplotlib zainstalowany lokalnie przez `pip install --user matplotlib`); uruchamiać z katalogu Zad2:
`python projekt/figures/access_patterns.py` i `python projekt/figures/results_plots.py` → `projekt/images/`.
Rysunki mają fizyczną szerokość ~17–18 cm (strona A4) – wstawiać w 100% szerokości tekstu, czcionki 7.5–11.5 pt.
Schematy: maks. 48 komórek w poziomie (dla T > 48 pokazany początek kafla, z adnotacją), układ paneli liczony w calach.
- [x] D1. `access_patterns.py` → `projekt/images/global_v{1,2}_BS{16,32}_R{8,36}.png`: ładowanie kafla, warp 0 i 1, iteracja k=0,
      lane w komórkach, granice sektorów 32 B / linii 128 B, liczba sektorów warpa i średnia w bloku. Indeksowanie 1:1 z kernels.cuh.
      Model daje 4.00 sektora/żądanie (v1) i 32.00 (v2) – zgodne z ncu (3.98–3.99 / 31.8–32.0).
- [x] D2. `access_patterns.py` → `projekt/images/shared_v{1,2}_BS{16,32}_R{8,36}.png`: lewy panel – układ wierszy shared (P słów,
      dopełnienie kreskowane, numer banku w komórce, odczyty warpa 0 jako t<lane>/b<bank>), prawy – banki × przebiegi.
      Stopień konfliktu z modelu: v1 = 1.00 wszędzie, v2 = 16 (BS16; BS32 R36) / 32 (BS32 R8) – zgodne z ncu.
- [x] D3. `results_plots.py` → `nstab.png` (obie karty, R=8/36, N_stab zaznaczone), `nstab_rozgrzewka.png` (z/bez rozgrzewki),
      `zegar_rtx4060.png` (zegar SM i moc w czasie – uzasadnienie rozgrzewki), `main_N4096.png`, `main_N8192.png`
      (CPU / v1 / v2 dla (R,BS), obie karty). Do dodania później: wykres energii (po ustaleniu wzorów, C3).

### E. Sprawozdanie (`projekt/sprawozdanie.md` → PDF)
- [x] E1. Szkielet sekcji.
- [x] E2. Opis systemów (tab. 1–2, parametry z cudaGetDeviceProperties), kod (listingi 1–4), rysunki 1–16 z analizą, konfiguracja, wzory.
- [x] E3. Tabele 6–12, rysunki 17–21, wnioski (model przepustowości shared: v1 GTX 94% limitu, v2 = limit/stopień konfliktu).
      Do uzupełnienia: data oddania (TODO w nagłówku).
- [ ] E4. Eksport do PDF, zip z kodem.

### F. Prezentacja (~15 min)
- [ ] F1. Krótkie slajdy / ściąga do omówienia wariantów i wyników.

## Otwarte pytania
- Autorzy: Mateusz Chlebosz, 151817 i Wojciech Durczak, 161609 → pliki do oddania `151817_161609.zip` / `151817_161609.pdf`.
- ~~CPU gpu vs gpu1~~ – ustalone: gpu1 = `gpu-6`, gpu = `gpu-5`; przyjmujemy i7-8700 dla obu. Pomiary preferencyjnie na gpu1.
