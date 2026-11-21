#!/usr/bin/env bash
# =====================================================================================
#  Eksperymenty obliczeniowe – zbiera linie "CSV,..." programu stencil do plikow CSV.
#
#  Uzycie (z katalogu Zad2):
#    projekt/run_experiments.sh nstab                 # wyznaczenie N_stab (wariant 1, BS=32, R1 i R2)
#    projekt/run_experiments.sh main  [N ...]         # CPU, v1, v2 dla BS 16/32, R1/R2 i podanych N
#    projekt/run_experiments.sh all   [N ...]         # nstab + main
#
#  Zmienne srodowiskowe (domyslne w nawiasach):
#    TAG     nazwa maszyny w nazwach plikow        (rtx4060 na Windows, gtx1080 na Linux)
#    BIN     sciezka do programu                    (build/stencil.exe lub build/stencil)
#    DEV     numer GPU (cudaSetDevice)              (0)
#    R1, R2  promienie: R1 < 32 = BS < R2           (8, 36)
#    REPS    liczba mierzonych powtorzen           (10; czas reprezentatywny = mediana)
#    NSTAB_N lista N dla serii N_stab               (patrz nizej)
#    OUTDIR  katalog wynikow                         (projekt/results)
#
#  Lokalnie (Windows) uruchamiac z Git Bash, w labie na gpu/gpu1 (nie na polluksie).
#  W trakcie pomiarow nvidia-smi zapisuje pobor mocy GPU do *_power.csv.
# =====================================================================================
set -euo pipefail

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) DEF_TAG=rtx4060; DEF_BIN=build/stencil.exe ;;
    *)                    DEF_TAG=gtx1080; DEF_BIN=build/stencil ;;
esac
TAG=${TAG:-$DEF_TAG}
BIN=${BIN:-$DEF_BIN}
DEV=${DEV:-0}
R1=${R1:-8}
R2=${R2:-36}
REPS=${REPS:-10}
OUTDIR=${OUTDIR:-projekt/results}
# Gesto dla malych N (kilka fal blokow na SM), rzadziej dla duzych.
NSTAB_N=${NSTAB_N:-"128 192 256 384 512 640 768 1024 1280 1536 2048 2560 3072 4096 6144 8192 12288 16384"}

HEADER="tag,device,variant,BS,N,R,threads,reps,data,t_med_ms,t_min_ms,t_mean_ms,gflops,max_abs_err,max_rel_err,ok,smem_B,pitch,conflict"

[ -x "$BIN" ] || { echo "Brak programu $BIN – najpierw zbuduj (patrz projekt/README.md)" >&2; exit 1; }
mkdir -p "$OUTDIR"

SMI_PID=""
start_power_log() {  # $1 = plik
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -i "$DEV" --query-gpu=timestamp,power.draw,power.limit,clocks.sm,temperature.gpu \
            --format=csv -lms 200 > "$1" 2>/dev/null &
        SMI_PID=$!
    fi
}
stop_power_log() {
    if [ -n "$SMI_PID" ]; then kill "$SMI_PID" 2>/dev/null || true; wait "$SMI_PID" 2>/dev/null || true; SMI_PID=""; fi
}
trap stop_power_log EXIT

# run <plik_csv> <argumenty programu...>  – dopisuje linie CSV (z kolumna tag), pokazuje postep
run() {
    local out=$1; shift
    local log rc=0
    log=$("$BIN" "$@" -dev="$DEV" -reps="$REPS") || rc=$?
    echo "$log" | grep '^\[' | sed 's/^/  /'
    echo "$log" | grep '^CSV,' | sed "s/^CSV,/$TAG,/" >> "$out"
    if [ $rc -ne 0 ]; then echo "  !! kod wyjscia $rc dla: $*" >&2; fi
}

save_sysinfo() {
    local f="$OUTDIR/${TAG}_sysinfo.txt"
    {
        echo "# $(date '+%Y-%m-%d %H:%M:%S')  host=$(hostname)"
        "$BIN" -N=64 -R=1 -mode=gpu1 -reps=1 -dev="$DEV" | grep '^GPU' || true
        command -v nvidia-smi >/dev/null && nvidia-smi -i "$DEV" -q -d POWER,CLOCK | grep -E 'Power Limit|Max Clocks|SM ' || true
        command -v lscpu >/dev/null && lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|max MHz' || true
    } > "$f"
    echo "Informacje o systemie: $f"
}

# --- N_stab: wariant 1 (efektywny), BS=32, R1 < BS < R2 ------------------------------
do_nstab() {
    local out="$OUTDIR/${TAG}_nstab.csv"
    echo "$HEADER" > "$out"
    start_power_log "$OUTDIR/${TAG}_nstab_power.csv"
    for R in $R1 $R2; do
        echo "== N_stab: BS=32 R=$R"
        for N in $NSTAB_N; do
            [ "$N" -gt $((2 * R)) ] || continue
            run "$out" -N="$N" -R="$R" -BS=32 -mode=gpu1
        done
    done
    stop_power_log
    echo "Wyniki: $out"
    python3 projekt/analyze_nstab.py "$out" 2>/dev/null || python projekt/analyze_nstab.py "$out"
}

# --- Glowny eksperyment: CPU, v1, v2 dla BS 16/32, R1/R2, N >= N_stab ------------------
do_main() {
    local Ns=("$@")
    [ ${#Ns[@]} -gt 0 ] || { echo "Podaj liste N (N >= max N_stab), np.: main 4096 8192" >&2; exit 1; }
    local out="$OUTDIR/${TAG}_main.csv"
    echo "$HEADER" > "$out"
    start_power_log "$OUTDIR/${TAG}_main_power.csv"
    for N in "${Ns[@]}"; do
        for R in $R1 $R2; do
            echo "== main: N=$N R=$R"
            # CPU nie zalezy od BS, ale jest liczony przy kazdym BS, zeby wyniki GPU
            # porownac z CPU element po elemencie (daje tez 2 niezalezne pomiary CPU).
            run "$out" -N="$N" -R="$R" -BS=16 -mode=cpu,gpu1,gpu2
            run "$out" -N="$N" -R="$R" -BS=32 -mode=cpu,gpu1,gpu2
        done
    done
    stop_power_log
    echo "Wyniki: $out"
}

cmd=${1:-}
shift || true
case "$cmd" in
    nstab) save_sysinfo; do_nstab ;;
    main)  save_sysinfo; do_main "$@" ;;
    all)   save_sysinfo; do_nstab; do_main "$@" ;;
    *) sed -n '2,22p' "$0"; exit 1 ;;
esac
