#!/usr/bin/env python3
"""Wykresy wynikow eksperymentow z projekt/results/*.csv.

Uzycie (z katalogu Zad2):  python projekt/figures/results_plots.py [katalog_wynikow] [katalog_wyjsciowy]
Domyslnie: projekt/results -> projekt/images/.
Rysunki maja szerokosc ok. 7.2 cala (~18 cm) – po wstawieniu na szerokosc strony A4 czcionki
zachowuja rozmiar (9–11 pt).

Tworzy:
  nstab.png           – GFLOP/s(N) wariantu 1, BS=32, R=8 i R=36, obie karty, z zaznaczonym N_stab
  nstab_rozgrzewka.png – RTX 4060: seria bez rozgrzewki vs z rozgrzewka 1 s (efekt zegara GPU)
  main_<N>.png        – CPU vs wariant 1 vs wariant 2 dla (R, BS), obie karty, dla danego N
  zegar_rtx4060.png   – zegar SM i pobor mocy RTX 4060 w czasie serii bez rozgrzewki
"""
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#b8b7b2"
# kolory przypisane do bytow (stale, niezalezne od kolejnosci):
C_R8, C_R36 = "#2a78d6", "#eb6834"                      # serie R
C_CPU, C_V1, C_V2 = "#1baf7a", "#2a78d6", "#eb6834"     # CPU / wariant 1 / wariant 2

CARDS = [("gtx1080", "GTX 1080 (CPU: i7-8700)"), ("rtx4060", "RTX 4060 Laptop (CPU: Ryzen 7 7840HS)")]

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.titlesize": 10.5, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def nstab_of(ns, g, thr=0.05):
    k = next(k for k in range(len(g)) if all(x <= g[k] * (1 + thr) for x in g[k + 1:]))
    return ns[k], g[k]


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_nstab_series(ax, rows, label_suffix="", ls="-", alpha=1.0, mark_stab=True):
    by_r = defaultdict(list)
    for r in rows:
        by_r[int(r["R"])].append((int(r["N"]), float(r["gflops"])))
    for R, color in ((8, C_R8), (36, C_R36)):
        pts = sorted(by_r[R])
        ns, g = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(ns, g, ls, color=color, linewidth=2, marker="o", markersize=4, alpha=alpha,
                label=f"R={R}{label_suffix}")
        if mark_stab:
            n0, g0 = nstab_of(ns, g)
            ax.axvline(n0, color=color, linestyle=":", linewidth=1.2)
            ax.annotate(f"N_stab={n0}", (n0, g0), textcoords="offset points", xytext=(6, -14),
                        color=INK2, fontsize=8)


def fig_nstab(resdir, outdir):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.6), sharex=True, sharey=True)
    for ax, (tag, title) in zip(axes, CARDS):
        path = os.path.join(resdir, f"{tag}_nstab.csv")
        if not os.path.exists(path):
            ax.set_visible(False)
            continue
        plot_nstab_series(ax, read_csv(path))
        ax.set_xscale("log", base=2)
        ax.set_title(title.split(" (")[0] + ": wariant 1, BS=32", loc="left")
        style_axes(ax)
        ax.legend(loc="lower right")
    for ax in axes:
        ax.set_ylabel("wydajność [GFLOP/s]")
    axes[0].set_ylim(bottom=0)
    axes[-1].set_xlabel("N (rozmiar tablicy N×N, skala log₂)")
    fig.suptitle("Wyznaczenie N_stab: najmniejsze N, po którym większe N\n"
                 "nie daje więcej niż 5% wzrostu wydajności",
                 x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout()
    path = os.path.join(outdir, "nstab.png")
    fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def fig_warmup(resdir, outdir):
    a = os.path.join(resdir, "rtx4060_nstab_bez_rozgrzewki.csv")
    b = os.path.join(resdir, "rtx4060_nstab.csv")
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plot_nstab_series(ax, read_csv(a), " – bez rozgrzewki", ls="--", alpha=0.75, mark_stab=False)
    plot_nstab_series(ax, read_csv(b), " – rozgrzewka 1 s", mark_stab=False)
    ax.set_xscale("log", base=2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("N (skala log₂)")
    ax.set_ylabel("wydajność [GFLOP/s]")
    ax.set_title("RTX 4060, wariant 1, BS=32: wpływ rozgrzewki na pomiar\n"
                 "(zegar GPU rośnie z 1890 do ~2475 MHz po ~1 s obciążenia)",
                 loc="left")
    style_axes(ax)
    ax.legend(loc="lower right", ncols=2, fontsize=8.5)
    fig.tight_layout()
    path = os.path.join(outdir, "nstab_rozgrzewka.png")
    fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def fig_clock(resdir, outdir):
    path = os.path.join(resdir, "rtx4060_nstab_bez_rozgrzewki_power.csv")
    if not os.path.exists(path):
        return None
    t, clk, pw = [], [], []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            p = float(row[1].strip().split()[0])
            if p > 200:  # bledny odczyt nvidia-smi na laptopie (590.01 W) – pomijamy
                continue
            t.append(ts)
            pw.append(p)
            clk.append(float(row[3].strip().split()[0]))
    t0 = t[0]
    sec = [(x - t0).total_seconds() for x in t]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    a1.plot(sec, clk, color=C_V1, linewidth=1.6)
    a1.set_ylabel("zegar SM [MHz]")
    a1.set_title("RTX 4060, seria N_stab bez rozgrzewki: zegar SM rośnie\n"
                 "dopiero po ~1 s ciągłego obciążenia", loc="left")
    a2.plot(sec, pw, color=C_V2, linewidth=1.6)
    a2.set_ylabel("moc GPU [W]")
    a2.set_xlabel("czas od startu serii [s]")
    for ax in (a1, a2):
        style_axes(ax)
    fig.tight_layout()
    out = os.path.join(outdir, "zegar_rtx4060.png")
    fig.savefig(out, dpi=250, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out


def fig_main(resdir, outdir, N):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.2), sharey=True)
    any_data = False
    for ax, (tag, title) in zip(axes, CARDS):
        path = os.path.join(resdir, f"{tag}_main.csv")
        if not os.path.exists(path):
            ax.set_visible(False)
            continue
        d = defaultdict(dict)
        for r in read_csv(path):
            if int(r["N"]) == N:
                d[(int(r["R"]), int(r["BS"]))][r["variant"]] = float(r["gflops"])
        if not d:
            ax.set_visible(False)
            continue
        any_data = True
        groups = sorted(d)
        width, gap = 0.26, 0.02
        series = [("omp", "CPU (OpenMP)", C_CPU), ("v1", "GPU wariant 1 (efektywny)", C_V1),
                  ("v2", "GPU wariant 2 (nieefektywny)", C_V2)]
        for si, (key, label, color) in enumerate(series):
            xs = [gi + (si - 1) * (width + gap) for gi in range(len(groups))]
            vals = [d[g].get(key, 0) for g in groups]
            ax.bar(xs, vals, width=width, color=color, label=label, edgecolor=SURFACE, linewidth=1)
            for x, v in zip(xs, vals):
                ax.text(x, v + 15, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"R={R}, BS={BS}" for R, BS in groups])
        ax.set_title(title, loc="left")
        style_axes(ax)
    if not any_data:
        plt.close(fig)
        return None
    for ax in axes:
        ax.set_ylabel("wydajność [GFLOP/s]")
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.08)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Porównanie CPU i wariantów GPU, N={N} (mediana 10 powtórzeń)", x=0.01, ha="left",
                 color=INK, fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(outdir, f"main_N{N}.png")
    fig.savefig(out, dpi=250, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out


def main():
    resdir = sys.argv[1] if len(sys.argv) > 1 else "projekt/results"
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join("projekt", "images")
    os.makedirs(outdir, exist_ok=True)
    made = [fig_nstab(resdir, outdir), fig_warmup(resdir, outdir), fig_clock(resdir, outdir)]
    made += [fig_main(resdir, outdir, N) for N in (4096, 8192)]
    for p in made:
        if p:
            print("Zapisano:", p)


if __name__ == "__main__":
    main()
