#!/usr/bin/env python3
"""Szacunkowa energia obliczen (CPU i GPU) oraz teoretyczny wspolczynnik CGMA.

Uzycie (z katalogu Zad2):  python projekt/analyze_energy.py [katalog_wynikow] [katalog_obrazow]
Domyslnie: projekt/results -> tabele projekt/results/energia_cgma.md, wykres projekt/images/energia_N8192.png

Wzory (zgodnie z opisem projektu):
  FLOP        = (N-2R)^2 * ((2R+1)^2 - 1)                       (same dodawania na danych)
  wydajnosc   = FLOP / t                                         [FLOP/s]
  energia     E = P_nom * t                                      [J]   (t = czas obliczen: kernel / funkcja CPU)
  efektywnosc energetyczna = FLOP / E = wydajnosc / P_nom        [FLOP/J]
  CGMA (teoretyczny, poziom kodu zrodlowego) = liczba operacji zmiennoprzecinkowych wykonanych
       przez watki / liczba dostepow tych watkow do pamieci globalnej (odczyty + zapisy):
    kernel 0 (bez shared):   na watek K^2 odczytow + 1 zapis, K^2-1 dodawan
                             CGMA = (K^2-1) / (K^2+1)  ~ 1
    warianty 1 i 2:          na blok T^2 odczytow (kafel, kazdy element raz) + BS^2 zapisow,
                             BS^2 * (K^2-1) dodawan
                             CGMA = BS^2 (K^2-1) / (T^2 + BS^2),  T = BS+2R, K = 2R+1
    (wariant 2 ma ten sam CGMA co wariant 1 – rozni sie sposobem dostepu, nie liczba dostepow;
     liczba transakcji/sektorow jest ~8x wieksza, co pokazuje ncu i rysunki global_v2_*)
    CGMA liczony dla pelnego bloku (bloki brzegowe maja mniej wynikow – pomijalne dla N >> BS).
"""
import csv
import os
import sys
from collections import defaultdict

# Moc nominalna [W] i jej zrodlo – do sprawozdania (opis systemow, wzory energii).
P_NOM = {
    ("gtx1080", "cpu"): (65.0, "Intel Core i7-8700, TDP 65 W (Intel ARK)"),
    ("gtx1080", "gpu"): (180.0, "NVIDIA GeForce GTX 1080, Graphics Card Power 180 W (NVIDIA; nvidia-smi: limit 180 W)"),
    ("rtx4060", "cpu"): (54.0, "AMD Ryzen 7 7840HS, default TDP 35–54 W, przyjęto 54 W (AMD)"),
    ("rtx4060", "gpu"): (105.0, "NVIDIA GeForce RTX 4060 Laptop, TGP 35–115 W (NVIDIA); w tym laptopie "
                                "maks. limit 105 W (nvidia-smi -q -d POWER)"),
}
CARDS = [("gtx1080", "GTX 1080 + i7-8700"), ("rtx4060", "RTX 4060 Laptop + Ryzen 7 7840HS")]
VNAME = {"omp": "CPU (OpenMP)", "v1": "GPU wariant 1", "v2": "GPU wariant 2"}


def cgma(variant, BS, R):
    K2 = (2 * R + 1) ** 2
    if variant == "v0":
        return (K2 - 1) / (K2 + 1)
    T = BS + 2 * R
    return BS * BS * (K2 - 1) / (T * T + BS * BS)


def read_main(resdir, tag):
    path = os.path.join(resdir, f"{tag}_main.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def energy_rows(resdir):
    out = []
    for tag, _ in CARDS:
        for r in read_main(resdir, tag):
            dev = r["device"]
            P, _ = P_NOM[(tag, dev)]
            t = float(r["t_med_ms"]) * 1e-3
            g = float(r["gflops"])
            out.append({"tag": tag, "variant": r["variant"], "BS": int(r["BS"]), "N": int(r["N"]),
                        "R": int(r["R"]), "t": t, "gflops": g, "P": P, "E": P * t, "gflop_per_J": g / P})
    return out


def tables(rows):
    lines = ["## Moc nominalna przyjęta do szacowania energii\n",
             "| układ | P_nom [W] | źródło |", "|---|---:|---|"]
    for (tag, dev), (P, src) in P_NOM.items():
        lines.append(f"| {tag} / {dev.upper()} | {P:.0f} | {src} |")

    lines += ["\n## Teoretyczny współczynnik CGMA (FLOP / dostęp do pamięci globalnej)\n",
              "| R | BS | T=BS+2R | kernel 0 (tylko global) | warianty 1 i 2 (kafel w shared) |",
              "|---:|---:|---:|---:|---:|"]
    for R in (8, 36):
        for BS in (16, 32):
            lines.append(f"| {R} | {BS} | {BS + 2 * R} | {cgma('v0', BS, R):.2f} | {cgma('v1', BS, R):.1f} |")

    by = defaultdict(dict)
    for r in rows:
        by[(r["tag"], r["N"], r["R"], r["BS"])][r["variant"]] = r
    for tag, title in CARDS:
        keys = sorted(k for k in by if k[0] == tag)
        if not keys:
            continue
        lines += [f"\n## Energia – {title}\n",
                  "| N | R | BS | t CPU [s] | t v1 [s] | t v2 [s] | E CPU [J] | E v1 [J] | E v2 [J] "
                  "| CPU [GFLOP/J] | v1 [GFLOP/J] | v2 [GFLOP/J] | E CPU / E v1 |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k in keys:
            d = by[k]
            c, a, b = d["omp"], d["v1"], d["v2"]
            lines.append(f"| {k[1]} | {k[2]} | {k[3]} | {c['t']:.4f} | {a['t']:.4f} | {b['t']:.4f} "
                         f"| {c['E']:.2f} | {a['E']:.2f} | {b['E']:.2f} | {c['gflop_per_J']:.2f} "
                         f"| {a['gflop_per_J']:.2f} | {b['gflop_per_J']:.2f} | {c['E'] / a['E']:.2f}× |")
    return "\n".join(lines) + "\n"


def figure(rows, imgdir, N=8192):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b8b7b2"
    COLORS = {"omp": "#1baf7a", "v1": "#2a78d6", "v2": "#eb6834"}
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.titlesize": 10.5, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.2), sharey=True)
    for ax, (tag, title) in zip(axes, CARDS):
        d = defaultdict(dict)
        for r in rows:
            if r["tag"] == tag and r["N"] == N:
                d[(r["R"], r["BS"])][r["variant"]] = r
        groups = sorted(d)
        width, gap = 0.26, 0.02
        for si, key in enumerate(("omp", "v1", "v2")):
            xs = [gi + (si - 1) * (width + gap) for gi in range(len(groups))]
            vals = [d[g][key]["E"] for g in groups]
            P = P_NOM[(tag, "cpu" if key == "omp" else "gpu")][0]
            ax.bar(xs, vals, width=width, color=COLORS[key], edgecolor=SURFACE, linewidth=1,
                   label=f"{VNAME[key]} (P_nom = {P:.0f} W)")
            for x, v in zip(xs, vals):
                ax.text(x, v * 1.08, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="bottom",
                        fontsize=8.5, color=INK2)
        ax.set_yscale("log")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"R={R}, BS={BS}" for R, BS in groups])
        ax.set_ylabel("energia E = P_nom · t [J] (skala log)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, which="major")
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8.5)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 6)
    fig.suptitle(f"Szacunkowa energia obliczeń, N={N} (mniej = lepiej)", x=0.01, ha="left", color=INK,
                 fontsize=11.5)
    fig.tight_layout()
    os.makedirs(imgdir, exist_ok=True)
    path = os.path.join(imgdir, f"energia_N{N}.png")
    fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    resdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("projekt", "results")
    imgdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join("projekt", "images")
    rows = energy_rows(resdir)
    text = tables(rows)
    md = os.path.join(resdir, "energia_cgma.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("Zapisano:", md)
    for N in (4096, 8192):
        print("Zapisano:", figure(rows, imgdir, N))


if __name__ == "__main__":
    main()
