#!/usr/bin/env python3
"""Analiza serii N_stab (tylko biblioteka standardowa).

Uzycie:  python projekt/analyze_nstab.py projekt/results/<tag>_nstab.csv [prog_procent]

Dla kazdego R tworzy tabele: N, M, liczba blokow, czas (mediana/min), GFLOP/s,
% maksimum serii, przyrost wzgledem poprzedniego N, i wyznacza N_stab:

  N_stab = najmniejsze N z serii, dla ktorego zadne wieksze N nie daje wydajnosci
           wiekszej o wiecej niz prog (domyslnie 5%) – tzn. dalsze zwiekszanie N
           nie powoduje istotnego wzrostu predkosci obliczen.

Tabele (Markdown) sa wypisywane i zapisywane obok CSV jako <tag>_nstab.md.
"""
import csv
import math
import sys
from collections import defaultdict


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # konsola Windows (cp1250) nie zna np. strzalek
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    thr = float(sys.argv[2]) / 100.0 if len(sys.argv) > 2 else 0.05

    series = defaultdict(list)
    tag = ""
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tag = row["tag"]
            series[(int(row["BS"]), int(row["R"]), row["variant"])].append(row)

    out = []
    for (bs, r, var), rows in sorted(series.items()):
        rows.sort(key=lambda x: int(x["N"]))
        g = [float(x["gflops"]) for x in rows]
        ns = [int(x["N"]) for x in rows]
        gmax = max(g)
        # N_stab: najmniejszy indeks k, dla ktorego max(g[k+1:]) <= g[k]*(1+thr)
        k_stab = next(k for k in range(len(g)) if all(x <= g[k] * (1 + thr) for x in g[k + 1:]))
        n_stab = ns[k_stab]

        out.append(f"### {tag}: wariant {var}, BS={bs}, R={r} (N_stab = {n_stab}, prog {thr*100:.0f}%)\n")
        out.append("| N | M=N-2R | bloki | t_med [ms] | t_min [ms] | GFLOP/s | % max | przyrost vs poprz. N |")
        out.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        prev = None
        for x, gi, n in zip(rows, g, ns):
            m = n - 2 * r
            blocks = math.ceil(m / bs) ** 2
            inc = "–" if prev is None else f"{(gi / prev - 1) * 100:+.1f}%"
            mark = " **←N_stab**" if n == n_stab else ""
            out.append(f"| {n}{mark} | {m} | {blocks} | {float(x['t_med_ms']):.4f} | {float(x['t_min_ms']):.4f} "
                       f"| {gi:.1f} | {gi / gmax * 100:.1f} | {inc} |")
            prev = gi
        out.append("")

    text = "\n".join(out)
    print(text)
    md = path[:-4] + ".md" if path.endswith(".csv") else path + ".md"
    with open(md, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Zapisano: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
