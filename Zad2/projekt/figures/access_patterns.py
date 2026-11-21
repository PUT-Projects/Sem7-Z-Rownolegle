#!/usr/bin/env python3
"""Schematy dostepu do pamieci globalnej i wspoldzielonej dla kerneli z kernels.cuh.

Uzycie (z katalogu Zad2):  python projekt/figures/access_patterns.py [katalog_wyjsciowy]
Domyslnie zapisuje PNG do projekt/images/.

Rysunki maja fizyczna szerokosc ok. 17 cm (szerokosc tekstu na stronie A4), wiec po
wstawieniu do sprawozdania w 100% szerokosci czcionki zachowuja podany rozmiar.

Indeksowanie jest odtworzone 1:1 z kernels.cuh (blok (0,0), i0 = j0 = 0):
  watek t = ty*BS + tx, warp w = t // 32, lane = t % 32
  ladowanie, iteracja k: idx = t + k*BS*BS
    wariant 1: r = idx // T, c = idx % T   -> TAB[r][c],  tile[r*P1 + c]
    wariant 2: c = idx // T, r = idx % T   -> TAB[r][c],  tileT[c*P2 + r]
  obliczenia, krok (u,v):
    wariant 1: tile [(ty+u)*P1 + tx+v]
    wariant 2: tileT[(tx+v)*P2 + ty+u]
Sektor pamieci globalnej = 32 B (8 floatow), linia = 128 B; zakladamy N podzielne przez 32
(jak w eksperymentach: 4096, 8192), wiec poczatek kazdego wiersza TAB jest wyrownany.
Bank pamieci wspoldzielonej = (indeks slowa) % 32.

Skrypt liczy tez (dla calego bloku, wszystkich iteracji i krokow u,v) srednia liczbe
sektorow na zadanie warpa i sredni stopien konfliktu – do porownania z ncu.
"""
import os
import sys
import textwrap
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- kolory (paleta referencyjna dataviz, tryb jasny) -------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9993"
GRID = "#d9d8d3"
BLUE = "#2a78d6"     # warp 0 / brak konfliktu
ORANGE = "#eb6834"   # warp 1 / konflikt
PAD = "#ecebe6"      # dopelnienie wiersza (padding)

SMEM_LIMIT = 48 * 1024
N_ASSUMED = 4096   # do adresow globalnych (wazne tylko N % 32 == 0)
MAX_COLS = 48      # tyle komorek miesci sie czytelnie w szerokosci strony
CELL = 0.142       # cala na komorke (48 komorek ~ 6.8 cala ~ 17 cm)
DPI = 250

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.labelsize": 9,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.titlesize": 9.5, "axes.titlecolor": INK,
})


# --- odtworzenie kernels.cuh ----------------------------------------------------------
def pad_to(T, want):
    return T + ((want - T % 32) + 32) % 32


def pitch_v1(BS, T):
    return T if BS == 32 else pad_to(T, 16)


def pitch_v2(T):
    return pad_to(T, 0) if T * pad_to(T, 0) * 4 <= SMEM_LIMIT else pad_to(T, 16)


def dims(BS, R, variant):
    T = BS + 2 * R
    P = pitch_v1(BS, T) if variant == 1 else pitch_v2(T)
    return T, P


def load_rc(variant, idx, T):
    if variant == 1:
        return idx // T, idx % T
    return idx % T, idx // T  # (r, c)


def warp_load(variant, BS, T, w, k):
    """Lista (lane, r, c) dla warpa w w iteracji k petli ladowania (tylko idx < T*T)."""
    out = []
    for lane in range(32):
        idx = 32 * w + lane + k * BS * BS
        if idx < T * T:
            r, c = load_rc(variant, idx, T)
            out.append((lane, r, c))
    return out


def sectors(acc, N=N_ASSUMED):
    return len({(r * N + c) * 4 // 32 for _, r, c in acc})


def warp_compute_addr(variant, BS, P, w, u, v):
    """Lista (lane, adres_slowa_shared) dla warpa w w kroku (u,v)."""
    out = []
    for lane in range(32):
        t = 32 * w + lane
        tx, ty = t % BS, t // BS
        a = (ty + u) * P + tx + v if variant == 1 else (tx + v) * P + ty + u
        out.append((lane, a))
    return out


def conflict_degree(addrs):
    per_bank = defaultdict(set)
    for a in addrs:
        per_bank[a % 32].add(a)  # ten sam adres = rozgloszenie (broadcast), nie konflikt
    return max(len(s) for s in per_bank.values())


def block_stats(variant, BS, R):
    """Srednia liczba sektorow/zadanie (ladowanie) i stopien konfliktu (obliczenia)."""
    T, P = dims(BS, R, variant)
    nwarps = BS * BS // 32
    iters = -(-T * T // (BS * BS))
    sec, req = 0, 0
    for k in range(iters):
        for w in range(nwarps):
            acc = warp_load(variant, BS, T, w, k)
            if acc:
                sec += sectors(acc)
                req += 1
    K = 2 * R + 1
    deg, n = 0, 0
    for w in range(nwarps):
        for u in range(K):
            for v in range(K):
                deg += conflict_degree([a for _, a in warp_compute_addr(variant, BS, P, w, u, v)])
                n += 1
    return sec / req, deg / n


# --- rysowanie --------------------------------------------------------------------------
HATCH = "#7d7c76"    # kolor kreskowania dopelnienia (wyraznie ciemniejszy od siatki)
plt.rcParams["hatch.linewidth"] = 0.7


def cell(ax, x, y, fc, ec=GRID, lw=0.5, hatch=None):
    kw = {"hatchcolor": HATCH} if hatch else {}
    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=fc, edgecolor=ec, linewidth=lw, hatch=hatch, **kw))


def label(ax, x, y, s, color="white", size=6.3):
    ax.text(x + 0.5, y + 0.52, s, ha="center", va="center", color=color, fontsize=size)


def setup_grid(ax, nx, ny, xlabel, ylabel, xstep=8, ystep=1):
    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x + 0.5 for x in range(0, nx, xstep)])
    ax.set_xticklabels([str(x) for x in range(0, nx, xstep)])
    ax.set_yticks([y + 0.5 for y in range(0, ny, ystep)])
    ax.set_yticklabels([str(y) for y in range(0, ny, ystep)])
    ax.tick_params(length=0, pad=2)


def wrap(s, width=110):
    return "\n".join(textwrap.wrap(s, width))


def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


# Uklad w calach: kazdy panel dostaje dokladnie nx*CELL x ny*CELL (komorki kwadratowe),
# a naglowek, tytuly paneli, opisy osi i notka maja zarezerwowane miejsce – nic nie zachodzi.
LEFT, RIGHT = 0.55, 0.15          # opis osi y + etykiety / prawy margines
PANEL_TITLE = 0.26                # tytul panelu nad osiami
XAXIS = 0.48                      # etykiety osi x + opis osi x pod osiami
HGAP = 0.75                       # odstep miedzy panelami obok siebie (miesci opis osi y)
LINE = 0.17                       # wysokosc linii tekstu 9 pt


LEG_ROW = 0.2      # wysokosc wiersza legendy
LEG_BOX = 0.13     # bok kwadracika koloru


def legend_rows(legend, W):
    """Rozklada wpisy legendy na wiersze o szerokosci W (cale)."""
    rows, cur, used = [], [], 0.0
    for fc, hatch, text in legend:
        w = LEG_BOX + 0.08 + 0.062 * len(text) + 0.3     # ~szerokosc tekstu 8 pt
        if cur and used + w > W - 0.1:
            rows.append(cur)
            cur, used = [], 0.0
        cur.append((fc, hatch, text, w))
        used += w
    if cur:
        rows.append(cur)
    return rows


def make_layout(panels, header, note, arrange="col", note_width=None, legend=None):
    """panels: lista (nx, ny); header: lista linii naglowka; legend: lista (kolor, kreskowanie,
    opis) rysowana pod naglowkiem; zwraca (fig, [ax, ...])."""
    if arrange == "col":
        W = LEFT + max(nx for nx, _ in panels) * CELL + RIGHT
    else:
        W = LEFT + sum(nx * CELL for nx, _ in panels) + HGAP * (len(panels) - 1) + RIGHT
    leg = legend_rows(legend or [], W)
    head_h = LINE * len(header) + 0.08 + LEG_ROW * len(leg) + (0.06 if leg else 0)
    note_width = note_width or max(60, int(W * 15))   # ~15 znakow 7.5 pt na cal szerokosci
    note_lines = wrap(note, note_width).split("\n") if note else []
    note_h = 0.15 * len(note_lines)
    if arrange == "col":
        H = head_h + sum(PANEL_TITLE + ny * CELL + XAXIS for _, ny in panels) + note_h
    else:
        H = head_h + PANEL_TITLE + max(ny for _, ny in panels) * CELL + XAXIS + note_h
    fig = plt.figure(figsize=(W, H))
    axes = []
    x, ytop = LEFT, H - head_h
    for nx, ny in panels:
        w, h = nx * CELL, ny * CELL
        y0 = ytop - PANEL_TITLE - h
        axes.append(fig.add_axes([x / W, y0 / H, w / W, h / H]))
        if arrange == "col":
            ytop = y0 - XAXIS
        else:
            x += w + HGAP
    for i, line in enumerate(header):
        fig.text(0.02 / W, (H - 0.05 - LINE * i) / H, line, ha="left", va="top", fontsize=9.5, color=INK)
    y = H - 0.05 - LINE * len(header) - 0.08
    for row in leg:
        x = 0.02
        for fc, hatch, text, w in row:
            kw = {"hatchcolor": HATCH} if hatch else {}
            fig.add_artist(Rectangle((x / W, (y - LEG_BOX) / H), LEG_BOX / W, LEG_BOX / H, facecolor=fc,
                                     edgecolor=MUTED, linewidth=0.5, hatch=hatch, transform=fig.transFigure, **kw))
            fig.text((x + LEG_BOX + 0.07) / W, (y - LEG_BOX / 2) / H, text, ha="left", va="center",
                     fontsize=8, color=INK)
            x += w
        y -= LEG_ROW
    if note_lines:
        fig.text(0.02 / W, (note_h + 0.02) / H, "\n".join(note_lines), ha="left", va="top", fontsize=7.5,
                 color=INK2, linespacing=1.25)
    return fig, axes


def fig_global(variant, BS, R, outdir):
    """Ladowanie kafla z pamieci globalnej: warp 0 i 1, iteracja k=0 (blok (0,0))."""
    T, _ = dims(BS, R, variant)
    w0, w1 = warp_load(variant, BS, T, 0, 0), warp_load(variant, BS, T, 1, 0)
    # Wariant 1: os pozioma = kolumna c (kolejne adresy w obrebie wiersza TAB).
    # Wariant 2: os pozioma = wiersz r (kazdy krok = +N*4 B), os pionowa = kolumna c.
    if variant == 1:
        pts = [(lane, c, r, col) for col, w in ((BLUE, w0), (ORANGE, w1)) for lane, r, c in w]
        ny = max(r for _, r, _ in w0 + w1) + 2
        xlabel = "kolumna c kafla (kolejne adresy co 4 B)"
        ylabel = "wiersz r"
    else:
        pts = [(lane, r, c, col) for col, w in ((BLUE, w0), (ORANGE, w1)) for lane, r, c in w]
        ny = max(c for _, _, c in w0 + w1) + 2
        xlabel = f"wiersz r kafla (kolejne adresy co N·4 B = {N_ASSUMED * 4 // 1024} KB dla N={N_ASSUMED})"
        ylabel = "kolumna c"
    nx = min(T, MAX_COLS)
    s0, s1 = sectors(w0), sectors(w1)
    avg_sec, _ = block_stats(variant, BS, R)
    name = "wariant 1 (dostęp łączony)" if variant == 1 else "wariant 2 (dostęp niełączony)"
    shown = "" if T <= MAX_COLS else f" Pokazano {MAX_COLS} z {T} " + ("kolumn kafla." if variant == 1 else "wierszy kafla.")
    note = ("Pionowe linie: granice sektorów 32 B (grube: linie 128 B)." if variant == 1 else
            "Pionowe linie: każdy wątek czyta inny wiersz TAB, czyli osobny sektor 32 B.")
    fig, (ax,) = make_layout(
        [(nx, ny)],
        [f"Pamięć globalna, {name}: BS={BS}, R={R}, kafel {T}×{T}, iteracja k=0",
         f"jedna instrukcja odczytu: warp 0 → {s0} sektorów 32 B, warp 1 → {s1} sektorów",
         f"średnio w całym bloku: {avg_sec:.2f} sektora na żądanie warpa (ideał: 4)"],
        "Liczba w komórce = lane (nr wątku w warpie, 0–31). " + note + " Kolejna iteracja k: idx += BS·BS." + shown,
        legend=[(BLUE, None, "warp 0 (wątki bloku t = 0–31) – element TAB, który pobiera"),
                (ORANGE, None, "warp 1 (wątki t = 32–63) – element TAB, który pobiera"),
                (SURFACE, None, "element kafla pobierany przez inne warpy lub w kolejnych iteracjach")])
    setup_grid(ax, nx, ny, xlabel, ylabel)
    for y in range(ny):
        for x in range(nx):
            cell(ax, x, y, SURFACE)
    for lane, x, y, col in pts:
        if x < nx and y < ny:
            cell(ax, x, y, col, ec=SURFACE)
            label(ax, x, y, str(lane))
    if variant == 1:  # granice sektorow 32 B (co 8 floatow), grubsze = linie 128 B
        for x in range(0, nx + 1, 8):
            ax.axvline(x, color=INK, linewidth=1.6 if x % 32 == 0 else 0.8, alpha=0.6)
    else:
        for x in range(0, nx + 1):
            ax.axvline(x, color=INK, linewidth=0.6, alpha=0.45)
    return save(fig, os.path.join(outdir, f"global_v{variant}_BS{BS}_R{R}.png"))


def shown_columns(variant, T, P, acc):
    """Kolumny wiersza shared do pokazania. Gdy wiersz (P slow) jest za dlugi, pokazujemy
    poczatek (z odczytami warpa), przerwe None ('…'), koniec kafla i dopelnienie (c >= T)."""
    limit = MAX_COLS if variant == 1 else 12
    if P <= limit:
        return list(range(P))
    used = max(a % P for _, a in acc) + 1             # kolumny z odczytami warpa 0
    head = max(used + (4 if variant == 1 else 2), 4)
    tail = list(range(T - 2, T)) + list(range(T, min(P, T + (8 if variant == 1 else 3))))
    return list(range(head)) + [None] + tail


def layout_size(variant, T, P, acc):
    ny = max(a // P for _, a in acc) + 2
    return len(shown_columns(variant, T, P, acc)), ny


def draw_layout(ax, variant, T, P, acc, deg):
    """Fragment pamieci wspoldzielonej: wiersze po P slow, nr banku w komorkach."""
    read = {a: lane for lane, a in acc}
    cols = shown_columns(variant, T, P, acc)
    nx, ny = len(cols), layout_size(variant, T, P, acc)[1]
    setup_grid(ax, nx, ny, "słowo w wierszu pamięci współdzielonej",
               "wiersz ty+u" if variant == 1 else "wiersz tx+v", ystep=1 if ny <= 8 else 4)
    # etykiety osi x = rzeczywiste numery slow (co 8 na poczatku + poczatek/koniec dopelnienia)
    ticks = [(x, c) for x, c in enumerate(cols) if c is not None and (c % 8 == 0 or c in (T - 1, T, P - 1))]
    if variant == 2:
        ticks = [(x, c) for x, c in enumerate(cols) if c is not None and (c in (0, T - 1, T) or c % 4 == 0 and c < T - 2)]
    ax.set_xticks([x + 0.5 for x, _ in ticks])
    ax.set_xticklabels([str(c) for _, c in ticks], fontsize=7)
    for r in range(ny):
        for x, c in enumerate(cols):
            if c is None:
                label(ax, x, r, "…", color=INK2, size=8)
                continue
            a = r * P + c
            if a in read:
                cell(ax, x, r, ORANGE if deg > 1 else BLUE, ec=SURFACE)
                label(ax, x, r, str(read[a]))
            else:
                pad = c >= T
                cell(ax, x, r, PAD if pad else SURFACE, hatch="///" if pad else None)
                label(ax, x, r, str(a % 32), color=INK2 if pad else MUTED, size=5.6)
    return nx, ny


def draw_banks(ax, acc, deg):
    """Banki (os x) x przebiegi (os y); w komorce lane(y) obslugiwane w danym przebiegu."""
    per_bank = defaultdict(list)
    for lane, a in acc:
        per_bank[a % 32].append((a, lane))
    setup_grid(ax, 32, deg, "numer banku", "przebieg", xstep=1, ystep=1 if deg <= 8 else 4)
    ax.set_xticklabels([str(b) for b in range(32)], fontsize=5.6)
    ax.set_yticklabels([str(p + 1) for p in range(0, deg, 1 if deg <= 8 else 4)])
    for b in range(32):
        addrs = sorted({a for a, _ in per_bank[b]})
        for p in range(deg):
            if p < len(addrs):
                lanes = [l for a, l in per_bank[b] if a == addrs[p]]
                cell(ax, b, p, ORANGE if deg > 1 else BLUE, ec=SURFACE)
                label(ax, b, p, ",".join(map(str, lanes)), size=6.3 if len(lanes) == 1 else 4.5)
            else:
                cell(ax, b, p, SURFACE)


def fig_shared(variant, BS, R, outdir):
    """Odczyt z pamieci wspoldzielonej przy obliczeniach: warp 0, krok (u,v)=(0,0)."""
    T, P = dims(BS, R, variant)
    acc = warp_compute_addr(variant, BS, P, 0, 0, 0)
    deg = conflict_degree([a for _, a in acc])
    _, avg_deg = block_stats(variant, BS, R)
    size_a = layout_size(variant, T, P, acc)
    expr = "tile[(ty+u)·P + tx+v]" if variant == 1 else "tileT[(tx+v)·P + ty+u]"
    name = "wariant 1 (bez konfliktów)" if variant == 1 else "wariant 2 (konflikty banków)"
    # wariant 1: panel a szeroki i niski -> panele jeden pod drugim; wariant 2: waski i wysoki -> obok siebie
    fig, (ax1, ax2) = make_layout(
        [size_a, (32, deg)],
        [f"Pamięć współdzielona, {name}: {expr}",
         f"BS={BS}, R={R}, warp 0, krok (u,v)=(0,0); średni stopień konfliktu w bloku: {avg_deg:.2f}"],
        "a) liczba w kolorowej komórce = lane wątku, który czyta to słowo; liczba w białej/kreskowanej komórce "
        "= numer banku słowa (adres % 32); „…” = pominięty środek wiersza. b) kolumna = bank, wiersz = kolejny "
        "przebieg (cykl) obsługi; w komórce lane obsłużone w tym przebiegu. Kolejne kroki (u,v) przesuwają "
        "wszystkie adresy o tę samą wartość – rozkład na banki się nie zmienia.",
        arrange="col" if variant == 1 else "row",
        legend=([(BLUE, None, "słowo czytane przez warp 0 – każdy wątek w innym banku (bez konfliktu)")]
                if deg == 1 else
                [(ORANGE, None, f"słowo czytane przez warp 0 – kilka wątków w tym samym banku "
                                f"(konflikt, {deg} kolejnych przebiegów)")]) +
               [(SURFACE, None, "słowo kafla nieczytane w tym kroku"),
                (PAD, "///", "dopełnienie wiersza (P − T słów, nieużywane)")])
    draw_layout(ax1, variant, T, P, acc, deg)
    draw_banks(ax2, acc, deg)
    if variant == 1:
        ta = f"a) układ: wiersz = P = {P} słów" + (f" (T={T} + {P - T} dopełnienia)" if P > T else " (= T)")
    else:
        ta = f"a) P = {P} słów" + (f" (T={T}+{P - T})" if P > T else "")
    ax1.set_title(ta, loc="left", fontsize=8.5, pad=4)
    ax2.set_title("b) warp 0 w bankach: " + ("1 przebieg – brak konfliktów" if deg == 1
                                             else f"{deg} przebiegów – konflikt {deg}-krotny"),
                  loc="left", fontsize=8.5, pad=4)
    return save(fig, os.path.join(outdir, f"shared_v{variant}_BS{BS}_R{R}.png"))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("projekt", "images")
    os.makedirs(outdir, exist_ok=True)
    print("| wariant | BS | R | T | P | sektory/zadanie (ladowanie) | stopien konfliktu (obliczenia) |")
    print("|---|---|---|---|---|---|---|")
    for variant in (1, 2):
        for BS in (16, 32):
            for R in (8, 36):
                T, P = dims(BS, R, variant)
                s, d = block_stats(variant, BS, R)
                print(f"| v{variant} | {BS} | {R} | {T} | {P} | {s:.2f} | {d:.2f} |")
                fig_global(variant, BS, R, outdir)
                fig_shared(variant, BS, R, outdir)
    print(f"Zapisano rysunki do {outdir}/")


if __name__ == "__main__":
    main()
