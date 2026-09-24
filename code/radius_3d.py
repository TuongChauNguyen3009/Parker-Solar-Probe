__author__ = "Chau Nguyen"

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, NullFormatter

ROOT = r"C:\SampleSPIS_chaunguyen_2026"
MORN = os.path.join(ROOT, "docs", "morning")
SET = sys.argv[1] if len(sys.argv) > 1 else "cap"
SETS = {"cap": [(0.8, ["WIRE3D_r08f", "WIRE3D_r08t"]), (1.6, ["WIRE3D_fast"]), (3.2, ["WIRE3D_r32f", "WIRE3D_r32t"]),
                (6.4, ["WIRE3D_r64t"]), (12.8, ["WIRE3D_r128t"]), (25.6, ["WIRE3D_r256t"]), (51.2, ["WIRE3D_r512t"])],
        "nocap": [(0.8, ["WIRE3D_r08n"]), (1.6, ["WIRE3D_noC_vr"]), (3.2, ["WIRE3D_r32n"]), (6.4, ["WIRE3D_r64n"]), (12.8, ["WIRE3D_r128n"])],
        "d16": [(0.8, ["WIRE3D_r08d16"]), (1.6, ["WIRE3D_r16d16"]), (3.2, ["WIRE3D_r32d16"]), (6.4, ["WIRE3D_r64d16"]), (12.8, ["WIRE3D_r128d16"])],
        "p20": [(0.8, ["WIRE3D_r08p20"]), (1.6, ["WIRE3D_r16p20"]), (3.2, ["WIRE3D_r32p20"]), (6.4, ["WIRE3D_r64p20"]), (12.8, ["WIRE3D_r128p20"]),
                (25.6, ["WIRE3D_r256p20"]), (51.2, ["WIRE3D_r512p20"])],
        "k": [(0.8, ["WIRE3D_k08t"]), (1.6, ["WIRE3D_k16t"]), (3.2, ["WIRE3D_k32t"]), (6.4, ["WIRE3D_k64t"]), (12.8, ["WIRE3D_k128t"])],
        "1au_n10": [(r, ["WIRE3D_1AU_r%sn10" % t]) for r, t in ((0.8, "08"), (1.6, "16"), (3.2, "32"), (6.4, "64"), (12.8, "128"), (25.6, "256"), (51.2, "512"))],
        "1au_c10": [(r, ["WIRE3D_1AU_r%sc10" % t]) for r, t in ((0.8, "08"), (1.6, "16"), (3.2, "32"), (6.4, "64"), (12.8, "128"), (25.6, "256"), (51.2, "512"))]}
T0 = 6e-4 if SET.startswith("1au") else 3e-5
TRUN = 2e-6
TSERIES = 1e-5
BODY, ANT = 0, 6
RADII = [0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2]
COL = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
MK = ["+", "x", "D", "o", "*", "v", "^"]
MS = {"+": 5.5, "x": 5.0, "D": 3.2, "o": 3.4, "*": 5.5, "v": 3.6, "^": 3.6}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 10, "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 3.4, "ytick.major.size": 3.4, "xtick.minor.size": 1.9, "ytick.minor.size": 1.9,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "axes.axisbelow": True, "grid.linestyle": "--", "grid.color": "0.6", "grid.linewidth": 0.6,
    "legend.fontsize": 9, "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "k",
    "legend.framealpha": 1.0, "legend.handlelength": 2.2, "legend.labelspacing": 0.35, "legend.borderpad": 0.4,
})


def monitor(proj, node):
    pat = os.path.join(ROOT, proj + ".spis5", "DefaultStudy", "Simulations", "Run1", "NumKernel", "Output",
                       "Average_surface_potential_of_node_%d_(V_,_s)__ElecNode%d_Potential.txt" % (node, node))
    hit = glob.glob(pat)
    if not hit:
        return None
    rows = [l.split(",") for l in open(hit[0]).read().splitlines()[1:] if l.strip()]
    return np.array([float(r[0]) for r in rows]), np.array([float(r[2]) for r in rows])


def read_pot(f, node):
    if not os.path.exists(f):
        return None
    t, v = [], []
    for ln in open(f).read().splitlines()[2:]:
        c = [x for x in ln.split(",") if x.strip()]
        try:
            w = [float(x) for x in c]
        except ValueError:
            continue
        if len(w) < 3:
            continue
        t.append(w[0]); v.append(w[1] if node == BODY else w[-1])
    return (np.array(t), np.array(v)) if len(t) > 2 else None


def project_pot(proj, node):
    return read_pot(os.path.join(ROOT, proj + ".spis5", "DefaultStudy", "Simulations", "Run1", "NumKernel", "Output", "potentials.txt"), node)


def interim(proj, node):
    return read_pot(os.path.join(MORN, proj + "_potentials_morning.txt"), node)


def final_copy(proj, node):
    return read_pot(os.path.join(ROOT, "docs", "potentials", proj + "_potentials.txt"), node)


def best_source(proj):
    for reader, status in ((project_pot, "done"), (monitor, "done"), (final_copy, "done"), (interim, "running")):
        a = reader(proj, ANT)
        if a is not None:
            return reader(proj, BODY), a, status
    return None


D, S, USED = {}, {}, []
for r, cands in SETS[SET]:
    got = None
    for proj in cands:
        src = best_source(proj)
        if src is not None and src[1][0].max() >= T0 + TRUN:
            got = (proj,) + src; break
    if got is None:
        for proj in cands:
            src = best_source(proj)
            if src is not None and src[1][0].max() >= TSERIES:
                got = (proj,) + src[:2] + ("early",); break
    if got is None:
        print("radius %.1f mm: no series yet" % r); continue
    proj, b, a, status = got
    for node, (t, v) in ((BODY, b), (ANT, a)):
        S[(r, node)] = (t, v)
        s = t >= T0
        if status != "early":
            D[(r, node)] = dict(mean=v[s].mean(), sd=v[s].std(ddof=1), se=v[s].std(ddof=1) / np.sqrt(s.sum()), n=int(s.sum()), t_end=t.max())
    USED.append((r, proj, status))
    if status != "early":
        print("%4.1f mm  %-14s %-8s t_end %.1f us  antenna %.2f +/- %.2f (sd)  body %.4f +/- %.4f" % (
            r, proj, status, a[0].max() * 1e6, D[(r, ANT)]["mean"], D[(r, ANT)]["sd"], D[(r, BODY)]["mean"], D[(r, BODY)]["sd"]))
    else:
        print("%4.1f mm  %-14s early    t_end %.1f us  (time series only, no plateau yet)" % (r, proj, a[0].max() * 1e6))
if not USED:
    raise SystemExit("nothing to plot for set " + SET)
RS = [r for r, _, _ in USED]
RP = np.array([r for r, _, st in USED if st != "early"])
TMAX = max(S[(r, ANT)][0].max() for r in RS) * 1e6
XMAX = max(80.0, np.ceil(TMAX / 10) * 10)

TR = 0.62 * max(1.0, (len(USED) + 1) / 6.0)
fig = plt.figure(figsize=(7.0, round(6.6 * (2.0 + TR) / 2.62, 3)))
gs = GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.0, TR], left=0.088, right=0.958, top=0.985, bottom=0.02, wspace=0.9, hspace=0.36)
ax = np.array([[fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4])], [fig.add_subplot(gs[1, 1:3]), None]])
for col, (node, ylab) in enumerate([(ANT, "Antenna Potential (V)"), (BODY, "Spacecraft Potential (V)")]):
    a = ax[0, col]
    vmax = max(S[(r, node)][1].max() for r in RS)
    for r in RS:
        i = RADII.index(r)
        t, v = S[(r, node)]
        a.plot(t * 1e6, v, color=COL[i], lw=1.1, marker=MK[i], ms=MS[MK[i]], mew=0.9, markevery=max(1, len(t) // 16), label="%.1f mm" % r)
    a.set_xlim(0, XMAX); a.set_ylim(0, np.ceil(vmax * 1.08))
    a.set_xlabel(r"Time ($\mu$s)", labelpad=2)
    a.set_ylabel(ylab)
ax[0, 0].legend(title="Wire Radius", loc="lower right" if SET.startswith("1au") else "upper right",
                ncol=2 if len(RS) > 3 else 1, columnspacing=1.0, title_fontsize=9)

a = ax[1, 0]
if len(RP):
    m = np.array([D[(r, ANT)]["mean"] for r in RP])
    a.plot(RP, m, "-s", color=COL[0], lw=1.2, ms=6.0, mfc=COL[0], mec=COL[0], zorder=3, clip_on=False)
RSHOW = [r for r in RADII if r <= max(12.8, max(RS))]
a.set_xscale("log"); a.set_xlim(0.62, RSHOW[-1] * 1.29)
a.xaxis.set_major_locator(FixedLocator(RSHOW)); a.xaxis.set_minor_formatter(NullFormatter())
a.set_xticklabels(["%g" % v for v in RSHOW])
a.set_xlabel("Wire Radius (mm)", labelpad=2); a.set_ylabel("Antenna Potential (V)")
if SET.startswith("1au") and len(RP):
    a.set_ylim(np.floor(m.min() - 0.5), np.ceil(m.max() + 0.5))
else:
    a.set_ylim(min(5, np.floor(m.min() - 0.3)) if len(RP) else 5, 10)

rows = []
for r, _, st in USED:
    if st == "early":
        rows.append(["%.1f" % r, "-", "-"])
    else:
        rows.append(["%.1f" % r,
                     "%.2f  $\\pm$  %.2f" % (D[(r, ANT)]["mean"], D[(r, ANT)]["sd"]),
                     "%.4f  $\\pm$  %.4f" % (D[(r, BODY)]["mean"], D[(r, BODY)]["sd"])])
axt = fig.add_subplot(gs[2, :]); axt.axis("off"); axt.grid(False); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
HEAD = ["Wire Radius (mm)", "Antenna Potential (V)", "Spacecraft Potential (V)"]
XB = [0.0, 0.275, 0.635, 1.0]
NR = len(rows) + 1
RH = min(0.178, 0.84 / NR)
YT = 0.955
YB = [YT - i * RH for i in range(NR + 1)]
CX = [(XB[i] + XB[i + 1]) / 2 for i in range(3)]
for i, y in enumerate(YB):
    axt.plot([XB[0], XB[-1]], [y, y], color="k", lw=1.0 if i in (0, 1, NR) else 0.55, solid_capstyle="butt", clip_on=False, zorder=4)
for x in XB:
    axt.plot([x, x], [YB[0], YB[-1]], color="k", lw=0.55, solid_capstyle="butt", clip_on=False, zorder=4)
for cx, lab in zip(CX, HEAD):
    axt.text(cx, (YB[0] + YB[1]) / 2, lab, ha="center", va="center", fontsize=9.5)
for i, row in enumerate(rows):
    yc = (YB[i + 1] + YB[i + 2]) / 2
    for cx, val in zip(CX, row):
        axt.text(cx, yc, val, ha="center", va="center", fontsize=9.5)

fig.canvas.draw(); rd = fig.canvas.get_renderer()
for a in [x for x in ax.ravel() if x is not None]:
    for axis in (a.xaxis, a.yaxis):
        bbs = [tt.get_window_extent(rd) for tt in axis.get_ticklabels() if tt.get_text()]
        assert not any(u.overlaps(w) for i, u in enumerate(bbs) for w in bbs[i + 1:]), "tick labels overlap"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radius_scan3d_" + SET)
fig.savefig(out + ".png", dpi=400)
fig.savefig(out + ".pdf")
with open(out + ".csv", "w") as f:
    f.write("wire_radius_mm,run,status,t_end_s,antenna_mean_V,antenna_sd_V,antenna_se_V,spacecraft_mean_V,spacecraft_sd_V,spacecraft_se_V,n_samples\n")
    for r, proj, st in USED:
        if st == "early":
            f.write("%.1f,%s,%s,%.3e,,,,,,,\n" % (r, proj, st, S[(r, ANT)][0].max())); continue
        A, B = D[(r, ANT)], D[(r, BODY)]
        f.write("%.1f,%s,%s,%.3e,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n" % (r, proj, st, A["t_end"], A["mean"], A["sd"], A["se"], B["mean"], B["sd"], B["se"], A["n"]))
print("da ghi radius_scan3d_%s.png / .pdf / .csv" % SET)
