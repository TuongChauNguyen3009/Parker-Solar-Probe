__author__ = "Chau Nguyen"

import numpy as np, glob, os, re, sys
from scipy.io import netcdf_file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.tri as mtri

_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def load_msh(path):
    L = open(path, encoding="utf-8", errors="ignore").read().split("\n")
    i = L.index("$Nodes")
    nn = int(L[i+1])
    P = np.zeros((nn+1, 3))
    for k in range(nn):
        p = L[i+2+k].split()
        P[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
    i = L.index("$Elements")
    ne = int(L[i+1])
    T = []
    for k in range(ne):
        p = L[i+2+k].split()
        if p[1] == "4":
            nt = int(p[2])
            T.append([int(x) for x in p[3+nt:3+nt+4]])
    return P, np.array(T)


def field_on_nodes(P, mask, phi):
    V = np.full(len(P), np.nan)
    V[mask + 1] = phi
    return V


def slice_x0(P, T, V, x0=0.0):
    x = P[T, 0] - x0
    x[x == 0.0] = 1e-12
    keep = (x.min(axis=1) < 0) & (x.max(axis=1) > 0)
    Tk, xk = T[keep], x[keep]
    ys, zs, vs, tris = [], [], [], []
    for tet, xs in zip(Tk, xk):
        pts = []
        for a, b in _EDGES:
            if (xs[a] < 0) != (xs[b] < 0):
                t = xs[a] / (xs[a] - xs[b])
                p = P[tet[a]] + t*(P[tet[b]] - P[tet[a]])
                v = V[tet[a]] + t*(V[tet[b]] - V[tet[a]])
                pts.append((p[1], p[2], v))
        if len(pts) < 3 or any(np.isnan(q[2]) for q in pts):
            continue
        if len(pts) == 4:
            q = np.array(pts)
            ang = np.arctan2(q[:, 1] - q[:, 1].mean(), q[:, 0] - q[:, 0].mean())
            q = q[np.argsort(ang)]
            polys = [q[[0, 1, 2]], q[[0, 2, 3]]]
        else:
            polys = [np.array(pts)]
        for q in polys:
            area = 0.5*abs((q[1, 0]-q[0, 0])*(q[2, 1]-q[0, 1])
                           - (q[2, 0]-q[0, 0])*(q[1, 1]-q[0, 1]))
            if area < 1e-10:
                continue
            k = len(ys)
            ys.extend(q[:, 0]); zs.extend(q[:, 1]); vs.extend(q[:, 2])
            tris.append([k, k+1, k+2])
    ys, zs, vs, tris = np.array(ys), np.array(zs), np.array(vs), np.array(tris)
    pts = np.round(np.column_stack([ys, zs]), 9)
    uniq, inv = np.unique(pts, axis=0, return_inverse=True)
    vals = np.zeros(len(uniq))
    vals[inv] = vs
    tris = inv[tris]
    good = ((tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2])
            & (tris[:, 0] != tris[:, 2]))
    tris = tris[good]
    a = uniq[tris[:, 0]]; b = uniq[tris[:, 1]]; c = uniq[tris[:, 2]]
    area = 0.5*np.abs((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1])
                      - (c[:, 0]-a[:, 0])*(b[:, 1]-a[:, 1]))
    tris = tris[area > 1e-12]
    tris = np.unique(np.sort(tris, axis=1), axis=0)
    tri = mtri.Triangulation(uniq[:, 0], uniq[:, 1], tris)
    return tri, vals


def cut_on_slice(tri, vs, pts_y, pts_z):
    if tri.mask is None:
        tri.set_mask(mtri.TriAnalyzer(tri).get_flat_tri_mask(0.001))
    return mtri.LinearTriInterpolator(tri, vs)(pts_y, pts_z).filled(np.nan)


CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "paraview_rainbow", [(0.0, (0, 0, 1)), (0.25, (0, 1, 1)), (0.5, (0, 1, 0)),
                         (0.75, (1, 1, 0)), (1.0, (1, 0, 0))])
for _a in sys.argv:
    if _a.startswith("--cmap="):
        CMAP = matplotlib.colormaps[_a.split("=", 1)[1]]
BODY_FC, BODY_EC = "0.62", "0.12"
VIEW_FROM_PLUS_X = True
RMAP = None
Z_WIRE, Z_BUS = 0.213, 1.70

args = [a for a in sys.argv[1:] if not a.startswith("--")]
LOG10 = "--log10" in sys.argv
WIDE = "--wide" in sys.argv
WIRE_POT = None
for a in sys.argv:
    if a.startswith("--wire-pot="):
        WIRE_POT = float(a.split("=", 1)[1])
VRANGE = None
for a in sys.argv:
    if a.startswith("--vrange="):
        VRANGE = tuple(float(x) for x in a.split("=", 1)[1].split(":"))
NWIN = None
for a in sys.argv:
    if a.startswith("--nwin="):
        NWIN = tuple(float(x) for x in a.split("=", 1)[1].split(":"))
ONLY = None
for a in sys.argv:
    if a.startswith("--only="):
        ONLY = [t.strip() for t in a.split("=", 1)[1].split(",") if t.strip()]
SYMLOG = "--symlog" in sys.argv
CLIP = "--full-range" not in sys.argv
GREY = "--grey-bands" in sys.argv
CLIP_POT = "--clip-potential" in sys.argv
PCT = 1.0
for a in sys.argv:
    if a.startswith("--pct="):
        PCT = float(a.split("=", 1)[1])
if not args:
    raise SystemExit("usage: python plot_spis.py <project.spis5> [--full-range] [--log10] [--snapshot] [--plateau] "
                     "[--prefix NAME] [--zoom METRES] [--clip-potential] [--pct=N] [--grey-bands] [--cmap=NAME]")
PROJ = args[0]
if not os.path.isdir(PROJ):
    raise SystemExit("no such project: " + PROJ)
PREFIX = (sys.argv[sys.argv.index("--prefix")+1] if "--prefix" in sys.argv
          else os.path.basename(PROJ.rstrip("\\/")).replace(".spis5", ""))
G = os.path.join(PROJ, "DefaultStudy")
D = os.path.join(G, "Simulations", "Run1", "OutputFolder", "DataFieldExtracted")
MON = os.path.join(G, "Simulations", "Run1", "OutputFolder", "DataFieldMonitored")


def gp(key, default=None):
    f = glob.glob(os.path.join(G, "Simulations", "Run1", "GlobalParameters", "globalParameters.xml"))
    if not f:
        return default
    s = open(f[0], encoding="utf-8", errors="ignore").read()
    m = re.search(r"<keyName>%s</keyName>(.{0,900}?)</GlobalParameter>" % re.escape(key), s, re.S)
    if not m:
        return default
    t = re.search(r"<typeAString>([^<]*)<", m.group(1))
    t = t.group(1) if t else "double"
    tag = {"double": "valueAsDouble", "int": "valueAsInt", "String": "valueAsString"}.get(t, "valueAsDouble")
    v = re.search(r"<%s>([^<]*)<" % tag, m.group(1))
    return v.group(1) if v else default


Te = float(gp("electronTemperature", 8.14))
Ti = float(gp("ionTemperature", 8.0))
n0 = float(gp("electronDensity", 6.93e6))
LD = 7430.0*np.sqrt(Te/n0)
METRES = "--metres" in sys.argv
LS = 1.0 if METRES else LD
UNIT = "m" if METRES else "$\\lambda_D$"
print("%s\n  Te=%g eV  Ti=%g eV  n=%.3g m^-3  ->  lambda_D=%.3f m" % (PREFIX, Te, Ti, n0, LD))
print("  electronDistrib=%s  ionDistrib=%s  avPartNbPerCell=%s  duration=%s  sunZ=%s  ionVy=%s"
      % (gp("electronDistrib"), gp("ionDistrib"), gp("avPartNbPerCell"), gp("duration"),
         gp("sunZ"), gp("ionVy")))

NICE = [("plasma_potential",                        "Potential (V)", "potential"),
        ("elec1_charge_density",                    "Electron charge density (e m$^{-3}$)", "elec"),
        ("ions1_MultipleVolDistrib=0_charge_density",
         "Ion charge density, analytic part (e m$^{-3}$)", "ion_analytic"),
        ("ions1_MultipleVolDistrib=1_charge_density",
         "Ion charge density, PIC perturbation (e m$^{-3}$)", "ion_perturb"),
        ("ions1_charge_density",                    "Ion charge density (e m$^{-3}$)", "ion"),
        ("photoElec_charge_density",                "Photoelectron charge density (e m$^{-3}$)", "photoelec"),
        ("secondElec_True_from_ambiant_electrons_charge_density",
         "Secondary electron charge density (e m$^{-3}$)", "secondelec"),
        ("secondElec_BS_from_ambiant_electrons_charge_density",
         "Backscattered electron charge density (e m$^{-3}$)", "bselec"),
        ("secondElec_from_ambiant_protons_charge_density",
         "Proton-induced secondary charge density (e m$^{-3}$)", "protonsec"),
        ("total_density",                           "Total charge density (e m$^{-3}$)", "total")]


SNAP = "--snapshot" in sys.argv
SNAP_T = None
if SNAP:
    _ts = sorted({m.group(1) for f in os.listdir(D)
                  for m in [re.search(r"_at_t_=_([0-9.eE+-]+)_s\.nc$", f)] if m}, key=float)
    if not _ts:
        raise SystemExit("--snapshot: no *_at_t_=*_s.nc files in " + D)
    _i = sys.argv.index("--snapshot")
    _want = None
    if _i + 1 < len(sys.argv) and not sys.argv[_i + 1].startswith("--"):
        try:
            _want = float(sys.argv[_i + 1])
        except ValueError:
            _want = None
    SNAP_T = _ts[-1] if _want is None else min(_ts, key=lambda z: abs(float(z) - _want))
    PREFIX = PREFIX + "_snap"
    print("  SNAPSHOT: instantaneous fields at t = %s s%s"
          % (SNAP_T, "" if _want is None else " (nearest to %g)" % _want))

PLATEAU = "--plateau" in sys.argv


def plateau_potential():
    key = lambda f: float(re.search(r"_at_t_=_([0-9.eE+-]+)_s", os.path.basename(f)).group(1))
    fs = sorted([f for f in glob.glob(os.path.join(D, "plasma_pot_at_t_=_*_s.nc"))
                 if "SC_ref" not in f and "Mask" not in f], key=key)
    if not fs:
        return None, 0, 0.0, 0.0
    tail = fs[int(0.75*len(fs)):]
    a = np.mean([netcdf_file(f, 'r', mmap=False).variables['dataArray'].data.ravel()
                 .astype(float) for f in tail], axis=0)
    return a, len(tail), key(tail[0]), key(tail[-1])


def discover(log10):
    if SNAP:
        pre = "log10_of_improved_" if log10 else "improved_"
        pat = pre + "*_at_t_=_" + SNAP_T + "_s.nc"
        pot = "plasma_pot_at_t_=_" + SNAP_T + "_s.nc"
    else:
        pre = "log10_of_final_" if log10 else "final_"
        pat = pre + "*.nc"
        pot = pre + "plasma_potential.nc"

    files = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(D, pat)))]
    out, used = [], set()

    if not log10 and os.path.exists(os.path.join(D, pot)):
        out.append((pot, "Potential (V)", "potential", True))

    for key, label, tag in NICE:
        if key == "plasma_potential":
            continue
        stem = key.replace("_charge_density", "").replace("total_density", "total")
        hit = None
        for b in files:
            if b in used or "current" in b or "charge_density" not in b:
                continue
            core = b[len(pre):].split("_charge_density")[0].lstrip("_")
            if core == stem or core == stem.rstrip("_"):
                hit = b
                break
        if hit:
            used.add(hit)
            out.append((hit, ("log$_{10}$ " + label) if log10 else label, tag, False))

    for b in files:
        if b in used or "current" in b or "charge_density" not in b:
            continue
        core = b[len(pre):].split("_charge_density")[0].lstrip("_")
        out.append((b, ("log$_{10}$ " if log10 else "") + core.replace("_", " ") +
                    " (e m$^{-3}$)", core[:18], False))
    return out


LIN = discover(False)
LG = discover(True)
if ONLY is not None:
    LIN = [f for f in LIN if f[2] in ONLY]
    LG = [f for f in LG if f[2] in ONLY]
print("  fields found: %d linear, %d log10%s"
      % (len(LIN), len(LG), "  (--only=%s)" % ",".join(ONLY) if ONLY else ""))


def _rough(P, T, V):
    e = np.vstack([T[:, [0,1]], T[:, [0,2]], T[:, [0,3]], T[:, [1,2]], T[:, [1,3]], T[:, [2,3]]])
    a, b = V[e[:, 0]], V[e[:, 1]]
    ok = np.isfinite(a) & np.isfinite(b)
    return np.nanmean(np.abs(a[ok] - b[ok]))


MESHES, MASKS, seen = [], [], set()
for m in glob.glob(os.path.join(G, "**", "*.msh"), recursive=True):
    try:
        P, T = load_msh(m)
        if len(T): MESHES.append((os.path.basename(m), P, T))
    except Exception:
        pass
for f in glob.glob(os.path.join(G, "**", "*Mask*.nc"), recursive=True):
    try:
        d = netcdf_file(f, 'r', mmap=False).variables['meshElmentId'].data.ravel().astype(int)
        if d.tobytes() not in seen:
            seen.add(d.tobytes()); MASKS.append((os.path.basename(f), d))
    except Exception:
        pass
print("  scanned %d meshes, %d distinct masks" % (len(MESHES), len(MASKS)))

_big = max(MESHES, key=lambda m: len(m[1]))
RMAP = float(np.linalg.norm(_big[1][1:], axis=1).max())*1.0125
if "--zoom" in sys.argv:
    RMAP = float(sys.argv[sys.argv.index("--zoom")+1])
    print("  ZOOM: plotting the inner %.1f m only (simulation domain unchanged)" % RMAP)
_x = _big[1][1:]
_centred = abs(_x[:, 1].min() + _x[:, 1].max()) < 0.35*abs(_x[:, 1]).max()
DRAW_BODY = _centred and RMAP > 12
print("  domain radius %.2f m -> %.2f lambda_D ; body silhouette overlay: %s"
      % (RMAP/1.0125, RMAP/LD/1.0125, "yes" if DRAW_BODY else "no (different frame)"))


def mesh_for(raw):
    best = None
    for mn, P, T in MESHES:
        if len(P) - 1 != raw.size:
            continue
        for kn, mk in MASKS:
            if mk.size != raw.size:
                continue
            try:
                r = _rough(P, T, np.abs(field_on_nodes(P, mk, raw)))
            except Exception:
                continue
            if best is None or r < best[0]:
                best = (r, P, T, mk, "%s + %s" % (mn, kn))
    if best is None:
        raise SystemExit("no mesh/mask matches field size %d" % raw.size)
    return best[1], best[2], best[3], best[4]


def symlog_ticks(ax, lo, hi, lt):
    dec = []
    if hi > lt:
        e = int(np.ceil(np.log10(lt)))
        while 10.0**e <= hi: dec.append(10.0**e); e += 1
    if lo < -lt:
        e = int(np.ceil(np.log10(lt)))
        while 10.0**e <= -lo: dec.append(-10.0**e); e += 1
    dec.append(0.0)
    dec = sorted(set(dec))
    while len(dec) > 9:
        dec = dec[::2] if 0.0 in dec[::2] else dec[1::2]
    ax.set_yticks(dec)
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def _surface_segments():
    cand = []
    for m in glob.glob(os.path.join(G, "**", "*.msh"), recursive=True):
        try:
            L = open(m, encoding="utf-8", errors="ignore").read().splitlines()
            i = L.index("$Nodes"); nn = int(L[i+1])
            P = np.zeros((nn+1, 3))
            for k in range(nn):
                q = L[i+2+k].split(); P[int(q[0])] = (float(q[1]), float(q[2]), float(q[3]))
            i = L.index("$Elements"); ne = int(L[i+1])
            tri = []
            for k in range(ne):
                q = L[i+2+k].split()
                if q[1] == "2":
                    nt = int(q[2]); tri.append(list(map(int, q[3+nt:3+nt+3])))
            if not tri:
                continue
            rmax = float(np.linalg.norm(P[1:], axis=1).max())
            cand.append((rmax, os.path.basename(m), P, tri))
        except Exception:
            continue
    if not cand:
        return []
    cand.sort(key=lambda c: c[0])
    rmax, name, P, faces = cand[0]
    global SIL_RMAX
    SIL_RMAX = rmax
    print("  silhouette source: %s (maxR %.2f m, %d surface tris)" % (name, rmax, len(faces)))
    segs = []
    for fc in faces:
        a, b, c = P[fc[0]], P[fc[1]], P[fc[2]]
        pts = []
        for u, v in ((a, b), (b, c), (c, a)):
            if (u[0] > 0) != (v[0] > 0):
                t = u[0]/(u[0] - v[0])
                pts.append((u[1] + t*(v[1] - u[1]), u[2] + t*(v[2] - u[2])))
        if len(pts) == 2:
            segs.append(pts)
    return segs


def _loops_from_segments(segs, tol=1e-6):
    key = lambda pt: (round(pt[0]/tol), round(pt[1]/tol))
    adj = {}
    for a, b in segs:
        adj.setdefault(key(a), []).append((key(b), b))
        adj.setdefault(key(b), []).append((key(a), a))
    pos = {}
    for a, b in segs:
        pos[key(a)] = a; pos[key(b)] = b
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop, cur, prev = [pos[start]], start, None
        seen.add(start)
        while True:
            nxt = None
            for k, pt in adj.get(cur, []):
                if k != prev and k not in seen:
                    nxt = (k, pt); break
            if nxt is None:
                if any(k == start for k, _ in adj.get(cur, [])) and len(loop) > 2:
                    loop.append(pos[start])
                break
            prev, cur = cur, nxt[0]
            seen.add(cur); loop.append(nxt[1])
        if len(loop) > 2:
            loops.append(np.array(loop))
    return loops


_SEGS = _surface_segments()
_LOOPS = _loops_from_segments(_SEGS)
print("  body silhouette: %d segments -> %d closed loop(s)" % (len(_SEGS), len(_LOOPS)))


def draw_body(ax):
    for lp in _LOOPS:
        ax.fill(lp[:, 0]/LS, lp[:, 1]/LS, facecolor=BODY_FC, edgecolor=BODY_EC,
                lw=0.8, zorder=4)
    if DRAW_BODY:
        ax.plot([-3.45/LS, -1.52/LS], [0.213/LS, 0.213/LS],
                color=BODY_EC, lw=1.6, solid_capstyle="butt", zorder=5)
        ax.plot([-3.45/LS, -1.52/LS], [0.213/LS, 0.213/LS],
                color=BODY_FC, lw=0.8, solid_capstyle="butt", zorder=6)

NODE_POT = []
for _f in sorted(glob.glob(os.path.join(MON, "ground_potential_on_node_*_V_versus_Time_s.nc"))):
    _y = netcdf_file(_f, 'r', mmap=False).variables['dataArray'].data.ravel()
    _y = _y[np.nonzero(_y)[0]]
    _y = _y[np.isfinite(_y)]
    if len(_y) < 16:
        continue
    _q = _y[int(0.75*len(_y)):]
    NODE_POT.append((int(re.search(r"node_(\d+)", os.path.basename(_f)).group(1)),
                     float(_q.mean()), float(_q.std())))

_gp = os.path.join(D, "Spacecraft_ground_potential_V_versus_time_s_(node_0).nc")
if os.path.exists(_gp):
    _g = netcdf_file(_gp, 'r', mmap=False).variables['dataArray'].data.ravel()
    _nz = _g[np.nonzero(_g)[0]]
    _q = _nz[int(.75*len(_nz)):]
    V_SC = float(_q.mean())
    rising = _nz[-1] > _nz[-5] if len(_nz) > 5 else False
    print("  Phi_SC = %+.4f +/- %.4f V  (trung binh quy cuoi, n=%d; mau cuoi %+.4f)  %s"
          % (V_SC, _q.std(), len(_q), _nz[-1],
             "STILL RISING - not converged" if rising else "flat"))
else:
    if NODE_POT:
        V_SC = NODE_POT[0][1]
        for idx, m, sd in NODE_POT:
            print("  node %-2d Phi = %+9.4f +/- %.4f V  (trung binh quy cuoi)" % (idx, m, sd))
    else:
        V_SC = float(netcdf_file(os.path.join(D, "final_plasma_potential.nc"), 'r', mmap=False)
                     .variables['dataArray'].data.ravel().max())
        print("  Phi_SC = %+.4f V  [from max(final_plasma_potential); no monitor]" % V_SC)

SIL_RMAX = globals().get('SIL_RMAX', 4.0)
BODY_POT = None
_sc = [f for f in glob.glob(os.path.join(D, "sc_pot_(surface-centered)*.nc"))
       if "Mask" not in f and "mesh" not in f]
if _sc:
    if PLATEAU and len(_sc) > 3:
        _k = lambda f: float(re.search(r"_at_t_=_([0-9.eE+-]+)_s", os.path.basename(f)).group(1))
        _o = sorted([f for f in _sc if "_at_t_=_" in f], key=_k)
        _o = _o[int(0.75*len(_o)):] or [max(_sc, key=os.path.getmtime)]
        _y = np.mean([netcdf_file(f, 'r', mmap=False).variables['dataArray'].data.ravel()
                      .astype(float) for f in _o], axis=0)
    else:
        _y = netcdf_file(max(_sc, key=os.path.getmtime), 'r', mmap=False).variables['dataArray'].data.ravel()
    _y = _y[np.isfinite(_y)]
    if len(_y):
        _v, _c = np.unique(np.round(_y, 4), return_counts=True)
        BODY_POT = float(_v[_c.argmax()])
        print("  body surface %+.4f V on %d of %d facets (surface field)"
              % (BODY_POT, _c.max(), len(_y)))
        _n0 = [m for i, m, _ in NODE_POT if i == 0]
        if _n0 and len(_v) > 1:
            _big = _v[_c >= max(50, 0.01 * len(_y))]
            if len(_big):
                _bp = float(_big[np.argmin(np.abs(_big - _n0[0]))])
                if abs(_bp - BODY_POT) > 1e-6:
                    print("  body surface %+.4f V (node 0 monitor %+.4f V, mode %+.4f V)" % (_bp, _n0[0], BODY_POT))
                    BODY_POT = _bp
if WIRE_POT is None and BODY_POT is not None:
    _far = [p for p in NODE_POT if abs(p[1] - BODY_POT) > 0.5]
    if _far:
        WIRE_POT = max(_far, key=lambda p: abs(p[1] - BODY_POT))[1]
        print("  wire node at %+.4f V, drawn at its own level" % WIRE_POT)

plt.rcParams["font.family"] = "serif"; plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]; plt.rcParams["mathtext.fontset"] = "stix"; plt.rcParams["font.size"] = 13
LBL = "$T_e = %g$ eV, $T_p = %g$ eV" % (Te, Ti)
s = np.linspace(-16.1, 16.1, 10000)
LOGN0 = np.log10(n0)

for item in (LG if LOG10 else LIN):
    fname, cblabel, tag, linear = item
    path = os.path.join(D, fname)
    if not os.path.exists(path):
        print("  MISSING: " + fname); continue
    raw = netcdf_file(path, 'r', mmap=False).variables['dataArray'].data.ravel().astype(float)
    if PLATEAU and tag == "potential":
        _a, _n, _t0, _t1 = plateau_potential()
        if _a is not None and _a.size == raw.size:
            raw = _a
            print("  PLATEAU: potential = mean of %d frames, t = %.3e..%.3e s" % (_n, _t0, _t1))
            for _idx, _m, _sd in sorted([p for p in (NODE_POT or []) if p[0] != 0]):
                _v, _c = np.unique(raw, return_counts=True)
                _grp = sorted([(int(cc), float(vv)) for vv, cc in zip(_v, _c) if cc >= 20],
                              reverse=True)
                if len(_grp) > 1:
                    _cc, _hit = max(_grp[1:], key=lambda g: abs(g[1] - _grp[0][1]))
                    print("  node %d: %d nodes at %+.3f V, monitor %+.3f V, diff %.3f V" % (_idx, _cc, _hit, _m, abs(_hit - _m)))
        else:
            print("  PLATEAU: no usable plasma_pot frames, keeping the single frame")
    P, T, mask, pairing = mesh_for(raw)
    tri, vals = slice_x0(P, T, field_on_nodes(P, mask, raw), 1e-4)
    vals = np.asarray(vals, dtype=float)
    vmin, vmax = float(raw.min()), float(raw.max())
    if tag == "potential" and linear:
        f = vals[np.isfinite(vals)]
        smin, smax = float(f.min()), float(f.max())
        SURF = BODY_POT if BODY_POT is not None else smax
        print("  slice range %+.4f .. %+.4f V (3D field max %+.4f V)" % (smin, smax, vmax))
        if len({round(m, 3) for _, m, _ in NODE_POT}) > 1:
            near = min(NODE_POT, key=lambda p: abs(p[1] - smax))
            print("  field surface %+.4f V = node %d (%+.4f V), %d nodes" % (smax, near[0], near[1], len(NODE_POT)))
        else:
            print("  field surface %+.4f V, Phi_SC %+.4f V, diff %.4f V" % (smax, V_SC, abs(smax - V_SC)))
        print("  colour range %+.4f .. %+.4f V (raw 3D field, matches the SPIS window); "
              "this slice spans %+.4f .. %+.4f V" % (vmin, vmax, smin, smax))
    nclip, clo, chi = 0.0, vmin, vmax
    if VRANGE is not None and (LOG10 or (tag == "potential" and linear)):
        vmin, vmax = VRANGE
        vals = np.clip(vals, vmin, vmax)
    if (CLIP and not LOG10 and tag != "potential") or (CLIP_POT and tag == "potential" and linear):
        f = vals[np.isfinite(vals)]
        clo, chi = (float(x) for x in np.percentile(f, [PCT, 100.0 - PCT]))
        if chi > clo:
            nclip = 100.0*float(np.mean((f < clo) | (f > chi)))
            vals = np.clip(vals, clo, chi)
            vmin, vmax = clo, chi
    cmin, cmax, vals_cut, WINNOTE = vmin, vmax, vals, ""
    if NWIN is not None and tag in ("ion", "elec"):
        if LOG10:
            wlo, whi = float(np.log10(max(NWIN[0], 0.1)*n0)), float(np.log10(NWIN[1]*n0))
        else:
            _sg = -1.0 if abs(vmin) > abs(vmax) else 1.0
            wlo, whi = sorted((_sg*NWIN[0]*n0, _sg*NWIN[1]*n0))
        _f = vals[np.isfinite(vals)]
        _out = 100.0*float(np.mean((_f < wlo) | (_f > whi)))
        vals = np.clip(vals, wlo, whi); vmin, vmax = wlo, whi
        WINNOTE = ("map colour scale limited to %g-%g $n_0$ ($n_0$ = %.3g m$^{-3}$); %.1f%% of the slice lies outside; "
                   "raw field %.3g to %.3g; cuts below show the raw values"
                   % (max(NWIN[0], 0.1) if LOG10 else NWIN[0], NWIN[1], n0, _out, float(raw.min()), float(raw.max())))
        _in = _f[(_f >= wlo) & (_f <= whi)]
        print("    colour window %s: %.4g .. %.4g, %.1f%% of the slice outside; inside the window the slice spans %.3g .. %.3g "
              "(5/50/95 pct %.3g / %.3g / %.3g)" % (tag, wlo, whi, _out, _in.min(), _in.max(), *np.percentile(_in, [5, 50, 95])))
    pad = (vmax - vmin)*1e-6 or 1e-12

    if WIDE:
        fig = plt.figure(figsize=(16, 8.8))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], hspace=0.62, wspace=0.26,
                              left=0.06, right=0.985, top=0.95, bottom=0.20)
        axA = fig.add_subplot(gs[:, 0])
        axB, axC, axD = (fig.add_subplot(gs[i, 1]) for i in range(3))
    else:
        fig, (axA, axB, axC, axD) = plt.subplots(4, 1, figsize=(9, 18),
                                             gridspec_kw={"height_ratios": [3.6, 1, 1, 1]},
                                             constrained_layout=True)
        fig.set_constrained_layout_pads(hspace=0.10)
    triL = matplotlib.tri.Triangulation(tri.x/LS, tri.y/LS, tri.triangles)
    im = axA.tricontourf(triL, vals, levels=np.linspace(vmin - pad, vmax + pad, 61), cmap=CMAP)
    draw_body(axA)
    axA.axvline(0, ls="--", color="w", lw=0.9, zorder=2)
    axA.axhline(Z_WIRE/LS, ls="--", color="w", lw=0.9, zorder=2)
    axA.axhline(Z_BUS/LS, ls="--", color="w", lw=0.9, zorder=2)
    ylim = (RMAP/LS, -RMAP/LS) if VIEW_FROM_PLUS_X else (-RMAP/LS, RMAP/LS)
    axA.set_xlim(*ylim); axA.set_ylim(-RMAP/LS, RMAP/LS); axA.set_aspect("equal")
    print("  chieu truc ngang: trai = %+.1f  ->  phai = %+.1f  (lambda_D)  [%s]"
          % (ylim[0], ylim[1], "trai DUONG, phai AM" if ylim[0] > ylim[1]
             else "trai AM, phai DUONG"))
    axA.set_xlabel("Location in Y (%s)" % UNIT); axA.set_ylabel("Location in Z (%s)" % UNIT)
    if WIDE:
        cb = fig.colorbar(im, cax=axA.inset_axes([0, -0.118, 1, 0.028]),
                          orientation="horizontal")
        cb.set_label(cblabel, labelpad=8)
    else:
        axA.set_title("Plasma Potential (V)" if tag == "potential" else cblabel,
                      fontsize=15, pad=8)
    cb = fig.colorbar(im, cax=axA.inset_axes([1.04, 0, 0.035, 1]))
    if vmin < 0 < vmax:
        _f = (0.0 - vmin) / (vmax - vmin)
        _nn = int(np.clip(round(_f * 5), 1, 4))
        tk = np.unique(np.concatenate([np.linspace(vmin, 0.0, _nn + 1),
                                       np.linspace(0.0, vmax, 5 - _nn + 1)]))
    else:
        tk = np.linspace(vmin, vmax, 6)
    if vmin < 0 < vmax:
        tk = np.array([t for t in tk if t == 0 or abs(t) > 0.03 * (vmax - vmin)])
    cb.set_ticks(list(tk))
    cb.set_ticklabels(["%.2f" % t if LOG10 or linear and abs(vmax) < 1e3 else
                       ("%.3g" % t if 1e-2 <= abs(t) < 1e4 or t == 0 else "%.2e" % t)
                       for t in tk])
    cb.ax.tick_params(labelsize=9)
    if nclip > 0.05:
        axA.text(0.5, -0.125, "colour range clipped at %g/%g percentile, %.1f%% of nodes "
                 "(raw %.3g to %.3g)" % (PCT, 100 - PCT, nclip, float(raw.min()), float(raw.max())),
                 transform=axA.transAxes, color="0.35", fontsize=8, ha="center", va="top")
    if WINNOTE:
        axA.text(0.5, -0.125, WINNOTE, transform=axA.transAxes, color="0.35", fontsize=8, ha="center", va="top")
    vmin, vmax = cmin, cmax

    short = ("log$_{10}$ Charge density (e m$^{-3}$)" if LOG10 else
             ("Potential (V)" if linear else "Charge density (e m$^{-3}$)"))
    for ax, (x, y, col, xlab) in zip(
            (axB, axC, axD),
            ((0*s, s, "#1f4e9c", "Location in Z along the spacecraft axis (%s)" % UNIT),
             (s, 0*s + Z_WIRE, "#b02418", "Location in Y at the antenna (%s)" % UNIT),
             (s, 0*s + Z_BUS, "#1a7a3c", "Location in Y through the bus (%s)" % UNIT))):
        v = cut_on_slice(tri, vals_cut, x, y)
        ax.plot(s/LS, v, color=col, lw=2)
        if LOG10:
            ax.axhline(LOGN0, color="0.45", lw=.9, ls="--")
            _sp = vmax - vmin
            _st = 1 if _sp <= 6 else (2 if _sp <= 10 else 3)
            base = [q for q in range(int(np.ceil(vmin/_st))*_st, int(np.floor(vmax)) + 1, _st)
                    if abs(q - LOGN0) > .09*_sp]
            ax.set_yticks(base + [LOGN0]); ax.set_yticklabels(["%d" % q for q in base] + ["%.2f" % LOGN0])
            ax._ref_ticks = [LOGN0]
            ax.set_ylim(vmin, vmax)
        elif linear:
            ax.axhline(0, color="#1f77b4", lw=1.3, ls="--", zorder=2)
            ax.text(0.995, 0.0, "ambient plasma, 0 V", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=9, color="#1f77b4", bbox=dict(facecolor="w", edgecolor="none", alpha=0.8, pad=1.5))
            ax.axhline(SURF, color="0.45", lw=.9, ls="--")
            ok = np.isfinite(v)
            j = 0
            while j < len(v):
                if not ok[j]:
                    k = j
                    while k < len(v) and not ok[k]:
                        k += 1
                    if j > 0 and k < len(v) and abs((s[j-1] + s[k]) / 2) <= SIL_RMAX + 0.1:
                        lev = SURF
                        if (ax is axC and WIRE_POT is not None
                                and -3.45 < (s[j-1] + s[k]) / 2 < -1.52):
                            lev = WIRE_POT
                        ax.plot([s[j-1]/LS, s[j-1]/LS, s[k]/LS, s[k]/LS], [v[j-1], lev, lev, v[k]],
                                color=col, lw=2.0, solid_capstyle="butt", zorder=3)
                    j = k
                else:
                    j += 1
            fin = v[np.isfinite(v)]
            lo_, hi_ = min(fin.min(), SURF, 0.), max(fin.max(), SURF, 0.)
            for _, m, _ in NODE_POT:
                lo_, hi_ = min(lo_, m), max(hi_, m)
            if ax is axC and WIRE_POT is not None:
                lo_, hi_ = min(lo_, WIRE_POT), max(hi_, WIRE_POT)
            sp = hi_ - lo_
            if SYMLOG:
                ax.set_yscale("symlog", linthresh=1.0, linscale=0.9)
                dec = [-(10.0**k) for k in range(0, 4) if -(10.0**k) > lo_*1.02]
                tk = [round(SURF, 2), 0.0, -1.0] + dec
                if lo_ < (min(dec) if dec else -1.0)*1.4:
                    tk.append(round(lo_, 1))
                tk = sorted({t for t in tk if lo_*1.05 <= t <= max(hi_, 0.) + .05})
                ax.set_yticks(tk)
                ax.set_yticklabels(["%.2f" % t if abs(t - SURF) < 1e-9 else
                                    ("0" if t == 0 else "%g" % t) for t in tk])
                ax.set_ylim(lo_*1.35, max(hi_, 0.) + 0.45)
            else:
                base = [q for q in sorted({round(t, 1) for t in np.linspace(lo_, hi_, 5)})
                        if abs(q) > .08*sp]
                ax.set_yticks(sorted(base + [0.0]))
                ax.set_ylim(lo_ - .05*sp, hi_ + .05*sp)
                levs = [round(SURF, 2)]
                if ax is axC and WIRE_POT is not None and abs(WIRE_POT - SURF) > .06*sp:
                    levs.append(round(WIRE_POT, 2))
                ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim()); ax2.set_yticks(levs)
                ax2.set_yticklabels(["%.2f" % t for t in levs]); ax2.tick_params(axis="y", labelsize=10, length=3)
        else:
            ax.axhline(0, color="k", lw=.6, ls=":")
            lt = max(abs(vmin), abs(vmax))/1e4
            ax.set_yscale("symlog", linthresh=lt)
            symlog_ticks(ax, min(vmin, 0.), max(vmax, 0.), lt)
            if tag in ("ion", "elec"):
                n0ref = n0 if vmax > 0 else -n0
                ax.axhline(n0ref, color="0.45", lw=.9, ls="--")
                ticks = sorted(set(list(ax.get_yticks()) + [n0ref]))
                def _lab(t):
                    if t == 0: return "0"
                    if t == n0ref:
                        m, e = ("%.2e" % abs(t)).split("e")
                        return ("$-%s{\\times}10^{%d}$" if t < 0 else
                                "$%s{\\times}10^{%d}$") % (m.rstrip("0").rstrip("."), int(e))
                    return ("$-10^{%d}$" if t < 0 else "$10^{%d}$") % round(np.log10(abs(t)))
                ax.set_yticks(ticks)
                ax.set_yticklabels([_lab(t) for t in ticks])
            if vmax <= 0:
                ax.set_ylim(min(vmin, 0.)*1.5, 0.5*lt)
            elif vmin >= 0:
                ax.set_ylim(-0.5*lt, max(vmax, 0.)*1.5)
        ax.set_xlim(*ylim)
        if ax is axC:
            if GREY: ax.axvspan(-3.45/LS, -1.52/LS, color="0.92", zorder=0)
        if GREY and ax is axD: ax.axvspan(-.5/LS, .5/LS, color="0.88", zorder=0)
        ax.set_xlabel(xlab); ax.set_ylabel(short); ax.grid(alpha=.2)
        ax.text(1.0, 1.02, LBL, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11)

    fig.canvas.draw(); _rd = fig.canvas.get_renderer()
    for _ax in (axB, axC, axD):
        _tk = list(_ax.get_yticks()); _tl = _ax.get_yticklabels()
        _lo, _hi = sorted(_ax.get_ylim())
        _ok = [i for i, t in enumerate(_tl) if t.get_text() and _lo <= _tk[i] <= _hi]
        _refs = getattr(_ax, "_ref_ticks", [0.0])
        _ok.sort(key=lambda i: 0 if any(abs(_tk[i] - r) < 1e-9*max(1.0, abs(r)) for r in _refs) else 1)
        _bb = {i: _tl[i].get_window_extent(_rd).expanded(1.0, 1.15) for i in _ok}
        _keep = []
        for i in _ok:
            if all(not _bb[i].overlaps(_bb[j]) for j in _keep):
                _keep.append(i)
        _keep.sort(key=lambda i: _tk[i])
        if len(_keep) < len(_ok):
            _txt = [_tl[i].get_text() for i in _keep]
            _ax.set_yticks([_tk[i] for i in _keep]); _ax.set_yticklabels(_txt)
    fig.canvas.draw(); _rd = fig.canvas.get_renderer()
    for _ax in (axB, axC, axD):
        _lo, _hi = sorted(_ax.get_ylim())
        _cur = sorted(((p, t) for p, t in zip(_ax.get_yticks(), _ax.get_yticklabels()) if t.get_text() and _lo <= p <= _hi), key=lambda x: x[0])
        _raw = [t.get_window_extent(_rd) for _, t in _cur]
        _gap = min([b.y0 - a.y1 for a, b in zip(_raw, _raw[1:])] or [float("nan")])
        print("    y labels %s: %s | min gap %.1f px" % ("BCD"[(axB, axC, axD).index(_ax)], [t.get_text() for _, t in _cur], _gap))
        assert not (_gap < 0), "overlapping y tick labels remain"
    out = "%s_%s%s%s.png" % (PREFIX, "log10_" if LOG10 else "",
                             tag, "_symlog" if SYMLOG and linear else "")
    plt.savefig(out, dpi=400); plt.close(fig)
    print("  %-11s %13.5g .. %-13.5g  %-44s -> %s%s"
          % (tag, vmin, vmax, pairing, out,
             "" if nclip <= 0.05 else "   [clipped from %.4g..%.4g, %.1f%% of nodes]"
             % (float(raw.min()), float(raw.max()), nclip)))
