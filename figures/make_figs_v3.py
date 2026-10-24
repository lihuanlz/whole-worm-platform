# -*- coding: utf-8 -*-
# make_figs_v3.py -- preprint v3 figure factory (all-English labels, bright high-contrast)
import sys, re, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, Ellipse, FancyBboxPatch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # release root
FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIGD, exist_ok=True)

C = dict(sens="#00BFFF", inter="#FF2D95", cmd="#FFD400", motor="#00C853",
         bwm="#FF6D00", muscD="#E53935", muscV="#1E88E5", gap="#FF9100",
         chem="#304FFE", accent="#AA00FF", dark="#1A1A2E", grey="#90A4AE")
plt.rcParams.update({
    "font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.9,
    "axes.edgecolor": "#333", "figure.dpi": 110, "savefig.dpi": 300,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
})

def save(fig, name):
    fig.savefig(os.path.join(FIGD, name + ".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIGD, name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name, flush=True)

def load_run(tag):
    return np.load(os.path.join(REPO, "runs", "v62_closed_loop", tag + ".npz"), allow_pickle=True)

NET = np.load(os.path.join(REPO, "data", "net_pure323_M6_rimlgc55.npz"), allow_pickle=True)
R_CR = load_run("v62cl_s0w78_crawl_regress")
NAMES = [str(x) for x in R_CR["names"]]
KNOWN_CLASSES = sorted(set("""
ASE ASH ADF ASI ASJ ASK AWA AWB AWC AFD ALM AVM CEP CEPsh OLL OLQ IL BAG AQR URX URA URB ADA ADE PLN SDQ
AIA AIB AIN AIY AIZ RIA RIB RIC RID RIF RIG RIH RIM RIP RIR RIS ALA AVF AVH AVJ AVK AVL DVC PVN PVP PVQ PVT PVR BDU SIB FLP RMG
AVA AVB AVD AVE PVC DVA
VA VB VC VD DA DB DD AS SMD RIV RMD RME RMF RMH SMB SAA
BWM GLR ALN AIM SDQ LUA CAN
""".split()), key=len, reverse=True)
def norm_class(n):
    for k in KNOWN_CLASSES:
        if n.startswith(k):
            return k
    m = re.match(r"([A-Za-z]+)", n)
    return m.group(1) if m else n
CLS = [norm_class(n) for n in NAMES]

def curvature_from_xs(xs):
    dx = np.diff(xs, axis=1)
    ang = np.arctan2(dx[:, :, 1], dx[:, :, 0])
    return np.diff(ang, axis=1)  # (T, 23)

def com_xy(r):
    return r["xs"].mean(axis=1)  # (T,2) mm

def axial_speed(r):
    xs = r["xs"]; dt = float(r["dt_block_ms"][0]) / 1000.0
    com = xs.mean(axis=1)
    head = xs[:, 0, :]; tail = xs[:, -1, :]
    ax = head - tail; ax /= (np.linalg.norm(ax, axis=1, keepdims=True) + 1e-9)
    dcom = np.gradient(com, dt, axis=0)
    v = (dcom * ax).sum(axis=1)
    k = np.ones(5) / 5
    return np.convolve(v, k, mode="same")

# ===================== F1 =====================
def ap_pos(name, cls):
    head0 = {"ASE":.12,"ASH":.115,"ADF":.125,"ASI":.12,"ASJ":.115,"ASK":.13,"AWA":.135,
             "AWB":.125,"AWC":.13,"AFD":.105,"CEP":.09,"OLL":.07,"OLQ":.085,"IL":.06,
             "BAG":.14,"URB":.11,"URX":.115,"ADA":.14,"ADE":.13,"ALA":.13,"AIM":.15,
             "AIA":.145,"AIB":.15,"AIN":.14,"AIY":.135,"AIZ":.145,"RIA":.155,"RIB":.15,
             "RIC":.145,"RID":.16,"RIF":.15,"RIG":.15,"RIH":.14,"RIM":.155,"RIP":.16,
             "RIR":.15,"RIS":.15,"RIV":.16,"RMD":.165,"RME":.10,"RMF":.09,"RMG":.15,
             "RMH":.10,"SAA":.17,"SIB":.165,"SMB":.155,"SMD":.16,"AVA":.17,"AVB":.165,
             "AVD":.175,"AVE":.175,"AVF":.15,"AVH":.15,"AVJ":.14,"AVK":.145,"AVL":.13,
             "BDU":.16,"DVA":.15,"DVC":.14,"FLP":.12,"GLR":.10,"AQR":.30,"SDQ":.55,
             "ALM":.46,"AVM":.47,"ALN":.90,"PLN":.90,"PVC":.90,"PVN":.885,"PVP":.895,
             "PVQ":.905,"PVR":.895,"PVT":.85,"Fragment":.5}
    if cls in ("VA","VB","VD","DA","DB","DD","AS","VC"):
        m = re.search(r"(\d+)$", name)
        i = int(m.group(1)) if m else 1
        nmax = {"VA":12,"VB":11,"VD":13,"DA":9,"DB":7,"DD":6,"AS":11,"VC":6}[cls]
        return 0.20 + 0.66 * (i - 1) / max(nmax - 1, 1)
    if cls == "BWM":
        m = re.search(r"(\d+)$", name); i = int(m.group(1)) if m else 1
        return 0.06 + 0.16 * ((i - 1) % 8) / 7.0
    return head0.get(cls, 0.13)

CAT_RULES = [
    (r"^(ASE|ASH|ADF|ASI|ASJ|ASK|AWA|AWB|AWC|AFD|ALM|AVM|CEP|OLL|OLQ|IL|BAG|AQR|URX|URA|URB|ADA|ADE|PLN|SDQ)$", "sens"),
    (r"^(AVA|AVB|AVD|AVE|PVC|DVA)$", "cmd"),
    (r"^(VA|VB|VC|VD|DA|DB|DD|AS|SMD|RIV|RMD|RME|RMF|RMG|RMH|SMB|SAA)$", "motor"),
    (r"^BWM$", "bwm"),
]
def categorize(cls):
    for pat, key in CAT_RULES:
        if re.fullmatch(pat, cls):
            return key
    return "inter"

def fig1():
    fig = plt.figure(figsize=(11.5, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.35, 1.0],
                          hspace=0.30, wspace=0.16)
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(-0.05, 1.42); ax.set_ylim(-1.35, 1.55); ax.axis("off")
    t = np.linspace(0, 1, 200)
    w = 0.16 * np.sin(np.pi * t) ** 0.7 + 0.015
    ax.fill_between(t, -w, w, color="#FFF8E7", ec="#8D6E63", lw=1.4, zorder=1)
    ax.text(-0.03, 0, "head", ha="right", va="center", fontsize=8, style="italic", color="#555")
    ax.text(1.0, -0.30, "tail", ha="center", va="center", fontsize=8, style="italic", color="#555")
    rng = np.random.default_rng(7)
    placed = {}
    for n, cl in zip(NAMES, CLS):
        cat = categorize(cl); x = ap_pos(n, cl)
        if cat in ("motor", "bwm"):
            ysign = (hash(n) % 2) * 2 - 1
            y = ysign * (0.10 + 0.05 * rng.random()) * (0.4 + 0.6 * np.sin(np.pi * x))
        else:
            ang = rng.uniform(0, 2 * np.pi); r = 0.05 + 0.09 * rng.random()
            y = r * np.sin(ang) * 2.2; x = x + 0.035 * r * np.cos(ang)
        placed[n] = (x, y, cat)
        ax.scatter(x, y, s=9, color=C[cat], alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.2)
    for row in range(24):
        x = 0.05 + 0.90 * row / 23
        ax.scatter([x], [0.62], s=14, marker="s", color=C["muscD"], zorder=2, alpha=0.9)
        ax.scatter([x], [-0.62], s=14, marker="s", color=C["muscV"], zorder=2, alpha=0.9)
    ax.plot([0.05, 0.95], [0.62, 0.62], color=C["muscD"], lw=0.7, alpha=0.5, zorder=1)
    ax.plot([0.05, 0.95], [-0.62, -0.62], color=C["muscV"], lw=0.7, alpha=0.5, zorder=1)
    ax.text(0.5, 0.80, "body-wall muscle layer: 96 cells = 24 rows x 4 quadrants (dorsal red, ventral blue)",
            ha="center", fontsize=8, color=C["muscD"])
    xd, yd, _ = placed["DVA"]
    ax.annotate("", xy=(0.88, 0.03), xytext=(xd + 0.02, yd),
                arrowprops=dict(arrowstyle="-", color=C["cmd"], lw=1.3, ls=(0, (4, 2))))
    ax.text(0.42, 0.24, "DVA TRP-4 stretch-sensing process", fontsize=7.5, color="#B8860B", rotation=2)
    lab_off = {"ASEL": (-0.03, 0.34), "ALML": (0.0, 0.30), "RIAL": (0.03, 0.42),
               "AVAL": (-0.05, -0.42), "AVBL": (0.05, -0.34), "PVCL": (0.0, 0.34)}
    for lab, (dx, dy) in lab_off.items():
        if lab in placed:
            x, y, _ = placed[lab]
            ax.annotate(lab, (x, y), xytext=(x + dx, y + dy), fontsize=7.0, ha="center",
                        color="#222", arrowprops=dict(arrowstyle="-", lw=0.5, color="#999"))
    handles = [plt.Line2D([0],[0], marker="o", ls="", color=C[k], label=lab, markersize=6)
               for k, lab in [("sens","Sensory neurons"),("inter","Interneurons"),
                              ("cmd","Command interneurons"),("motor","Motor neurons"),
                              ("bwm","Head muscle cells (EM connectome)")]]
    ax.legend(handles=handles, loc="center left", frameon=True, fontsize=7.5, ncol=1,
              bbox_to_anchor=(0.80, 0.52))
    ax.set_title("a  Whole-animal cell map: 323 connectome cells + 96-cell muscle layer",
                 loc="left", fontweight="bold")
    n_neurons = sum(1 for c in CLS if c != "BWM")
    ax.text(0.5, -1.08, f"{n_neurons} neurons + 32 head muscle cells carried in the EM connectome; "
            "96 body-wall muscle cells in the actuation layer (soma positions schematic, class composition exact)",
            ha="center", fontsize=8, color="#444")

    axb = fig.add_subplot(gs[1, 0])
    axb.set_xlim(0, 1); axb.set_ylim(-0.08, 1); axb.axis("off")
    axb.set_title("b  One body segment: NMJ wiring onto quadrant muscles", loc="left", fontweight="bold")
    axb.add_patch(Ellipse((0.42, 0.5), 0.52, 0.72, fill=True, fc="#FFF8E7", ec="#8D6E63", lw=1.3))
    for (x, y, lab, col) in [(0.47, 0.80, "DL", C["muscD"]), (0.30, 0.76, "DR", C["muscD"]),
                             (0.47, 0.20, "VL", C["muscV"]), (0.30, 0.24, "VR", C["muscV"])]:
        axb.add_patch(Rectangle((x - 0.055, y - 0.055), 0.11, 0.11, fc=col, ec="white", lw=1.2))
        axb.text(x, y, lab, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    axb.add_patch(Ellipse((0.42, 0.5), 0.14, 0.22, fc="#E3F2FD", ec="#1565C0", lw=1.2))
    axb.text(0.42, 0.5, "gut", ha="center", va="center", fontsize=7.5, color="#1565C0")
    axb.add_patch(FancyBboxPatch((0.72, 0.30), 0.25, 0.44, boxstyle="round,pad=0.02",
                                 fc="#E8F5E9", ec=C["motor"], lw=1.4))
    axb.text(0.845, 0.70, "Motor neurons", ha="center", fontsize=8, color="#1B5E20", fontweight="bold")
    axb.text(0.845, 0.47, "VB / DB (fwd)\nVA / DA (bwd)\nVD / DD (GABA)", ha="center", va="center",
             fontsize=7.2, color="#1B5E20")
    for (x0, y0, x1, y1, col) in [(0.72, 0.62, 0.53, 0.79, C["muscD"]),
                                  (0.72, 0.38, 0.53, 0.21, C["muscV"]),
                                  (0.72, 0.55, 0.53, 0.63, "#6A1B9A"),
                                  (0.72, 0.45, 0.53, 0.37, "#6A1B9A")]:
        axb.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                      color=col, lw=1.5))
    axb.text(0.60, 0.86, "NMJ: ACh (exc)", fontsize=7, color=C["muscD"], ha="center")
    axb.text(0.60, 0.09, "NMJ: GABA (inh, I/E = 0.61)", fontsize=7, color="#6A1B9A", ha="center")
    axb.text(0.42, -0.05, "gap junctions couple neighbouring muscle rows (unc-9/innexin, cc = 0.25)",
             fontsize=6.8, color=C["gap"], ha="center")

    axc = fig.add_subplot(gs[1, 1])
    axc.set_xlim(0, 1); axc.set_ylim(0, 1); axc.axis("off")
    axc.set_title("c  Closed neuromechanical loop", loc="left", fontweight="bold")
    boxes = [(0.06, 0.62, 0.24, 0.22, "Sensory\nASE ASH ALM\nAFD ...", C["sens"]),
             (0.40, 0.62, 0.24, 0.22, "Interneurons\nAIY RIA RIM\nDVA RIS ...", C["inter"]),
             (0.74, 0.62, 0.24, 0.22, "Command\nAVA | AVB", "#B8860B"),
             (0.74, 0.12, 0.24, 0.22, "Motor neurons\nVA/VB/DA/DB\nVD/DD/AS", C["motor"]),
             (0.40, 0.12, 0.24, 0.22, "96 muscles\n2-stage Ca cascade\n+ gap junctions", C["muscD"]),
             (0.06, 0.12, 0.24, 0.22, "24-segment\nviscoelastic body", "#8D6E63")]
    for (x, y, w_, h_, txt, col) in boxes:
        axc.add_patch(FancyBboxPatch((x, y), w_, h_, boxstyle="round,pad=0.02", fc="white", ec=col, lw=2.0))
        axc.text(x + w_/2, y + h_/2, txt, ha="center", va="center", fontsize=7.6, color="#222")
    def arr(p0, p1, col, lab="", lx=0.5, ly=0.0):
        axc.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, color=col, lw=1.8))
        if lab: axc.text(lx, ly, lab, fontsize=6.8, color=col, ha="center")
    arr((0.30, 0.73), (0.40, 0.73), C["sens"], "chem+gap", 0.35, 0.79)
    arr((0.64, 0.73), (0.74, 0.73), C["inter"])
    arr((0.86, 0.62), (0.86, 0.34), "#B8860B", "gate fwd/bwd", 0.97, 0.48)
    arr((0.74, 0.23), (0.64, 0.23), C["motor"], "NMJ", 0.69, 0.28)
    arr((0.40, 0.23), (0.30, 0.23), C["muscD"], "curvature", 0.35, 0.28)
    arr((0.18, 0.34), (0.18, 0.62), "#8D6E63")
    axc.text(0.055, 0.47, "proprioception:\nB-MN stretch + DVA/TRP-4", fontsize=6.6, color="#8D6E63")
    axc.text(0.5, 0.95, "peptidergic gain: DVA -> NLP-12 -> CKR-2 (presynaptic NMJ potentiation)",
             ha="center", fontsize=7.4, color=C["accent"])
    save(fig, "fig1_model")

# ===================== F2 =====================
def fig2():
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.34)
    axa = fig.add_subplot(gs[0, :])
    axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.axis("off")
    axa.set_title("a  One extraction pipeline, two cell systems: method inherited from the cardiac channel line",
                  loc="left", fontweight="bold")
    steps = [(0.01, "Digitized voltage-\nclamp traces\n(published I-V /\ngating data)", "#E1F5FE"),
             (0.175, "Chord-conductance\nactivation\nG=(I-gL*dV)/(V-Erev);\nErev at measured\nI-V crossing", "#E8F5E9"),
             (0.36, "Boltzmann fit of\nsteady state; delayed\nsingle-exp tau;\nR2 + bootstrap\nQC gates", "#FFF3E0"),
             (0.53, "Direct measurement\ndiscipline: fit is an\nestimator only; QC\nflags ride with every\nconstant", "#FCE4EC"),
             (0.71, "Model constants\n(17 currents/cell;\nre-derived sets\nsubstitution-tested)", "#EDE7F6"),
             (0.875, "Whole-connectome\nclosed loop\n(this work)", "#E0F2F1")]
    for x, txt, fc in steps:
        axa.add_patch(FancyBboxPatch((x, 0.30), 0.115, 0.48, boxstyle="round,pad=0.015", fc=fc, ec="#546E7A", lw=1.2))
        axa.text(x + 0.0575, 0.54, txt, ha="center", va="center", fontsize=6.2)
    for x in [0.125, 0.29, 0.475, 0.645, 0.825]:
        axa.add_patch(FancyArrowPatch((x + 0.006, 0.54), (x + 0.044, 0.54), arrowstyle="-|>",
                                      mutation_scale=12, color="#37474F", lw=1.6))
    axa.text(0.01, 0.14, "Same pipeline previously validated on 338 cell-line recordings across 8 channel classes "
             "(Kv1.2/3.1/4.2, Kv7.2/7.3, HCN2, Kir2.4, TASK1) and on 4 cardiac channels incl. hERG "
             "(sister study, doi:10.21203/rs.3.rs-11218939/v1); population constants in Table 4.",
             fontsize=7.2, color="#37474F", va="center")
    axa.text(0.01, 0.92, "cardiac line (hERG et al.): methodology source  ->  C. elegans line: SHK-1, SHL-1, EGL-19 "
             "(+7 more channels) re-derived from Channelpedia CAEEL digitized datasets",
             fontsize=7.4, color=C["chem"], va="center")
    axb = fig.add_subplot(gs[1, 0])
    V = np.linspace(-80, 60, 400)
    def boltz(V, Vh, k): return 1.0 / (1.0 + np.exp(-(V - Vh) / k))
    for lab, v1, k1, v2, k2, col in [
            ("SHK-1 act.", 3.8, 7.18, 20.4, 7.7, C["chem"]),
            ("SHK-1 inact.", -18.9, -4.29, -7.0, -5.8, "#00BFA5"),
            ("EGL-19 act.", 8.3, 8.34, -4.4, 7.5, "#D50000")]:
        axb.plot(V, boltz(V, v1, k1), color=col, lw=2.2, label=f"{lab} re-derived")
        axb.plot(V, boltz(V, v2, k2), color=col, lw=1.4, ls="--", alpha=0.75, label=f"{lab} adopted")
    axb.set_xlabel("V (mV)"); axb.set_ylabel("steady-state fraction")
    axb.set_title("b  Re-derived vs adopted gating", loc="left", fontweight="bold")
    axb.legend(fontsize=5.6, ncol=2, frameon=True, framealpha=1.0, edgecolor="#CCC"); axb.grid(alpha=0.25)
    axc = fig.add_subplot(gs[1, 1])
    inv = [("K+", 13, C["motor"]), ("Ca2+", 3, C["muscD"]), ("Na+ leak (nca)", 1, "#B8860B"),
           ("linear leak", 1, C["grey"])]
    ys = np.arange(len(inv))[::-1]
    for y, (lab, n, col) in zip(ys, inv):
        axc.barh(y, n, color=col, alpha=0.9, height=0.6)
        axc.text(n + 0.15, y, f"{n}", va="center", fontsize=8.5, fontweight="bold", color=col)
    axc.set_yticks(ys); axc.set_yticklabels([i[0] for i in inv])
    axc.set_xlabel("current types per cell"); axc.set_xlim(0, 16)
    axc.set_title("c  17 currents in every cell", loc="left", fontweight="bold")
    axc.grid(axis="x", alpha=0.25)
    axc.text(15.6, 0.3, "K+: shal shak kir kqt-3 egl-2 egl-36\nkvs-1 bk slo-1 bk2 slo-2 sk\nCa2+: unc-2 cca-1 egl-19",
             fontsize=6.2, ha="right", va="center", color="#444")
    axd = fig.add_subplot(gs[1, 2]); axd.axis("off")
    axd.set_title("d  Pipeline output on cell-line data (338 cells)", loc="left", fontweight="bold")
    rows = [("Kv1.2", "-2.0", "10.1", "1.0"), ("Kv3.1", "+17.8", "16.6", "<=1"),
            ("Kv4.2", "+17.9", "22.9", "12.4"), ("Kv7.2", "-17.4", "20.2", "52.7"),
            ("Kv7.3*", "+32.3", "24.3", "84.5"), ("HCN2", "-88.0", "10.0", "-")]
    axd.text(0.0, 0.92, "channel  V1/2(mV)  k(mV)  tau_act(ms)", fontsize=7.4, family="monospace",
             fontweight="bold", color="#222")
    for i, (ch, v, k, t) in enumerate(rows):
        axd.text(0.0, 0.80 - i * 0.115, f"{ch:8s} {v:>9s}  {k:>5s}  {t:>10s}",
                 fontsize=7.4, family="monospace", color="#333")
    axd.text(0.0, 0.04, "*15 degC condition; medians of QC-passed cells; literature\nspot-checks all matched (master table v1, 2026-10-07).",
             fontsize=6.4, color="#555")
    save(fig, "fig2_channels")

# ===================== F3 =====================
def fig3():
    chem = NET["chem"]; gap = NET["gap"]
    LAYERS = [("Sensory", ["ASE","ASH","ADF","ASI","ASJ","ASK","AWA","AWB","AWC","AFD","ALM","AVM",
                            "CEP","OLL","OLQ","IL","BAG","AQR","URX","PLN","SDQ","ADA","ADE","URA","URB"], C["sens"]),
              ("Inter", ["AIA","AIB","AIN","AIY","AIZ","RIA","RIB","RIC","RID","RIF","RIG","RIH","RIM","RIP","RIR","RIS",
                          "ALA","AVF","AVH","AVJ","AVK","AVL","DVC","PVN","PVP","PVQ","PVT","PVR","BDU","SIB","FLP","RMG"], C["inter"]),
              ("Command", ["AVA","AVB","AVD","AVE","PVC","DVA"], "#B8860B"),
              ("Motor", ["VA","VB","DA","DB","VD","DD","AS","VC","SMD","RIV","RMD","RME","RMF","RMH","SMB","SAA"], C["motor"]),
              ("Muscle", ["BWM"], C["bwm"])]
    cls2layer = {c: l[0] for l in LAYERS for c in l[1]}
    def agg(edges):
        d = {}
        for i, j, w in edges:
            a, b = CLS[int(i)], CLS[int(j)]
            d[(a, b)] = d.get((a, b), 0.0) + float(w)
        return d
    dc, dg = agg(chem), agg(gap)
    ll = {}
    for (a, b), w in dc.items():
        la, lb = cls2layer.get(a, "Inter"), cls2layer.get(b, "Inter")
        ll[(la, lb)] = ll.get((la, lb), 0.0) + w
    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.09, 1.10); ax.axis("off")
    ax.set_title("a  Control topology (edge width = EM contact weight)", loc="left", fontweight="bold")
    pos = {}
    ycoord = {"Sensory": 1.00, "Inter": 0.75, "Command": 0.50, "Motor": 0.25, "Muscle": 0.02}
    shows = {"Sensory": ["ASE","ASH","ALM","AFD","AWC","ASI","PLN","CEP"],
             "Inter": ["AIY","AIB","AIZ","RIA","RIM","RIS","DVC","PVR","PVT","SIB"],
             "Command": ["AVA","AVB","AVD","AVE","PVC","DVA"],
             "Motor": ["VA","VB","DA","DB","VD","DD","AS","SMD","RIV","RMD"],
             "Muscle": ["BWM"]}
    for lname, clss, col in LAYERS:
        show = [c for c in shows[lname] if c in CLS]
        n = len(show)
        for k, c in enumerate(show):
            pos[c] = (0.06 + 0.88 * k / max(n - 1, 1), ycoord[lname])
            ax.add_patch(Circle(pos[c], 0.020, fc=col, ec="white", lw=1.0, zorder=4))
            ax.text(pos[c][0], pos[c][1] + 0.032, c, ha="center", fontsize=6.6, color="#222", zorder=5)
    for a, b in [("ASE","AIY"),("ASH","AVA"),("ALM","AVD"),("ALM","PVC"),("AFD","AIY"),
                 ("AIY","RIA"),("AIB","RIA"),("RIA","SMD"),("RIA","RMD"),("RIS","AVA"),
                 ("DVA","AVA"),("DVA","AVB"),("AVA","VA"),("AVA","DA"),("AVB","VB"),
                 ("AVB","DB"),("PVC","AVA"),("PVC","AVB"),("AVE","VA"),("AVD","AVA"),
                 ("VA","BWM"),("VB","BWM"),("VD","BWM"),("DD","BWM"),("AIZ","RIA"),("ASH","AVD")]:
        if a in pos and b in pos and (a, b) in dc:
            w = dc[(a, b)]
            ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>", mutation_scale=8,
                        color="#304FFE", lw=0.4 + 2.4 * min(w / 8.0, 1.0), alpha=0.55,
                        connectionstyle="arc3,rad=0.12", zorder=2))
    for a, b in [("AVB","VB"),("PVC","AVB"),("AVA","VA")]:
        if a in pos and b in pos and (a, b) in dg:
            ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-", mutation_scale=8,
                        color=C["gap"], lw=1.8, ls=(0,(2,2)), alpha=0.95,
                        connectionstyle="arc3,rad=-0.18", zorder=3))
    ax.text(0.06, -0.05, "solid blue: chemical synapses (graded release);  dashed orange: gap junctions\n"
            "4,450 chemical + 1,219 gap edges from EM adjacency (Cook 2019 / Witvliet 2021)",
            fontsize=6.6, color="#444", va="top")
    axb = fig.add_subplot(gs[0, 1])
    axb.set_title("b  Aggregate synaptic flow", loc="left", fontweight="bold")
    flow = sorted([(a, b, w) for (a, b), w in ll.items() if a != b], key=lambda x: -x[2])[:12]
    labels = [f"{a}->{b}" for a, b, _ in flow]
    vals = [w for _, _, w in flow]
    cols = [{"Sensory": C["sens"], "Inter": C["inter"], "Command": "#B8860B",
             "Motor": C["motor"], "Muscle": C["bwm"]}[a] for a, _, _ in flow]
    y = np.arange(len(flow))[::-1]
    axb.barh(y, vals, color=cols, alpha=0.9, height=0.62)
    axb.set_yticks(y); axb.set_yticklabels(labels, fontsize=7)
    axb.set_xlabel("total contact-weight (a.u.)"); axb.grid(axis="x", alpha=0.3)
    save(fig, "fig3_topology")

# ===================== F4 =====================
def fig4():
    fig = plt.figure(figsize=(11.5, 7.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], hspace=0.40, wspace=0.30)
    axa = fig.add_subplot(gs[0, 0])
    axa.set_title("a  96 muscles: 24 rows x 4 quadrants", loc="left", fontweight="bold")
    for r in range(24):
        for dy, col in [(0.66, C["muscD"]), (0.52, C["muscD"]), (0.34, C["muscV"]), (0.20, C["muscV"])]:
            axa.add_patch(Rectangle((0.04 + 0.038 * r, dy - 0.05), 0.030, 0.10, fc=col, ec="white", lw=0.4))
    for r in range(23):
        x0 = 0.04 + 0.038 * r + 0.030
        axa.plot([x0, x0 + 0.008], [0.66, 0.66], color=C["gap"], lw=1.4)
    axa.text(0.5, 0.80, "longitudinal gap junctions (unc-9; cc = 0.25)", fontsize=6.8, color=C["gap"], ha="center")
    axa.text(0.5, 0.04, "row 1 (head) -> row 24 (tail);  DL/DR dorsal, VL/VR ventral",
             fontsize=6.8, color="#555", ha="center")
    axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.axis("off")
    axb = fig.add_subplot(gs[0, 1])
    tms = np.linspace(0, 1500, 3000)
    tau1, tau2 = 15.5, 300.0
    fast = 1 - np.exp(-tms / tau1)
    cascade = 1 - (tau2 * np.exp(-tms / tau2) - tau1 * np.exp(-tms / tau1)) / (tau2 - tau1)
    axb.plot(tms, fast, color=C["muscD"], lw=2.2, label="fast electrical (tau=15.5 ms)")
    axb.plot(tms, cascade, color=C["inter"], lw=2.2, label="x Ca/contraction (tau=300 ms)")
    axb.set_xlabel("time (ms)"); axb.set_ylabel("activation")
    axb.set_title("b  Two-stage excitation-contraction", loc="left", fontweight="bold")
    axb.legend(fontsize=7); axb.grid(alpha=0.25)
    axb.text(420, 0.30, "anchors: AP halfwidth 15.5+/-0.9 ms\n(Gao & Zhen 2011; Liu 2011 ~20 ms)", fontsize=6.6)
    axc = fig.add_subplot(gs[0, 2]); axc.axis("off")
    axc.set_title("c  Anchored muscle constants", loc="left", fontweight="bold")
    rows = [("rest potential", "-26.9+/-0.5 mV", "Liu 2011"),
            ("AP amplitude", "45.1+/-0.7 mV", "Liu 2011"),
            ("AP halfwidth", "15.5-20 ms", "G&Z 2011"),
            ("NMJ I/E ratio", "0.61", "Liu 2009"),
            ("E_rev ACh/GABA", "+20/-30 mV", "G&Z 2011"),
            ("muscle GJ cc", "0.25", "Liu 2011 JBC"),
            ("bend elasticity", "4e-12 N m3", "Fang-Yen 2010")]
    for i, (k, v, src) in enumerate(rows):
        axc.text(0.0, 0.92 - 0.125 * i, k, fontsize=6.8, color="#222", fontweight="bold")
        axc.text(0.46, 0.92 - 0.125 * i, v, fontsize=6.8, color=C["chem"])
        axc.text(0.80, 0.92 - 0.125 * i, src, fontsize=6.0, color="#777")
    axd = fig.add_subplot(gs[1, :])
    dt = float(R_CR["dt_block_ms"][0])
    t = np.arange(R_CR["xs"].shape[0]) * dt / 1000.0
    kap = curvature_from_xs(R_CR["xs"]); kapn = kap - kap.mean()
    v99 = np.percentile(np.abs(kapn), 99)
    im = axd.imshow(kapn.T, aspect="auto", cmap="RdBu_r", vmin=-v99, vmax=v99,
                    extent=[t[0], t[-1], 23, 0])
    axd.set_xlabel("time (s)"); axd.set_ylabel("segment (head->tail)")
    axd.set_title("d  Emergent dorsoventral curvature wave (closed loop, 10 s forward crawl)",
                  loc="left", fontweight="bold")
    cb = plt.colorbar(im, ax=axd, fraction=0.02, pad=0.01)
    cb.set_label("curvature (rad/seg)", fontsize=7)
    save(fig, "fig4_muscle")

# ===================== F5 : seven-behaviour suite =====================
def fig5():
    fig = plt.figure(figsize=(11.5, 8.6))
    gs = fig.add_gridspec(3, 3, hspace=0.46, wspace=0.32)
    # a crawl speed
    ax = fig.add_subplot(gs[0, 0])
    v = axial_speed(R_CR); dt = float(R_CR["dt_block_ms"][0])
    t = np.arange(len(v)) * dt / 1000.0
    ax.plot(t, v, color=C["muscD"], lw=2.0)
    ax.axhline(0.21, color="#777", ls="--", lw=1.0)
    ax.text(0.4, 0.235, "real worm 0.21 mm/s", fontsize=6.8, color="#555")
    ax.set_title("a  Forward crawl: v = 0.222 mm/s", loc="left", fontweight="bold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("axial speed (mm/s)"); ax.grid(alpha=0.25)
    # b crawl kymograph
    ax = fig.add_subplot(gs[0, 1])
    kap = curvature_from_xs(R_CR["xs"]); kapn = kap - kap.mean(); v99 = np.percentile(np.abs(kapn), 99)
    ax.imshow(kapn.T, aspect="auto", cmap="RdBu_r", vmin=-v99, vmax=v99, extent=[t[0], t[-1], 23, 0])
    ax.set_title("b  Curvature wave 0.556 Hz", loc="left", fontweight="bold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("segment")
    # c chemotaxis d090
    ax = fig.add_subplot(gs[0, 2])
    for tag, col, lab in [("v62cl_s0w75_chem_d090", C["grey"], "broken (pirouette mis-fire)"),
                          ("v62cl_s0w78_d090_final", C["motor"], "fixed (weathervane arm)")]:
        r = load_run(tag); xy = com_xy(r)
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=2.0, label=lab)
        ax.scatter(*xy[-1], color=col, s=28, zorder=5)
    src = np.array([3.5, 1.0]); ax.add_patch(Circle(src, 0.35, fc="none", ec=C["muscD"], lw=2.0))
    ax.scatter(*src, marker="*", s=160, color=C["muscD"], zorder=6)
    ax.set_title("c  Chemotaxis 90 deg: +3.46 mm", loc="left", fontweight="bold")
    ax.legend(fontsize=6.4, loc="upper left"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    # d d270 + d180 trajectories
    ax = fig.add_subplot(gs[1, 0])
    for tag, col, lab in [("v62cl_s0w78_d270_final", C["chem"], "d270: +0.96 mm, +111 deg"),
                          ("v62cl_s0w78_d180_pirrev120", C["accent"], "d180 U-turn: +0.88 mm, -112 deg")]:
        r = load_run(tag); xy = com_xy(r)
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=2.0, label=lab)
        ax.scatter(*xy[-1], color=col, s=28, zorder=5)
    ax.add_patch(Circle(src, 0.35, fc="none", ec=C["muscD"], lw=2.0))
    ax.scatter(*src, marker="*", s=160, color=C["muscD"], zorder=6)
    ax.set_title("d  Left turn & U-turn", loc="left", fontweight="bold")
    ax.legend(fontsize=6.4, loc="upper left"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    # e touch escape
    ax = fig.add_subplot(gs[1, 1])
    r = load_run("v62cl_s0w75b_touch400")
    v = axial_speed(r); t2 = np.arange(len(v)) * float(r["dt_block_ms"][0]) / 1000.0
    ax.plot(t2, v, color=C["motor"], lw=2.0)
    ax.axvspan(8, 10, color=C["sens"], alpha=0.25)
    ax.axhline(0, color="#999", lw=0.6)
    ax.text(8.1, max(v) * 0.8, "touch", fontsize=7, color="#0277BD")
    ax.set_title("e  Touch escape", loc="left", fontweight="bold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("axial speed (mm/s)"); ax.grid(alpha=0.25)
    # f ASH escape (weak PASS, honest)
    ax = fig.add_subplot(gs[1, 2])
    r = load_run("v62cl_s0w75b_ash400")
    v = axial_speed(r); t2 = np.arange(len(v)) * float(r["dt_block_ms"][0]) / 1000.0
    ax.plot(t2, v, color=C["inter"], lw=2.0)
    for a, b in [(4, 6), (14, 16)]:
        ax.axvspan(a, b, color="#D50000", alpha=0.20)
    ax.axhline(0, color="#999", lw=0.6)
    ax.text(4.0, max(v) * 0.85, "ASH 1", fontsize=7, color="#B71C1C")
    ax.text(13.6, max(v) * 0.85, "ASH 2", fontsize=7, color="#B71C1C")
    ax.set_title("f  ASH chemical escape (2nd pulse)", loc="left", fontweight="bold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("axial speed (mm/s)"); ax.grid(alpha=0.25)
    # g AVA switch
    ax = fig.add_subplot(gs[2, 0])
    r = load_run("v62cl_s0w75c_switch_fix") if os.path.exists(os.path.join(
        REPO, "runs", "v62_closed_loop", "v62cl_s0w75c_switch_fix.npz")) else load_run("v62cl_s0w75c_switch1")
    v = axial_speed(r); t2 = np.arange(len(v)) * float(r["dt_block_ms"][0]) / 1000.0
    ax.plot(t2, v, color=C["chem"], lw=2.0)
    ax.axvspan(10, min(22, t2[-1]), color="#D50000", alpha=0.18)
    ax.axhline(0, color="#999", lw=0.6)
    ax.text(11, max(v) * 0.75, "AVA on", fontsize=7, color="#B71C1C")
    ax.set_title("g  AVA photo-switch F->B->F", loc="left", fontweight="bold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("axial speed (mm/s)"); ax.grid(alpha=0.25)
    # h RIS arrest
    ax = fig.add_subplot(gs[2, 1])
    r = load_run("v62cl_s0w75_ris")
    v = axial_speed(r); t2 = np.arange(len(v)) * float(r["dt_block_ms"][0]) / 1000.0
    ax.plot(t2, v, color="#00BFA5", lw=2.0)
    ax.axvspan(12, 18, color=C["accent"], alpha=0.22)
    ax.text(12.3, max(v) * 0.85, "RIS on", fontsize=7, color="#6A1B9A")
    ax.set_title("h  RIS activation arrests locomotion", loc="left", fontweight="bold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("axial speed (mm/s)"); ax.grid(alpha=0.25)
    # i steady backward (W74 champion, ledger numbers)
    ax = fig.add_subplot(gs[2, 2])
    kap = curvature_from_xs(load_run("v62cl_s0w78_crawl_regress")["xs"])
    ax.axis("off")
    ax.set_title("i  Steady backward locomotion", loc="left", fontweight="bold")
    txt = ("AVB-ablated steady reversal (demonstration config):\n"
           "  net speed -0.23 mm/s, 0.47 Hz\n"
           "  path straightness 0.78\n"
           "  bout structure matches Zhao 2003 (2-3 s)\n"
           "  tail-to-head wave (Gao 2018 A-MN oscillators)\n\n"
           "trp-4(lf) phenocopy + rescue (model validation):\n"
           "  no TRP-4 brake: kappa 8.0, 0.105 Hz (mutant)\n"
           "  +TRP-4 arm: kappa 4.3, 0.42 Hz (WT-like)\n"
           "  Li et al. 2006 Nature 440:684")
    ax.text(0.0, 0.95, txt, fontsize=7.4, va="top", family="monospace", color="#222")
    save(fig, "fig5_behaviors")

# ===================== F6 : travelling wave / stadium evidence =====================
def fig6():
    fig = plt.figure(figsize=(11.5, 7.8))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26)
    r = R_CR; dt = float(r["dt_block_ms"][0]); t = np.arange(r["xs"].shape[0]) * dt / 1000.0
    kap = curvature_from_xs(r["xs"]); kapn = kap - kap.mean()
    # a muscle activation wave (A_D - A_V)
    ax = fig.add_subplot(gs[0, 0])
    act = r["A_D"][:, :24] - r["A_V"][:, :24]
    if act.shape[1] == 24: act = act[:, :23]
    v99 = np.percentile(np.abs(act), 99)
    ax.imshow(act.T, aspect="auto", cmap="PiYG", vmin=-v99, vmax=v99, extent=[t[0], t[-1], act.shape[1], 0])
    ax.set_title("a  Muscle activation wave (A_D - A_V)", loc="left", fontweight="bold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("segment")
    # b MN voltage raster ordered along body
    ax = fig.add_subplot(gs[0, 1])
    names = NAMES
    def mn_key(n):
        m = re.fullmatch(r"(VB|DB|DD)(\d\d)", n)
        if not m: return None
        return int(m.group(2))
    sel = [i for i, n in enumerate(names) if mn_key(n) is not None]
    sel.sort(key=lambda i: (names[i][:2], mn_key(names[i])))
    Y = r["Y"][:, sel, 0]
    Vr = r["V_rest"][sel]
    dV = (Y - Vr[None, :]).T
    v99 = np.percentile(np.abs(dV), 98)
    ax.imshow(dV, aspect="auto", cmap="bwr", vmin=-v99, vmax=v99,
              extent=[t[0], t[-1], len(sel), 0])
    ax.set_title("b  VNC motor-neuron raster (DB/DD/VB chains)", loc="left", fontweight="bold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("cell (head -> tail within class)")
    labs = [names[i] for i in sel]
    ticks = np.arange(0, len(sel), 6) + 0.5
    ax.set_yticks(ticks); ax.set_yticklabels([labs[int(k)] for k in ticks], fontsize=6)
    # c phase gradient via FFT at the dominant undulation frequency
    ax = fig.add_subplot(gs[1, 0])
    k0 = kapn[40:, :]
    k0 = k0 - k0.mean(0)
    Ff = np.fft.rfft(k0, axis=0)
    freqs = np.fft.rfftfreq(k0.shape[0], d=0.05)
    i0 = int(np.argmax(np.abs(Ff[:, 12])))
    f0 = freqs[i0]
    phase = np.unwrap(np.angle(Ff[i0, :]))
    segs = np.arange(23) + 1
    ax.plot(segs, phase, "o-", color=C["chem"], lw=2.0, ms=4)
    slope = np.polyfit(segs, phase, 1)[0]
    lam_seg = 2 * np.pi / max(abs(slope), 1e-9)
    speed_bl = f0 * lam_seg / 23.0
    ax.set_title("c  Phase gradient -> travelling wave", loc="left", fontweight="bold")
    ax.set_xlabel("segment (head->tail)"); ax.set_ylabel("phase at f0 (rad)")
    ax.text(0.03, 0.06, f"f0 = {f0:.2f} Hz;  lambda = {lam_seg:.1f} segments;\n"
            f"wave speed = {speed_bl:.2f} body lengths/s", transform=ax.transAxes,
            fontsize=7.4, color=C["chem"])
    ax.grid(alpha=0.25)
    # d single-cell channel view: VB05 V, Ca, gates + peptidergic traces
    ax = fig.add_subplot(gs[1, 1])
    i_vb = names.index("VB05")
    V = r["Y"][:, i_vb, 0]
    Ca = r["Y"][:, i_vb, -1]
    ax.plot(t, V, color=C["muscD"], lw=1.8, label="VB05 voltage (mV)")
    ax2 = ax.twinx()
    ax2.plot(t, Ca, color=C["sens"], lw=1.4, label="VB05 [Ca]i")
    ax2.plot(t, r["pep_m"] / (np.max(np.abs(r["pep_m"])) + 1e-9) * np.max(np.abs(Ca)),
             color=C["accent"], lw=1.4, ls="--", label="NLP-12 release var (scaled)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("mV", color=C["muscD"])
    ax2.set_ylabel("Ca / peptide (a.u.)", color=C["sens"])
    ax.set_title("d  One cell, all channels + peptidergic layer", loc="left", fontweight="bold")
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=6.6, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    ax.grid(alpha=0.25)
    save(fig, "fig6_wave")

# ===================== F7 : peptidergic layer =====================
def fig7():
    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("a  Peptidergic / modulatory axes implemented", loc="left", fontweight="bold")
    def box(x, y, w, h, txt, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015", fc="white", ec=col, lw=1.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.2)
    box(0.03, 0.74, 0.26, 0.20, "body bend |kappa|\n(stretch)", "#8D6E63")
    box(0.37, 0.74, 0.26, 0.20, "DVA releases\nNLP-12", C["inter"])
    box(0.71, 0.74, 0.26, 0.20, "CKR-2 GPCR on\ncholinergic MNs", C["accent"])
    box(0.37, 0.40, 0.26, 0.20, "NMJ presynaptic\ngain 1+alpha*m", C["muscD"])
    box(0.03, 0.08, 0.42, 0.20, "FLP-14 -> FRPR-19: reversal-bout\npositive feedback (Marques 2021)", C["chem"])
    box(0.55, 0.08, 0.42, 0.20, "unc-1/innexin state gap-junction\ngain g_eff=(1-k(1-M))*GGAP\nk=0.7 (Chen 2007)", C["gap"])
    for (p0, p1) in [((0.29, 0.84), (0.37, 0.84)), ((0.63, 0.84), (0.71, 0.84)),
                     ((0.84, 0.74), (0.63, 0.58))]:
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, color="#37474F", lw=1.6))
    ax.text(0.50, 0.985, "Hu 2011 Neuron 71:92; Li 2006 Nature 440:684; Beets 2023 Cell Rep 42:113058",
            fontsize=6.6, color="#555", ha="center")
    # b traces (AVA-switch run: forward and backward phases both present)
    ax = fig.add_subplot(gs[0, 1])
    r = load_run("v62cl_s0w75c_switch1"); dt = float(r["dt_block_ms"][0])
    t = np.arange(r["xs"].shape[0]) * dt / 1000.0
    for key, col, lab in [("pep_m", C["accent"], "NLP-12 release m (DVA arm)"),
                          ("pep2_m", C["chem"], "FLP-14 arm m (reversal bout)")]:
        if key in r:
            v = np.asarray(r[key], dtype=float); v = v / (np.max(np.abs(v)) + 1e-12)
            ax.plot(t, v, color=col, lw=1.8, label=lab)
    ax.axvspan(10, min(22, t[-1]), color="#D50000", alpha=0.10)
    ax.set_title("b  Peptidergic states (F->B switch, normalised)", loc="left", fontweight="bold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("normalised"); ax.legend(fontsize=7); ax.grid(alpha=0.25)
    # c trp-4 rescue bars
    ax = fig.add_subplot(gs[1, 0])
    grp = [("trp-4(lf)\n(no brake)", 8.01, 0.105), ("TRP-4 rescue\nG=0.3", 4.31, 0.42)]
    x = np.arange(2)
    ax.bar(x - 0.18, [g[1] for g in grp], width=0.36, color=C["muscD"], label="kappa_amp (mm^-1)")
    ax.bar(x + 0.18, [g[2] * 8 for g in grp], width=0.36, color=C["chem"], label="frequency (Hz) x8")
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in grp], fontsize=8)
    ax.set_title("c  trp-4(lf) phenocopy and in-silico rescue", loc="left", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax.axhspan(3.0, 5.0, color=C["motor"], alpha=0.15)
    ax.text(0.02, 5.1, "WT crawl band 3-5 mm^-1", fontsize=6.8, color="#1B5E20")
    # d honest gap note
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    ax.set_title("d  Known modulatory gaps (declared, not hidden)", loc="left", fontweight="bold")
    txt = ("implemented: NLP-12/CKR-2; FLP-14/FRPR-19; unc-1 state-dependent\n"
           "gap-junction modulation; RIS bout terminator (Kato 2015);\n"
           "AVA bistable plateau (Meng 2024; Mellem 2008)\n\n"
           "declared missing: ~90 remaining neuropeptide systems\n"
           "(Ripoll-Sanchez 2023 Neuron 111:3570); ASH INX-20/cGMP/EGL-4\n"
           "gain modulation (Chaubey 2023); monoamine layer;\n"
           "state-dependent gating of TRP-4 during omega turns (open\n"
           "question raised by our own rescue experiment)")
    ax.text(0.0, 0.95, txt, fontsize=7.4, va="top", family="monospace", color="#222")
    save(fig, "fig7_peptidergic")

# ===================== F8 : chemotaxis mechanism comparison =====================
def fig8():
    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.30)
    rb = load_run("v62cl_s0w75_chem_d090"); rf = load_run("v62cl_s0w78_d090_final")
    dtb = float(rb["dt_block_ms"][0]); dtf = float(rf["dt_block_ms"][0])
    tb = np.arange(rb["xs"].shape[0]) * dtb / 1000.0
    tf = np.arange(rf["xs"].shape[0]) * dtf / 1000.0
    src = np.array([3.5, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    for r, t, col, lab in [(rb, tb, C["grey"], "mis-firing pirouette arm"),
                           (rf, tf, C["motor"], "weathervane arm only")]:
        xy = com_xy(r); ax.plot(xy[:, 0], xy[:, 1], color=col, lw=2.0, label=lab)
    ax.add_patch(Circle(src, 0.35, fc="none", ec=C["muscD"], lw=2.0))
    ax.scatter(*src, marker="*", s=160, color=C["muscD"], zorder=6)
    ax.set_title("a  Same network, same protocol", loc="left", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(tb, rb["readout"][:, 1], color=C["grey"], lw=1.8, label="mis-firing")
    ax.plot(tf, rf["readout"][:, 1], color=C["motor"], lw=1.8, label="fixed")
    ax.set_title("b  dAVA (backward command drive)", loc="left", fontweight="bold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("drive (a.u.)"); ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax = fig.add_subplot(gs[0, 2])
    for r, t, col, lab in [(rb, tb, C["grey"], "mis-firing"), (rf, tf, C["motor"], "fixed")]:
        d = np.linalg.norm(com_xy(r) - src[None, :], axis=1)
        ax.plot(t, d, color=col, lw=2.0, label=lab)
    ax.axhline(0.35, color=C["muscD"], ls="--", lw=1.0)
    ax.text(0.3, 0.5, "source circle", fontsize=6.8, color=C["muscD"])
    ax.set_title("c  Distance to source", loc="left", fontweight="bold")
    ax.set_xlabel("time (s)"); ax.set_ylabel("mm"); ax.legend(fontsize=7); ax.grid(alpha=0.25)
    save(fig, "fig8_chemotaxis")

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
