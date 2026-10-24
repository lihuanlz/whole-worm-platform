# Supplementary real-time demonstration movies for preprint v3.
# Renders multi-panel "oscilloscope" animations from ARCHIVED runs only
# (no engine execution): worm posture, curvature kymograph, single-cell
# V/Ca scope, VNC motor-neuron raster, peptidergic state, command readout.
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

ROOT = Path(__file__).resolve().parent.parent  # release root
RUNS = ROOT / "runs" / "v62_closed_loop"
OUT = ROOT / "movies"
OUT.mkdir(exist_ok=True)

BG = "#0b1020"; PANEL = "#101830"; FG = "#e8ecff"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": "#3a4670", "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "text.color": FG,
    "font.size": 8.5, "axes.titlesize": 9.5,
})

def motor_indices(names, prefix):
    idx = [(i, int(''.join(c for c in n if c.isdigit()) or 0))
           for i, n in enumerate(names) if n.startswith(prefix)]
    return [i for i, _ in sorted(idx, key=lambda t: t[1])]

def render(npz_path, out_path, title):
    d = np.load(npz_path, allow_pickle=True)
    xs = d["xs"]; A_D = d["A_D"]; A_V = d["A_V"]
    Y = d["Y"]; names = [str(x) for x in d["names"]]
    dt = float(d["dt_block_ms"][0]) / 1000.0
    T = xs.shape[0]; t = np.arange(T) * dt
    curv = (A_D - A_V).T                       # (24, T)
    vb05 = names.index("VB05")
    V = Y[:, :, 0]; Ca = Y[:, :, -1]
    print("V range", V.min(), V.max(), "| Ca range", Ca.min(), Ca.max())
    raster_idx = (motor_indices(names, "DB") + motor_indices(names, "DD")
                  + motor_indices(names, "VB") + motor_indices(names, "VD"))
    raster = Y[:, raster_idx, 0].T             # (n_cells, T)
    rlbl = [names[i] for i in raster_idx]
    pep = d["pep_m"]; pep2 = d["pep2_m"]; rd = d["readout"]

    fig = plt.figure(figsize=(15.4, 8.6), dpi=110)
    gs = fig.add_gridspec(3, 3, hspace=0.52, wspace=0.34,
                          left=0.055, right=0.985, top=0.90, bottom=0.075)
    ax_worm = fig.add_subplot(gs[0:2, 0])
    ax_kymo = fig.add_subplot(gs[0, 1:])
    ax_rast = fig.add_subplot(gs[1, 1:])
    ax_cell = fig.add_subplot(gs[2, 0])
    ax_pep  = fig.add_subplot(gs[2, 1])
    ax_cmd  = fig.add_subplot(gs[2, 2])

    fig.suptitle(title, color=FG, fontsize=12, fontweight="bold")

    # --- worm posture, fixed frame ---
    pad = 0.6
    x0, x1 = xs[:, :, 0].min() - pad, xs[:, :, 0].max() + pad
    y0, y1 = xs[:, :, 1].min() - pad, xs[:, :, 1].max() + pad
    ax_worm.set_xlim(x0, x1); ax_worm.set_ylim(y0, y1)
    ax_worm.set_aspect("equal"); ax_worm.set_title("Body posture (fixed frame)")
    ax_worm.set_xlabel("x (mm)"); ax_worm.set_ylabel("y (mm)")
    spine, = ax_worm.plot([], [], lw=3.5, color="#33e0ff", solid_capstyle="round")
    head, = ax_worm.plot([], [], "o", ms=7, color="#ff4d6d")
    ax_worm.plot(xs[:, 0, 0], xs[:, 0, 1], ":", lw=0.8, color="#ff4d6d", alpha=0.5)

    # --- curvature kymograph ---
    vm = np.abs(curv).max()
    ax_kymo.imshow(curv, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm,
                   extent=[0, t[-1], 24, 0])
    cur1 = ax_kymo.axvline(0, color="#ffe95e", lw=1.4)
    ax_kymo.set_title(r"Curvature wave  $A_D - A_V$  (kymograph)")
    ax_kymo.set_xlabel("time (s)"); ax_kymo.set_ylabel("segment (head→tail)")
    ax_kymo.set_yticks([0, 12, 23])

    # --- VNC raster ---
    ax_rast.imshow(raster, aspect="auto", cmap="turbo", vmin=-80, vmax=0,
                   extent=[0, t[-1], len(rlbl), 0])
    cur2 = ax_rast.axvline(0, color="#ffffff", lw=1.2)
    ax_rast.set_title("Ventral-cord motor neurons, V (DB/DD/VB/VD, head→tail)")
    ax_rast.set_xlabel("time (s)"); ax_rast.set_ylabel("cell")
    ticks = np.linspace(0, len(rlbl) - 1, 6).astype(int)
    ax_rast.set_yticks(ticks + 0.5); ax_rast.set_yticklabels([rlbl[i] for i in ticks])

    # --- single-cell scope ---
    ax_cell.set_title("Single cell VB05: V & intracellular Ca²⁺")
    ax_cell.set_xlim(0, t[-1]); ax_cell.set_ylim(-105, 70)
    ax_cell.set_xlabel("time (s)"); ax_cell.set_ylabel("V (mV)", color="#39ff88")
    lV, = ax_cell.plot([], [], color="#39ff88", lw=1.6)
    axc2 = ax_cell.twinx(); axc2.set_ylim(0, max(0.5, float(Ca.max()) * 1.15))
    axc2.set_ylabel("Ca²⁺ (a.u.)", color="#ff9f1c")
    axc2.tick_params(colors="#ff9f1c")
    lCa, = axc2.plot([], [], color="#ff9f1c", lw=1.4)

    # --- peptidergic ---
    ax_pep.set_title("Peptidergic state variables")
    ax_pep.set_xlim(0, t[-1]); ax_pep.set_ylim(-0.03, 1.05)
    ax_pep.set_xlabel("time (s)"); ax_pep.set_ylabel("activation")
    lp1, = ax_pep.plot([], [], color="#ff5df2", lw=1.8, label="NLP-12 arm (DVA→CKR-2)")
    lp2, = ax_pep.plot([], [], color="#ffe95e", lw=1.8, label="FLP-14 arm (FRPR-19)")
    ax_pep.legend(loc="lower right", fontsize=7, facecolor=PANEL, labelcolor=FG)

    # --- command readout ---
    ax_cmd.set_title("Command drive readout")
    ax_cmd.set_xlim(0, t[-1])
    rmax = max(1e-9, float(np.abs(rd).max()) * 1.1)
    ax_cmd.set_ylim(-0.05 * rmax, rmax)
    ax_cmd.set_xlabel("time (s)"); ax_cmd.set_ylabel("drive (a.u.)")
    lc1, = ax_cmd.plot([], [], color="#33e0ff", lw=1.6, label="AVB (forward)")
    lc2, = ax_cmd.plot([], [], color="#ff4d6d", lw=1.6, label="AVA (backward)")
    ax_cmd.legend(loc="upper right", fontsize=7, facecolor=PANEL, labelcolor=FG)

    def frame(k):
        spine.set_data(xs[k, :, 0], xs[k, :, 1])
        head.set_data([xs[k, 0, 0]], [xs[k, 0, 1]])
        cur1.set_xdata([t[k], t[k]]); cur2.set_xdata([t[k], t[k]])
        lV.set_data(t[:k + 1], V[:k + 1, vb05])
        lCa.set_data(t[:k + 1], Ca[:k + 1, vb05])
        lp1.set_data(t[:k + 1], pep[:k + 1]); lp2.set_data(t[:k + 1], pep2[:k + 1])
        lc1.set_data(t[:k + 1], rd[:k + 1, 0]); lc2.set_data(t[:k + 1], rd[:k + 1, 1])
        return spine, head, cur1, cur2, lV, lCa, lp1, lp2, lc1, lc2

    w = FFMpegWriter(fps=20, codec="libx264", bitrate=3200,
                     extra_args=["-pix_fmt", "yuv420p"])
    FuncAnimation(fig, frame, frames=T, interval=50, blit=True).save(out_path, writer=w)
    plt.close(fig)
    print("saved", out_path)

render(RUNS / "v62cl_s0w78_crawl_regress.npz",
       OUT / "suite_S9_crawl_oscilloscope.mp4",
       "Whole-worm platform, real-time observability — forward crawl (frozen configuration)")
render(RUNS / "v62cl_s0w75c_switch1.npz",
       OUT / "suite_S10_switch_peptidergic.mp4",
       "Whole-worm platform, real-time observability — forward↔backward switch with peptidergic arms")
