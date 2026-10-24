# Supplementary assay movies S1-S8 for preprint v3, from ARCHIVED runs only.
# Layout per assay: fixed-frame body posture (left) + axial-speed trace with
# cursor (right top) + assay context panel (right bottom). English labels.
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

_HERE = Path(__file__).resolve()
if _HERE.parent.name == "figures":
    ROOT = _HERE.parent.parent                       # model_release_v3/figures/ -> release root
    RUNS = ROOT / "runs" / "v62_closed_loop"
    OUT = ROOT / "movies"
else:                                                # preprint working dir
    ROOT = _HERE.parent
    RUNS = ROOT / "model_release_v3" / "runs" / "v62_closed_loop"
    OUT = ROOT / "movies"
OUT.mkdir(exist_ok=True)

C = dict(worm="#1565C0", head="#D81B60", trail="#90CAF9", sp="#00897B",
         stim="#EF6C00", src="#C62828", cmd="#5E35B1")

def axial_speed(xs):
    com = xs.mean(axis=1)
    hv = xs[:, 0] - xs[:, 5]
    hn = hv / (np.linalg.norm(hv, axis=1, keepdims=True) + 1e-12)
    step = np.diff(com, axis=0)
    return np.sum(step * hn[:-1], axis=1) / 0.05

def render(tag, out, title, stim_windows=(), chem_src=None, extra=None):
    d = np.load(RUNS / f"{tag}.npz", allow_pickle=True)
    xs = d["xs"]; T = xs.shape[0]; dt = float(d["dt_block_ms"][0]) / 1000.0
    t = np.arange(T) * dt
    v = np.concatenate([[0.0], axial_speed(xs)])
    com = xs.mean(axis=1)

    fig = plt.figure(figsize=(12.6, 5.4), dpi=110)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.42,
                          left=0.06, right=0.98, top=0.86, bottom=0.11)
    axw = fig.add_subplot(gs[:, 0]); axs = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, 1])
    fig.suptitle(title, fontsize=11.5, fontweight="bold")

    pad = 0.8
    axw.set_xlim(xs[:, :, 0].min() - pad, xs[:, :, 0].max() + pad)
    axw.set_ylim(xs[:, :, 1].min() - pad, xs[:, :, 1].max() + pad)
    axw.set_aspect("equal"); axw.set_title("Body posture (fixed frame)")
    axw.set_xlabel("x (mm)"); axw.set_ylabel("y (mm)")
    if chem_src is not None:
        axw.add_patch(plt.Circle(chem_src, 0.35, fc="none", ec=C["src"], lw=2.0))
        axw.scatter(*chem_src, marker="*", s=200, color=C["src"], zorder=6,
                    label="chemoattractant source")
        axw.legend(loc="upper left", fontsize=8)
    axw.plot(com[:, 0], com[:, 1], ":", color=C["trail"], lw=1.4, zorder=2)
    spine, = axw.plot([], [], lw=4.0, color=C["worm"], solid_capstyle="round", zorder=4)
    head, = axw.plot([], [], "o", ms=8, color=C["head"], zorder=5)

    axs.set_title("Axial speed (head->tail positive)")
    axs.set_xlim(0, t[-1])
    vm = max(0.25, float(np.abs(v).max()) * 1.15)
    axs.set_ylim(-vm, vm)
    axs.set_xlabel("time (s)"); axs.set_ylabel("mm/s")
    axs.axhline(0, color="#999", lw=0.7)
    for a, b in stim_windows:
        axs.axvspan(a, min(b, t[-1]), color=C["stim"], alpha=0.18)
    lV, = axs.plot([], [], color=C["sp"], lw=2.0)
    cur = axs.axvline(0, color="#333", lw=1.0)

    axc.set_title("Command drive (AVB fwd / AVA bwd)")
    rd = d["readout"]
    axc.set_xlim(0, t[-1]); axc.set_ylim(-0.05, float(rd.max()) * 1.15 + 1e-6)
    axc.set_xlabel("time (s)"); axc.set_ylabel("a.u.")
    lB, = axc.plot([], [], color="#1E88E5", lw=1.6, label="AVB")
    lA, = axc.plot([], [], color="#E53935", lw=1.6, label="AVA")
    axc.legend(loc="upper right", fontsize=7.5)
    if extra:
        extra(axc, d, t)

    def frame(k):
        spine.set_data(xs[k, :, 0], xs[k, :, 1])
        head.set_data([xs[k, 0, 0]], [xs[k, 0, 1]])
        lV.set_data(t[:k + 1], v[:k + 1]); cur.set_xdata([t[k], t[k]])
        lB.set_data(t[:k + 1], rd[:k + 1, 0]); lA.set_data(t[:k + 1], rd[:k + 1, 1])
        return spine, head, lV, cur, lB, lA

    w = FFMpegWriter(fps=20, codec="libx264", bitrate=2400, extra_args=["-pix_fmt", "yuv420p"])
    FuncAnimation(fig, frame, frames=T, interval=50, blit=True).save(OUT / out, writer=w)
    plt.close(fig)
    print("saved", out, flush=True)

SRC = (3.5, 1.0)
render("v62cl_s0w78_crawl_regress", "suite_S1_forward_crawl.mp4",
       "S1  Forward crawl — 0.222 mm/s, 0.556 Hz, emergent dorsoventral wave")
render("v62cl_s0w78_d090_final", "suite_S2_chemotaxis_d090.mp4",
       "S2  Salt chemotaxis, 90 deg assay — weathervane steering onto the source (+3.46 mm)",
       chem_src=SRC)
render("v62cl_s0w78_d270_final", "suite_S3_chemotaxis_d270.mp4",
       "S3  Salt chemotaxis, 270 deg assay — smooth left turn onto the source (+0.96 mm)",
       chem_src=SRC)
render("v62cl_s0w78_d180_pirrev120", "suite_S4_chemotaxis_d180.mp4",
       "S4  Salt chemotaxis, 180 deg assay — pirouette-arm U-turn (+0.88 mm)",
       chem_src=SRC)
render("v62cl_s0w75b_touch400", "suite_S5_touch_escape.mp4",
       "S5  Gentle-touch escape — AVA pulse 6-8 s drives a reversal bout (Wicks 1995)",
       stim_windows=[(6, 8)])
render("v62cl_s0w75b_ash400", "suite_S6_ash_escape.mp4",
       "S6  ASH chemical escape, two-pulse protocol — response on the 2nd pulse (weak pass, see text)",
       stim_windows=[(4, 6), (14, 16)])
render("v62cl_s0w75c_switch1", "suite_S7_ava_switch.mp4",
       "S7  AVA photo-switch F->B->F — command-state switch within +/-1 s (Piggott 2011)",
       stim_windows=[(10, 22)])
render("v62cl_s0w75_ris", "suite_S8_ris_arrest.mp4",
       "S8  RIS activation arrests locomotion — speed 0.17 -> ~0.01 mm/s, recovery on offset",
       stim_windows=[(12, 18)])
