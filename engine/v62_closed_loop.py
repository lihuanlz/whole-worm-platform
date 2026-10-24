# -*- coding: utf-8 -*-
# v62_closed_loop.py — closed-loop whole-worm simulation entry point.
#
# Full-channel nervous system (323 cells, conductance-based, 29 ODEs/cell)
# coupled to an original 96-cell muscle layer (muscle96.py) and an original
# 24-segment viscoelastic rod body (rod2d_own.py). Synaptic and gap-junction
# weights are anatomical EM contact counts (theta=0; nothing is trained).
# The undulatory wave is sustained by closed-loop stretch feedback through
# the body; the network supplies gating / steering / RIV readout.
# Full network state (292x29) is logged every 50 ms.
#
# Legacy mode (--rod2d borrowed, muscle96 off) requires the third-party
# worm-whisperer package (https://github.com/kairess/worm-whisperer);
# it is NOT needed for the champion configuration reported in the
# manuscript and its modules are imported lazily only in that mode.
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
_WW = os.environ.get("WORM_WHISPERER_PATH", str(HERE.parent / "worm-whisperer"))
if os.path.isdir(_WW):
    sys.path.insert(0, _WW)

DT_MS = 0.025
BLOCK_MS = 50.0
CTRL_HZ = 30
NET_NPZ = HERE.parent / "data" / "net_gapfix.npz"
OUT = HERE.parent / "runs" / "v62_closed_loop"


GSYN = 0.005   # global chemical-synapse conductance gain (override with --gsyn)
GGAP = 0.3     # global electrical-synapse gain (override with --ggap)
GNCA_A_ARG = -1.0  # class-A NCA conductance override passed via --gncaA (v84)
GLEAK_GABA_ARG = 1.0  # DD/VD gleak multiplier passed via --gabaleak (v86, cross-inhibition oscillation fix)
GEGL36_GABA_ARG = -1.0  # DD/VD gegl36 override passed via --gabaegl36 (v87, plateau repolarisation fix)
UNC2PED_A_ARG = -1.0   # class-A UNC-2 non-inactivating fraction passed via --unc2pedA (W13, Gao 2018 oscillation window current)
GSK_A_ARG = -1.0       # class-A KCNL/SK conductance passed via --gskA (W13, slow brake, restored underdetermined direction)
CATAUSCALE_A_ARG = -1.0  # class-A somatic-Ca clearance multiplier passed via --catauscaleA (W13, microdomain 50 ms -> somatic 2 s)
GUNC2_A_ARG = -1.0     # class-A gunc2 override passed via --gunc2A (W13, network-scale window current)
GUNC2_AVA_ARG = -1.0   # AVA gunc2 override passed via --gunc2ava (W52, Meng 2024 UNC-2 baseline near opening threshold)
UNC2PED_AVA_ARG = -1.0 # AVA UNC-2 non-inactivating fraction passed via --unc2pedava (W52, regenerative window current)
GSK_AVA_ARG = -1.0     # AVA KCNL/SK conductance passed via --gskava (W52, plateau self-terminates in seconds)
CATAUSCALE_AVA_ARG = -1.0  # AVA somatic-Ca clearance multiplier passed via --catauscaleava (W52)


# ---- Chemotaxis world layer (architecture after worm-whisperer worm/env/chem.py, Lee 2026, MIT) ----
# Key difference: their ASE current is relayed into the command layer by an external RL-agent loop (their network is silent from sensory to command);
# here ASEL/ASER currents are injected directly into the full-channel network, walking the real anatomical path ASEL/ASER->AIY/AIZ->command layer.
class ChemField:
    """2D Gaussian concentration field (attractant source)"""
    def __init__(self, source_xy, sigma=2.0, peak=1.0):
        self.src = np.asarray(source_xy, float); self.sigma = sigma; self.peak = peak
    def conc(self, xy):
        d2 = ((np.asarray(xy) - self.src) ** 2).sum(-1)
        return self.peak * np.exp(-d2 / (2 * self.sigma ** 2))


class ASEInput:
    """dC/dt of the head-node concentration after low-pass(τ): rising->ASEL (ON), falling->ASER (OFF)
    physiology: Suzuki et al. 2008 (ASEL/ASER ON-OFF asymmetry);
          Pierce-Shimomura 1999; Iino & Yoshida 2009 (chemotaxis depends on dC/dt)"""
    def __init__(self, field, gain=50.0, tau=0.5):
        self.f, self.gain, self.tau = field, gain, tau
        self.prev = None; self.dcdt = 0.0
    def currents(self, head_xy, dt):
        c = float(self.f.conc(head_xy))
        if self.prev is not None:
            self.dcdt += (dt / self.tau) * ((c - self.prev) / dt - self.dcdt)
        self.prev = c
        return max(self.dcdt, 0.0) * self.gain, max(-self.dcdt, 0.0) * self.gain, c


def build_v62():
    import anchored_engine_v57 as v57
    v57.GUNC2_AB = 0.4
    if GNCA_A_ARG >= 0.0:
        v57.GNCA_A = GNCA_A_ARG
        print(f"[gncaA] class-A (VA/DA) NCA Na+ leak overridden to {GNCA_A_ARG:g} nS "
              f"(Gao 2018 + Frontiers 2022: NCA is the necessary engine of the A-MN intrinsic oscillation)", flush=True)
    if GLEAK_GABA_ARG != 1.0:
        v57.GLEAK_GABA = GLEAK_GABA_ARG
        print(f"[gabaleak] DD/VD gleak x{GLEAK_GABA_ARG:g} "
              f"(C2.6: VD/DD calcium should oscillate with the crawl rhythm; fixes the tonic-saturation clamp)", flush=True)
    if GEGL36_GABA_ARG >= 0.0:
        v57.GEGL36_GABA = GEGL36_GABA_ARG
        print(f"[gabaegl36] DD/VD gegl36 = {GEGL36_GABA_ARG:g} nS "
              f"(C2.6: EGL-36 repolarises the EGL-19 plateau, restoring VD/DD wave-frequency oscillation; VD5 anchor holds)", flush=True)
    if UNC2PED_A_ARG >= 0.0:
        v57.UNC2PED_A = UNC2PED_A_ARG
        print(f"[unc2pedA] class-A (VA/DA) UNC-2 non-inactivating fraction = {UNC2PED_A_ARG:g} "
              f"(W13: Gao 2018 oscillation window current; restores A-MN bistable regenerative excitability)", flush=True)
    if GSK_A_ARG >= 0.0:
        v57.GSK_A = GSK_A_ARG
        print(f"[gskA] class-A (VA/DA) KCNL/SK = {GSK_A_ARG:g} nS (W13: slow brake)", flush=True)
    if CATAUSCALE_A_ARG >= 0.0:
        v57.CATAUSCALE_A = CATAUSCALE_A_ARG
        print(f"[catauscaleA] class-A (VA/DA) Ca clearance x{CATAUSCALE_A_ARG:g} (W13: somatic Ca 2 s)", flush=True)
    if GUNC2_A_ARG >= 0.0:
        v57.GUNC2_A = GUNC2_A_ARG
        print(f"[gunc2A] class-A (VA/DA) gunc2 = {GUNC2_A_ARG:g} nS (W13: network-scale window current)", flush=True)
    if GUNC2_AVA_ARG >= 0.0:
        v57.GUNC2_AVA = GUNC2_AVA_ARG
        print(f"[gunc2ava] AVA gunc2 = {GUNC2_AVA_ARG:g} nS "
              f"(W52: Meng 2024 PMC11006214 AVA baseline near the UNC-2 opening threshold / multistable -30~-10)", flush=True)
    if UNC2PED_AVA_ARG >= 0.0:
        v57.UNC2PED_AVA = UNC2PED_AVA_ARG
        print(f"[unc2pedava] AVA UNC-2 non-inactivating fraction = {UNC2PED_AVA_ARG:g} "
              f"(W52: regenerative window current, same Gao 2018 anchor)", flush=True)
    if GSK_AVA_ARG >= 0.0:
        v57.GSK_AVA = GSK_AVA_ARG
        print(f"[gskava] AVA KCNL/SK = {GSK_AVA_ARG:g} nS (W52: plateau self-terminates in seconds)", flush=True)
    if CATAUSCALE_AVA_ARG >= 0.0:
        v57.CATAUSCALE_AVA = CATAUSCALE_AVA_ARG
        print(f"[catauscaleava] AVA Ca clearance x{CATAUSCALE_AVA_ARG:g} (W52: somatic Ca 2 s)", flush=True)
    import anchored_engine_v58 as v58
    v58.VREL_OFF = -10.0
    import v62_fast_circuit as vf
    import v62_numba_circuit as vn
    vf.DT = DT_MS; vn.DT = DT_MS
    vf.eng = v58; vn.eng = v58
    net = vn.Worm62NB(GSYN, GGAP, g_wl=0.0, netnpz=NET_NPZ, e_inh=-90.0, vclip=(-100.0, 60.0))
    from scipy.sparse import csr_matrix
    net.M_wlex = csr_matrix((net.N, net.N)); net.M_wlin = csr_matrix((net.N, net.N))
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", type=float, default=10.0, help="AVB stimulus amplitude pA")
    ap.add_argument("--initdir", type=float, default=180.0,
                    help="initial heading deg (azimuth the animal faces, 0=+x 90=+y; default 180 = legacy behaviour bit-identical, "
                         "the heading of Rod2D.initial is the rod-extension direction = opposite of facing, hence the internal +180)")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--smoke", action="store_true", help="run only 2 s baseline + 3 s stimulus")
    ap.add_argument("--tag", default="")
    ap.add_argument("--block", action="append", default=[], help="channel block as name:multiplier, e.g. gunc2:0")
    ap.add_argument("--kmax", type=float, default=6.0, help="motor-layer maximum curvature 1/mm")
    ap.add_argument("--gsr", type=float, default=2.0, help="stretch-receptor gain")
    ap.add_argument("--span", type=float, default=0.25, help="stretch-receptor spatial span fraction")
    ap.add_argument("--pep", type=float, default=0.0,
                    help="peptidergic gain strength alpha (DVA->NLP-12->CKR-2 axis, 0=virtual ckr-2 KO)")
    ap.add_argument("--peptau", type=float, default=3.0, help="peptidergic modulation time constant s")
    ap.add_argument("--ckr1", type=float, default=0.0,
                    help="phase-2 CKR-1 peptidergic gain G0 (pA), 0=off (legacy behaviour bit-identical). "
                         "Chen 2022 Cell Rep 38:110330 (PMID 35139370) + Chen 2024 (PMC10952047, "
                         "Mendeley 10.17632/jm9yt2hdnk.1 raw data): the same Galpha-q receptor CKR-1 integrates two motor states at SMD — "
                         "NLP-12 arm (DVA stretch->SMD, body-curvature gain; nlp-12 loss -21.6%% / ckr-1 -17.9%%) + "
                         "NLP-18 arm (ASI->SMD+AIB, omega-turn gain; nlp-18 loss -11.6%% / ckr-1 -12.3%%). "
                         "Fig3: NMJ minis/EPSCs are unaffected by CKR-1 -> the gain acts on SMD/AIB excitability, not on synaptic transmission")
    ap.add_argument("--ckr1tau12", type=float, default=12.9,
                    help="NLP-12 arm time constant s (Chen 2024 Fig5E measured Ca2+ 10-90%% rise time 12.93+-0.98 s)")
    ap.add_argument("--ckr1tau18", type=float, default=22.1,
                    help="NLP-18 arm time constant s (same Fig5E, measured 22.08+-3.25 s)")
    ap.add_argument("--ckr1m0", type=float, default=0.0,
                    help="initial value M0 in [0,1] of the CKR-1 two-arm modulation state (warm-up at the same level, as for ggapmod; 0=legacy behaviour)")
    ap.add_argument("--ggapmod", type=float, default=0.0,
                    help="phase-2 mechanisation (2026-10-21): gap-junction conductance modulated by a state variable. When !=0, Lg is scaled per block by "
                         "g_eff=(1-k(1-M))xGGAP; M=activity of the anterior mechanosensory layer (ALM/AVM gentle touch + dopaminergic "
                         "CEP/ADE) via a DOP-2 GPCR slow variable. Anchors: Chen 2007 unc-1 gating lowers UNC-9 coupling by "
                         "~70%% (PMC1976340); LeBoeuf 2015 DOP-2->UNC-7 (PMC4495247); Schwarz 2018 "
                         "EGL-4/PKG->UNC-7/9 (PMC6283151); Kawano 2011 UNC-7/9 shunt (PMID 22099460). "
                         "the escape/crawl 3.3x conductance ratio falls exactly at the two ends of the unc-1 measured modulation range. Requires --ggap 1.0 "
                         "(anchored full scale): M~0->0.3 crawl gear, M~1->1.0 escape gear. 0=off=legacy behaviour bit-identical")
    ap.add_argument("--ggapmodtau", type=float, default=3.0,
                    help="ggapmod state-variable M time constant s (GPCR signalling magnitude anchor, same 1-10 s class as peptau)")
    ap.add_argument("--ggapmodk", type=float, default=0.7,
                    help="ggapmod modulation depth (Chen 2007 measured: disrupting the unc-1;unc-9 pathway lowers coupling by ~70%%)")
    ap.add_argument("--ggapmodm0", type=float, default=0.0,
                    help="ggapmod initial/set state M0 in [0,1] (slow-modulation-state semantics: 0=crawl gear 1=escape gear; "
                         "with --ggapmodtau 600 (Schwarz 2018 EGL-4/PKG sleep-wake minutes-scale) the state is locked. "
                         "acute sensory-driven switching was falsified on 2026-10-21 (dose response at 0.6/0.75/0.9 three levels "
                         "frac_bwd=0.02/0.05/0.00 — the coupling state must be pre-established over seconds): escape/crawl are two "
                         "slow modulation states, not acute switches; acute touch escape is executed by the chemical pathway (Chalfie 1985)")
    ap.add_argument("--ggapmodash", type=float, default=0.0,
                    help="B5: 0=drive pool excludes ASH=legacy behaviour bit-identical; !=0 adds ASHL/R to the ggapmod drive pool "
                         "(Chaubey 2023 PMC10319955: state-dependent innexin/INX-20 modulation is located at ASH itself; "
                         "the drive pool of the 2026-10-21 acute-switch falsification experiment was ALM/AVM/CEP/ADE without ASH — "
                         "M cannot rise during chemical escape; this flag fills that structural gap)")
    ap.add_argument("--avarect", type=float, default=0.0,
                    help="W49 (2026-10-23): AVA<->A-MN rectifying gap-junction strength in [0,1], 0=off=legacy behaviour bit-identical. "
                         "anchor Shui 2020 Sci Adv 6:eabb3076 (PMC7455501): UNC-7b (AVA side)/UNC-9 (A-MN side) "
                         "heterotypic-channel ball-receptor gating — Vj=V_AVA-V_AMN>0 (AVA more depolarised) ball plugs pore=closed, "
                         "Vj<0 ball released=open (current flows only in the A->AVA reverse direction); Liu 2017 Nat Commun 8:14818 "
                         "in-vivo dual-electrode unidirectional rectification; Kawano 2011 Neuron 72:572 shunt resting inhibition; "
                         "Gao 2018 eLife e29915 state-dependent inhibition+facilitation; Starich 2009 Neural Dev 4:16 composition. "
                         "implementation: per block, Boltzmann gating from the previous block's voltages (50 ms delay, as in ggapmod) rebuilds Lg")
    ap.add_argument("--avarectv0", type=float, default=17.0,
                    help="W49 Boltzmann half-activation Vj midpoint mV (Tier-2 calibration 2026-10-23: crawl-state Vj=+9.2 must pass/"
                         "bout-state Vj=20-77 must shut; separation solution V0=17; biological mapping ~3x voltage compression -> ~+50 mV; "
                         "Shui 2020 Table S3 fitted value pending; V0=0 measured to collapse crawling v=0.064 on record)")
    ap.add_argument("--avarecta", type=float, default=0.22,
                    help="W49 Boltzmann slope /mV (Tier-2 calibration: =engine ~3x voltage compression x Liu 2006 innexin "
                         "gating slope 0.10 mapping; Table S3 pending)")
    ap.add_argument("--avarectmin", type=float, default=0.05,
                    help="W49 Gjmin residual conductance ratio (Tier-2: Liu 2017 reverse coupling ~0, 'strongest rectification')")
    ap.add_argument("--cellablmode", type=str, default="clamp", choices=["clamp", "edge"],
                    help="cellabl ablation protocol: clamp=legacy behaviour (-300 pA constant-current hyperpolarisation, edges intact); "
                         "edge=true laser-ablation semantics (delete every chemical+electrical edge touching the cell, no injection). "
                         "under the clamp protocol a clamped cell drags its neighbours backward through residual gap edges = artefact; "
                         "reproducing the Gao 2018 ablation preparation requires edge mode")
    ap.add_argument("--mnprop", type=float, default=0.0,
                    help="class-B motor-neuron proprioception gain pA/(1/mm) (Wen et al. 2012, Neuron 76:750: "
                         "class-B MN unbranched neurites sense adjacent body-segment bending; activity driven phase-locked to the body wave); "
                         "0=off=legacy behaviour bit-identical; D/V antisymmetric injection, polarity verified against the preferred_curvature source: "
                         "κ>0=dorsal contraction -> DB +κ, VB -κ (v66 fix; v65 assigned the opposite, locking statistics unaffected), injected per block (50 ms)")
    ap.add_argument("--mnoffset", type=int, default=0,
                    help="anterior offset of the class-B MN proprioceptive sensing position (in body segments; Wen et al. 2012 experimental anchor: "
                         "VB/DB depolarisation correlates with ipsilateral bending ~200 um anterior of the soma -> 22 segments ~50 um/seg -> ~4 segs; "
                         "Bryden & Cohen 2008 theory: anterior sensing + time delay -> travelling-wave dispersion, minimal B loop oscillates). "
                         "0=same segment=legacy behaviour bit-identical")
    ap.add_argument("--mnproptau", type=float, default=0.0,
                    help="EMA time constant ms of the first-channel (Wen anterior positive-feedback) curvature signal (Bryden & Cohen 2008 "
                         "Biol Cybern 98:339 travelling-wave dispersion mechanism — anterior sensing + time delay -> head-tail travelling wave; "
                         "instantaneous positive feedback synchronises without propagating; the delay provides the inter-segment phase gradient; "
                         "0=off=passthrough=legacy behaviour bit-identical; active only when mnprop!=0)")
    ap.add_argument("--mnprop2", type=float, default=0.0,
                    help="second proprioceptive channel gain pA/(1/mm) (dual-sign coexistence test: Wen 2012 measured = ipsilateral-contraction positive feedback"
                         "(this parameter positive, anterior sensing); the Cohen group modelling assumption = ipsilateral-stretch negative feedback (negative, posterior sensing); "
                         "when both channels coexist they superpose linearly with their own gains; criterion = Wen Fig7: B-MN activity must correlate positively with ipsilateral bending; "
                         "0=off=legacy behaviour bit-identical)")
    ap.add_argument("--mnoffset2", type=int, default=0,
                    help="second-channel sensing-position offset (body segments, negative=posterior; active only when mnprop2!=0)")
    ap.add_argument("--mnprop2tau", type=float, default=0.0,
                    help="first-order low-pass time constant ms of the second-channel curvature signal (Zhao et al. 2025 PLOS Comput Biol "
                         "21(6):e1013171: negative feedback comes from the delayed current after muscle contracts to a set length; muscle contraction/relaxation itself has "
                         "hundred-ms dynamics -> EMA proxies the muscle's mechanical delay, removing the 10 Hz artefact of the 1-block (50 ms) pure delay; "
                         "0=off=passthrough=legacy behaviour bit-identical; active only when mnprop2!=0)")
    ap.add_argument("--mnprop2thr", type=float, default=0.0,
                    help="second-channel negative-feedback contraction threshold 1/mm (Zhao 2025: the negative-feedback current returns to the motor neuron only after the muscle contracts to a set length"
                         "-> half-wave rectification max(ipsilateral contraction - threshold, 0), avoiding linear cancellation against the positive-feedback channel; "
                         "0=off=purely linear=legacy behaviour bit-identical; active only when mnprop2!=0)")
    ap.add_argument("--mnback", type=int, default=0,
                    help="mirror-reversal of the second proprioceptive channel's receptive field during backward (root-cause-A fix: "
                         "mndrv is not direction-gated -> with backward gating on, the B-MN loop still pushes forward tuned tail->head; the two forces fight and only slow the animal; "
                         "biological counterpart: in backward the wave initiates at the head (tail->head = forward, head->tail = backward), "
                         "loop mirroring = sign-reversing the sensing offset (posterior sensing <-> anterior sensing); "
                         "1=when backward gating (relg/rel dir_state<0) is active mnoffset2 is negated, 0=off=legacy behaviour bit-identical; "
                         "active only when mnprop2!=0)")
    ap.add_argument("--bmnclip", type=str, default="clip0", choices=["clip0", "reflect"],
                    help="class-B sensing-site boundary semantics (clip0=legacy behaviour: out-of-range sensing sites clamp to the boundary segment "
                         "=posterior class B (VB09-11/DB06-07) all clamped to segment 23 with tail-tip κ~0 -> self-consistent dead zone (crawl tail wave disappears), "
                         "under backward mirroring anterior class B clamps to segment 0 = forced in-phase -> head synchronised block = standing-wave-isation of the backward wave (+2°/seg); "
                         "reflect=out-of-range sensing sites mirror back into the body (same fix as W57 class A), boundary cells still receive travelling-wave signal; "
                         "default clip0=legacy behaviour bit-identical)")
    ap.add_argument("--mnanat", type=int, default=0,
                    help="class-B MN soma-to-segment mapping mode (fix of the tail dead zone): "
                         "1=located by the anatomical innervation table (neuron_muscle.csv: DB01->muscle rows 07-09->segment ~6, "
                         "DB07->muscle rows 22-24->segment ~21; proprioceptive positions align with muscle-innervation positions, closing the local feedback loop), "
                         "0=old uniform sprinkling over nseg segments by index=legacy behaviour bit-identical")
    ap.add_argument("--rivabl", type=float, default=0.0,
                    help="RIVL/R constant injection pA (negative=functional ablation: Gray et al. 2005 "
                         "PNAS 102:3184 doi:10.1073/pnas.0409009101, RIV ablation -> post-reversal ventral-turn bias disappears; "
                         "0=off=legacy behaviour bit-identical)")
    ap.add_argument("--cellabl", type=str, default="",
                    help="functional-ablation cell list, comma-separated, prefix wildcards supported (e.g. 'AVA*,AVB*,DB*,VB*'): "
                         "constant strong hyperpolarisation = HisCl-style silencing (Gao 2018 eLife 7:e29915 premotor/B-MN ablation protocol); "
                         "empty=off=legacy behaviour bit-identical")
    ap.add_argument("--cellablp", type=float, default=-300.0,
                    help="cellabl injection amplitude pA (default -300, strong hyperpolarisation)")
    ap.add_argument("--gncaA", type=float, default=-1.0,
                    help="class-A (VA/DA) NCA Na+ leak conductance override nS (v84, Gao 2018 oscillation engine; "
                         "-1=off=legacy behaviour bit-identical)")
    ap.add_argument("--unc2pedA", type=float, default=-1.0,
                    help="class-A (VA/DA) UNC-2 inactivation-gate non-inactivating fraction [0,1) (W13 2026-10-23, Gao 2018: "
                         "the A-MN endogenous oscillation requires the UNC-2 window current, unc-2(gf)=weakened inactivation; single-cell probes adjudicate "
                         "the original cell is monostable without hysteresis and cannot oscillate; ped 0.20 restores the bistable window + AVA shunt gating, all three phases pass; "
                         "-1=off=legacy behaviour bit-identical)")
    ap.add_argument("--gskA", type=float, default=-1.0,
                    help="class-A (VA/DA) KCNL/SK calcium-activated potassium conductance nS (W13: restored underdetermined direction of anatomical expression, "
                         "slow brake; 0.05 passes anchors, >=0.1 clamps ignition; -1=off=legacy behaviour bit-identical)")
    ap.add_argument("--catauscaleA", type=float, default=-1.0,
                    help="class-A (VA/DA) intracellular calcium clearance time-constant multiplier (W13: 50 ms microdomain -> somatic seconds, "
                         "Gao 2018 DA9 Ca oscillation ~50 s; 40 taken (=2 s); -1=off=legacy behaviour bit-identical)")
    ap.add_argument("--gunc2A", type=float, default=-1.0,
                    help="class-A (VA/DA) gunc2 override nS (W13: E2d measured the GUNC2_AB=0.4 window current under network conductances"
                         "<1%% swamped, needs network scale 1.0+; VB/DB untouched; -1=off=legacy behaviour bit-identical)")
    ap.add_argument("--gunc2ava", type=float, default=-1.0,
                    help="AVA gunc2 override nS (W52, Meng 2024 Sci Adv 10:adk0002: AVA baseline near the UNC-2 opening threshold, "
                         "bistable regimes -30/-20 and -15/-10; the local engine's AVA gunc2=0 = underdetermined clamped-fit direction, "
                         "the command layer lacks a regenerative carrier; -1=off=legacy behaviour bit-identical)")
    ap.add_argument("--unc2pedava", type=float, default=-1.0,
                    help="AVA UNC-2 inactivation-gate non-inactivating fraction [0,1) (W52: hinf~0 at -25 mV, "
                         "the regenerative window current requires ped; -1=off=legacy behaviour bit-identical)")
    ap.add_argument("--gskava", type=float, default=-1.0,
                    help="AVA KCNL/SK calcium-activated potassium conductance nS (W52: activity-dependent termination, same discipline as W13 A-MN; "
                         "-1=off=legacy behaviour bit-identical)")
    ap.add_argument("--catauscaleava", type=float, default=-1.0,
                    help="AVA intracellular calcium clearance time-constant multiplier (W52: somatic seconds-scale termination; -1=off=legacy behaviour bit-identical)")
    ap.add_argument("--gabaleak", type=float, default=1.0,
                    help="DD/VD gleak multiplier (v86, eLife 2021 PMC7986531: VD/DD calcium should oscillate with the crawl rhythm; "
                         "measured tonic saturation clamps it silent -> ventral overdrive, left drift; 1.0=legacy behaviour bit-identical)")
    ap.add_argument("--gabaegl36", type=float, default=-1.0,
                    help="DD/VD gegl36 override nS (v87, same anchor: VD recipe gegl36=0 is an underdetermined clamped-fit direction, "
                         "the EGL-19 L-type plateau (-25..-5 mV negative slope) lacks high-voltage K repolarisation -> parks at a ~-4 mV plateau, wave frequency dies; "
                         "EGL-36 closed at rest spares the VD5 anchor; single-cell probes 0.6/1.0 pass anchors; -1=legacy behaviour bit-identical)")
    ap.add_argument("--amnprop", type=float, default=0.0,
                    help="class-A MN (VA/DA) proprioceptive channel gain pA/(1/mm) (v85, Gao et al. 2018 eLife 7:e29915: "
                         "class A are backward endogenous oscillators and accept proprioceptive entrainment; the loop mirrors the B-chain backward morphology = anterior sensing; "
                         "0=off=legacy behaviour bit-identical)")
    ap.add_argument("--amnoffset", type=int, default=6,
                    help="class-A channel sensing-position offset toward the head, in body segments (positive=anterior/head-side sensing -> head->tail propagation = backward wave, "
                         "isomorphic to the B chain after mnback mirroring; active only when amnprop!=0)")
    ap.add_argument("--amngate", type=int, default=1,
                    help="class-A channel direction gating (Kato et al. 2015 Cell 163:656 forward/backward subcircuit division): "
                         "1=active only in the backward state (dir_state<0), 0=always active; active only when amnprop!=0)")
    ap.add_argument("--amnclip", type=str, default="clip0", choices=["clip0", "edge0", "reflect"],
                    help="class-A sensing boundary semantics (W55: clip0=legacy behaviour — head-side soma sensing sites clip to segment 0 "
                         "=anterior 8 segments forced in-phase (numerical artefact); edge0=zero injection beyond the rostral boundary = anterior class-A pure endogenous oscillators "
                         "(Gao 2018 eLife 7:e29915 class-A endogenous-oscillator property), wave seeds; default clip0=legacy behaviour bit-identical; "
                         "W57 reflect=out-of-range sensing sites mirror back into the body (fix for edge0 starvation letting tail VA09-12 be pinned down by VD GABA, "
                         "no in-phase artefact: after reflection the per-segment sensing sites stay dispersed)")
    ap.add_argument("--amndir", type=str, default="anterior", choices=["anterior", "posterior"],
                    help="class-A proprioception direction (W56 2026-10-23 reversal: Gao 2018 Fig1B/2B hard evidence that the backward wave propagates posterior->anterior "
                         "(tail->head), Fig5 VA11 leads VA10 -> the relay must sense the tail side (posterior); v85b's anterior sensing produced a head->tail wave"
                         "=the forward direction — the reversed direction was the structural root cause of the J2 failure; anterior=legacy behaviour (v85b) bit-identical, "
                         "posterior=tail-side sensing (v85a direction, tail->head backward wave)")
    ap.add_argument("--amnhp", type=float, default=0.0,
                    help="class-A proprioceptive loop high-pass time constant, seconds (backward sawtooth-lock fix: stretch receptors are phasic/"
                         "adaptive — respond only to curvature change, static-bend DC is subtracted, breaking the strong-tonic-drive + negative-feedback"
                         "Turing-style static lock; signal=kappa_EMA80ms - kappa_EMA(amnhp); "
                         "0=off=legacy behaviour bit-identical; evidence level: mechanistic analogy, bounded estimate)")
    ap.add_argument("--bpropbwd", type=float, default=0.0,
                    help="backward-state B-chain (mnprop2) gain attenuation ratio (v85, Kato 2015 subcircuit mutual-inhibition analog: "
                         "when backward gating is active B-chain gain x(1-bpropbwd); 0=no attenuation=legacy behaviour bit-identical)")
    ap.add_argument("--bmnhp", type=float, default=0.0,
                    help="W58: backward-state mnprop2 (B-chain proprioceptive loop) high-pass time constant, seconds (stretch-receptor phasic-tonic mix/"
                         "tonic adaptation decay — textbook spindle-adaptation mechanism Matthews 1963 + worm DVA/TRP-4 phasic preference "
                         "Li 2006 Nature; cuts the DC channel of the bout self-sustaining loop: static S-bend -> mnprop2 DC -> B-chain tonic reversal"
                         "->AVA plateau supply -> bout never exits; active only when dir_state<0, crawl regression bit-unchanged; "
                         "reuses the amnhp slow-EMA reference; 0=off=legacy behaviour bit-identical)")
    ap.add_argument("--abx", type=float, default=1.0,
                    help="backward-state A-MN(VA/DA)->B-MN(VB/DB) excitatory chemical crosstalk gain (W54 2026-10-23, "
                         "Kato 2015 subcircuit mutual-inhibition analog, same family as bpropbwd: in the backward state the forward module should be suppressed; "
                         "anatomical audit: VA->VB w=5.508 locks VA into the forward program = VA synchronous without a wave (DA->DB only 1.65, "
                         "DA free -> phase gradient 0.31 rad/seg = lambda 20-seg travelling wave); when backward gating is active the A->B chemical-edge gain is multiplied by abx; "
                         "1.0=no attenuation=legacy behaviour bit-identical)")
    ap.add_argument("--avdx", type=float, default=1.0,
                    help="backward-state A-MN(VA/DA)<->VD/DD inhibitory-MN gap-junction gain (W54b 2026-10-23: "
                         "after abx 0.3 went live VA still synchronous -> next candidate locks VA<->VD gap 1.188 / DD->VA 0.532 "
                         "(DA side only 0.192); archived measurements show VD/DD/VB all in the synchronous block, VA dragged in; "
                         "mechanistic analogy = state-dependent innexin modulation (peptidergic tonic-calibration candidate, honestly labelled phenomenological); "
                         "when backward gating is active this edge set's gain is multiplied by avdx; 1.0=no attenuation=legacy behaviour bit-identical)")
    ap.add_argument("--sloa", type=float, default=0.0,
                    help="class-A MN SLO-1/SLO-2-type activity-dependent potassium conductance nS (v92, Yu/Wang/Gao 2022 Front Mol Neurosci "
                         "15:845733: slo-1(lf)/slo-2(lf) -> larger tonic/long-lasting rPSC + faster backward = the activity-dependent negative-feedback brake of the class-A oscillation;"
                         "UNC-2 Ca influx -> Ca/Na-activated K efflux terminates the burst, homologous to lamprey VGCC->KCa->AHP; "
                         "engine implementation: ca=somatic activity trace ((softplus(dV)-ln2)/25 influx, tau clearance), I_K=-G*ca*(V-E_K) outward; "
                         "sustained static drive is adapted away (DC suppressed), the 0.6 Hz rhythmic component passes = the soma-level mechanised realisation of the amnhp high-pass; "
                         "during crawling class A is silent dV≈0 -> ca→0 -> zero current; 0=off=legacy behaviour bit-identical)")
    ap.add_argument("--sloatau", type=float, default=1.0,
                    help="sloa calcium-trace clearance time constant s (Yu 2022: rPSC tonic rise 0.82 s / duration 3.9-9.9 s, seconds-scale adaptation; "
                         "1.0 s taken, same validation window as amnhp; sensitivity band on record)")
    ap.add_argument("--sloath", type=float, default=15.0,
                    help="sloa calcium-influx activation threshold dV mV (UNC-2=high-voltage-activated VGCC, opens only on strong depolarisation: Yu 2022/Gao 2018; "
                         "measured distribution: crawl class-A dV≤+11 / backward-bout peak median +56, threshold 15 sits in the gap -> crawl influx identically zero, "
                         "regression bit-unchanged; the hard rectification is a reduction of the steep HVA Boltzmann)")
    ap.add_argument("--amnosc", type=float, default=0.0,
                    help="class-A MN UNC-2 regenerative influx-arm gain pA (v92 steady-backward structural piece: the depolarising arm of the same UNC-2 channel of Yu 2022 —"
                         "HVA Ca2+ influx itself regeneratively depolarises and together with the sloa SLO brake arm forms a relaxation oscillator "
                         "(Gao 2018: class A = backward endogenous oscillator); oscillatory drive replaces tonic plateau supply -> the static Turing mode"
                         "cannot stand without continuous energy, and the proprioceptive chain entrains phase into a gentle travelling wave; reuses _infl=max(dV-sloath,0)/25 "
                         "-> identically zero during crawling, regressions naturally bit-unchanged; requires --sloa; 0=off=legacy behaviour bit-identical)")
    ap.add_argument("--amngrad", type=float, default=0.0,
                    help="class-A UNC-2 drive tail-high-head-low multiplicative gradient (W56, 2026-10-23; Gao 2018 eLife 7:e29915: "
                         "DA9, the most caudal unit, is most active = leading oscillator; the lamprey fastest-oscillator entrainment mechanism -> tail->head travelling wave; "
                         "factor=1+g*(2*seg/(nseg-1)-1): linear head(1-g) tail(1+g); gradient magnitude = underdetermined parameter (on record), "
                         "0=uniform=legacy behaviour bit-identical; active only when amnosc!=0)")
    ap.add_argument("--dvaprop", type=float, default=0.0,
                    help="DVA/TRP-4 amplitude-limiting brake gain pA/(1/mm) (v92, Li/Feng/Sternberg/Xu 2006 Nature 440:684 "
                         "PMC2865900: trp-4(lf) = exaggerated-bending phenotype -> DVA stretch sensing negatively inhibits ventral-cord MNs against overbending; "
                         "engine implementation: |kappa slow EMA| hyperpolarising injection at the A-MN soma segments; the travelling wave averages ~0 at the dvatau scale = AC passes, "
                         "static-bend DC passes fully = selectively dismantles the Turing static lock; active only in the backward state (dir_state<0) -> crawl regression untouched; "
                         "0=off=legacy behaviour bit-identical; evidence level: measured mechanism (phenotype direction) + bounded-estimate gain)")
    ap.add_argument("--dvatau", type=float, default=1.0,
                    help="dvaprop slow-EMA time constant s (peptidergic/dopamine amplitude regulation is a seconds-scale slow signal: Tao 2019 PVD/NLP-12, "
                         "PDE-dopamine-AVK-FLP-1 loop; 1.0 s taken, same-order window as the ~2 s wave period; sensitivity band on record)")
    ap.add_argument("--dvatarget", default="mn", choices=["mn", "muscle"],
                    help="dvaprop brake action point: mn=hyperpolarise class-A MNs (measured counterproductive, weakens the travelling wave's unlocking force); "
                         "muscle=scale muscle-row activation by 1/(1+G*|DC|) (Li 2006: DVA inhibits body-wall muscle contraction; "
                         "G in (1/mm)^-1, recommended 0.1-0.4) — the neural-side travelling wave is unharmed; the static bend's contraction is released directly")
    ap.add_argument("--avamaster", type=float, default=0.0,
                    help="AVA master-neuron slow tonic excitation of AVB, gain pA/mV (v93, Meng 2024 PMC11006214: "
                         "AVA tonic depolarisation slowly excites AVB via extrasynaptic ACh, tau on the order of minutes, making forward the default state and providing"
                         "an endogenous backward->forward exit clock; engine implementation: EMA(max(0,dV_AVA-floor)) excitation injected into AVB, "
                         "dV taken from the previous block (50 ms delay, as for pep2/ggapmod); 0=off=legacy behaviour bit-identical; "
                         "evidence level: measured mechanism (two-timescale model + TWK-40 perturbation validation) + bounded-estimate gain)")
    ap.add_argument("--avamastertau", type=float, default=30.0,
                    help="avamaster slow-EMA time constant s (Meng 2024: slow-excitation timescale on the order of minutes; 30 s taken as conservative lower bound)")
    ap.add_argument("--avamasterfloor", type=float, default=5.0,
                    help="avamaster counting floor mV: only depolarising activity of dV_AVA above this floor is accumulated (resting jitter not counted)")
    ap.add_argument("--sloe", type=float, default=-80.0,
                    help="sloa potassium equilibrium potential mV (physiological K+ reversal -70~-90, standard -80 taken)")
    ap.add_argument("--mnspan", type=int, default=0,
                    help="class-B MN proprioceptive window half-width, body segments (Wen et al. 2012 Neuron 76:750: "
                         "receptors are distributed along the neurite, not a soma point -> sensing = mean curvature over soma position ±span segments, "
                         "under the anatomical mapping point sensing leaves only 6 discrete sites and freezes the oscillation; the cross-segment window mimics neurite span; "
                         "0=point sensing=legacy behaviour bit-identical)")
    ap.add_argument("--mndrive", type=float, default=0.0,
                    help="class-B MN->muscle drive gain (v66 loaded loop): anatomical innervation table neuron_muscle.csv, "
                         "DB->dorsal muscle rows, VB->ventral muscle rows, drive = G * (W @ (softplus(dV)-ln2)); "
                         "0=off=legacy behaviour bit-identical")
    ap.add_argument("--dmndrive", type=float, default=0.0,
                    help="class-D MN->muscle cross-inhibition gain (White 1986 adult wiring: VD->ventral muscles, "
                         "DD->dorsal muscles, GABAergic inhibitory, subtracted from ipsilateral activation; anatomical placement: DD/VD have no long neurites "
                         "(Cohen group review), hence local without offset; Ji 2021 eLife: cross-inhibition maintains high frequency and wave form; "
                         "the neuron-level VD->VB reset is already in the connectome (sign=-1); this parameter supplies the muscle-level half; "
                         "0=off=legacy behaviour bit-identical)")
    ap.add_argument("--muscletau", type=float, default=0.0,
                    help="first-order integration time constant of body-wall muscle activation, ms (v68 muscle-layer dynamics: "
                         "Zhao et al. 2025 PLOS Comput Biol 21(6):e1013171 showed the crawl generator = "
                         "local negative-feedback loops of an MN + the muscles it innervates, delay coming from muscle contraction/relaxation dynamics; "
                         "Shen et al. 2016 Neuron 90:869 DINOs measured muscle calcium rhythm). "
                         "muscle activation m approaches the instantaneous drive drv exponentially with tau: dm/dt=(drv-m)/tau, "
                         "integrated on motor_dt (5 ms) substeps; 0=off=passthrough=legacy behaviour bit-identical")
    ap.add_argument("--smddrive", type=float, default=0.0,
                    help="SMD->head-muscle drive gain (v68 R6, Yeon et al. 2018 PLoS Biol 16(6):e2004929 "
                         "doi:10.1371/journal.pbio.2004929: SMDD/V are sensory-motor dual-role neurons, "
                         "innervating dorsal/ventral head muscles; Zhao et al. 2025 anterior oscillation loop = SMD + head muscles; "
                         "anatomical table neuron_muscle.csv: SMDDL/R->DR/DL01-06, SMDVL/R->VR/VL01-06, "
                         "same formula as class-B mndrive, softplus(dV)-ln2; 0=off=legacy behaviour bit-identical)")
    ap.add_argument("--smdprop", type=float, default=0.0,
                    help="SMD proprioceptive feedback gain pA/(1/mm); the sensed signal = mean curvature of the first 6 head segments "
                         "(Yeon 2018: SMDD senses head-muscle stretch via TRP-1/TRP2 = stretch reflex = negative-feedback form, "
                         "negative values taken: dorsal bend (κ>0) inhibits SMDD / ventral bend inhibits SMDV; isomorphic to the Zhao 2025 local negative-feedback loop; "
                         "0=off=legacy behaviour bit-identical)")
    ap.add_argument("--smdpropseg", type=str, default="0:6",
                    help="v82: SMD proprioceptive receptive-field segment range 'a:b' (default '0:6'=legacy behaviour bit-identical; "
                         "Yeon 2018 original text = neck muscle stretch, neck = segments 3:9, "
                         "that span is directly driven by the travelling-wave loop DB01/02 innervating segments 7-9 = the loop's natural ignition source)")
    ap.add_argument("--smdabl", type=str, default="",
                    help="SMD feedback ablation (trp-1,2 double-mutant simulation; Yeon 2018: mutation/ablation -> ventral circling): "
                         "D=shut SMDD feedback, V=shut SMDV feedback, empty=no ablation; active only when smdprop!=0")
    ap.add_argument("--smdpropbw", type=int, default=0,
                    help="W57: auto-shut the SMD reflex in the backward state (1=on, smdprop gain zeroed when dir_state<0; "
                         "w57d/e verdict: an always-on reflex is the main cause of backward spinning; 0=off=legacy behaviour bit-identical)")
    ap.add_argument("--muscle96", type=float, default=0.0,
                    help="v81 original physiological-grouped muscle-layer master gain (engine/muscle96.py): 96 body-wall muscles each integrate in first order, "
                         "drive = network-wide release table rel=clip((V-vrel)/25,0,1.5) through the anatomical innervation matrix "
                         "(neuron_muscle.csv, ACh positive / DD-VD GABA negative); constants all literature-anchored "
                         "(data/muscle96_constants.csv); when !=0 it replaces the Boyle bang-bang layer + the mndrive/"
                         "dmndrv/smddrive drive halves (kappa0 comes from A_D-A_V; the koff steering bias is retained); "
                         "0=off=legacy behaviour bit-identical")
    ap.add_argument("--m96tau", type=float, default=80.0,
                    help="muscle96 muscle-activation time constant ms (Liu 2011 AP half-width ~20 ms + "
                         "Gao&Zhen 2011 contraction timing + calibration; never used as a frequency knob)")
    ap.add_argument("--m96tau2", type=float, default=0.0,
                    help="muscle96 two-layer cascade slow-layer (calcium/contraction) time constant ms (P10, 2026-10-22: "
                         ">0 gives fast layer = m96tau at the Gao&Zhen 2011 AP half-width anchor 15.5 ms, slow layer = calcium-transient decay anchor "
                         "~1.0-1.57 s (J R Soc Interface 2015;12:20140963 measured); "
                         "DC gain unchanged, only dynamics/phase change; 0=off=old single layer bit-identical)")
    ap.add_argument("--m96amax", type=float, default=1.5,
                    help="muscle96 per-muscle activation saturation ceiling (linear smooth saturation, matches the old path's clip ceiling; "
                         "submaximal design, Liu 2021 Sci Adv)")
    ap.add_argument("--m96sig", default="rel", choices=["rel", "sp"],
                    help="muscle96 drive signal: rel=vrel release table clip((V-vrel)/25,0,1.5) (default, physiological form); "
                         "sp=softplus(dV)-ln2 deviation drive (same form as mndrive, immune to vrel mismatch; "
                         "v81 P3 measurement: rel saturates broadly under vrel-table mismatch, dynamic range compressed)")
    ap.add_argument("--m96gate", type=float, default=1.0,
                    help="muscle96 direction-gating switch: 1=forward state passes only B/DD classes, backward state passes only A/VD classes "
                         "(command-level gating, same semantics as the old Boyle g_f/g_b); 0=all classes always pass "
                         "(v81 P3 control: under the mnback architecture the backward wave is also carried in reverse by the B loop, "
                         "gating would cut the backward execution end; default-on validated only in pure-forward scenarios)")
    ap.add_argument("--m96homeo", type=float, default=0.0,
                    help="muscle96 homeostatic tension-calibration time constant, seconds (v86: baseline sig0_h slowly tracks each neuron's "
                         "tonic release, absorbing tonic dorsoventral asymmetry -> the static bend bias (left-drifting crawl) disappears, "
                         "only wave-frequency components pass; 0=off=legacy behaviour bit-identical)")
    ap.add_argument("--m96ei", type=float, default=0.6,
                    help="P7 constant anchoring (2026-10-21): NMJ inhibition/excitation current ratio E:I. Measured 0.61 "
                         "(Liu/Hollopeter/Jorgensen PNAS 2009 PMC2705609: whole-ACh evoked 2.01 nA "
                         "vs whole-GABA evoked 1.23 nA) → default 0.6; 1.0=legacy behaviour (no anchor, control only)")
    ap.add_argument("--m96gj", type=float, default=0.0,
                    help="P11 muscle-muscle gap-junction layer strength multiplier (2026-10-23, Liu/Chen/Wang 2011 JBC "
                         "286:44285 PMC3243499: neighbouring-muscle AP/Ca synchrony is carried by GJs independent of neural input; longitudinal same-quadrant"
                         "coupling coefficient measured 0.25 (unc-9 knockout -> 0.1 desynchrony + locomotion defects); Liu 2006 JBC 281:7881 "
                         "belly-belly + arm-arm coupling; White 1986 same-quadrant belly GJs). 1.0=anchored strength (longitudinal c=1/3, "
                         "transverse x0.73 bounded estimate), steady-state tridiagonal analytic solution; 0=off=legacy behaviour bit-identical)")
    ap.add_argument("--turnsmd", type=float, default=0.0,
                    help="pulse current for turns executed via SMD, pA (v68 R9: Gray et al. 2005 PNAS 102:3184 "
                         "doi:10.1073/pnas.0409009101: SMD encodes the steep amplitude of omega turns [turn angle after ablation"
                         "137->72 degrees]; on event-pulse trigger the drive becomes a differential injection of SMDV excitation + SMDD inhibition, "
                         "executing the turn through smddrive's anatomical head-muscle drive, replacing the artificial koff curvature bias; "
                         "0=off=legacy behaviour bit-identical [still uses koff])")
    ap.add_argument("--fc1", type=float, default=0.0,
                    help="FC1 turn-execution endogenous gain (v74 R18): on event trigger, no artificial pulse is injected; "
                         "instead fc1 times the in-situ synaptic budget of RIA->SMD is used as the differential drive amplitude (the budget is reconstructed online"
                         "from the previous block's voltages, state-dependent); mechanism anchors: Kramer 2026 RIM tyramine gating of the head motor chain + "
                         "Hendricks 2012 RIA turn encoding + Gray 2005 SMD turn-angle encoding; "
                         "0=off=legacy behaviour bit-identical (still uses the turnsmd artificial pulse))")
    ap.add_argument("--fc1tau2", type=float, default=3.0,
                    help="FC1 gating decay time constant s (pirouette-bout timescale, Pierce-Shimomura 1999)")
    ap.add_argument("--fc1dir", default="steer", choices=["steer", "random", "fixed", "weath"],
                    help="FC1 direction source: steer=sign of AIY(R-L) (R19 verdict: encodes only the temporal derivative, not spatial direction; mirror falsified) | "
                         "random=pirrand re-draw | fixed=+1 ventral | weath=v77 R22: sign of the w_weath correlated integral "
                         "(kap carrier agreement 0.92, the only measured signal carrying spatial direction; pulse_dir=-sign(w), "
                         "w<0=ventral up-gradient->ventral bend +1; continuous injection still governed by the weath gain; the decision layer is independent of the gain)")
    ap.add_argument("--wrst", type=float, default=0.0,
                    help="W68 (2026-10-24): bout-end w_weath reset + freeze window s (efference-copy extension: "
                         "the omega deep bend of a pirouette bout changes heading, and the curvature remnant needs ~1 wave period to dissipate; "
                         "the old w's spatial knowledge based on the old heading is void, and remnant-period sampling is polluted -> reset to zero at bout end and freeze accumulation "
                         "for wrst seconds; anchors: Pierce-Shimomura 1999 straight post-bout run + the v77 R22b manoeuvre-period freeze, same idea; "
                         "0=off=legacy behaviour bit-identical)")
    ap.add_argument("--fc2", type=float, default=0.0,
                    help="FC2 forward tonic endogenous gain pA (v75 R20): AVA extrasynaptic tonic ACh tone -> AVB "
                         "(Meng et al. Sci Adv 2024, 10.1126/sciadv.adk0002: AVA resting ~-23 mV constitutive"
                         "release, maintained by TWK-40 K2P, two-phase gating of AVB); voltage-gated: comes online when AVA>-45 mV, "
                         "clamping it away (HisCl simulation) shuts it -> testable prediction; 0=off=legacy behaviour bit-identical (uses stim))")
    ap.add_argument("--fc2v0", type=float, default=-45.0,
                    help="FC2 voltage-gating threshold mV (master-cell paper Fig1: boundary between hyperpolarised -75..-30 / depolarised -30..-10 zones)")
    ap.add_argument("--fc2vs", type=float, default=15.0, help="FC2 gating ramp width mV")
    ap.add_argument("--fc2cap", type=float, default=1.3, help="FC2 gating ceiling (when AVA depolarises further during reversal)")
    ap.add_argument("--avaclamp", type=float, default=0.0,
                    help="AVA constant clamp current pA (negative=hyperpolarisation, HisCl/Fig6A dose-experiment simulation; 0=no clamp)")
    ap.add_argument("--fc2mn", type=float, default=0.0,
                    help="FC2 second arm: the same AVA voltage-gated tonic -> class-B MN pool (master-cell paper: extrasynaptic cholinergic"
                         "transmission tonically strengthens the ventral motor subcircuit [L1 larva evidence]; replaces the mntonic artificial tetanus; 0=off)")
    ap.add_argument("--avaplat", type=float, default=0.0,
                    help="FC3 AVA regenerative plateau-current conductance pA/mV (v76 R21): VGCC-style voltage-gated positive feedback "
                         "(Mellem 2008 Nat Neurosci 11:865 UNC-2/EGL-19 plateau potentials; Meng 2024 Sci Adv "
                         "Fig1C: AVA bistable zones -30..-20 / -15..-10 mV); fast activation m + slow inactivation n (bout self-terminates); "
                         "replaces the avahold artificial sustainment; 0=off=legacy behaviour bit-identical")
    ap.add_argument("--avaplatth", type=float, default=-16.0,
                    help="plateau-current activation half-threshold mV (bistable-zone boundary, Meng 2024 phase diagram)")
    ap.add_argument("--avaplatk", type=float, default=1.5, help="activation ramp steepness mV")
    ap.add_argument("--avaplattau", type=float, default=0.2, help="m activation time constant s")
    ap.add_argument("--avaplattau2", type=float, default=4.0, help="n inactivation time constant s (bout duration)")
    ap.add_argument("--avaplate", type=float, default=60.0, help="Ca reversal potential mV (driving force)")
    ap.add_argument("--avaplat2", type=float, default=0.0,
                    help="v91: gate-edge-ignited AVA plateau-current conductance pA/mV — the voltage-threshold path is unusable "
                         "under the engine's compressed voltage scale (engine AVA excursion ~2-5 mV, whole-run peak -30.6 vs threshold -16); "
                         "the backward-gate flip (dir_state<0) is used as the ignition signal instead = internalisation of the v77 R23 pirrevp event ignition "
                         "(the event is the network's own gate flip, not an external protocol). Structure/τ/E as in avaplat (Meng 2024 upper stability regime + "
                         "UNC-2/EGL-19-like VGCC regenerative current); 0=off=legacy behaviour bit-identical")
    ap.add_argument("--avaplat2tau", type=float, default=0.2, help="avaplat2 m activation time constant s")
    ap.add_argument("--avaplat2tau2", type=float, default=4.0,
                    help="avaplat2 n inactivation time constant s (=bout self-termination duration, Hilliard 2005 backward bouts on the 1-3 s scale)")
    ap.add_argument("--bwrefr", type=float, default=0.0,
                    help="v91: post-bout re-entry refractory period, seconds (Wicks & Rankin 1995: post-backward response suppression); "
                         "timed from the bout end edge; within the window the relg forward->backward gate flip is blocked so the AVB forward program rebuilds,"
                         "the body unwinds its deep bend, and the proprioception-backward-state reverberation is cut (v91 short-pulse experiment: without it the bout-omega "
                         "limit cycle does not fall back to the crawl state within 50 s); 0=off=legacy behaviour bit-identical")
    ap.add_argument("--risbrk", type=float, default=0.0,
                    help="W59: coupling coefficient of the RIS brake into the relg backward pool (Kato 2015 Cell 163:656 bout terminator; "
                         "dA pool -= risbrk*max(dRIS-risbrkth,0); 0=off=legacy behaviour bit-identical)")
    ap.add_argument("--risdrive", type=float, default=0.0,
                    help="W59b: RIS bout-termination integral ignition current pA (Kato 2015: RIS is excited at reversal end; "
                         "charges in the backward state / discharges in forward, terminating the bout naturally through in-network GABA synapses; 0=off=legacy behaviour bit-identical)")
    ap.add_argument("--riskill", type=float, default=0.0,
                    help="W69 (2026-10-24): RIS full-charge self-limiting fuse s — in the backward state, if ris_m stays fully charged (>0.99) longer than "
                         "riskill seconds, the discharge path is forced. Kato 2015: RIS excitation is a transient peak at the bout end edge (Ca peak width ~1-2 s), "
                         "not a sustained state; fixes the freeze in the AVB-ablated state (AVA dead -> forward state never reached -> RIS stuck at full charge + "
                         "risbrk keeps pressing the AVA pool -> both gates shut, frozen 17 s); doubles as a bout-duration ceiling fuse "
                         "(Zhao 2003: bouts rarely exceed 6 s). 0=off=legacy behaviour bit-identical)")
    ap.add_argument("--ristau", type=float, default=1.5,
                    help="W59b: RIS integral charge time constant s (anchor Zhao 2003 bout median 2-3 s, 1.5 taken)")
    ap.add_argument("--ristau2", type=float, default=0.3,
                    help="W59b: RIS integral discharge time constant s (RIS withdraws quickly after bout end; Kato 2015 transient is short)")
    ap.add_argument("--risbrkth", type=float, default=0.0,
                    help="W59: RIS brake onset threshold dV mV (RIS crawl tonic ≈-6.6, bout peaks +50~86, threshold 0 taken)")
    ap.add_argument("--mntonic", type=float, default=0.0,
                    help="class-B MN constant depolarising injection pA (v67 R1 diagnosis: ventral-cord MN voltages are weakly coupled, |dV|<~0.1 mV, "
                         "sitting at the resting point, softplus(0)-ln2=0 -> the mndrive loop has zero gain at the origin and cannot ignite; "
                         "in the real animal class-B MNs carry superimposed AVB/PVC tonic drive during crawling; this parameter mimics the missing tonic operating point; "
                         "0=off=legacy behaviour bit-identical)")
    ap.add_argument("--vdsp", type=float, default=0.0,
                    help="W62 (2026-10-24): class-A MN -> class-D MN sp release-channel gain (works around the vrel mismatch). "
                         "lesion: on the v62 base, A-MNs run below the release threshold the whole time (crawl+bout) (dV~-35), rel-table release=0 "
                         "-> VD/DD silent throughout -> muscle-inhibition sculpting missing -> bout static net bend coils 2.0. "
                         "implementation: Iex[D] += vdsp * (W_ad @ (softplus(dV_prev)-ln2)), W_ad=A->D chemical edges"
                         "anatomically distributed (White 1986/Cook 2019, in the net); the sp form follows the muscle96/mndrive example "
                         "(NMJ release follows presynaptic depolarisation = physiological first principle). 0=off=legacy behaviour bit-identical)")
    ap.add_argument("--vdspx", type=float, default=0.0,
                    help="W64 (2026-10-24): vdsp subclass tiering — when non-zero, edges targeting VD12/VD13 are removed"
                         "from the A->D sp channel (bout-active VD12/13 suppress the tail ventral driver VA12 via the real anatomical edge VD12/13->VA12"
                         "(GABA ~19 EM contacts, sign=-1), and VD13 additionally inhibits ventral tail muscles directly -> "
                         "the higher the vdsp dose the heavier the tail dorsal bend = the 0.6-backfire mechanism; the VD1-11 midbody sculpting benefit is kept. "
                         "0=off=legacy behaviour bit-identical)")
    ap.add_argument("--headosc", type=float, default=0.0,
                    help="head class-B MN sinusoidal seed-injection amplitude pA (v67 R1b, Wen 2012 chain-of-reflexes test: "
                         "the wave is initiated by a head rhythm and propagated by the B-MN reflex chain; DB/VB in antiphase, acting on class-B cells whose somas sit in the first 2 segments; "
                         "0=off=legacy behaviour bit-identical)")
    ap.add_argument("--headfreq", type=float, default=0.3,
                    help="headosc frequency Hz (real-animal crawl wave frequency 0.2-0.5 Hz)")
    ap.add_argument("--boylescale", type=float, default=1.0,
                    help="Boyle motor-layer output multiplier (default 1=milestone behaviour; <1 weakens the muscle-layer stretch feedback, "
                         "used to test whether MN drive is truly load-bearing)")
    ap.add_argument("--kb", type=float, default=5.0,
                    help="Rod2D bending stiffness k_b = EI/ell (material constant, energy term (k_b/2)(dphi-ell*kappa0)^2; "
                         "default 5.0=bit-identical. Anchors: Fang-Yen 2010 PNAS internal-elasticity dynamic measurement -> k_b~0.7; "
                         "Backholm 2013 PNAS three-point bending -> 0.04-50; crawl-regime constraint (elasticity~drag) -> O(1); 5.0 sits inside the window)")
    ap.add_argument("--cn", type=float, default=40.0,
                    help="Rod2D normal-drag ratio Cn/Ct (Ct=1 normalised). Anchors: Boyle 2012 K_agar=30-40, 40 taken, "
                         "Berri 2009 locomotion-simulator fit; water=1.5 (Gray&Lissmann 1964)")
    ap.add_argument("--rod2d", default="borrowed", choices=["borrowed", "own"],
                    help="body-layer source: borrowed=original worm-whisperer modules (default, legacy behaviour bit-identical); "
                         "own=original analytic-gradient module engine/rod2d_own.py (alignment verification 5 PASS, difference ~1e-7 mm)")
    ap.add_argument("--ald", type=float, default=0.0,
                    help="aldicarb strength: ceiling of the ACh-accumulation gain from AChE blockade (0=no drug)")
    ap.add_argument("--aldtau", type=float, default=5.0, help="aldicarb onset time constant s")
    ap.add_argument("--proto", default="none",
                    help="optogenetic protocol: switch=AVB10s->AVA12s->AVB8s | ris=AVB throughout+RIS12-18s | region=silence a partition 8-16s | ashchem=ASH chemical-transduction front end 8-16s")
    ap.add_argument("--ashchem", type=float, default=0.0,
                    help="ASH chemical front-end full-amplitude current pA (active when proto=ashchem; "
                         "Hilliard 2005 EMBO J 24:63 PMC544906: ASH responds to repellent 'presence' sustained (<10 s no decay), "
                         "transduction chain GPCR->ODR-3/GPA-3->OSM-9(TRPV)->EGL-19 calcium; 0=off)")
    ap.add_argument("--ashchemtau", type=float, default=1.0,
                    help="ASH transduction onset time constant, seconds (Hilliard 2005 Fig1D/3A: fully rises within a 3 s window, "
                         "rise criterion >=2%%/s -> tau~1 s bounded estimate)")
    ap.add_argument("--ashchemtaua", type=float, default=20.0,
                    help="ASH adaptation charge time constant, seconds (Hilliard 2005 Fig3B: 60 s exposure decays after ~30 s "
                         "-> tau_adapt~20-30 s; GPC-1 (Ggamma)-mediated, gpc-1 mutants show no decay to repeated stimuli)")
    ap.add_argument("--ashchemtaur", type=float, default=100.0,
                    help="ASH adaptation recovery time constant, seconds (Hilliard 2005 Fig5F: after 1 min exposure still suppressed at 1 min, "
                         "largely recovered by 5 min -> tau_recov~100 s magnitude)")
    ap.add_argument("--ashchemwin", default="8:16",
                    help="ASH concentration square-wave window 'start:end' seconds (default 8:16 = same timing as the region protocol; "
                         "v91 short-pulse experiments use 8:10 etc. — Hilliard 2005 paradigm: brief aversive stimulus->a single backward bout")
    ap.add_argument("--ashmod", type=float, default=0.0,
                    help="P13: ASH volume-knob depth K (Chaubey 2023 PMC10319955: INX-20/cGMP/EGL-4 down-regulation layer; "
                         "gain=1/(1+K*ash_mod); 0=off=legacy behaviour bit-identical)")
    ap.add_argument("--ashstarve", type=float, default=0.0,
                    help="P13: starvation drive 0-1 (Chaubey 2023: food deprivation -> ODR-1 produces cGMP -> INX network into ASH "
                         "-> sensitivity down-regulated; ash_mod rises to this value with tau=60 s, falls back on refeeding with tau=300 s)")
    ap.add_argument("--region", default="",
                    help="partition for proto=region: head_sensory|chemo_sensory|head_motor|tail_sensory|mod_ala_hsn")
    ap.add_argument("--pamp", type=float, default=50.0, help="optogenetic stimulus amplitude pA")
    ap.add_argument("--phasic", type=float, default=0.0,
                    help="phasic pulse width ms for the region protocol (O'Hagan et al. 2005 Nat Neurosci 8:43: TRNs adapt fast, "
                         "responding only to force onset/offset, not sustained force -> the physiological stimulus is a pulse, not tonic DC; "
                         ">0 applies region current only at the window start [8.0, 8.0+w] and end [16.0-w, 16.0]; "
                         "0=tonic throughout=legacy behaviour bit-identical)")
    ap.add_argument("--avahold", type=float, default=0.0,
                    help="AVA plateau self-sustaining current amplitude pA (v70 R16, Meng et al. 2024 Sci Adv 10:adk0002: "
                         "AVA is the tonically depolarised master neuron, plateau-potential in nature -> the backward bout self-sustains after stimulus end; "
                         "the backward trigger edge charges it to this amplitude, then it decays unconditionally with avaholdt exponential -> the bout terminates spontaneously; "
                         "0=off=legacy behaviour bit-identical)")
    ap.add_argument("--avaholdt", type=float, default=1500.0,
                    help="decay time constant of the AVA plateau self-sustaining current, ms (control knob of bout duration; "
                         "real-animal touch backward bouts run ~2-3 s, Wicks & Rankin 1995)")
    ap.add_argument("--avatonc", type=float, default=0.0,
                    help="AVA constant tonic depolarising current pA (v92 steady-backward protocol: Kato 2015 Cell 163:656 "
                         "steady backward = a sustained behaviour driven by moderate tonic AVA dominance, not an escape bout; constant injection into AVA throughout, "
                         "no touch trigger, no avaplat2 strong plateau -> a gentle whole-body reverse wave; 0=off=legacy behaviour bit-identical)")
    ap.add_argument("--pep2", type=float, default=0.0,
                    help="backward-loop peptidergic positive-feedback gain pA (v73 R17: "
                         "FLP-14/FRPR-19 positive-feedback loop Marques 2021 PLoS Genet 17:e1009880; "
                         "MGL-2/mGluR5 presynaptic gain Katz 2019 Nat Commun 10:1701; "
                         "RIM->AVA backward-state reinforcement Sordillo 2021 eLife 10:e67723; "
                         "m2 is driven by slow accumulation of depolarisation in the backward-group pool (AVA+AVD+AVE), current re-injected into the pool; "
                         "0=off=legacy behaviour bit-identical)")
    ap.add_argument("--pep2tau", type=float, default=3.0,
                    help="peptidergic positive-feedback slow-variable time constant s (peptidergic signalling is seconds-scale, same magnitude as peptau)")
    ap.add_argument("--pep2th", type=float, default=0.8,
                    help="feedback ignition threshold mV: pool depolarisation below this value does not accumulate "
                         "(design: above crawl fluctuation ~0.3, below the ASH sensory seed ~1.2)")
    ap.add_argument("--pep2ref", type=float, default=1.0,
                    help="feedback saturation scale mV: m2 tops out when pool depolarisation exceeds the threshold by this amount")
    ap.add_argument("--pep2tau2", type=float, default=8.0,
                    help="negative-arm n2 time constant s (v73 R17b: mimics AVR-14 chloride-channel delayed inhibition, "
                         "Sordillo 2021 eLife; n2 slowly tracks m2, net drive=m2-n2 -> the bout terminates spontaneously)")
    ap.add_argument("--pep2inh", type=float, default=1.0,
                    help="negative-arm weight: net re-injection = pep2*(m2 - pep2inh*n2); 1=full cancellation")
    ap.add_argument("--chemsrc", default="",
                    help="proto=chem concentration-source coordinates 'x,y' mm (empty=no-source control)")
    ap.add_argument("--chemsigma", type=float, default=2.0,
                    help="v72: concentration-field Gaussian radius mm (default 2.0=legacy behaviour; the effective gradient range of an overnight-diffused point source on a real chemotaxis plate"
                         "is centimetre-scale, Pierce-Shimomura 1999 / Iino & Yoshida 2009)")
    ap.add_argument("--chemgain", type=float, default=50.0,
                    help="ASE current gain pA per (concentration/s)")
    ap.add_argument("--net", default="", help="network npz path (default data/net_gapfix.npz)")
    ap.add_argument("--gsyn", type=float, default=0.0, help="global chemical-synapse conductance gain (0=default 0.005)")
    ap.add_argument("--tauex", type=float, default=0.0,
                    help="excitatory synaptic-conductance time constant ms (P12 sensitivity test; 0=baseline 2.0, NMJ-bracket fast-edge anchor)")
    ap.add_argument("--tauinh", type=float, default=0.0,
                    help="inhibitory synaptic-conductance time constant ms (P12 sensitivity test; 0=baseline 5.0)")
    ap.add_argument("--ggap", type=float, default=0.0, help="global electrical-synapse gain (0=default 0.3)")
    ap.add_argument("--out", default="", help="output-directory override (default runs/v62_closed_loop; pure path infrastructure, zero numerical impact)")
    ap.add_argument("--vrel", default="", help="in-situ release-threshold table .npy (v59: vrel=V_rest in situ -10; empty=legacy isolated-equilibrium behaviour)")
    ap.add_argument("--vrelset", default="",
                    help="per-cell vrel override 'AVAL:-43.3,AVAR:-44.7' (W50: fix for AVA vrel=-83 causing release saturation, "
                         "requires --vrel; default=bit-identical)")
    ap.add_argument("--kturn", type=float, default=0.0, help="steering-gain override (0=default 0.03)")
    ap.add_argument("--gate", default="abs", choices=["abs", "rel", "relg"], help="gating scheme: abs=legacy absolute threshold 3 mV; rel=v60 relative (AVA vs AVB); relg=v61 group-pool relative (backward side AVA+AVD+AVE, Chalfie 1985)")
    ap.add_argument("--turnmode", default="linear", choices=["linear", "pirouette", "event"],
                    help="steering map: linear=legacy behaviour head_bias=k_turn*dSMD biphasic fixed steering; "
                         "pirouette=v63 continuous gating, run up-gradient and turn down-gradient (has a -90° orbit fixed point); "
                         "event=v64 event-based pirouette, dSMD upward threshold crossing triggers a fixed-length turn pulse + refractory, "
                         "breaking the orbit fixed point -> spiralling approach (Pierce-Shimomura 1999 discrete pirouettes)")
    ap.add_argument("--smdthresh", type=float, default=0.0005, help="pirouette/event mode dSMD trigger threshold mV")
    ap.add_argument("--steerthresh", type=float, default=1.0, help="turnsrc=aiy mode AIY(R-L) trigger threshold mV")
    ap.add_argument("--pulsebias", type=float, default=0.6, help="event-mode turn-pulse head-curvature bias")
    ap.add_argument("--pulsedur", type=float, default=3.0, help="event-mode pulse duration s")
    ap.add_argument("--refract", type=float, default=2.0, help="event-mode refractory period s")
    ap.add_argument("--turnsrc", default="smd", choices=["smd", "aiy", "dwn"],
                    help="event turn-signal source: smd=SMD(D-V) (legacy, contaminated by the head-swing motor program, agreement 0.70); "
                         "aiy=AIY(R-L) (v64c, agreement 1.00 with ASER-ASEL, separation 19.8 mV; "
                         "AIY is the biological chemotaxis comparator, Iino & Yoshida 2009); "
                         "dwn=decline-rate driven (v83, Pierce-Shimomura 1999: pirouette rate rises with dC/dt<0; "
                         "steer=-ase.dcdt, sustained suprathreshold while drifting away allows retriggering, fixing the AIY difference signal returning to zero at steady state and never firing)")
    ap.add_argument("--dwnthresh", type=float, default=0.005,
                    help="turnsrc=dwn decline trigger threshold (units=-dcdt, concentration/s; a turn fires only when dC/dt decline exceeds threshold)")
    ap.add_argument("--escomega", type=float, default=0.0,
                    help="B4 escape-omega series switch: 0=off=legacy behaviour bit-identical; !=0 triggers an additional event turn pulse at the backward bout end edge "
                         "(direction-state -1->+1 transition) (real-animal ASH avoidance = "
                         "backward bout + omega in series, Hilliard 2002/2005; the coupling carrier is undecided in the literature "
                         "(candidate: post-bout RIV/SMD disinhibition rebound); this implementation triggers on the endogenous direction-state edge, "
                         "execution reuses the v64 event pulse layer — semi-endogenous, disclosed in the limitations section)")
    ap.add_argument("--dwntau", type=float, default=3.0,
                    help="v83b: slow low-pass time constant s applied to -dcdt before dwn triggering (kills the 0.7 Hz head-swing leak:"
                         "the head swing's lateral velocity modulates instantaneous dcdt by ~±0.03 > the net chemotaxis signal ±0.025, "
                         "unfiltered, turn triggering phase-locks to head-swing phase; PS99 turn-rate modulation is itself a seconds-scale slow variable)")
    ap.add_argument("--steerbi", type=float, default=0.0,
                    help="R19 decision-layer bidirectional triggering (2026-10-20): 0=unidirectional steer>+thr=legacy behaviour bit-identical; "
                         "1=|steer|>thr bidirectional triggering; with fc1dir=steer, dir=sign(steer); "
                         "verdict = negative: the steer sign encodes the temporal derivative (away/toward), not spatial direction, "
                         "falsified by mirror geometry (approach -2.55 mm vs -0.57 mm), and the negative branch never ignited under either geometry "
                         "(steer never < -thr) — knob retained for review only)")
    ap.add_argument("--weath", type=float, default=0.0,
                    help="v71: weathervane bidirectional steering gain pA (0=off; klinotaxis, Iino & Yoshida 2009)")
    ap.add_argument("--weathtau", type=float, default=2.0, help="weathervane head-swing × dC/dt correlation low-pass time constant s")
    ap.add_argument("--weathmode", default="dkap", choices=["dkap", "kap"],
                    help="v77 R22: weathervane carrier: dkap=dκ0/dt proxy (legacy=legacy behaviour bit-identical; the sign audit"
                         "showed it unlocks from the true head-swing phase, w agreement only 0.06 = sign-flipped garbage) | "
                         "kap=head-joint bend position tanh(κ0/3) (carrier offline verdict: agreement 0.92, w direction correct; "
                         "the nose-tip heading samples the concentration field; proprioceptive joint angles are directly available, Yeon 2018; "
                         "the world-frame swing-velocity carrier vlat is retired: the body-frame projection cancels the swing, agreement ~0.58, negligible)")
    ap.add_argument("--pirrevp", type=float, default=0.0,
                    help="v77 R23: pirouette-backward plateau coupling (0=off=legacy behaviour bit-identical): the turn event ignites and charges the AVA plateau gate "
                         "m_plat -> the FC3 regenerative VGCC plateau drives a backward bout that kills forward momentum "
                         "(Pierce-Shimomura 1999: pirouette = backward + omega turn)."
                         "verdict = negative (2026-10-20): both 0.5/1.0 points drive bout duty ~75%%+ and collapse the forward regime "
                         "(v_fwd~0.005, freq 0.18 Hz, κ~8), and the SMD drive τ3 s decays ahead of the ~4 s bout, "
                         "a reorientation-timing mismatch — knob retained only for long-window run-and-tumble review, not in the default stack")
    ap.add_argument("--pirrand", type=int, default=0,
                    help="v71: random-direction pirouette seed (0=fixed ventral=legacy behaviour; >0=each turn randomly dorsal/ventral, Pierce-Shimomura 1999)")
    ap.add_argument("--pirrev", type=float, default=0.0,
                    help="v72: pirouette backward-coupling AVA charge amplitude pA (0=off=legacy behaviour bit-identical; "
                         "Pierce-Shimomura 1999 J Neurosci 19:9557: pirouette = backward bout + omega turn, "
                         "the backward kills forward momentum so the head bend completes at small radius; reuses the avaholdt decay, bout ~2-3 s, "
                         "Wicks & Rankin 1995)")
    ap.add_argument("--pirrevang", type=float, default=0.0,
                    help="v76: pirrev bearing-error gating threshold deg (0=gating off=legacy behaviour bit-identical; "
                         ">0 allows pirouette AVA ignition only when |heading - source bearing error| >= threshold, "
                         "protocol blocks use 90 = ignite only in the away-from-source hemisphere; Pierce-Shimomura 1999 pirouettes handle large-error "
                         "reorientation / Iino 2009 small-error weathervane division of labour; 999=never ignite; "
                         "W75 diagnosis: at d090, medium-error pirrev misfires left dAVA ignited 57.5%% of the time, degrading approach "
                         "from +3.46 mm to +0.91 mm; active only when proto chem has a source (ase not None))")
    ap.add_argument("--refrmod", type=float, default=0.0,
                    help="v73: turn-frequency modulation reference level mV (0=fixed refractory=legacy behaviour; >0 gives "
                         "refract_eff=refract*clip(refrmod/steer,0.3,1.0) — the faster the decline, the denser the turns, "
                         "Pierce-Shimomura 1999 pirouette frequency modulation)")
    ap.add_argument("--pirsup", type=float, default=999.0,
                    help="v78 R25: pirouette approach-suppression threshold (default 999=off=legacy behaviour bit-identical; "
                         "otherwise a turn event may retrigger only when low-passed dC/dt < pirsup (concentration/s, peak-normalised), "
                         "the Pierce-Shimomura 1999 core: rising concentration (approaching the source) suppresses pirouettes for straight runs, "
                         "falling concentration (decline) triggers turns; the temporal gate for near-source dwelling/arrest)")
    ap.add_argument("--pirc", type=float, default=999.0,
                    help="v78 R25b: absolute-concentration dwelling gate (default 999=off=legacy behaviour bit-identical; otherwise when "
                         "instantaneous concentration >= pirc (peak-normalised, 0~1) turn triggering is suppressed — when the in-ring gradient is zeroed, the concentration level itself "
                         "drives dwelling; Sawin 2000 basal slowing / Hills 2004 area-restricted search / Chalasani 2007 "
                         "AWC area-restricted-search absolute-concentration channel)")
    ap.add_argument("--pirmag", type=float, default=0.0,
                    help="v78 R25b: turn-sharpness error-calibration reference mV (0=fixed-length pulse=legacy behaviour; >0 gives "
                         "pulsedur_eff=pulsedur*clip(|steer|/pirmag,1,3), large error -> deep long omega bend, "
                         "Pierce-Shimomura 1999 turn-angle/bearing-error correlation)")
    ap.add_argument("--arrc", type=float, default=999.0,
                    help="v78 R25c: absolute-concentration slowing threshold (default 999=off=legacy behaviour bit-identical; when low-passed concentration "
                         ">= arrc (peak-normalised), mndrive is multiplied by arrg to slow — Sawin 2000 Neuron basal slowing "
                         "response (dopamine CEP/ADE/PDE, high concentration/food -> slowed dwelling), Hills 2004 area-restricted search)")
    ap.add_argument("--arrg", type=float, default=0.35,
                    help="v78 R25c: mndrive gain while slowing is active (biological basal-slowing amplitude ~down to 1/3-1/2)")
    ap.add_argument("--arrtau", type=float, default=2.0,
                    help="v78 R25c: concentration low-pass time constant s (dopamine tonic-signal timescale, debounce)")
    ap.add_argument("--asad", type=float, default=0.0,
                    help="v79 R26: GPCR-level sensory-adaptation strength (default 0=off=legacy behaviour bit-identical; in sustained high-concentration zones the adapted state"
                         "rises -> ASEL/ASER injection and the weathervane correlation signal decay together -> the on-peak tether fades -> "
                         "spontaneous dispersal — ARR-1/beta-arrestin early desensitisation (Palmitessa 2005 JBC 280:24649, "
                         "Merritt 2022 PNAS 119:e2119950119), AWA fast desensitisation preserves derivative representation (Larsch 2013), "
                         "behavioural olfactory adaptation (Colbert & Bargmann 1995 Neuron 14:803)")
    ap.add_argument("--asadtau", type=float, default=30.0,
                    help="v79 R26: adapted-state time constant s (effective composite of ARR-1 early + intermediate timescales, honestly labelled a phenomenological timescale)")
    ap.add_argument("--asadth", type=float, default=0.5,
                    help="v79 R26: adaptation activation threshold (peak-normalised low-passed concentration; only sustained high-concentration zones drive adaptation, far-field/approach segments unaffected)")
    ap.add_argument("--offg", type=float, default=0.0,
                    help="v79 R27: OFF-response/area-search turn-enhancement gain (default 0=off=legacy behaviour bit-identical; near the source"
                         "(low-passed concentration >offth) when a turn triggers: pulse duration x=ars and retrigger interval /=ars, ars=1+K*normalised concentration "
                         "-> high-turn-rate milling geometry — Hills 2004 Neuron 43:349 elevated off-food turn rate (dopamine-dependent), "
                         "Chalasani 2007 Nature 450:63 AWC OFF-resp->omega, Gray 2005 PNAS 102:3184, "
                         "Lopez-Cruz 2019 Neuron off-food local search; dwelling geometry = low speed + high turning (Flavell 2013)")
    ap.add_argument("--offth", type=float, default=0.5,
                    help="v79 R27: turn-enhancement activation threshold (peak-normalised low-passed concentration, same semantics as asadth)")
    ap.add_argument("--savestate", default="", help="save the full closed-loop state pickle after the run (checkpoint resume)")
    ap.add_argument("--loadstate", default="", help="resume from a pickle state (skip warm-up; t/body/gating/ASE all continuous)")
    # ---- RIA three-compartment module (spec: RIA three-compartment prototype mechanism specification, 2026-10-19) ----
    ap.add_argument("--riacomp", type=int, default=0,
                    help="RIA three-compartment regenerative dendrites (0=off=legacy behaviour bit-identical; 1=on: "
                         "RIA->SMD D/V output and SMD->RIA feedback are rerouted into the nrD/nrV compartments, "
                         "AIY inhibition gates through an ACC-2-equivalent shunt, Liu 2018/Hendricks 2012 mechanism)")
    ap.add_argument("--riagax", type=float, default=0.05, help="compartment axial coupling conductance nS (cable anchor 0.005-0.5)")
    ap.add_argument("--riaacc2", type=float, default=5.0, help="ACC-2-equivalent shunt gain (Liu 2018 sensory suppression)")
    ap.add_argument("--riaroute", type=float, default=-1.0,
                    help="compartment output routing: -1=crossed (nrV->SMDD, miniSOG hard evidence, default); +1=ipsilateral (control)")
    # ---- v3 calcium-store local-release gating (Hendricks 2012 mechanism; requires riacomp=1) ----
    ap.add_argument("--riaca3", type=int, default=0,
                    help="v3 calcium-store edition: compartment activity = GAR-3 slow calcium variable, additively gating local release (default 0=v2 voltage compartments)")
    ap.add_argument("--riakmod", type=float, default=1.0,
                    help="v3 release-gating gain (O(1) anchor: rig batch20 measured rel_typ/c_typ=48.7)")
    ap.add_argument("--riatau", type=float, default=1.0,
                    help="v3 calcium-store time constant s (GPCR->ER slow scale, literature magnitude 0.3-3)")
    ap.add_argument("--riasomaadd", type=float, default=0.0,
                    help="v3b=0 compartment edges dominated by the local calcium store (default, eliminates permanent S_MAX saturation); v3a=1 soma+store additive (control)")
    args = ap.parse_args()

    global NET_NPZ, GSYN, GGAP, OUT, GNCA_A_ARG, GLEAK_GABA_ARG, GEGL36_GABA_ARG
    global UNC2PED_A_ARG, GSK_A_ARG, CATAUSCALE_A_ARG, GUNC2_A_ARG
    global GUNC2_AVA_ARG, UNC2PED_AVA_ARG, GSK_AVA_ARG, CATAUSCALE_AVA_ARG
    if args.out:
        OUT = Path(args.out)
    if args.gsyn > 0: GSYN = args.gsyn
    if args.ggap > 0: GGAP = args.ggap
    GNCA_A_ARG = args.gncaA
    GLEAK_GABA_ARG = args.gabaleak
    GEGL36_GABA_ARG = args.gabaegl36
    UNC2PED_A_ARG = args.unc2pedA
    GSK_A_ARG = args.gskA
    CATAUSCALE_A_ARG = args.catauscaleA
    GUNC2_A_ARG = args.gunc2A
    GUNC2_AVA_ARG = args.gunc2ava
    UNC2PED_AVA_ARG = args.unc2pedava
    GSK_AVA_ARG = args.gskava
    CATAUSCALE_AVA_ARG = args.catauscaleava
    if args.net:
        NET_NPZ = Path(args.net) if Path(args.net).is_absolute() else Path(args.net).resolve()
        print(f"[net] using network: {NET_NPZ}", flush=True)

    import jax
    import jax.numpy as jnp
    from jax import lax
    jax.config.update("jax_enable_x64", True)

    t0 = time.time()
    net = build_v62()
    if args.vrel:
        net.vrel = np.load(args.vrel)
        print(f"[vrel] in-situ release-threshold table {args.vrel} (n={len(net.vrel)}, v59 design intent s_rest=0.4)", flush=True)
    z = np.load(NET_NPZ, allow_pickle=False)
    chem, sign, gap = z["chem"], z["sign"], z["gap"]
    names = [str(s) for s in z["neurons"]]
    if args.vrelset:
        assert args.vrel, "--vrelset requires --vrel"
        _idx0 = {n: i for i, n in enumerate(names)}
        for _kv in args.vrelset.split(","):
            _kn, _vv = _kv.split(":")
            net.vrel[_idx0[_kn.strip()]] = float(_vv)
        print(f"[vrelset] per-cell release-threshold override: {args.vrelset} "
              f"(W50: AVA vrel=-83 release-saturation fix, sV@rest 2.32 -> graded zone)", flush=True)
    W = {"chem": chem, "sign": sign, "gap": gap, "cmasks": [], "gmasks": []}

    from muscle_maps import load_muscle_maps
    muscle_maps = load_muscle_maps(net)

    # ---- v81: original physiological-grouped muscle layer (instantiated when muscle96!=0; default off, old path bit-identical) ----
    m96 = None
    if args.muscle96 != 0.0:
        from muscle96 import Muscle96
        m96 = Muscle96(muscle_maps, tau_m_ms=args.m96tau, amax=args.m96amax,
                       neuron_names=[str(n) for n in names], ei_nmj=args.m96ei,
                       tau_ca_ms=args.m96tau2, gj_cc=args.m96gj)
        print(f"[muscle96] v81 original muscle layer ON: G={args.muscle96:g}, tau_m={args.m96tau:g}ms, "
              f"Amax={args.m96amax:g}, 96-muscle graded integration (constant table data/muscle96_constants.csv); "
              f"the Boyle layer/mndrive/dmndrv/smddrive drive halves are replaced", flush=True)
        if args.m96tau2 > 0.0:
            print(f"[muscle96] P10 two-layer cascade ON: fast layer (electrical excitation) tau={args.m96tau:g}ms "
                  f"(Gao&Zhen 2011 AP half-width anchor 15.5) -> slow layer (calcium/contraction) tau={args.m96tau2:g}ms "
                  f"(RSI2015 calcium-transient decay anchor 1.0-1.57 s); DC gain unchanged, only dynamics change", flush=True)
        print(f"[muscle96] P7 anchoring: E:I={m96.ei_nmj:g} (NMJ measured 0.61, PMC2705609); "
              f"absolute scale g_abs={m96.g_abs:.3f} nA/unit drive — ACh full amplitude ≈2.01 nA, "
              f"GABA full amplitude ≈{m96.ei_nmj * 2.01:.2f} nA (measured 1.23); "
              f"current->contraction mapping unmeasured (field blank), mA dimensionless on the kmax calibration path (bounded estimate)", flush=True)

    import jax_circuit as jc
    if args.tauex > 0.0 or args.tauinh > 0.0:
        # P12 (2026-10-22): synaptic-conductance time-constant sensitivity test — module constants must be overridden before the first jit compile
        # (folded during jax tracing); position is after import / before the first build_step call, same discipline as the GSYN override
        if args.tauex > 0.0: jc.TAU_EX = args.tauex
        if args.tauinh > 0.0: jc.TAU_INH = args.tauinh
        print(f"[tau] synaptic time-constant override: TAU_EX={jc.TAU_EX:g}ms TAU_INH={jc.TAU_INH:g}ms "
              f"(baseline 2.0/5.0, NMJ-bracket fast-edge anchor)", flush=True)
    P = jc.extract(net, W, muscle_maps, DT_MS)

    # ---- in-silico channel block: --block gunc2:0 (repeatable); the multiplier acts on the P parameter matrix ----
    for spec_ in args.block:
        cname, cval = spec_.split(":")
        cval = float(cval)
        if cname not in jc.PK:
            raise SystemExit(f"unknown channel parameter {cname}; choices {list(jc.PK)}")
        ci = jc.PK[cname]
        P["Pawc"] = P["Pawc"].copy(); P["Prmd"] = P["Prmd"].copy()
        P["Pawc"][:, ci] *= cval; P["Prmd"][:, ci] *= cval
        print(f"[block] {cname} x {cval}  (network-wide)", flush=True)

    # ---- RIA three-compartment extension (enabled when riacomp=1; default off, legacy behaviour bit-identical) ----
    riacomp = None
    if args.riacomp:
        import jax_circuit_riacomp as jrc
        ria_spec = jrc.build_comp_spec(NET_NPZ)
        ria_mats = jrc.build_riacomp_mats(P, ria_spec, NET_NPZ)
        if args.riaca3:
            ria_cp3 = dict(tau_store=args.riatau, k_mod=args.riakmod,
                           s_acc2=args.riaacc2, route=args.riaroute,
                           soma_add=args.riasomaadd)
            step, assemble_V, init_state2 = jrc.build_step_riacomp3(P, ria_spec, ria_mats, ria_cp3,
                                                                    ctrl_hz=CTRL_HZ)
            print(f"[riacomp-v3] calcium-store release gating ON: tau={args.riatau:g}s k_mod={args.riakmod:g} "
                  f"s_acc2={args.riaacc2:g} soma_add={args.riasomaadd:g} "
                  f"route={'crossed' if args.riaroute < 0 else 'ipsilateral'} "
                  f"(Hendricks 2012 mechanism)", flush=True)
        else:
            ria_cp = dict(g_ax=args.riagax, s_acc2=args.riaacc2, route=args.riaroute)
            step, assemble_V, init_state2 = jrc.build_step_riacomp(P, ria_spec, ria_mats, ria_cp,
                                                                   ctrl_hz=CTRL_HZ)
            print(f"[riacomp] RIA three-compartment ON: g_ax={args.riagax:g}nS s_acc2={args.riaacc2:g} "
                  f"route={'crossed' if args.riaroute < 0 else 'ipsi'}", flush=True)
        riacomp = True
    else:
        step, assemble_V = jc.build_step(P, fwd_idx=(), n2w_idx=(), ctrl_hz=CTRL_HZ)

    N = P["N"]
    if riacomp:
        Mex = ria_mats["Mex_m"]   # re-routed edges removed (G_SYN multiplier included)
    else:
        Mex = jnp.asarray(P["G_SYN"] * P["Mex_u"][0])   # theta=0: all-unit multiplier -> anatomical weights
    Min = jnp.asarray(P["G_INH"] * P["Min_u"][0])
    Lg = jnp.asarray(P["G_GAP"] * P["Lg_u"][0])
    # W54 (2026-10-23): A->B chemical crosstalk edge split (backward-state attenuation; abx=1.0 = full weight = bit-identical)
    Mex_abx_fwd_np = Mex_abx_bwd_np = None
    if args.abx != 1.0:
        _nm = [str(n) for n in names]
        def _isc(n, pfx):
            return n.startswith(pfx) and any(ch.isdigit() for ch in n)
        _Mex_full_np = np.asarray(P["G_SYN"] * P["Mex_u"][0])
        _abmask = np.zeros_like(_Mex_full_np)
        for e in range(chem.shape[0]):
            if sign[e] > 0 and (_isc(_nm[int(chem[e, 0])], "VA") or _isc(_nm[int(chem[e, 0])], "DA")) \
                    and (_isc(_nm[int(chem[e, 1])], "VB") or _isc(_nm[int(chem[e, 1])], "DB")):
                _abmask[int(chem[e, 1]), int(chem[e, 0])] = 1.0
        Mex_abx_fwd_np = _Mex_full_np
        Mex_abx_bwd_np = _Mex_full_np * (1.0 - (1.0 - args.abx) * _abmask)
        print(f"[W54] A->B crosstalk backward attenuation ON: abx={args.abx:g} (Kato2015 sub-loop mutual-inhibition analog; "
              f"edges={int(_abmask.sum())}, VA->VB 5.508 chain included)", flush=True)
    # W54b: A-MN<->VD/DD gap edge-set Laplacian (backward-state attenuation; disabled when avdx=1.0)
    Lg_avd_np = None
    if args.avdx != 1.0:
        _nm2 = [str(n) for n in names]
        def _isc2(n, pfx):
            return n.startswith(pfx) and any(ch.isdigit() for ch in n)
        _A = {"VA", "DA"}; _I = {"VD", "DD", "VB", "DB"}
        def _cls2(n):
            for p in ("VA", "DA", "VD", "DD", "VB", "DB"):
                if _isc2(n, p):
                    return p
            return None
        _rows = []
        for e in range(gap.shape[0]):
            ca, cb = _cls2(_nm2[int(gap[e, 0])]), _cls2(_nm2[int(gap[e, 1])])
            if (ca in _A and cb in _I) or (ca in _I and cb in _A):
                _rows.append(e)
        Lg_avd_np = np.zeros_like(np.asarray(P["G_GAP"] * P["Lg_u"][0]))
        for e in _rows:
            a, b = int(gap[e, 0]), int(gap[e, 1])
            w = float(P["G_GAP"]) * float(gap[e, 2])
            Lg_avd_np[a, a] += w; Lg_avd_np[a, b] -= w
            Lg_avd_np[b, a] -= w; Lg_avd_np[b, b] += w
        print(f"[W54b] A<->VD/DD gap backward attenuation ON: avdx={args.avdx:g} "
              f"(edges={len(_rows)}; phenomenological state gating, innexin modulation candidate)", flush=True)
    EMPTY = jnp.zeros(0, jnp.float64)
    NSTEPS = int(round(BLOCK_MS / DT_MS))

    def _blk(c, _):
        return step(c, Mex, Min, Lg, _IEX, False, EMPTY), None

    # inside the scan closure, Iex is passed in as a tracer
    def make_run_block():
        def run_block(state, Iex):
            def f(c, _):
                return step(c, Mex, Min, Lg, Iex, False, EMPTY), None
            return lax.scan(f, state, None, length=NSTEPS)[0]
        return jax.jit(run_block)
    run_block = make_run_block()

    # stage-2 ggapmod: runtime Lg scaling channel (off by default; run_block_lv=None -> legacy path bit-identical)
    run_block_lv = None
    run_block_lvm = None
    if args.ggapmod != 0.0 or args.avarect != 0.0 or args.abx != 1.0 or args.avdx != 1.0:
        Lg_full_np = np.asarray(P["G_GAP"] * P["Lg_u"][0])   # full-scale base matrix (G_GAP multiplier included)
        def make_run_block_lv():
            def run_block(state, Iex, Lg_t):
                def f(c, _):
                    return step(c, Mex, Min, Lg_t, Iex, False, EMPTY), None
                return lax.scan(f, state, None, length=NSTEPS)[0]
            return jax.jit(run_block)
        run_block_lv = make_run_block_lv()
        if Mex_abx_bwd_np is not None:
            # W54: Lg+Mex dual per-block channels (A->B crosstalk backward attenuation)
            def make_run_block_lvm():
                def run_block(state, Iex, Lg_t, Mex_t):
                    def f(c, _):
                        return step(c, Mex_t, Min, Lg_t, Iex, False, EMPTY), None
                    return lax.scan(f, state, None, length=NSTEPS)[0]
                return jax.jit(run_block)
            run_block_lvm = make_run_block_lvm()
        if args.ggapmod != 0.0:
            print(f"[ggapmod] stage-2 gap modulation ON: g_eff=(1-{args.ggapmodk:g}(1-M))xGGAP, "
                  f"tau={args.ggapmodtau:g}s; anchors Chen2007 unc-1(0.7)/LeBoeuf2015 DOP-2/"
                  f"Schwarz2018 EGL-4/Kawano2011 shunt; with --ggap 1.0: M~0->0.3 crawl, M~1->1.0 escape",
                  flush=True)

    rows_awc = np.asarray(P["rows_awc"]); rows_rmd = np.asarray(P["rows_rmd"])
    if riacomp:
        state = init_state2()
    else:
        state = (jnp.asarray(P["Y0awc"]), jnp.asarray(P["Y0rmd"]),
                 jnp.zeros(N, jnp.float64), jnp.zeros(N, jnp.float64))

    def Yfull(state):
        Yawc, Yrmd = np.asarray(state[0]), np.asarray(state[1])
        Y = np.zeros((N, Yawc.shape[1]))
        Y[rows_awc] = Yawc; Y[rows_rmd] = Yrmd
        return Y

    idx = {n: i for i, n in enumerate(names)}
    AVB = [idx["AVBL"], idx["AVBR"]]; AVA = [idx["AVAL"], idx["AVAR"]]
    # ggapmod driver set: anterior mechanosensory layer (ALM/AVM gentle touch) + dopaminergic neurons (CEP/ADE, DA-signal proxy;
    # DA acts by volume transmission and the engine has no wireless layer (g_wl=0), so DA->DOP-2 input is proxied by these cells' depolarising activity;
    # CEP/ADE/PDE are all the dopaminergic neurons of C. elegans; qualitative anchor Sulston 1975/WormAtlas)
    DAT = [idx[n] for n in ("ALML", "ALMR", "AVM", "CEPDL", "CEPDR",
                            "CEPVL", "CEPVR", "ADEL", "ADER") if n in idx]
    if args.ggapmodash != 0.0:
        # B5: ASH into the driver pool — Chaubey 2023 (PMC10319955): state-dependent innexin (INX-20) modulation
        # localised to ASH itself; on chemical escape ASH depolarises -> M->1 -> g_eff->GGAP (escape regime),
        # during crawl/chemotaxis ASH rests -> M~0 -> g_eff~0.3xGGAP (crawl regime). The two manual regimes are thus unified into
        # a single baseline GGAP + endogenous state modulation (depth k=0.7 anchored on Chen 2007 unc-1;unc-9 70% coupling reduction)
        DAT = DAT + [idx[n] for n in ("ASHL", "ASHR") if n in idx]
    AVD = [idx["AVDL"], idx["AVDR"]]; AVE = [idx["AVEL"], idx["AVER"]]
    SMDD = [idx["SMDDL"], idx["SMDDR"]]; SMDV = [idx["SMDVL"], idx["SMDVR"]]
    RIV = [idx["RIVL"], idx["RIVR"]]; RIS = idx["RIS"]
    ASEL = idx["ASEL"]; ASER = idx["ASER"]
    AIYL_i = idx["AIYL"]; AIYR_i = idx["AIYR"]   # v64c turning-comparator readout (AIY L/R pair)
    ASH_i = [idx["ASHL"], idx["ASHR"]]; ASI_i = [idx["ASIL"], idx["ASIR"]]  # stage-2 CKR-1: NLP-18 sensory driver pool
    AIB_i = [idx["AIBL"], idx["AIBR"]]   # stage-2 CKR-1: NLP-18 arm target (turn-initiating interneurons, Chen 2022)
    # ---- W62: A->D sp release-channel precompute (VA/DA->VD/DD edges screened from the chem table, anatomical distribution) ----
    W_ad_t, W_ad_s, W_ad_w = [], [], []
    if args.vdsp != 0.0:
        _ct = np.asarray(chem)
        for _e in range(_ct.shape[0]):
            _a, _b, _w = int(_ct[_e, 0]), int(_ct[_e, 1]), float(_ct[_e, 2])
            _na, _nb = names[_a], names[_b]
            if (_na.startswith(("VA", "DA")) and _nb.startswith(("VD", "DD"))
                    and any(ch.isdigit() for ch in _na) and any(ch.isdigit() for ch in _nb)):
                W_ad_t.append(_b); W_ad_s.append(_a); W_ad_w.append(_w)
        W_ad_t = np.array(W_ad_t); W_ad_s = np.array(W_ad_s); W_ad_w = np.array(W_ad_w)
        print(f"[W62] vdsp A->D sp release channel ON: {len(W_ad_t)} edges (VA/DA->VD/DD), G={args.vdsp:g}", flush=True)
        if args.vdspx != 0.0:
            _keep = np.array([names[_b] not in ("VD12", "VD13") for _b in W_ad_t])
            _drop = int((~_keep).sum())
            W_ad_t = W_ad_t[_keep]; W_ad_s = W_ad_s[_keep]; W_ad_w = W_ad_w[_keep]
            print(f"[W64] vdspx subclass tiering ON: dropped {_drop} VD12/VD13-target edges, kept {len(W_ad_t)}", flush=True)
    if args.rivabl != 0.0:
        print(f"[rivabl] RIVL/R constant injection {args.rivabl:g} pA (Gray 2005 functional-ablation protocol)", flush=True)

    # ---- generic functional ablation (Gao 2018 protocol): prefix wildcard -> constant strong hyperpolarisation ----
    abl_idx = []
    if args.cellabl:
        for p in args.cellabl.split(","):
            p = p.strip()
            if not p:
                continue
            if p.endswith("*"):
                abl_idx += [idx[n] for n in names if n.startswith(p[:-1]) and n in idx]
            else:
                abl_idx.append(idx[p])
        abl_idx = sorted(set(abl_idx))
        print(f"[cellabl] functional ablation of {len(abl_idx)} cells ({args.cellablp:g} pA constant): "
              f"{','.join(names[i] for i in abl_idx)}", flush=True)

    # ---- edge-deletion true ablation (cellablmode=edge): remove all chemical+electrical edges touching the ablated cells ----
    abl_set = set(abl_idx) if (args.cellablmode == "edge" and abl_idx) else set()
    if abl_set:
        Mex_np = np.asarray(Mex).copy(); Min_np = np.asarray(Min).copy(); Lg_np = np.asarray(Lg).copy()
        for _M in (Mex_np, Min_np, Lg_np):
            _M[abl_idx, :] = 0.0; _M[:, abl_idx] = 0.0
        Mex = jnp.asarray(Mex_np); Min = jnp.asarray(Min_np); Lg = jnp.asarray(Lg_np)
        if "Lg_full_np" in dir():
            Lg_full_np[abl_idx, :] = 0.0; Lg_full_np[:, abl_idx] = 0.0
        print(f"[cellabl-edge] true laser-ablation semantics for {len(abl_idx)} cells: all chemical+electrical edges deleted, no current injection "
              f"(Gao 2018 ablation preparation; under a clamp protocol, residual gap edges drag neighbours = artefact; this mode replaces that)",
              flush=True)

    # ---- W49: AVA<->A-MN rectifying gap-junction edge-set precompute (Shui 2020 ball-and-receptor gating) ----
    AR = None
    if args.avarect != 0.0:
        gap_tab = np.asarray(W["gap"])
        amn_ids = set(idx[n] for n in names
                      if n in idx and (n.startswith("DA") or n.startswith("VA"))
                      and (any(ch.isdigit() for ch in n) or n == "VAn"))
        ava_ids = set(AVA)
        ar_a, ar_b, ar_ava, ar_amn, ar_w = [], [], [], [], []
        for e in range(gap_tab.shape[0]):
            a = int(gap_tab[e, 0]); b = int(gap_tab[e, 1]); w = float(gap_tab[e, 2])
            if a in abl_set or b in abl_set:
                continue   # with edge ablation, drop deleted edges
            if a in ava_ids and b in amn_ids:
                ar_a.append(a); ar_b.append(b); ar_ava.append(a); ar_amn.append(b); ar_w.append(w)
            elif b in ava_ids and a in amn_ids:
                ar_a.append(a); ar_b.append(b); ar_ava.append(b); ar_amn.append(a); ar_w.append(w)
        if ar_a:
            ar_a = np.array(ar_a); ar_b = np.array(ar_b)
            ar_ava = np.array(ar_ava); ar_amn = np.array(ar_amn)
            ar_gw = np.array(ar_w) * float(P["G_GAP"])
            _base = np.asarray(Lg)   # edge-ablation handling already applied (if enabled)
            Lg_ar_ohm = np.zeros_like(_base)
            np.add.at(Lg_ar_ohm, (ar_a, ar_a), ar_gw); np.add.at(Lg_ar_ohm, (ar_a, ar_b), -ar_gw)
            np.add.at(Lg_ar_ohm, (ar_b, ar_a), -ar_gw); np.add.at(Lg_ar_ohm, (ar_b, ar_b), ar_gw)
            AR = dict(a=ar_a, b=ar_b, ava=ar_ava, amn=ar_amn, gw=ar_gw,
                      Lg_rest=_base - Lg_ar_ohm)
            print(f"[W49] AVA<->A-MN rectification ON: {len(ar_a)} edges, strength={args.avarect:g}, "
                  f"Boltzmann A={args.avarecta:g} V0={args.avarectv0:g} Gmin={args.avarectmin:g} "
                  f"(Shui2020 PMC7455501 ball-and-receptor: Vj=V_AVA-V_AMN>0 plugged=closed, <0 open=A->AVA; "
                  f"Liu2017 in-vivo unidirectional; voltage taken from the previous block, 50ms delay, as for ggapmod)", flush=True)
        else:
            print("[W49] WARNING: no AVA<->A-MN gap edges found (or all removed by edge ablation); rectification inactive", flush=True)

    def _Lg_w49(Vabs):
        """W49: assemble the rectifying Lg from the absolute-voltage vector. gate=Boltzmann(Vj), Vj=V_AVA-V_AMN:
        Vj>0 (AVA more depolarised = backward-state plateau) -> ball plugs the pore, gate->Gmin (A-MN released);
        Vj<0 (A-MN more depolarised) -> gate->1 (A->AVA conducts, Kawano shunt/reverse amplification)."""
        Vj = Vabs[AR["ava"]] - Vabs[AR["amn"]]
        gate = args.avarectmin + (1.0 - args.avarectmin) / (
            1.0 + np.exp(args.avarecta * (Vj - args.avarectv0)))
        gw = ((1.0 - args.avarect) + args.avarect * gate) * AR["gw"]
        Lt = AR["Lg_rest"].copy()
        np.add.at(Lt, (AR["a"], AR["a"]), gw); np.add.at(Lt, (AR["a"], AR["b"]), -gw)
        np.add.at(Lt, (AR["b"], AR["a"]), -gw); np.add.at(Lt, (AR["b"], AR["b"]), gw)
        return Lt

    # ---- silent-partition optogenetics: class prefix -> cell indices ----
    REGION_PREFIX = {
        "head_sensory":  ["IL", "CEP", "OLL", "OLQ"],
        "chemo_sensory": ["ASE", "ASG", "ASH", "ASI", "ASJ", "ASK", "ADF",
                          "AWA", "AWB", "AWC", "AUA", "BAG", "AFD", "FLP", "SDQ"],
        "head_motor":    ["RIV", "RMD", "RME", "RMH", "SMB", "SMD"],
        "tail_sensory":  ["PVD", "PLN", "PVN", "PVQ"],
        "mod_ala_hsn":   ["ALA", "HSN"],
        # --- escape reflex circuit (after net323 recovered missing cells) ---
        "touch_ant":     ["ALM", "AVM"],          # anterior touch (literature: -> AVD -> backward)
        "touch_post":    ["PLM"],                  # posterior touch (literature: -> PVC/AVA -> forward acceleration)
        "ash":           ["ASH"],                  # nociception (literature: -> AVA -> backward)
        "afd":           ["AFD"],                  # thermosensation (Narayan & Sternberg 2011 PMC3111291: AFD->AIY gain probe)
        "ava":           ["AVA"],                  # command-layer direct drive (Roberts 2016 style: ChR2::AVA -> backward + AVB suppression)
        "avb":           ["AVB"],                  # command-layer direct-drive control (Roberts 2016 style)
        "asel":          ["ASEL"],                 # chemotaxis ON pathway (literature: responds to rising concentration, sustains the forward run)
        "aser":          ["ASER"],                 # chemotaxis OFF pathway (literature: responds to falling concentration, promotes turning/reversal)
        "pvd":           ["PVD"],                  #  harsh touch (literature: -> escape)
        "phb":           ["PHB"],                  # tail sensation (literature: -> AVA -> backward)
        "tail_all":      ["PHA", "PHB", "PHC", "PQR", "PLM", "PDE"],
    }
    RC = np.array([], dtype=int)
    if args.proto == "region":
        ps = tuple(REGION_PREFIX.get(args.region, ()))
        RC = np.array([i for i, n in enumerate(names)
                       if n.startswith(ps) and "sh" not in n], dtype=int)
        print(f"[region] {args.region}: {len(RC)} cells -> {[names[i] for i in RC][:14]}", flush=True)
        if len(RC) == 0:
            raise SystemExit(f"unknown/empty region: {args.region}")

    # ---- Body layer (rod2d=own: original analytic-gradient rod, rod2d_own.py) ----
    if args.rod2d == "own":
        from rod2d_own import Rod2DOwn as Rod2D, preferred_curvature_own as preferred_curvature
    else:
        from worm.body.rod2d import Rod2D
        from worm.body.muscle_map import preferred_curvature
    body = Rod2D(Cn=args.cn, k_b=args.kb)
    x = body.initial(heading=np.radians(args.initdir + 180.0))   # v83: initdir = facing azimuth
    fwd_m = bwd_m = None
    if args.muscle96 == 0.0:
        # Legacy Boyle bang-bang motor layer (only exercised when muscle96 is off;
        # requires the third-party worm-whisperer package).
        from worm.body.boyle_motor import BoyleMotor
        kw = dict(G_SR=args.gsr, sr_span_frac=args.span, I_avb_D=0.5, I_avb_V=0.5)
        fwd_m = BoyleMotor(direction=+1, **kw)
        bwd_m = BoyleMotor(direction=-1, **dict(kw, sr_span_frac=0.5))
    motor_dt = 5e-3; kappa_max = args.kmax
    gate_thresh = 3.0; k_turn = 0.03; head_rows = 12
    if args.kturn != 0: k_turn = args.kturn
    k_omega = 0.36; riv_thresh = 30.0; omega_T = 2.5; omega_sigma = 4.0
    # v60 relative-gating parameters (only with --gate rel): floor = silence floor (below it neither side opens, rest state held),
    # hyst = hysteresis margin (a flip requires the winner to exceed the loser by this margin; anti-chatter)
    gate_floor = 0.5; gate_hyst = 1.0; dir_state = +1
    last_bout_end_ms = -1.0e18   # v91 bwrefr: time of the previous bout end (initialised to the distant past)

    # ---- v65 proprioceptive loop: B-class motor neurons <- neighbouring segment curvature (Wen et al. 2012, Neuron 76:750) ----
    # anatomy: B-class motor neurons (DB/VB) extend unbranched neurites along the ventral cord and were proposed to sense body-wall bending directly;
    # Wen 2012 calcium imaging + optogenetics showed B-class MN activity is phase-locked to the body bending wave, a feedback necessary for forward-wave propagation.
    # construction: cells of each class map uniformly by ordinal onto body segments; D/V antisymmetric;
    # injected once per block, kappa taken from the previous block's end value -> fixed 1-block (50ms) feedback delay (about 5-10% of the wave period, stated as-is).
    # polarity (v66 fix): in preferred_curvature source, kappa>0 = dorsal contraction -> DB +kappa (dorsal bend excites dorsal B-class),
    # VB -kappa; v65 had it reversed (VB +kappa / DB -kappa) — locking amplitude/correlation unaffected (the analysis folds the sign per class),
    # only the inter-class assignment swapped; on the output side (mndrive) the absolute polarity must be as stated to close a positive feedback.
    # default --mnprop 0: this block does not execute at all, legacy behaviour bit-identical.
    BMN_idx = np.zeros(0, dtype=int); BMN_sgn = np.zeros(0); BMN_seg = np.zeros(0, dtype=int)
    BMN_seg_eff2 = np.zeros(0, dtype=int)
    BMN_seg_eff2_rev = BMN_seg_eff2
    AMN_idx = np.zeros(0, dtype=int); AMN_sgn = np.zeros(0); AMN_seg_eff = np.zeros(0, dtype=int)
    if args.mnprop != 0.0 or args.mntonic != 0.0 or args.headosc != 0.0 or args.mnprop2 != 0.0 or args.mnanat != 0 or args.amnprop != 0.0 or args.dvaprop != 0.0:
        nseg = body.n - 1
        _ci, _sg, _se = [], [], []
        if args.mnanat != 0:
            # v69 R10a: anatomical mapping — soma segment = weighted mean position of the muscle rows this cell innervates (neuron_muscle.csv);
            # preferred_curvature: segment-j curvature is driven by muscle rows j, j+1 (0-based) -> muscle row r acts on segments r-1, r,
            # innervation centre (in muscle-row units) r_mean corresponds to segment r_mean-0.5; sensing position aligned with innervation position,
            # so the Zhao 2025 local negative-feedback loop closes at the true anatomical position (fixes the tail dead zone);
            # cells without muscle-innervation records fall back to the uniform mapping
            import re as _re2
            _musc_a, _wout_a = muscle_maps[0], muscle_maps[1]
            for cls, s in (("DB", +1.0), ("VB", -1.0)):
                ks = [k for k in range(1, 30) if f"{cls}{k:02d}" in idx]
                for rank, k in enumerate(ks):
                    ci = idx[f"{cls}{k:02d}"]
                    wrow = np.zeros(24)
                    for mi, mname in enumerate(_musc_a):
                        mm = _re2.fullmatch(r"([DV])([LR])(\d\d)", mname)
                        if not mm:
                            continue
                        if (cls == "DB") != (mm.group(1) == "D"):
                            continue
                        wrow[int(mm.group(3)) - 1] += _wout_a[ci, mi]
                    if wrow.sum() > 0:
                        r_mean = float((wrow * np.arange(24)).sum() / wrow.sum())
                        seg = int(np.clip(round(r_mean - 0.5), 0, nseg - 1))
                    else:
                        seg = min(int((rank + 0.5) / len(ks) * nseg), nseg - 1)
                    _ci.append(ci); _sg.append(s); _se.append(seg)
            print(f"[mnanat] B-class MN soma->segment anatomical mapping ON: placed by innervated rows in neuron_muscle.csv, "
                  f"seg range {min(_se)}..{max(_se)}", flush=True)
        else:
            for cls, s in (("DB", +1.0), ("VB", -1.0)):
                ks = [k for k in range(1, 30) if f"{cls}{k:02d}" in idx]
                for rank, k in enumerate(ks):
                    _ci.append(idx[f"{cls}{k:02d}"]); _sg.append(s)
                    _se.append(min(int((rank + 0.5) / len(ks) * nseg), nseg - 1))
        BMN_idx = np.array(_ci, dtype=int); BMN_sgn = np.array(_sg); BMN_seg = np.array(_se, dtype=int)
        # v67 R1: sensing position = soma segment shifted mnoffset segments anteriorly (anterior = head = decreasing segment index; Rod2D xs[0] = head);
        # clipped to 0 when a head MN has no segment anterior to it. With mnoffset=0, BMN_seg_eff==BMN_seg, bit-identical.
        BMN_seg_eff = np.clip(BMN_seg - args.mnoffset, 0, nseg - 1)
        # v67 R4: second-channel sensing position (negative offset = posterior, paired with the Cohen stretch sign)
        BMN_seg_eff2 = np.clip(BMN_seg - args.mnoffset2, 0, nseg - 1)
        # v70 R14: backward mirroring — when backward gating is active the offset is negated (posterior sensing <-> anterior sensing = wave-direction reversal);
        # with default mnback=0, BMN_seg_eff2_rev equals BMN_seg_eff2, bit-identical
        if args.mnback != 0:
            BMN_seg_eff2_rev = np.clip(BMN_seg + args.mnoffset2, 0, nseg - 1)
        else:
            BMN_seg_eff2_rev = BMN_seg_eff2
        if args.bmnclip == "reflect":
            # W57c (2026-10-23): out-of-range sensing positions mirrored back into the body — with clip0 all tail-side B-class clamp to segment 23
            # (tail-tip kappa~0 = self-consistent dead zone: tip does not bend -> no signal -> tail B dies -> tip bends even less); under backward mirroring the head side
            # clamps to segment 0 = forced in-phase = head synchronised block becomes a standing wave; after reflection, boundary cells receive dispersed travelling-wave signals
            def _refl(a):
                r = a.copy()
                over = r > nseg - 1
                r[over] = 2 * (nseg - 1) - r[over]
                under = r < 0
                r[under] = -r[under]
                return np.clip(r, 0, nseg - 1)
            _raw2 = BMN_seg - args.mnoffset2
            BMN_seg_eff2 = _refl(_raw2)
            if args.mnback != 0:
                BMN_seg_eff2_rev = _refl(BMN_seg + args.mnoffset2)
            print(f"[bmnclip] reflect ON: forward-state out-of-range {int((_raw2 > nseg - 1).sum() + (_raw2 < 0).sum())} / "
                  f"backward-state out-of-range {int(((BMN_seg + args.mnoffset2) > nseg - 1).sum() + ((BMN_seg + args.mnoffset2) < 0).sum())} "
                  f"B-class sensing positions mirrored back into the body", flush=True)
        # v67 R1b: head B-class mask (soma mapped onto the first 2 segments), for --headosc sine-seed injection
        BMN_head = BMN_seg <= 1
        print(f"[mnprop] B-class MN proprioceptive loop ON: G={args.mnprop:g} pA/(1/mm), "
              f"{len(_ci)} cells (DB/VB) -> {nseg} segs, 1-block (50ms) feedback delay, "
              f"anterior offset={args.mnoffset} segs, head cells={int(BMN_head.sum())}", flush=True)
        if args.mnprop2 != 0.0:
            print(f"[mnprop2] second proprioceptive channel ON: G2={args.mnprop2:g} pA/(1/mm), "
                  f"offset2={args.mnoffset2} segs (negative = posterior)", flush=True)
            if args.mnback != 0:
                print(f"[mnback] backward mirroring ON: offset2 negated while backward gating is active "
                      f"({args.mnoffset2}->{-args.mnoffset2} segs, wave direction reverses with the directional state)", flush=True)
        # ---- v85: A-class MN (VA/DA) proprioceptive loop (Gao et al. 2018 eLife 7:e29915) ----
        # A-class cells are endogenous backward oscillators and accept proprioceptive entrainment; the loop mirrors the B-chain backward form: anterior sensing (-amnoffset segs,
        # head side) -> head->tail propagation = backward wave (v85b fix: v85a mistakenly took posterior sensing = tail->head = forward wave, fighting the B-chain forward state);
        # the soma->segment mapping reuses the B-chain anatomical/uniform logic (DA -> dorsal muscle rows, VA -> ventral)
        if args.amnprop != 0.0 or args.dvaprop != 0.0:
            _ciA, _sgA, _seA = [], [], []
            if args.mnanat != 0:
                for cls, s in (("DA", +1.0), ("VA", -1.0)):
                    ks = [k for k in range(1, 30) if f"{cls}{k:02d}" in idx]
                    for rank, k in enumerate(ks):
                        ci = idx[f"{cls}{k:02d}"]
                        wrow = np.zeros(24)
                        for mi, mname in enumerate(_musc_a):
                            mm = _re2.fullmatch(r"([DV])([LR])(\d\d)", mname)
                            if not mm:
                                continue
                            if (cls == "DA") != (mm.group(1) == "D"):
                                continue
                            wrow[int(mm.group(3)) - 1] += _wout_a[ci, mi]
                        if wrow.sum() > 0:
                            r_mean = float((wrow * np.arange(24)).sum() / wrow.sum())
                            seg = int(np.clip(round(r_mean - 0.5), 0, nseg - 1))
                        else:
                            seg = min(int((rank + 0.5) / len(ks) * nseg), nseg - 1)
                        _ciA.append(ci); _sgA.append(s); _seA.append(seg)
            else:
                for cls, s in (("DA", +1.0), ("VA", -1.0)):
                    ks = [k for k in range(1, 30) if f"{cls}{k:02d}" in idx]
                    for rank, k in enumerate(ks):
                        _ciA.append(idx[f"{cls}{k:02d}"]); _sgA.append(s)
                        _seA.append(min(int((rank + 0.5) / len(ks) * nseg), nseg - 1))
            AMN_idx = np.array(_ciA, dtype=int); AMN_sgn = np.array(_sgA)
            AMN_seg = np.array(_seA, dtype=int)   # v92: soma segments (site of dvaprop braking)
            # W56 (2026-10-23): amndir=posterior -> sense the tail side (+amnoffset) -> tail->head relay = backward wave (Gao 2018);
            # anterior = legacy behaviour (v85b): sense the head side -> head->tail wave = forward direction
            _off = args.amnoffset if args.amndir == "anterior" else -args.amnoffset
            _rawA = np.array(_seA, dtype=int) - _off
            # W55 (2026-10-23): with amnclip=edge0, out-of-boundary injection is zeroed — out-of-boundary A-class cells
            # escape the clip in-phase artefact and become a pure endogenous-oscillator population (Gao 2018), seeding the wave
            if args.amnclip == "reflect":
                # W57 (2026-10-23): out-of-range sensing positions mirrored back into the body — the edge0 starvation experiment showed tail VA09-12
                # once injection is lost, are shut down by bout-activated VD (GABA) (bout hyperpolarisation -11mV);
                # after reflection the sensing positions stay dispersed (no clip in-phase artefact); tail A-class ride the B-chain carrier and rejoin
                _refl = _rawA.copy()
                _over = _refl > nseg - 1
                _refl[_over] = 2 * (nseg - 1) - _refl[_over]
                _under = _refl < 0
                _refl[_under] = -_refl[_under]
                AMN_seg_eff = np.clip(_refl, 0, nseg - 1)
                AMN_emask = np.ones(len(_seA), dtype=float)
                print(f"[amnclip] reflect ON: out-of-range {int((_over | _under).sum())}/{len(_seA)} A-class sensing positions "
                      f"mirrored back into the body (dir={args.amndir})", flush=True)
            else:
                AMN_seg_eff = np.clip(_rawA, 0, nseg - 1)
                if args.amnclip == "edge0":
                    AMN_emask = ((_rawA >= 0) & (_rawA <= nseg - 1)).astype(float)
                    print(f"[amnclip] edge0 ON: out-of-boundary {int((AMN_emask == 0).sum())}/{AMN_emask.size} A-class somata "
                          f"zero injection (pure endogenous-oscillator ends, dir={args.amndir})", flush=True)
                else:
                    AMN_emask = np.ones(len(_seA), dtype=float)
            # W56 (2026-10-23): UNC-2-driven tail-high/head-low gradient (Gao 2018: DA9, the most caudal, is the pacemaker oscillator)
            if args.amngrad != 0.0:
                AMN_gfac = 1.0 + args.amngrad * (2.0 * np.array(_seA, dtype=float) / (nseg - 1) - 1.0)
                print(f"[amngrad] UNC-2 tail-high/head-low gradient ON: g={args.amngrad:g} head x{AMN_gfac.min():.2f} "
                      f"tail x{AMN_gfac.max():.2f} (Gao2018 DA9 pacemaker)", flush=True)
            else:
                AMN_gfac = np.ones(len(_seA), dtype=float)
            if args.amnprop != 0.0:
                print(f"[amnprop] A-class MN proprioceptive loop ON: G={args.amnprop:g} pA/(1/mm), "
                      f"{len(_ciA)} cells (DA/VA) -> {nseg} segs, anterior sensing offset=-{args.amnoffset} segs (head side, head->tail backward wave), "
                      f"gating={'backward-only' if args.amngate else 'always'} (Gao2018 A-class oscillator entrainment)", flush=True)
            if args.dvaprop != 0.0:
                print(f"[dvaprop] DVA/TRP-4 amplitude brake ON: G={args.dvaprop:g} pA/(1/mm) tau={args.dvatau:g}s, "
                      f"|kappa slow-EMA| hyperpolarising injection into {len(_ciA)} A-class MN soma segments, backward-state only "
                      f"(Li 2006 Nature PMC2865900: trp-4(lf) = over-bending, DVA negative limiting)", flush=True)

    # ---- v66 loading loop: B-class MN -> anatomical muscle drive (lower half of the Wen 2012 closed loop) ----
    # innervation weights = anatomical table neuron_muscle.csv (BAAIWorm export, 96 body-wall muscles x MN, unit = contact count);
    # DB->dorsal muscle rows, VB->ventral rows (B-class cholinergic excitation of ipsilateral body-wall muscle); L/R pair averaged into a single row (x0.5);
    # each muscle normalised by total innervation (n_inn); drive function softplus(dV)-ln2: dV=0 -> 0 (no contribution at rest),
    # dV=+10mV -> ~9.3, slightly negative below (graded release below rest, physiologically reasonable).
    # default --mndrive 0: matrix not built, legacy behaviour bit-identical.
    Wdrv_D = np.zeros((24, N)); Wdrv_V = np.zeros((24, N))
    if args.mndrive != 0.0 or args.dmndrive != 0.0:
        import re as _re
        _musc, _wout, _inhib, _ninn = muscle_maps
        for cls, M in (("DB", Wdrv_D), ("VB", Wdrv_V)):
            rows_i = [idx[f"{cls}{k:02d}"] for k in range(1, 30) if f"{cls}{k:02d}" in idx]
            for mi, mname in enumerate(_musc):
                mm = _re.fullmatch(r"([DV])([LR])(\d\d)", mname)
                if not mm:
                    continue
                if (cls == "DB") != (mm.group(1) == "D"):
                    continue
                row = int(mm.group(3)) - 1
                M[row, rows_i] += _wout[rows_i, mi] / _ninn[mi] * 0.5
        if args.mndrive != 0.0:
            print(f"[mndrive] B-class MN->muscle drive ON: G={args.mndrive:g}, anatomical-table innervation "
                  f"D-side nonzeros {int((Wdrv_D > 0).sum())} / V-side {int((Wdrv_V > 0).sum())}, "
                  f"boylescale={args.boylescale:g}", flush=True)
    # v67 R2: D-class MN -> muscle cross-inhibition matrix (White 1986 adult: DD->dorsal muscle, VD->ventral muscle)
    Wdm_D = np.zeros((24, N)); Wdm_V = np.zeros((24, N))
    if args.dmndrive != 0.0:
        import re as _re2
        _musc2, _wout2, _inhib2, _ninn2 = muscle_maps
        for cls, M, side in (("DD", Wdm_D, "D"), ("VD", Wdm_V, "V")):
            rows_i = [idx[f"{cls}{k:02d}"] for k in range(1, 30) if f"{cls}{k:02d}" in idx]
            for mi, mname in enumerate(_musc2):
                mm = _re2.fullmatch(r"([DV])([LR])(\d\d)", mname)
                if not mm or mm.group(1) != side:
                    continue
                row = int(mm.group(3)) - 1
                M[row, rows_i] += _wout2[rows_i, mi] / _ninn2[mi] * 0.5
        print(f"[dmndrive] D-class MN->muscle cross-inhibition ON: G={args.dmndrive:g}, anatomical-table innervation "
              f"DD->D-side nonzeros {int((Wdm_D > 0).sum())} / VD->V-side {int((Wdm_V > 0).sum())}, "
              f"local, no offset (D-class has no long neurites)", flush=True)

    # ---- v68 R6: SMD -> head-muscle anatomical drive + SMD proprioceptive loop (Yeon 2018 / Zhao 2025 anterior loop) ----
    # anatomy: SMDDL/R->DR/DL01-06 (dorsal head muscles), SMDVL/R->VR/VL01-06 (ventral head muscles), table verified;
    # drive formula identical to the B-class mndrive; feedback = mean curvature of the 6 head segments, negative gain = stretch reflex (isomorphic to the Zhao negative-feedback loop).
    # default --smddrive/--smdprop 0: no matrix, no injection, legacy behaviour bit-identical.
    Wsmd_D = np.zeros((24, N)); Wsmd_V = np.zeros((24, N))
    SMD_idx = np.zeros(0, dtype=int); SMD_sgn = np.zeros(0)
    SMDD_idx = np.zeros(0, dtype=int); SMDV_idx = np.zeros(0, dtype=int)
    if args.smddrive != 0.0 or args.smdprop != 0.0 or args.turnsmd != 0.0 or args.fc1 != 0.0:
        import re as _re3
        _musc3, _wout3, _inhib3, _ninn3 = muscle_maps
        for cls, M, side in (("SMDD", Wsmd_D, "D"), ("SMDV", Wsmd_V, "V")):
            cells = [f"{cls}{s}" for s in ("L", "R") if f"{cls}{s}" in idx]
            rows_i = [idx[c] for c in cells]
            for mi, mname in enumerate(_musc3):
                mm = _re3.fullmatch(r"([DV])([LR])(\d\d)", mname)
                if not mm or mm.group(1) != side:
                    continue
                row = int(mm.group(3)) - 1
                M[row, rows_i] += _wout3[rows_i, mi] / _ninn3[mi] * 0.5
        _ci2, _sg2 = [], []
        for cls, s in (("SMDD", +1.0), ("SMDV", -1.0)):
            for sd in ("L", "R"):
                if f"{cls}{sd}" in idx:
                    _ci2.append(idx[f"{cls}{sd}"]); _sg2.append(s)
        SMD_idx = np.array(_ci2, dtype=int); SMD_sgn = np.array(_sg2)
        SMDD_idx = SMD_idx[SMD_sgn > 0]; SMDV_idx = SMD_idx[SMD_sgn < 0]  # R9 turning class masks
        if args.smdabl.upper() == "D":    # trp-1,2 double-mutant mimicry: SMDD feedback cut (Yeon 2018)
            SMD_sgn[SMD_sgn > 0] = 0.0
        elif args.smdabl.upper() == "V":
            SMD_sgn[SMD_sgn < 0] = 0.0
        print(f"[smd] SMD loop ON: drive G={args.smddrive:g} (D-side nonzeros {int((Wsmd_D > 0).sum())} / "
              f"V-side {int((Wsmd_V > 0).sum())}), prop G={args.smdprop:g} (mean of the 6 head segments), "
              f"abl='{args.smdabl}', cells={len(SMD_idx)}", flush=True)

    # ---- v74 R18 (FC1): RIA->SMD in-situ synaptic-budget edge table (amplitude baseline for event-gated endogenous turning drive) ----
    # survey 2026-10-20: RIA->SMD tonic budget ~20pA (8 edges); a turnsmd=40 pulse is only a 2x gap;
    # D/V asymmetry given by anatomy (RIAL favours SMDV 4.68v4.00 / RIAR favours SMDD 6.19v5.10)
    FC1_pre = np.zeros(0, dtype=int); FC1_post = np.zeros(0, dtype=int); FC1_w = np.zeros(0)
    if args.fc1 != 0.0:
        _ria = [idx[c] for c in ("RIAL", "RIAR") if c in idx]
        _smdset = set(SMDD_idx.tolist()) | set(SMDV_idx.tolist())
        _e = [(int(c[0]), int(c[1]), c[2]) for c, sg_ in zip(chem, sign)
              if int(c[0]) in _ria and int(c[1]) in _smdset and sg_ > 0]
        FC1_pre = np.array([e[0] for e in _e], dtype=int)
        FC1_post = np.array([e[1] for e in _e], dtype=int)
        FC1_w = np.array([e[2] for e in _e])
        print(f"[fc1] endogenous turning drive ON: G={args.fc1:g} x RIA->SMD budget ({len(_e)} edges), "
              f"dir={args.fc1dir}, tau2={args.fc1tau2:g}s", flush=True)

    # ---- checkpoint resume (v63b): loading a snapshot skips warm-up; neural/body/motor/gating/ASE states all restored ----
    if args.ckr1 != 0.0:
        print(f"[ckr1] stage-2 CKR-1 peptidergic gain ON: G0={args.ckr1:g} pA, "
              f"tau12={args.ckr1tau12:g}s tau18={args.ckr1tau18:g}s M0={args.ckr1m0:g}; "
              f"anchors Chen2022 (PMID 35139370)/Chen2024 (PMC10952047): NLP-12 arm->SMD body curvature / "
              f"NLP-18 arm->SMD+AIB omega turns; w18/w12=1.42 (Ca2+ peaks 0.502/0.353)", flush=True)
    ckpt = None
    if args.loadstate:
        import pickle
        with open(args.loadstate, "rb") as fh:
            ckpt = pickle.load(fh)
        print(f"[ckpt] loaded {args.loadstate} (snapshot t={ckpt['t_ms']/1000:.1f}s)", flush=True)

    # ---- 2 s baseline warm-up (skipped on resume; snapshot state restored directly) ----
    if ckpt is None:
        print(f"[init] build {time.time()-t0:.1f}s; warmup 2s baseline...", flush=True)
        Iex0 = jnp.zeros(N, jnp.float64)
        if args.ckr1 != 0.0 and args.ckr1m0 != 0.0:
            # CKR-1 warm-up in the same regime: M=M0 constant injection (as for ggapmod warm-up; otherwise V_rest is captured in a peptide-free state and the regime-shift transient contaminates)
            Iex0 = Iex0.at[np.asarray(SMDD + SMDV)].add(
                jnp.asarray(args.ckr1 * (1.0 + 1.42) * float(args.ckr1m0), jnp.float64))
            Iex0 = Iex0.at[np.asarray(AIB_i)].add(
                jnp.asarray(args.ckr1 * 1.42 * float(args.ckr1m0), jnp.float64))
        n_warm = int(round(2000 / BLOCK_MS))
        for b in range(n_warm):
            if run_block_lv is not None:
                # ggapmod: warm-up also runs in the modulated regime (M=M0 -> g_eff=(1-k(1-M0))xGGAP);
                # otherwise warm-up equilibrates in the wrong coupling state, V_rest is captured at the wrong operating point, and the regime-shift transient causes spasm (observed)
                _gmod0 = 1.0 - args.ggapmodk * (1.0 - float(args.ggapmodm0)) \
                    if args.ggapmod != 0.0 else 1.0
                if AR is not None:
                    # W49: warm-up likewise runs rectified (previous-block voltage; first block = Y0 isolated anchor: AVA-25/AMN-72 -> nearly closed)
                    _Vw = np.asarray(assemble_V(state[0], state[1]))
                    state = run_block_lv(state, Iex0,
                                         jnp.asarray(_gmod0 * _Lg_w49(_Vw), jnp.float64))
                else:
                    state = run_block_lv(state, Iex0,
                                         jnp.asarray(_gmod0 * Lg_full_np, jnp.float64))
            else:
                state = run_block(state, Iex0)
            if b == 0:
                print(f"[init] first block compiled+run, {time.time()-t0:.1f}s", flush=True)
        V_rest = np.asarray(assemble_V(state[0], state[1])).copy()
        print(f"[warm] V mean {V_rest.mean():.2f} min {V_rest.min():.2f} max {V_rest.max():.2f} "
              f"nan={np.isnan(V_rest).any()}", flush=True)
        print(f"[warm] dAVB0={V_rest[AVB].mean():.2f} dAVA0={V_rest[AVA].mean():.2f}", flush=True)
    else:
        state = tuple(jnp.asarray(a) for a in ckpt["state"])
        if riacomp and len(state) == 4:
            # old 4-field snapshot + riacomp=1: add compartment initial states (Yc/gAc/gIaiy)
            _Y0 = init_state2()
            state = state + _Y0[4:]
            print(f"[ckpt] old snapshot: compartment initial states added (riacomp)", flush=True)
        V_rest = ckpt["V_rest"]
        x = jnp.asarray(ckpt["x"])
        fwd_m = ckpt["fwd_m"]; bwd_m = ckpt["bwd_m"]
        print(f"[ckpt] state restored, warm-up skipped (V_rest carried over from the snapshot)", flush=True)

    if m96 is not None:
        # v81: recompute the tonic baseline from true resting voltages + actual vrel table (dorsoventral anatomical asymmetry, see muscle96.py)
        _rel0 = m96.calibrate_rest(V_rest, net.vrel, mode=args.m96sig)
        print(f"[muscle96] resting baseline I0 calibrated (mode={args.m96sig}): rel0 mean={_rel0.mean():.3f} "
              f"min={_rel0.min():.3f} max={_rel0.max():.3f}", flush=True)

    # ---- stimulation phase ----
    secs = 3.0 if args.smoke else args.secs
    n_blocks = int(round(secs * 1000 / BLOCK_MS))

    # ---- chemotaxis protocol initialisation ----
    ase = None
    chem_c = [0.0]          # v78 R25b: per-block instantaneous concentration (for the pirc absolute-concentration dwell gate)
    arr_clp = [0.0]         # v78 R25c: low-passed concentration/peak (dopamine tonic signal for Sawin basal slowing)
    asad = [0.0]            # v79 R26: GPCR-level sensory-adaptation state 0-1 (ARR-1-style desensitisation; updated only when asad!=0)
    asad_gain = [1.0]       # v79 R26: adaptation gain 1/(1+K*a), attenuating weathervane-related signals in step
    # P11 (2026-10-22): ASH chemical-transduction front-end state (anchor Hilliard 2005 PMC544906)
    #   m = transduction drive state (tau_on~1s); a = GPC-1-style adaptation state (charge tau_a~20s, recovery tau_r~100s)
    #   I_ASH = ashchem * m * max(1-a, 0); concentration square wave c=1 in window 8-16s (same timing as the region protocol, as control)
    ash_m = [0.0]; ash_a = [0.0]
    ash_mod = [0.0]     # P13: INX-20/cGMP modulation state 0-1 (Chaubey 2023; starvation-driven, tau 60s up / 300s recovery)
    if args.proto == "chem":
        if args.chemsrc:
            sx, sy = (float(v) for v in args.chemsrc.split(","))
            ase = ASEInput(ChemField((sx, sy), sigma=args.chemsigma), gain=args.chemgain)
            print(f"[chem] source=({sx:g},{sy:g}) sigma={args.chemsigma:g} gain={args.chemgain:g} pA/(C/s) "
                  f"tau=0.5s | ASEL/ASER via the anatomical pathway", flush=True)
        else:
            print("[chem] source-free control (AVB operating point only)", flush=True)

    def make_Iex(t_s):
        """return the externally applied current vector at the current time, per the optogenetic protocol"""
        I = np.zeros(N)
        if args.proto == "switch":        # forward->backward->forward state switch
            if t_s < 10.0:      I[AVB] = args.pamp
            elif t_s < 22.0:    I[AVA] = args.pamp
            else:               I[AVB] = args.pamp
        elif args.proto == "avapulse":    # v93: AVA short pulse (touch mimic, Wicks 1995 bout 2-3s),
            if 6.0 <= t_s < 8.0:  I[AVA] = args.pamp   # no external drive afterwards -> tests the endogenous exit clock
        elif args.proto == "ris":         # RIS point excitation during forward -> abrupt stop
            I[AVB] = args.pamp
            if 12.0 <= t_s < 18.0: I[RIS] = args.pamp
        elif args.proto == "region":      # silent-partition point excitation during forward, 8-16s
            I[AVB] = args.stim
            if args.phasic > 0.0:
                # v70 R15: fast-adapting phasic pulse (onset+offset double pulse, O'Hagan 2005)
                _w = args.phasic / 1000.0
                if (8.0 <= t_s < 8.0 + _w) or (16.0 - _w <= t_s < 16.0):
                    I[RC] = args.pamp
            elif 8.0 <= t_s < 16.0:
                I[RC] = args.pamp
        elif args.proto == "ashchem":     # P11: ASH chemical-transduction front end (anchor Hilliard 2005)
            I[AVB] = args.stim
            _dt = BLOCK_MS / 1000.0
            _w0, _w1 = (float(x) for x in args.ashchemwin.split(":"))
            _c = 1.0 if (_w0 <= t_s < _w1) else 0.0   # concentration square wave (default 8-16s, same timing as region)
            ash_m[0] += (_dt / args.ashchemtau) * (_c - ash_m[0])          # transduction onset tau~1s
            if ash_m[0] > ash_a[0]:
                ash_a[0] += (_dt / args.ashchemtaua) * (ash_m[0] - ash_a[0])  # adaptation charge 20s
            else:
                ash_a[0] += (_dt / args.ashchemtaur) * (ash_m[0] - ash_a[0])  # recovery 100s
            _i = args.ashchem * ash_m[0] * max(1.0 - ash_a[0], 0.0)
            # P13 (2026-10-22): INX-20/cGMP/EGL-4 volume knob (Chaubey 2023 PMC10319955)
            #   starvation -> ODR-1 produces cGMP (AWB/AWC/ASI) -> via the INX network (ADL->RMG/ASK->ASH) into ASH
            #   -> EGL-4(PKG)->RGS-2/3 -> down-regulates GPCR signalling -> ASH sensitivity drops (inx-20ko = hypersensitive)
            #   tau: minute scale (cGMP gap-junction diffusion + PKG phosphorylation; Hilliard 5min recovery, same order)
            if args.ashstarve > 0.0:
                ash_mod[0] += (_dt / 60.0) * (args.ashstarve - ash_mod[0])
            else:
                ash_mod[0] += (_dt / 300.0) * (0.0 - ash_mod[0])
            _i *= 1.0 / (1.0 + args.ashmod * ash_mod[0])
            I[ASH_i[0]] += _i; I[ASH_i[1]] += _i
        elif args.proto == "chem":        # chemotaxis: AVB operating point + ASE sensory closed loop
            I[AVB] = args.stim
            if ase is not None:
                on_, off_, _c = ase.currents(np.asarray(x)[0], BLOCK_MS / 1000.0)
                I[ASEL] += on_; I[ASER] += off_
                chem_c[0] = _c                 # v78 R25b: instantaneous concentration for the absolute-concentration dwell gate
                # v78 R25c: low-passed concentration/peak (dopamine tonic timescale; MGL-1-type GPCR persistent-state transduction)
                arr_clp[0] += (BLOCK_MS / 1000.0 / args.arrtau) * (_c / max(ase.f.peak, 1e-12) - arr_clp[0])
                if args.asad != 0.0:
                    # v79 R26: GPCR-level sensory adaptation (ARR-1-style desensitisation) — in sustained high concentration (>asadth) the adaptation state rises,
                    # ASEL/ASER injection gain attenuates as 1/(1+K*a) -> tether fades on the peak -> spontaneous dispersal;
                    # with asad=0 this block does not execute, legacy behaviour bit-identical
                    _dr = max(arr_clp[0] - args.asadth, 0.0) / max(1.0 - args.asadth, 1e-12)
                    asad[0] += (BLOCK_MS / 1000.0 / args.asadtau) * (_dr - asad[0])
                    asad_gain[0] = 1.0 / (1.0 + args.asad * asad[0])
                    I[ASEL] += on_ * (asad_gain[0] - 1.0)
                    I[ASER] += off_ * (asad_gain[0] - 1.0)
        else:
            I[AVB] = args.stim
        return jnp.asarray(I, jnp.float64)

    Iex = make_Iex(0.0)

    log = []; Ytr = []
    kappa = np.zeros(body.n - 1)
    kap2_f = np.zeros(body.n - 1)   # v67 R5: second-channel EMA filter state (muscle mechanical-delay proxy)
    kap1_f = np.zeros(body.n - 1)   # v68 R7: first-channel EMA filter state (Bryden-Cohen travelling-wave dispersion delay)
    kap2_hp = np.zeros(body.n - 1)  # v91: A-class loop high-pass reference state (used only with --amnhp>0)
    kap2_dva = np.zeros(body.n - 1)  # v92: DVA brake slow-EMA state (used only with --dvaprop!=0)
    am_ema = 0.0                     # v93: AVA master-neuron slow-EMA state (used only with --avamaster!=0)
    mA_D = np.zeros(24); mA_V = np.zeros(24)   # v68 R7: muscle-activation integral states (24 muscle rows each side, D/V)
    t_ms = 0.0; omega_t0 = None
    hold_I = 0.0   # v70 R16: AVA plateau self-sustaining current state (charged on the trigger edge, unconditional exponential decay)
    m2_pep = 0.0          # v73 R17: backward-loop peptidergic positive-feedback slow variable (inactive when pep2=0)
    n2_pep = 0.0          # v73 R17b: negative-arm slow variable (AVR-14-style delayed inhibition, bout self-termination)
    m_fc1 = 0.0           # v74 R18: turn-execution gating slow variable (charged to 1 on event, exponential decay; inactive when fc1=0)
    m_plat = np.zeros(len(AVA))  # v76 R21: AVA plateau-current fast activation gate (L/R separate, VGCC-style)
    n_plat = np.zeros(len(AVA))  # v76 R21: slow inactivation gate (tracks m, bout self-termination)
    mp2 = np.zeros(len(AVA))     # v91: avaplat2 gated edge-triggered plateau, fast activation gate
    ris_m = 0.0                  # W59b: RIS bout-termination integrator state (charges in backward, discharges in forward)
    ris_hold_ms = 0.0            # W69: ris_m full-charge hold timer (for the riskill fuse)
    np2 = np.zeros(len(AVA))     # v91: avaplat2 slow inactivation gate
    slo_ca = np.zeros(len(AMN_idx))  # v92: A-class SLO calcium trajectory (activity-dependent negative feedback, active with --sloa>0)
    dV_prev = np.zeros(N)  # v73 R17: previous-block voltage (feedback signal, 1 block = 50ms delay, stated as-is)
    dir_state_prev = +1   # v70 R16c: previous-block directional state for trigger-edge detection
    fwd_since = -1.0e9    # v70 R16e: time of the last bout end (2 s re-arm timer)
    pulse_until = -1.0; refr_until = 0.0; dsmd_prev = 0.0   # v64 event-pirouette states
    smd_turn_on = False   # v68 R9: pulse state of SMD-executed turning (takes effect with 1-block delay)
    w_weath = 0.0; kap0_prev = 0.0   # v71 W1: weathervane correlation integrator state + previous head-joint curvature
    w_frz_until_ms = 0.0             # W68: end of the post-bout w freeze window (never active when wrst=0)
    dwn_lp = 0.0          # v83b: slow low-pass state of the worsening signal (tau=dwntau, suppresses head-swing leakage)
    pulse_dir = 1.0   # v71 W2: current turning-pulse direction (+1=ventral, -1=dorsal; re-drawn on each trigger when pirrand>0)
    rng_pir = np.random.default_rng(args.pirrand) if args.pirrand > 0 else None
    # ---- peptidergic layer state (DVA->NLP-12->CKR-2): DVA stretch-driven, CKR-2 presynaptic NMJ gain ----
    pep_m = 0.0                      # NLP-12/CKR-2 activation level m(t) in [0,1]
    gm_m = float(args.ggapmodm0)     # ggapmod: modulation state M(t) in [0,1], initial value M0 (stage-2, see --ggapmod)
    ckr_m12 = float(args.ckr1m0); ckr_m18 = float(args.ckr1m0)   # stage-2 CKR-1 two-arm modulation states (see --ckr1)
    KAP_REF = 3.0                    # DVA saturation stretch reference curvature, 1/mm
    pep_dt = BLOCK_MS / 1000.0
    if ckpt is not None:
        kappa = ckpt["kappa"]; t_ms = ckpt["t_ms"]; omega_t0 = ckpt["omega_t0"]
        kap2_f = ckpt.get("kap2_f", np.zeros(body.n - 1))
        kap1_f = ckpt.get("kap1_f", np.zeros(body.n - 1))
        kap2_hp = ckpt.get("kap2_hp", np.zeros(body.n - 1))
        mA_D = ckpt.get("mA_D", np.zeros(24)); mA_V = ckpt.get("mA_V", np.zeros(24))
        pep_m = ckpt["pep_m"]; dir_state = ckpt["dir_state"]
        last_bout_end_ms = ckpt.get("last_bout_end_ms", -1.0e18)
        gm_m = float(ckpt.get("gm_m", 0.0))   # ggapmod state resume (0 if an old snapshot lacks this key)
        ckr_m12 = float(ckpt.get("ckr_m12", 0.0)); ckr_m18 = float(ckpt.get("ckr_m18", 0.0))   # CKR-1 resume
        hold_I = ckpt.get("hold_I", 0.0)
        dir_state_prev = ckpt.get("dir_state_prev", dir_state)
        fwd_since = ckpt.get("fwd_since", -1.0e9)
        pulse_until = ckpt.get("pulse_until", -1.0); refr_until = ckpt.get("refr_until", 0.0)
        dsmd_prev = ckpt.get("dsmd_prev", 0.0)
        w_weath = ckpt.get("w_weath", 0.0); kap0_prev = ckpt.get("kap0_prev", 0.0)
        w_frz_until_ms = ckpt.get("w_frz_until_ms", 0.0)
        dwn_lp = ckpt.get("dwn_lp", 0.0)
        pulse_dir = ckpt.get("pulse_dir", 1.0)
        m_fc1 = ckpt.get("m_fc1", 0.0)
        m_plat = ckpt.get("m_plat", np.zeros(len(AVA))); n_plat = ckpt.get("n_plat", np.zeros(len(AVA)))
        mp2 = ckpt.get("mp2", np.zeros(len(AVA))); np2 = ckpt.get("np2", np.zeros(len(AVA)))
        slo_ca = ckpt.get("slo_ca", np.zeros(len(AMN_idx)))   # v92: SLO calcium trajectory resume (default 0 in old snapshots)
        if m96 is not None and ckpt.get("mA96") is not None:
            m96.mA = np.asarray(ckpt["mA96"], dtype=float).copy()   # v81: 96-muscle integrator-state continuity
            if ckpt.get("mE96") is not None:
                m96.mE = np.asarray(ckpt["mE96"], dtype=float).copy()  # P10: fast-layer state continuity
            print(f"[ckpt] muscle96 integrator state restored (mean mA={m96.mA.mean():.3f})", flush=True)
        ash_m[0] = float(ckpt.get("ash_m", 0.0)); ash_a[0] = float(ckpt.get("ash_a", 0.0))  # P11
        ash_mod[0] = float(ckpt.get("ash_mod", 0.0))  # P13
        if ase is not None and ckpt.get("ase_prev") is not None:
            ase.prev = ckpt["ase_prev"]; ase.dcdt = ckpt["ase_dcdt"]
            arr_clp[0] = ckpt.get("arr_clp", 0.0)
            asad[0] = ckpt.get("asad", 0.0)         # v79 R26: adaptation-state continuity (default 0 in old snapshots)
            if args.asad != 0.0:
                asad_gain[0] = 1.0 / (1.0 + args.asad * asad[0])
    print(f"[run] stim AVB {args.stim} pA x {secs}s ({n_blocks} blocks) pep={args.pep}", flush=True)
    ris_off = False  # v70 R18n: RIS stop rule gates the second proprioceptive channel (triggered only in the RIS protocol)
    for b in range(n_blocks):
        Iex = make_Iex(t_ms / 1000.0)
        if args.mnprop != 0.0:
            # v65 proprioceptive injection: previous-block end curvature -> B-class MN applied current (D/V antisymmetric)
            # v67 R1: sensing position taken mnoffset segments anterior to the soma (Wen 2012: ~200um anterior)
            # v68 R7: optional EMA delay (Bryden-Cohen 2008 travelling-wave dispersion: anterior sensing + time delay -> propagation)
            if args.mnproptau > 0.0:
                kap1_f = kap1_f + (BLOCK_MS / args.mnproptau) * (kappa - kap1_f)
            else:
                kap1_f = kappa
            if args.mnspan > 0:
                # v69 R10a2: neurite-span window (Wen 2012: sensing distributed along the neurite, not a soma point)
                _w1 = 2 * args.mnspan + 1
                kap1_s = np.convolve(kap1_f, np.ones(_w1) / _w1, mode="same")
            else:
                kap1_s = kap1_f
            Iex = Iex.at[BMN_idx].add(
                jnp.asarray(args.mnprop * BMN_sgn * kap1_s[BMN_seg_eff], jnp.float64))
        if args.mnprop2 != 0.0 or args.amnprop != 0.0:
            # v67 R4: second proprioceptive channel (Cohen stretch sign = negative gain, posterior sensing = negative offset)
            # v67 R5: optional EMA low-pass (Zhao 2025 muscle mechanical-delay proxy), removing the 10Hz block-delay artefact of local negative feedback;
            #         optional contraction-threshold half-wave rectification (Zhao 2025: negative feedback fires only past a set shortening), guarding against linear cancellation
            if args.mnprop2tau > 0.0:
                kap2_f = kap2_f + (BLOCK_MS / args.mnprop2tau) * (kappa - kap2_f)
            else:
                kap2_f = kappa
            if args.mnspan > 0:
                _w2 = 2 * args.mnspan + 1
                kap2_s = np.convolve(kap2_f, np.ones(_w2) / _w2, mode="same")
            else:
                kap2_s = kap2_f
            # v70 R14: backward mirroring — dir_state holds the previous block's end value; in the backward state sensing positions use the mirrored array
            # (wave direction reverses with the directional state; switching delay = 1 block = 50ms, same order as the existing feedback delay)
            _eff2 = BMN_seg_eff2_rev if (args.mnback != 0 and dir_state < 0) else BMN_seg_eff2
            # v91/W58 high-pass reference: slow EMA (= static-bend DC) — shared by the A loop (amnhp) and the B loop in the backward state (bmnhp);
            # tau prefers amnhp (already used by the A loop); bmnhp is used when amnhp=0; identical time-constant semantics
            _hp_s = None
            _hp_tau = args.amnhp if args.amnhp > 0.0 else args.bmnhp
            if _hp_tau > 0.0:
                kap2_hp = kap2_hp + (BLOCK_MS / (_hp_tau * 1000.0)) * (kappa - kap2_hp)
                if args.mnspan > 0:
                    _hp_s = np.convolve(kap2_hp, np.ones(2 * args.mnspan + 1) / (2 * args.mnspan + 1), mode="same")
                else:
                    _hp_s = kap2_hp
            if args.bmnhp > 0.0 and dir_state < 0:
                # W58: B-chain backward-state high-pass — cuts the DC channel of the bout self-sustaining loop
                # (static S-bend -> mnprop2 DC -> B-chain tonic inversion -> AVA plateau feed -> bout never exits; phasic entrainment preserved)
                d2 = BMN_sgn * (kap2_s - _hp_s)[_eff2]
            else:
                d2 = BMN_sgn * kap2_s[_eff2]
            if args.mnprop2thr > 0.0:
                d2 = np.maximum(d2 - args.mnprop2thr, 0.0)
            if not ris_off:
                # v70 R18n: when the RIS rule fires, this channel is shut together with the Boyle loop — otherwise, in the unified stack,
                # the wave is maintained by mnprop2 and RIS gate clearing closes only the boylescale half, turning a stop into acceleration (r18m)
                if args.mnprop2 != 0.0:
                    _g2 = args.mnprop2
                    if args.bpropbwd != 0.0 and dir_state < 0:
                        # v85: B-chain gain attenuation during backward (Kato et al. 2015 Cell 163:656 anterior/posterior sub-loop mutual-inhibition analog)
                        _g2 = args.mnprop2 * (1.0 - args.bpropbwd)
                    Iex = Iex.at[BMN_idx].add(jnp.asarray(_g2 * d2, jnp.float64))
                if args.amnprop != 0.0 and (args.amngate == 0 or dir_state < 0):
                    # v85: A-class MN proprioceptive injection (Gao 2018: A-class backward oscillators accept proprioceptive entrainment);
                    # mirrors the B-chain backward form = anterior sensing (AMN_seg_eff = soma - amnoffset), reusing the same kap2_s filter/span products;
                    # with amngate=1 it is active only in the backward state — wave drive during backward is taken over by the A chain
                    if args.amnhp > 0.0:
                        # v91: high-pass phasic feedback — subtracts the slow-EMA reference (= static-bend DC), passing only curvature change;
                        # breaks the static locking of negative feedback under strong tonic drive (backward sawtooth-lock fix, 2026-10-22)
                        # since W58: kap2_hp/_hp_s are updated before the d2 computation (shared with B-loop bmnhp) and used directly here
                        _sigA = kap2_s - _hp_s
                    else:
                        _sigA = kap2_s
                    Iex = Iex.at[AMN_idx].add(
                        jnp.asarray(args.amnprop * AMN_sgn * AMN_emask * _sigA[AMN_seg_eff], jnp.float64))
                if args.dvaprop != 0.0 and dir_state < 0:
                    # v92: DVA/TRP-4 amplitude-limiting brake (Li 2006 Nature PMC2865900: trp-4(lf) = over-bending ->
                    # DVA stretch sensing negatively suppresses ventral-cord MNs/body-wall muscle to prevent over-bending). Slow-EMA curvature = static-bend DC readout (travelling wave averages ~0 = AC passed).
                    # active only in the backward state (dir_state<0) -> crawl regression bit-untouched
                    kap2_dva = kap2_dva + (BLOCK_MS / (args.dvatau * 1000.0)) * (kappa - kap2_dva)
                    if args.dvatarget == "mn":
                        # |DC| hyperpolarising injection into A-class MN soma segments (note: this weakens travelling-wave unlocking)
                        Iex = Iex.at[AMN_idx].add(
                            jnp.asarray(-args.dvaprop * np.abs(kap2_dva[AMN_seg]), jnp.float64))
        if args.mntonic != 0.0:
            # v67 R1 diagnostic: B-class MN constant depolarisation (mimics the real worm's tonic drive, sets the loop operating point)
            Iex = Iex.at[BMN_idx].add(jnp.asarray(args.mntonic, jnp.float64))
        if args.vdsp != 0.0 and len(W_ad_t) > 0:
            # W62: A->D sp release channel — Iex[D] += G * W_ad @ (softplus(dV_prev)-ln2)
            # dV taken from the previous block (50ms delay, as for ggapmod); sp form as in muscle96/mndrive (NMJ release follows
            # presynaptic depolarisation = physiological first principle, bypassing the vrel mismatch on A->D group edges; anatomical distribution = chem edge weights)
            _spd = np.logaddexp(0.0, np.asarray(dV_prev)[W_ad_s]) - 0.6931471805599453
            _id = np.zeros(N); np.add.at(_id, W_ad_t, W_ad_w * _spd)
            _tgt = np.unique(W_ad_t)
            Iex = Iex.at[jnp.asarray(_tgt)].add(jnp.asarray(args.vdsp * _id[_tgt], jnp.float64))
        if args.avatonc != 0.0:
            # v92: AVA constant tonic injection (Kato 2015: steady backward = AVA tonic dominance; no trigger, no plateau)
            Iex = Iex.at[np.asarray(AVA)].add(jnp.asarray(args.avatonc, jnp.float64))
        if args.avamaster != 0.0:
            # v93: AVA master-neuron slow tonic excitation of AVB (Meng 2024 PMC11006214 two-timescale model:
            # fast phasic inhibition via RIM/LGC-55 already carried by the M6 net; here we implement the slow excitatory arm = endogenous clock for backward exit,
            # aiming to replace the AVB ablation probe so forward/backward switching is fully endogenous). dV taken from the previous block (50ms delay, as for pep2)
            am_ema += (max(0.0, float(dV_prev[AVA].mean()) - args.avamasterfloor) - am_ema) \
                      * (BLOCK_MS / 1000.0 / args.avamastertau)
            Iex = Iex.at[np.asarray(AVB)].add(jnp.asarray(args.avamaster * am_ema, jnp.float64))
        if (args.avahold != 0.0 or args.pirrev != 0.0) and hold_I != 0.0:
            # v70 R16: AVA plateau self-sustaining injection (Meng 2024: AVA tonic depolarisation -> bout self-sustains after stimulus withdrawal)
            # v72: pirrev charging also uses this injection
            Iex = Iex.at[np.asarray(AVA)].add(jnp.asarray(hold_I, jnp.float64))
        if args.rivabl != 0.0:
            # v69 R10b: RIV functional ablation (Gray et al. 2005 PNAS 102:3184: RIV ablation ->
            # loss of the ventral bias of post-reversal turning); constant hyperpolarisation clamps RIVL/R off the operating point
            Iex = Iex.at[np.asarray(RIV)].add(jnp.asarray(args.rivabl, jnp.float64))
        if abl_idx and args.cellablmode == "clamp":
            # v84: generic functional ablation (Gao 2018 eLife 7:e29915 reproduction probe)
            # with cellablmode=edge the edges are already deleted; no injection (true laser-ablation semantics)
            Iex = Iex.at[np.asarray(abl_idx)].add(jnp.asarray(args.cellablp, jnp.float64))
        if args.smdprop != 0.0:
            # v68 R6: SMD proprioceptive injection: mean curvature of the 6 head segments, negative gain = stretch reflex
            # (Yeon 2018: TRP-1/2-mediated; dorsal bend kappa>0 -> suppress SMDD / ventral bend -> suppress SMDV)
            # v82: receptive field adjustable via --smdpropseg (default 0:6 legacy; neck = 3:9, Yeon 2018 original text "neck")
            _sa, _sb = (int(x) for x in args.smdpropseg.split(":"))
            _smd_g = args.smdprop
            if args.smdpropbw != 0 and dir_state < 0:
                # W57: automatically close the SMD neck reflex in the backward state (w57d/e judgement: an always-on reflex in the backward state
                # leaves the head without travelling-wave constraint -> main cause of spinning; biological counterpart = directional-state-dependent reflex-gain modulation,
                # analogous to descending modulation of stretch-reflex gain at gait initiation, Lamont 2018 review)
                _smd_g = 0.0
            Iex = Iex.at[SMD_idx].add(
                jnp.asarray(_smd_g * SMD_sgn * kappa[_sa:_sb].mean(), jnp.float64))
        if args.turnsmd != 0.0 and smd_turn_on:
            # v68 R9: turning pulse executed via SMD (Gray 2005: SMD encodes omega-turn angle amplitude);
            # SMDV excitation (ventral contraction) + SMDD inhibition = deep ventral bend; acts on head muscles via the smddrive anatomical pathway
            Iex = Iex.at[SMDV_idx].add(jnp.asarray(args.turnsmd * pulse_dir, jnp.float64))
            Iex = Iex.at[SMDD_idx].add(jnp.asarray(-args.turnsmd * pulse_dir, jnp.float64))
        if args.fc1 != 0.0:
            # v74 R18: endogenous-budget turning drive — amplitude = fc1 x m_fc1 x RIA->SMD in-situ synaptic budget
            # (budget rebuilt online from previous-block voltages: state-dependent, no drive while RIA is silent; 1 block = 50ms delay, as for pep2)
            if m_fc1 > 1.0e-3:
                _Vp = dV_prev + V_rest
                _sV = np.clip((_Vp[FC1_pre] - net.vrel[FC1_pre]) / 25.0, 0.0, 1.5)
                _bud = float(args.gsyn * np.sum(FC1_w * _sV * np.abs(_Vp[FC1_post])))
                _amp = args.fc1 * m_fc1 * _bud
                if args.offg != 0.0:
                    # v79 R27b: near-source turning-execution gain (first-round diagnosis: lengthening duration does not fix the root cause — execution drive too weak,
                    # ~24 deg/event; _amp also multiplied by ars -> near-source turns truly reorient; far-field ars=1 bit-identical)
                    _amp = _amp * (1.0 + args.offg * max(arr_clp[0] - args.offth, 0.0) / max(1.0 - args.offth, 1e-12))
                Iex = Iex.at[SMDV_idx].add(jnp.asarray(_amp * pulse_dir, jnp.float64))
                Iex = Iex.at[SMDD_idx].add(jnp.asarray(-_amp * pulse_dir, jnp.float64))
            m_fc1 *= float(np.exp(-(BLOCK_MS / 1000.0) / args.fc1tau2))
        if args.weath != 0.0 and ase is not None:
            # v71 W1: weathervane bidirectional steering (klinotaxis, Iino & Yoshida 2009 J Neurosci 29:5370)
            # single-point ASE carries only a temporal signal, but undulation sweeps the head across the gradient dorsoventrally (= laterally):
            # head swinging to one side while dC/dt rises => that side is up-gradient => keep bending toward it.
            # w = lowpass(s*g, tau_w): w>0 = dorsal up-gradient -> SMDD excited/SMDV suppressed -> dorsal bend (kappa>0 = dorsal bend);
            # w<0 the reverse -> ventral bend. Direction set by the sign of s*g => bidirectional. g = ase.dcdt (signed filtered value).
            if args.weathmode == "kap":
                # v77 R22 carrier-judgement winner: s = tanh(kappa0/3) head-joint bend position — snout orientation is the concentration-sampling phase,
                # correlated with dC/dt it yields the up/down-gradient side (offline judgement: agreement 0.92, w direction correct, |w| p95 ~0.0034;
                # joint-angle proprioception directly available, Yeon 2018 Nat Commun 9 somatosensory gating)
                _s = float(np.tanh(kappa[0] / 3.0))
            else:
                # dkap (legacy): s = tanh(dkappa0/dt/8) = head-swing-direction proxy — a sign audit proved it phase-unlocked from the head swing,
                # w agreement 0.06 = sign-flipped garbage; kept only for bit-identical legacy behaviour
                _dk0 = (kappa[0] - kap0_prev) / (BLOCK_MS / 1000.0)
                kap0_prev = float(kappa[0])
                _s = float(np.tanh(_dk0 / 8.0))
            w_update = (BLOCK_MS / 1000.0 / args.weathtau) * (_s * ase.dcdt * asad_gain[0] - w_weath)
            if m_fc1 < 0.5 and t_ms >= w_frz_until_ms:
                # v77 R22b efference-copy-style gating: during the main turning-manoeuvre peak (m_fc1>0.5, ~2s) kappa0/dcdt are contaminated
                # by the manoeuvre itself (ventral-bend execution while still receding -> s*g gives a false "dorsal up-gradient" signal; observed to cause dir=-1 misturns, |err| amplified);
                # the correlation integrator is frozen during the main peak, so w accumulates mainly self-sampling from normal crawl phases
                # W68: likewise frozen for wrst seconds after a bout end (curvature-remnant dissipation period, same idea)
                w_weath += w_update
            Iex = Iex.at[SMDD_idx].add(jnp.asarray(args.weath * w_weath, jnp.float64))
            Iex = Iex.at[SMDV_idx].add(jnp.asarray(-args.weath * w_weath, jnp.float64))
        if args.headosc != 0.0:
            # v67 R1b: head B-class sine seed (D/V anti-phase -> alternating head-neck dorsoventral bending -> tests reflex-chain propagation)
            _hs = args.headosc * np.sin(2.0 * np.pi * args.headfreq * (t_ms / 1000.0))
            Iex = Iex.at[BMN_idx[BMN_head]].add(
                jnp.asarray(_hs * BMN_sgn[BMN_head], jnp.float64))
        if args.pep2 != 0.0:
            # v73 R17: backward-loop peptidergic positive feedback (FLP-14/FRPR-19 loop, Marques 2021 PLoS Genet;
            # MGL-2/mGluR5 presynaptic gain, Katz 2019 Nat Commun; RIM->AVA backward-state reinforcement, Sordillo 2021 eLife).
            # group pool (AVA+AVD+AVE) depolarisation -> slow m2 accumulation -> current injected back into the pool: a sensory seed (ASH ~1.2mV) crossing
            # pep2th can ignite it; when pool voltage falls, m2 decays -> self-limiting. Signal taken from the previous block (1 block = 50ms delay, stated as-is).
            _pool = float(dV_prev[AVA].mean() + dV_prev[AVD].mean() + dV_prev[AVE].mean())
            _M2inf = min(max((_pool - args.pep2th) / args.pep2ref, 0.0), 1.0)
            m2_pep += (_M2inf - m2_pep) * (BLOCK_MS / 1000.0 / args.pep2tau)
            n2_pep += (m2_pep - n2_pep) * (BLOCK_MS / 1000.0 / args.pep2tau2)
            Iex = Iex.at[np.asarray(AVA + AVD + AVE)].add(
                jnp.asarray(args.pep2 * (m2_pep - args.pep2inh * n2_pep), jnp.float64))
        if args.ckr1 != 0.0:
            # stage-2 CKR-1 peptidergic gain layer (Chen 2022 Cell Rep 38:110330 PMID 35139370;
            # Chen 2024 PMC10952047 + Mendeley 10.17632/jm9yt2hdnk.1 raw-data anchors):
            # the same Galpha-q receptor CKR-1 integrates two motor states on SMD:
            #  NLP-12 arm: DVA (stretch-activated mechanosensation) -> NLP-12 -> CKR-1 on SMD -> body-curvature gain
            #    mutants: body curvature WT 4.380 -> nlp-12 3.435 (-21.6%) / ckr-1 3.596 (-17.9%);
            #    tau12=12.9s (Fig5E measured Ca2+ 10-90% rise); drive = DVA stretch proxy mean|kappa|/KAP_REF
            #    (as in the pep layer, KAP_REF=3.0 already anchored); effect = G-alpha-q depolarising injection into the four SMD cells
            #  NLP-18 arm: ASI (multimodal sensation) -> NLP-18 -> CKR-1 on SMD + AIB -> omega-turn gain
            #    mutants: omega curvature WT 7.072 -> nlp-18 6.250 (-11.6%) / ckr-1 6.203 (-12.3%);
            #    tau18=22.1s; drive = ASH+ASI sensory-pool depolarisation (softplus(dV)-ln2)/25mV, as for ggapmod,
            #    signal taken from the previous block (50ms delay, as for pep2); dV~0 during crawl -> M18~0, rising during escape stimulation
            #  arm weight ratio w18/w12 = 1.42 = Ca2+ peak ratio 0.502/0.353 (Fig5D measured activation capacity);
            #  EC50 NLP-18 13.6 nM vs NLP-12 26.5 nM (Fig5B dose-response, Hill fit)
            #  Fig3: NMJ minis/EPSC unaffected by CKR-1 -> the gain acts only on SMD/AIB excitability, not on synaptic transmission
            #  G0 amplitude = bounded estimate: validated against mutant phenotype scores by in-silico mutant comparison (on/off curvature difference),
            #  sensitive interval recorded in the constant-anchoring ledger (stage-2 CKR-1 section)
            _M12inf = min(float(np.abs(kappa).mean()) / KAP_REF, 1.0)
            ckr_m12 += (_M12inf - ckr_m12) * (pep_dt / args.ckr1tau12)
            _dv18 = 0.5 * (float(dV_prev[ASH_i].mean()) + float(dV_prev[ASI_i].mean()))
            _M18inf = min(max((float(np.logaddexp(0.0, _dv18)) - 0.6931471805599453) / 25.0, 0.0), 1.0)
            ckr_m18 += (_M18inf - ckr_m18) * (pep_dt / args.ckr1tau18)
            _Ickr = float(args.ckr1 * (ckr_m12 + 1.42 * ckr_m18))
            Iex = Iex.at[np.asarray(SMDD + SMDV)].add(jnp.asarray(_Ickr, jnp.float64))
            Iex = Iex.at[np.asarray(AIB_i)].add(
                jnp.asarray(float(args.ckr1 * 1.42 * ckr_m18), jnp.float64))
        if args.fc2 != 0.0:
            # v75 R20 (FC2): AVA extrasynaptic tonic ACh tone -> AVB (Meng et al. Sci Adv 2024 master
            # neuron: AVA depolarised rest ~-23mV -> constitutive ACh release tonically excites the forward circuit; maintained by TWK-40).
            # voltage-gated ramp: AVA tonic comes online in the depolarised zone (above fc2v0=-45) and shuts off automatically when clamped below -50,
            # matching the progressive forward-speed decline of the HisCl dose experiment (Fig6A of that paper) — a testable prediction.
            # note: the vrel table's AVA=-83mV saturates chemical release and cannot carry tonic state dependence, hence a voltage-gated implementation (documented).
            _Va = float((dV_prev[AVA] + V_rest[AVA]).mean())
            _g2 = min(max((_Va - args.fc2v0) / args.fc2vs, 0.0), args.fc2cap)
            Iex = Iex.at[np.asarray(AVB)].add(jnp.asarray(args.fc2 * _g2, jnp.float64))
            if args.fc2mn != 0.0:
                # v75 R20c (FC2 second arm): the same AVA voltage-gated tonic -> B-class MN pool (extrasynaptic ACh tonic
                # reinforcement of the ventral motor sub-circuit; AVA depolarisation during backward keeps the gate open -> the mnback mirrored wave gets the same operating point)
                Iex = Iex.at[BMN_idx].add(jnp.asarray(args.fc2mn * _g2, jnp.float64))
        if args.avaclamp != 0.0:
            # v75 R20b: AVA constant clamp (HisCl mimicry; negative current = hyperpolarisation)
            Iex = Iex.at[np.asarray(AVA)].add(jnp.asarray(args.avaclamp, jnp.float64))
        if args.risdrive != 0.0:
            # W59b: RIS bout-termination integrator firing (Kato 2015 Cell 163:656: RIS excites specifically at reversal end/
            # forward start; in this net RIS was silent throughout = the terminator never activated, the observed root cause of non-exiting bouts).
            # dual of avaplat2: charges in the backward state (ristau), discharges in forward (ristau2) -> true RIS depolarisation ->
            # natural inhibition of AVA/A-class via in-network GABA synapses -> bout termination; tau anchored on Zhao2003 bout median 2-3s.
            # honest note: the upstream drive of RIS bout activity in the real worm is unknown; this is a functionally equivalent implementation (bounded estimate).
            # W69 riskill: forced discharge on full-charge timeout (RIS transient peak self-limits, Kato 2015; fixes the AVB-ablation freeze)
            _ris_charging = (dir_state < 0) and (
                args.riskill == 0.0 or ris_hold_ms < args.riskill * 1000.0)
            if _ris_charging:
                ris_m += (BLOCK_MS / 1000.0 / args.ristau) * (1.0 - ris_m)
                if args.riskill != 0.0 and ris_m > 0.99:
                    ris_hold_ms += BLOCK_MS
            else:
                ris_m += (BLOCK_MS / 1000.0 / args.ristau2) * (0.0 - ris_m)
                if dir_state >= 0:
                    ris_hold_ms = 0.0   # reset the fuse timer in the forward state
            Iex = Iex.at[np.asarray(RIS)].add(jnp.asarray(args.risdrive * ris_m, jnp.float64))
        if args.avaplat != 0.0:
            # v76 R21 (FC3): AVA regenerative plateau current — fast activation m (sigmoid voltage gate, threshold -16mV) +
            # slow inactivation n (tracks m, tau2=4s) + Ca driving-force self-limit; positive feedback sustains the bout, self-terminates when n catches m.
            # per cell (AVAL/AVAR independent); signal taken from the previous block (50ms delay, as declared for pep2).
            _Va3 = dV_prev[AVA] + V_rest[AVA]
            _minf = 1.0 / (1.0 + np.exp(-(_Va3 - args.avaplatth) / args.avaplatk))
            m_plat = m_plat + (BLOCK_MS / 1000.0 / args.avaplattau) * (_minf - m_plat)
            n_plat = n_plat + (BLOCK_MS / 1000.0 / args.avaplattau2) * (m_plat - n_plat)
            _Iplat = -args.avaplat * m_plat * np.clip(1.0 - n_plat, 0.0, None) * (_Va3 - args.avaplate)
            Iex = Iex.at[np.asarray(AVA)].add(jnp.asarray(_Iplat, jnp.float64))
        if args.avaplat2 != 0.0:
            # v91 (ASH completion): gate-edge-ignited AVA plateau current — same structure as avaplat (VGCC-like regenerative
            # Ca current + fast activation m + slow inactivation n self-termination), but the ignition signal is the backward gate state (dir_state<0,
            # previous block, 50ms delay as for pep2/ggapmod): the engine's compressed voltage scale (~3x, AVA excursion 2-5mV)
            # makes a soma-voltage-threshold path unreachable; the gate flip is the network-level "AVA active" signature (endogenisation of
            # the v77 R23 pirrevp event ignition). During crawl the gate is always forward -> m=0 -> zero current -> baseline bit-identical;
            # in a bout m charges fast (0.2s), n follows slowly (4s) -> plateau self-sustains -> self-terminates when n catches up (Hilliard
            # 2005 bout 1-3s order); structure/tau/E anchored on Meng 2024 + Mellem 2008; conductance G is
            # a bounded estimate (no direct measurement; calibration target = a coherent self-terminating bout; recorded in the ledger)
            _on2 = 1.0 if dir_state < 0 else 0.0
            mp2 += (BLOCK_MS / 1000.0 / args.avaplat2tau) * (_on2 - mp2)
            np2 += (BLOCK_MS / 1000.0 / args.avaplat2tau2) * (mp2 - np2)
            _Va4 = dV_prev[AVA] + V_rest[AVA]
            _Iplat2 = -args.avaplat2 * mp2 * np.clip(1.0 - np2, 0.0, None) * (_Va4 - args.avaplate)
            Iex = Iex.at[np.asarray(AVA)].add(jnp.asarray(_Iplat2, jnp.float64))
        if args.sloa != 0.0 and AMN_idx.size > 0:
            # v92: SLO-1/SLO-2-type activity-dependent potassium current (Yu/Wang/Gao 2022 Front Mol Neurosci 15:845733
            # measurements: slo-1(lf) raises tonic rPSC frequency+charge and speeds backward, slo-2(lf) tonic ~4x / long-lasting ~10x,
            # = activity-dependent negative-feedback brake on A-class oscillators; UNC-2 Ca influx -> KCa efflux terminates the burst; conserved lamprey homolog);
            # implementation: ca = soma activity trace, influx = max(dV-sloath, 0)/25 (UNC-2 high-voltage activation threshold: crawl dV<=+11 always zero,
            # bout peak +56 opens it = reduced model of steep HVA activation; engine voltage scale /25 as for ggapmod, no new knob),
            # cleared with tau=sloatau; I_K=-G*ca*(V-E_K) outward hyperpolarising;
            # sustained static drive -> ca accumulates -> DC is adaptively suppressed; the 0.6Hz rhythmic component outruns tau -> passes
            # = mechanised soma-level counterpart of the amnhp signal-level high-pass; signal taken from the previous block (50ms delay, as for pep2);
            # during crawl A-class is silent, dV~0 -> ca->0 -> zero current -> zero impact on forward (verified by the regression gate)
            _dVA = dV_prev[AMN_idx]
            _infl = np.maximum(_dVA - args.sloath, 0.0) / 25.0
            slo_ca += (BLOCK_MS / 1000.0 / args.sloatau) * (_infl - slo_ca)
            _VaA = _dVA + V_rest[AMN_idx]
            _Islo = -args.sloa * np.clip(slo_ca, 0.0, None) * (_VaA - args.sloe)
            Iex = Iex.at[AMN_idx].add(jnp.asarray(_Islo, jnp.float64))
            if args.amnosc != 0.0:
                # v92: UNC-2 regenerative arm (depolarising arm of the same channel, Yu 2022; Gao 2018 A-class endogenous oscillators):
                # regenerative influx ignition -> voltage upswing -> ca accumulation -> SLO brake extinguishes -> reset = relaxation oscillation;
                # oscillatory drive replaces the tonic plateau-fed static Turing mode; _infl is always zero during crawl -> regression untouched
                Iex = Iex.at[AMN_idx].add(jnp.asarray(args.amnosc * AMN_gfac * _infl, jnp.float64))
        if run_block_lv is not None:
            # stage-2 ggapmod: M <- anterior sensory/dopaminergic layer activity (softplus(dV)-ln2, dV from the previous block, 50ms delay
            # as for pep2; V_SCALE=25 is the engine's own voltage scale, no new knob; at rest dV~0 (V_rest = warm-up end state),
            # softplus convexity gives M~0.03 micro-jitter during crawl, conductance impact <2%, ignored)
            if args.ggapmod != 0.0:
                _spda = float((np.logaddexp(0.0, dV_prev[DAT]) - 0.6931471805599453).mean())
                gm_inf = min(max(_spda / 25.0, 0.0), 1.0)
                gm_m += (gm_inf - gm_m) * (BLOCK_MS / 1000.0 / args.ggapmodtau)
                _gmod = 1.0 - args.ggapmodk * (1.0 - gm_m)
            else:
                _gmod = 1.0
            if AR is not None:
                # W49: rectified regime — per block, from previous-block absolute voltage (dV_prev+V_rest, 50ms delay as for ggapmod),
                # assemble the Boltzmann-gated Lg; stackable with ggapmod (rectified edges replaced first, then _gmod scales globally)
                _Vp = dV_prev + V_rest
                _Lgt = jnp.asarray(_gmod * _Lg_w49(_Vp), jnp.float64)
            else:
                _Lgt = jnp.asarray(_gmod * Lg_full_np, jnp.float64)
            if Lg_avd_np is not None:
                # W54b: A<->VD/DD gap switched by directional state (attenuated in backward, restored in forward = add zero)
                _fvd = args.avdx if dir_state < 0 else 1.0
                _Lgt = _Lgt + jnp.asarray(_gmod * (_fvd - 1.0) * Lg_avd_np, jnp.float64)
            if run_block_lvm is not None:
                # W54: A->B crosstalk switched by directional state (attenuated in backward, restored in forward)
                _Mext = jnp.asarray(Mex_abx_bwd_np if dir_state < 0 else Mex_abx_fwd_np,
                                    jnp.float64)
                state = run_block_lvm(state, Iex, _Lgt, _Mext)
            else:
                state = run_block_lv(state, Iex, _Lgt)
        else:
            state = run_block(state, Iex)
        V = np.asarray(assemble_V(state[0], state[1]))
        dV = V - V_rest
        dV_prev = dV
        dB = dV[AVB].mean(); dA = dV[AVA].mean()
        dSMD = dV[SMDD].mean() - dV[SMDV].mean(); dRIS = dV[RIS]; dRIV = dV[RIV].mean()
        g_f = 1.0 if (dB > gate_thresh and dB > dA) else 0.0
        g_b = 1.0 if (dA > gate_thresh and dA >= dB) else 0.0
        if args.gate == "rel":
            # v60 relative gating: direction = AVA/AVB relative comparison + hysteresis; gate closed (rest) when both sides are below floor
            if dir_state > 0 and dA > dB + gate_hyst and dA > gate_floor:
                dir_state = -1
            elif dir_state < 0 and dB > dA + gate_hyst and dB > gate_floor:
                dir_state = +1
            act = max(dB, dA)
            g_f = 1.0 if (dir_state > 0 and act > gate_floor) else 0.0
            g_b = 1.0 if (dir_state < 0 and act > gate_floor) else 0.0
        elif args.gate == "relg":
            # v61 group-pool relative gating: backward side = AVA+AVD+AVE pool (Chalfie 1985 command-tuple anatomy),
            # forward side = AVB; same hysteresis/floor rules
            dA = dV[AVA].mean() + dV[AVD].mean() + dV[AVE].mean()
            if args.risbrk != 0.0:
                # W59: RIS brake counted into the backward pool (Kato 2015 Cell 163:656: RIS excites specifically at reversal end,
                # GABAergic inhibition of AVA/A-class = bout terminator; a single-cell in-net RIS->AVA synapse
                # cannot overpower the in-phase resonance of a 3-cell pool — the gating layer honestly counts RIS's functional inhibition of the backward command;
                # 0=off=legacy behaviour bit-identical)
                # W75 (2026-10-24, seven-behaviour regression fix): the AVA phase (10-22s) of the switch protocol is an experimenter
                # sustained clamp (Gao 2018 AVA::ChR2 continuous illumination = continuous backward) and must override endogenous RIS termination
                # (risbrk's physiological semantics = spontaneous bout terminator, Kato 2015 spontaneous reversal->forward transient);
                # all other protocols/windows bit-identical (region/avapulse post-stimulus bout terminating in 2-3s is correct, Wicks 1995)
                _risbrk_on = not (args.proto == "switch" and 10.0 <= (t_ms / 1000.0) < 22.0)
                if _risbrk_on:
                    dA = dA - args.risbrk * max(float(dRIS) - args.risbrkth, 0.0)
            if dir_state > 0 and dA > dB + gate_hyst and dA > gate_floor \
                    and (t_ms - last_bout_end_ms) > args.bwrefr * 1000.0:
                # v91 bwrefr: forward->backward flips blocked inside the post-bout refractory window (Wicks & Rankin 1995)
                dir_state = -1
            elif dir_state < 0 and dB > dA + gate_hyst and dB > gate_floor:
                dir_state = +1
            act = max(dB, dA)
            g_f = 1.0 if (dir_state > 0 and act > gate_floor) else 0.0
            g_b = 1.0 if (dir_state < 0 and act > gate_floor) else 0.0
        dSMDmax = max(dV[SMDD].mean(), dV[SMDV].mean())
        ris_off = bool(dRIS > gate_thresh and dRIS > 1.5 * max(dB, dA, dSMDmax))
        if ris_off:
            g_f = g_b = 0.0
        if args.avahold != 0.0 or args.pirrev != 0.0:
            # v70 R16e: trigger-edge charging + unconditional decay + 2s re-arming — after a bout ends, if the gate is still
            # jittering on residual current it flash-flips forward->backward within one block; without re-arming it re-triggers endlessly;
            # require >=2s of continuous forward before the next charge (Wicks & Rankin 1995: post-bout response refractory period)
            # v72: pirrev charges at the event pulse; here we only handle decay and re-arm bookkeeping
            if args.avahold != 0.0 and dir_state_prev > 0 and dir_state < 0 and (t_ms - fwd_since) > 2000.0:
                hold_I = args.avahold
            else:
                hold_I *= np.exp(-BLOCK_MS / args.avaholdt)
            if dir_state > 0 and dir_state_prev < 0:
                fwd_since = t_ms
        _bout_end = (dir_state_prev < 0 and dir_state > 0)   # B4: backward-bout end edge (escape-omega chaining trigger signal)
        if _bout_end:
            last_bout_end_ms = t_ms   # v91 bwrefr: bout-end timing (start of the refractory period)
            if args.wrst != 0.0:
                # W68: the bout has changed the heading, the old w is voided; accumulation frozen during the curvature-remnant dissipation period
                w_weath = 0.0
                w_frz_until_ms = t_ms + args.wrst * 1000.0
                # W68b: bout end = end of the turning manoeuvre, fc1 drive cleared without tail (Pierce-Shimomura 1999:
                # a pirouette is followed by a straight run; the m_fc1 tau2=3s long tail would pin post-bout crawling in a deep-bend spinning state,
                # observed in w68_wrst2_d180: post-bout |kappa|max 17-24 persisting for 6s)
                m_fc1 = 0.0
        dir_state_prev = dir_state
        if args.turnmode == "event":
            # v64 event-based pirouette: dSMD crossing threshold upward -> fixed-length turning pulse + refractory (discrete pirouette,
            # Pierce-Shimomura 1999); radial re-approach during runs breaks the -90 deg orbital fixed point of continuous gating
            t_s = t_ms / 1000.0
            if args.turnsrc == "aiy":
                steer = (dV[AIYR_i] - dV[AIYL_i])
                thr = args.steerthresh
            elif args.turnsrc == "dwn":
                # v83: worsening-rate trigger (Pierce-Shimomura 1999): steer = worsening level (-dcdt),
                # the sensory layer's own filtered dcdt (tau=0.5s), not an external signal; persistently above threshold while receding ->
                # re-triggers as soon as the refractory ends (the biological pirouette-bout rhythm); direction still given by the w_weath network correlation
                # v83b: one more dwntau slow low-pass — the modulation of instantaneous dcdt by head-swing lateral velocity (+/-0.03)
                # exceeds the net chemotactic signal (+/-0.025); without filtering, triggering phase-locks to the head swing (serial false triggers observed at d000)
                _raw = float(-ase.dcdt) if ase is not None else 0.0
                dwn_lp += (BLOCK_MS / 1000.0 / args.dwntau) * (_raw - dwn_lp)
                steer = dwn_lp
                thr = args.dwnthresh
            else:
                steer = dSMD
                thr = args.smdthresh
            if t_s < pulse_until:
                # v68 R9: with turnsmd!=0 turning is executed via the SMD->muscle pathway; the artificial koff bias is off
                head_bias = (args.pulsebias * pulse_dir) if (args.turnsmd == 0.0 and args.fc1 == 0.0) else 0.0
            elif (steer > thr or (args.steerbi != 0.0 and steer < -thr)
                    or (args.escomega != 0.0 and _bout_end)) and t_s >= refr_until \
                    and (ase is None or (ase.dcdt < args.pirsup
                                         and (args.pirc >= 999.0
                                              or chem_c[0] / max(ase.f.peak, 1e-12) < args.pirc))):
                # v64b fix: level + refractory re-triggering (no rising edge — under a sustained falling gradient dSMD never recrosses the threshold,
                # edge detection would fire one pulse then run straight away); biological repeated pirouettes are exactly rhythmic re-triggering
                # v78 R25: approach-suppression gate (when pirsup<999) — low-passed dC/dt >= threshold = approaching the source,
                # turning re-trigger forbidden (Pierce-Shimomura 1999: pirouettes fire only when conditions worsen);
                # slides straight toward the peak; past the peak dC/dt flips negative -> corrective turning released at once -> dwell near the source
                # v78 R25b: absolute-concentration dwell gate (when pirc<999) — instantaneous concentration >= pirc (peak-normalised) suppresses triggering,
                # dwelling by concentration level after the in-ring gradient vanishes (Sawin 2000/Hills 2004/Chalasani 2007)
                _pdur = args.pulsedur
                if args.pirmag > 0.0:
                    # v78 R25b: turn angle scaled by bearing error (Pierce-Shimomura 1999 turn-angle/error correlation,
                    # large error -> deep long omega bend; capped at 3x to prevent regime collapse)
                    _pdur = args.pulsedur * float(np.clip(abs(steer) / args.pirmag, 1.0, 3.0))
                if args.offg != 0.0:
                    # v79 R27: OFF-response/local-search turning boost — near the source (low-passed concentration > ofth)
                    # the refractory is divided by ars -> denser turning re-triggers (first-round lesson: lengthening pulses does not fix the root cause and overshoots,
                    # the execution-gain boost is applied at fc1 _amp by the same ars); with offg=0 this block does not execute,
                    # far-field/approach ars=1 bit-identical (Hills 2004/Chalasani 2007/Gray 2005/
                    # Lopez-Cruz 2019; dwelling geometry = low speed + high turning, Flavell 2013)
                    _ars = 1.0 + args.offg * max(arr_clp[0] - args.offth, 0.0) / max(1.0 - args.offth, 1e-12)
                pulse_until = t_s + _pdur
                refr_eff = args.refract
                if args.refrmod != 0.0 and (steer > 0.0 or (args.steerbi != 0.0 and steer < 0.0)):
                    # v73: turning-frequency modulation (Pierce-Shimomura 1999 core finding: pirouette frequency rises with
                    # the worsening rate dC/dt<0); steer amplitude ~ v*sin(error angle)*|grad C|; during a large-error U-turn
                    # the refractory shortens -> dense pulses (biological pirouette bouts, 40% are consecutive multi-turns); restored after lock-on
                    # R19: with bidirectional triggering, scale by |steer| (the negative branch = large swings during fast approach)
                    _sref = abs(steer) if args.steerbi != 0.0 else steer
                    refr_eff = args.refract * float(np.clip(args.refrmod / _sref, 0.3, 1.0))
                if args.offg != 0.0:
                    refr_eff = refr_eff / _ars          # v79 R27: near-source re-trigger densification
                refr_until = pulse_until + refr_eff
                if rng_pir is not None:
                    # v71 W2: random-direction pirouette (Pierce-Shimomura 1999: timing encodes the gradient, direction re-randomised)
                    pulse_dir = 1.0 if rng_pir.random() < 0.5 else -1.0
                if args.pirrev != 0.0:
                    # v72: pirouette-backward coupling (Pierce-Shimomura 1999: pirouette = backward bout + omega
                    # turn coupled; the reversal first cancels forward momentum, the head bend then completes at small radius; reusing the hold_I decay,
                    # bout length set by avaholdt, charging on the pulse trigger, unconditional decay -> bout self-terminates)
                    # v76: bearing-error gating (when pirrevang>0) — fires only when |heading - source bearing| >= pirrevang,
                    # small errors left to the weathervane/SMD arm (Pierce-Shimomura 1999 / Iino 2009 division of labour;
                    # in W75 d090, misfiring at 57.5% dAVA duty = hard evidence of degeneration, vs pirrev=0 approach +3.46mm);
                    # heading/source bearing taken from the previous block's body state (50ms delay as for ggapmod/amnprop); ase=None passes through
                    _fire = True
                    if ase is not None and args.pirrevang > 0.0:
                        _hd = np.asarray(x)[0] - np.asarray(x)[-1]
                        _sv = np.asarray(ase.f.src) - np.asarray(x)[0]
                        _be = float(np.degrees(np.arccos(np.clip(
                            float(_hd @ _sv) / (np.linalg.norm(_hd) * np.linalg.norm(_sv) + 1e-12),
                            -1.0, 1.0))))
                        _fire = _be >= args.pirrevang
                    if _fire:
                        hold_I = args.pirrev
                if args.pirrevp != 0.0:
                    # v77 R23: event-ignited AVA plateau gate (m_plat charging -> FC3 regenerative VGCC plateau drives the backward bout,
                    # killing forward momentum -> omega small-radius turn; m_plat itself decays in 0.2s and needs plateau regeneration to persist,
                    # the bout self-terminates via n_plat slow inactivation — the execution side fully endogenous; only the "event->ignition" coupling is an engineering rule)
                    m_plat = np.maximum(m_plat, args.pirrevp)
                if args.fc1 != 0.0:
                    # v74 R18: endogenous direction + gated charging (fc1dir=steer: the AIY(R-L) sign is the turning direction;
                    # random: re-drawn by rng_pir above; fixed: always +1 ventral;
                    # weath: v77 R22, pulse_dir=-sign(w_weath), the only observed carrier of spatial direction)
                    if args.fc1dir == "steer":
                        pulse_dir = 1.0 if steer >= 0.0 else -1.0
                    elif args.fc1dir == "fixed":
                        pulse_dir = 1.0
                    elif args.fc1dir == "weath":
                        pulse_dir = -1.0 if w_weath > 0.0 else 1.0
                    m_fc1 = 1.0
                    print(f"[fc1] t={t_s:.1f}s event charge dir={pulse_dir:+.0f} steer={steer:+.2f}mV",
                          flush=True)
                if rng_pir is not None:
                    print(f"[pir] t={t_s:.1f}s pulse_dir={pulse_dir:+.0f} steer={steer:+.1f}mV"
                          + (f" rev={args.pirrev:g}pA" if args.pirrev != 0.0 else "")
                          + (f" refr={refr_eff:.1f}s" if args.refrmod != 0.0 else ""), flush=True)
                if args.escomega != 0.0 and _bout_end:
                    print(f"[escomega] t={t_s:.1f}s bout-end triggered omega turning pulse "
                          f"dir={pulse_dir:+.0f} dur={_pdur:.1f}s", flush=True)
                head_bias = (args.pulsebias * pulse_dir) if (args.turnsmd == 0.0 and args.fc1 == 0.0) else 0.0
            else:
                head_bias = 0.0
            smd_turn_on = args.turnsmd != 0.0 and t_s < pulse_until
            dsmd_prev = dSMD
        elif args.turnmode == "pirouette":
            # v63 pirouette gating: a single-point chemosensor cannot measure the gradient's left/right (Pierce-Shimomura 1999),
            # the biological strategy = suppress turning and run straight up-gradient, trigger turning down-gradient; dSMD<0 = ASEL phase, >0 = ASER phase (observed signs)
            head_bias = k_turn * max(dSMD - args.smdthresh, 0.0)
        else:
            head_bias = k_turn * dSMD
        omega_bias = -k_omega * max(dRIV - riv_thresh, 0.0)
        if omega_bias != 0 and omega_t0 is None: omega_t0 = t_ms
        elif omega_bias == 0: omega_t0 = None
        koff = np.zeros(24); koff[:head_rows] += head_bias
        if omega_bias != 0 and omega_T > 0:
            c = -omega_sigma + (24 + 2 * omega_sigma) * ((t_ms - omega_t0) / 1000.0) / omega_T
            koff += omega_bias * np.exp(-((np.arange(24) - c) / omega_sigma) ** 2)
        koff_j = 0.5 * (koff[1:] + koff[:-1])
        # peptidergic slow-variable update: DVA reads the whole-body mean |curvature| -> NLP-12 level (second-scale time constant)
        M_inf = min(np.abs(kappa).mean() / KAP_REF, 1.0)
        pep_m += (M_inf - pep_m) * (pep_dt / args.peptau)
        g_pep = 1.0 + args.pep * pep_m    # CKR-2 presynaptic gain (Hu 2011: enhanced ACh release)
        # aldicarb: AChE blockade -> ACh accumulates over time (pharmacodynamic compressed timescale)
        g_ald = 1.0 + args.ald * (1.0 - np.exp(-(t_ms / 1000.0) / args.aldtau))
        g_tot = g_pep * g_ald
        # v66 loading-loop drive (once per block, dV constant within a block): softplus(dV)-ln2 -> anatomical innervation matrix -> muscle rows
        if args.mndrive != 0.0 or args.dmndrive != 0.0 or args.smddrive != 0.0:
            sp = np.logaddexp(0.0, dV) - 0.6931471805599453
        if args.mndrive != 0.0:
            # v78 R25c: absolute-concentration slowing (Sawin 2000 basal slowing; Hills 2004 ARS; dopamine
            # DOP-2/3 inhibit RIC/SIA -> less ACh/octopamine; DA->DVA promotes dwelling) — when the low-passed concentration crosses threshold,
            # the forward drive is multiplied by arrg; smddrive is not slowed (local search needs the steering force preserved)
            _mnd = args.mndrive
            if ase is not None and args.arrc < 999.0 and arr_clp[0] >= args.arrc:
                _mnd = args.mndrive * args.arrg
            mndrv_D = _mnd * (Wdrv_D @ sp)
            mndrv_V = _mnd * (Wdrv_V @ sp)
        else:
            mndrv_D = 0.0; mndrv_V = 0.0
        if args.smddrive != 0.0:
            # v68 R6: SMD->head-muscle drive (Yeon 2018: SMD's dual role = sensing + driving)
            smddrv_D = args.smddrive * (Wsmd_D @ sp)
            smddrv_V = args.smddrive * (Wsmd_V @ sp)
        else:
            smddrv_D = 0.0; smddrv_V = 0.0
        if args.dmndrive != 0.0:
            # v67 R2: D-class cross-inhibition, subtracted from ipsilateral activation (GABA relaxes contralateral contraction);
            # inhibition does not enter g_tot (aldicarb/pep are ACh-specific); with dmndrv=0 it subtracts 0.0, bit-identical
            dmndrv_D = args.dmndrive * (Wdm_D @ sp)
            dmndrv_V = args.dmndrive * (Wdm_V @ sp)
        else:
            dmndrv_D = 0.0; dmndrv_V = 0.0
        for _ in range(int(round(BLOCK_MS / 1000 / motor_dt))):
            kr = np.concatenate([[kappa[0]], 0.5 * (kappa[1:] + kappa[:-1]), [kappa[-1]]])
            if m96 is not None:
                # v81: own muscle-layer drive — drive signal through the anatomical matrix, 96 muscles integrating independently;
                # dV constant within a block (50ms block granularity); g_tot amplifies only ACh edges
                _gf, _gb = (g_f, g_b) if args.m96gate != 0.0 else (1.0, 1.0)
                if args.m96sig == "sp":
                    sig96 = m96.sp_signal(dV)
                else:
                    sig96 = m96.release(dV + V_rest, net.vrel)
                A_D, A_V = m96.step(sig96, motor_dt * 1000.0, g_tot,
                                    gate_f=_gf, gate_b=_gb, gain=args.muscle96,
                                    homeo_tau_s=args.m96homeo)
                A_D = np.clip(A_D, 0, 1.5); A_V = np.clip(A_V, 0, 1.5)
                if args.dvaprop != 0.0 and args.dvatarget == "muscle" and dir_state < 0:
                    # v92: DVA brake, muscle side (Li 2006: DVA suppresses body-wall muscle contraction) — joint-j curvature <-> muscle rows j, j+1
                    # (same mapping as preferred_curvature); the deeper |DC|, the looser the contraction; travelling-wave AC component unaffected
                    _bk = np.abs(kap2_dva)
                    _bkr = 0.5 * (np.concatenate([_bk, _bk[-1:]]) + np.concatenate([_bk[:1], _bk]))[:24]
                    _dg = 1.0 / (1.0 + args.dvaprop * _bkr)
                    A_D = A_D * _dg; A_V = A_V * _dg
            else:
                fD, fV = fwd_m.step(kr, motor_dt, g_f, 0.0, kappa_offset=koff)
                bD, bV = bwd_m.step(kr, motor_dt, g_b, 0.0, kappa_offset=koff)
                # with boylescale=1, mndrv=0 and dmndrv=0 this degenerates bit-identically to the milestone form (f+b)*g_tot
                drv_D = ((fD + bD) * args.boylescale + mndrv_D + smddrv_D) * g_tot - dmndrv_D
                drv_V = ((fV + bV) * args.boylescale + mndrv_V + smddrv_V) * g_tot - dmndrv_V
                if args.muscletau > 0.0:
                    # v68 R7: first-order muscle-activation integration (Zhao 2025: the negative-feedback delay comes from muscle contraction/relaxation dynamics;
                    # this supplies the phase resource the local MN-muscle negative-feedback loop needs to oscillate)
                    _am = (motor_dt * 1000.0) / args.muscletau
                    mA_D = mA_D + _am * (drv_D - mA_D); mA_V = mA_V + _am * (drv_V - mA_V)
                    A_D = np.clip(mA_D, 0, 1.5); A_V = np.clip(mA_V, 0, 1.5)
                else:
                    A_D = np.clip(drv_D, 0, 1.5); A_V = np.clip(drv_V, 0, 1.5)
            kappa0 = preferred_curvature(A_D, A_V, kappa_max) + koff_j
            steps = int(round(motor_dt / body.dt))
            x = body.run(x, np.broadcast_to(kappa0, (steps, len(kappa0))))
            kappa = np.asarray(body.curvature(x))
        t_ms += BLOCK_MS
        log.append(dict(t=t_ms, x=np.asarray(x).copy(), kappa=kappa.copy(),
                        A_D=A_D, A_V=A_V, gates=(g_f, g_b), pep_m=pep_m, pep2_m=m2_pep, pep2_n=n2_pep,
                        fc1_m=m_fc1, plat_m=float(m_plat.mean()),
                        ckr_m12=ckr_m12, ckr_m18=ckr_m18,
                        pulse=1.0 if (t_ms / 1000.0) < pulse_until else 0.0,
                        readout=dict(dAVB=float(dB), dAVA=float(dA), dSMD=float(dSMD), dRIV=float(dRIV))))
        Ytr.append(Yfull(state))
        if (b + 1) % 40 == 0:
            print(f"[run] t={t_ms/1000:.1f}s dAVB={dB:.2f} dAVA={dA:.2f} g=({g_f:.0f},{g_b:.0f}) "
                  f"({time.time()-t0:.0f}s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_smoke" if args.smoke else f"_s{args.stim:g}{args.tag}"
    np.savez_compressed(OUT / f"v62cl{tag}.npz",
                        log=np.array([r["kappa"] for r in log]),
                        A_D=np.array([r["A_D"] for r in log]),   # v81 P3: muscle-row activation trajectories (for calibration probes)
                        A_V=np.array([r["A_V"] for r in log]),
                        xs=np.array([r["x"] for r in log]),
                        gates=np.array([r["gates"] for r in log]),
                        pep_m=np.array([r["pep_m"] for r in log]),
                        pep2_m=np.array([r["pep2_m"] for r in log]),
                        pep2_n=np.array([r["pep2_n"] for r in log]),
                        fc1_m=np.array([r["fc1_m"] for r in log]),
                        ckr_m12=np.array([r["ckr_m12"] for r in log]),
                        ckr_m18=np.array([r["ckr_m18"] for r in log]),
                        plat_m=np.array([r["plat_m"] for r in log]),
                        pulse=np.array([r["pulse"] for r in log]),
                        readout=np.array([[r["readout"]["dAVB"], r["readout"]["dAVA"],
                                           r["readout"]["dSMD"], r["readout"]["dRIV"]] for r in log]),
                        Y=np.stack(Ytr), names=np.array(names), V_rest=V_rest,
                        stim=np.array([args.stim]), mnprop=np.array([args.mnprop]),
                        dt_block_ms=np.array([BLOCK_MS]))
    print(f"[done] saved {OUT / ('v62cl' + tag + '.npz')} ({time.time()-t0:.0f}s)", flush=True)

    if args.savestate:
        import pickle
        with open(args.savestate, "wb") as fh:
            pickle.dump(dict(state=[np.asarray(s) for s in state], V_rest=V_rest,
                             x=np.asarray(x), kappa=kappa, kap2_f=kap2_f, kap1_f=kap1_f, kap2_hp=kap2_hp,
                             mA_D=mA_D, mA_V=mA_V, t_ms=t_ms, omega_t0=omega_t0,
                             pep_m=pep_m, gm_m=gm_m, ckr_m12=ckr_m12, ckr_m18=ckr_m18, dir_state=dir_state, hold_I=hold_I, dir_state_prev=dir_state_prev, fwd_since=fwd_since, fwd_m=fwd_m, bwd_m=bwd_m,
                             pulse_until=pulse_until, refr_until=refr_until, dsmd_prev=dsmd_prev,
                             w_weath=w_weath, kap0_prev=kap0_prev, pulse_dir=pulse_dir, m_fc1=m_fc1,
                             w_frz_until_ms=w_frz_until_ms,
                             dwn_lp=dwn_lp,
                             m_plat=m_plat, n_plat=n_plat, mp2=mp2, np2=np2, last_bout_end_ms=last_bout_end_ms,
                             slo_ca=slo_ca,
                             mA96=(m96.mA.copy() if m96 is not None else None),
                             mE96=(m96.mE.copy() if m96 is not None else None),
                             ash_m=ash_m[0], ash_a=ash_a[0],
                             ash_mod=ash_mod[0],
                             ase_prev=(ase.prev if ase is not None else None),
                             ase_dcdt=(ase.dcdt if ase is not None else 0.0),
                             arr_clp=arr_clp[0], asad=asad[0]), fh)
        print(f"[ckpt] saved {args.savestate} @ t={t_ms/1000:.1f}s", flush=True)

    # ---- behavioural metrics ----
    xs = np.stack([r["x"] for r in log]); n0 = int(1.0 * 1000 / BLOCK_MS)
    if len(xs) > n0 + 8:
        xs2 = xs[n0:]; dt = BLOCK_MS / 1000
        c = xs2.mean(1); v = np.gradient(c, dt, axis=0)
        hd = xs2[:, 0] - xs2[:, -1]; hd /= np.linalg.norm(hd, axis=1, keepdims=True) + 1e-12
        v_ax = np.sum(v * hd, 1)
        kap = np.stack([r["kappa"] for r in log[n0:]])
        kmid = kap[:, kap.shape[1] // 2]
        f = np.fft.rfftfreq(len(kmid), dt); Pp = np.abs(np.fft.rfft(kmid - kmid.mean())) ** 2
        freq = float(f[Pp[1:].argmax() + 1]) if len(kmid) > 8 else 0.0
        print(f"[metrics] v_forward={v_ax.mean():+.4f} mm/s  frac_bwd={(v_ax < -0.02).mean():.2f}  "
              f"freq={freq:.3f} Hz  kappa_amp={kap.std():.3f}"
              + (f"  ggapmod_M_end={gm_m:.3f} g_eff_end={1.0 - args.ggapmodk * (1.0 - gm_m):.3f}xGGAP"
                 if run_block_lv is not None else "")
              + (f"  ckr1_M12={ckr_m12:.3f} M18={ckr_m18:.3f} kamp_head={kap[:, :6].std():.3f}"
                 if args.ckr1 != 0.0 else ""), flush=True)

    # ---- chemotaxis criteria (output format aligned with worm-whisperer phase3_chemotaxis.py) ----
    if args.proto == "chem" and len(log) > 8:
        cs = np.stack([np.asarray(r["x"]).mean(0) for r in log])      # centroid trajectory
        heads = np.stack([np.asarray(r["x"])[0] for r in log])        # head-node trajectory
        hd0 = xs[0][0] - xs[0][-1]; hd1 = xs[-1][0] - xs[-1][-1]
        cross = float(hd0[0] * hd1[1] - hd0[1] * hd1[0])
        dot = float(hd0 @ hd1) / (np.linalg.norm(hd0) * np.linalg.norm(hd1) + 1e-12)
        ang = float(np.degrees(np.arctan2(cross, np.clip(dot, -1.0, 1.0))))
        if ase is not None:
            src = ase.f.src
            d0 = float(np.linalg.norm(cs[0] - src)); d1 = float(np.linalg.norm(cs[-1] - src))
            rep = ASEInput(ase.f, gain=args.chemgain)                 # deterministic replay for the peak
            imax = 0.0
            for hy in heads:
                o_, f_, _ = rep.currents(hy, BLOCK_MS / 1000.0)
                imax = max(imax, o_, f_)
            print(f"[chem] source distance {d0:.2f} -> {d1:.2f} mm (approach {d0 - d1:+.2f} mm)  "
                  f"net turn {ang:+.1f} deg  max ASE {imax:.1f} pA"
                  + (f"  asad={asad[0]:.2f} gain={asad_gain[0]:.2f}" if args.asad != 0.0 else ""), flush=True)
        else:
            print(f"[chem] source-free control  net turn {ang:+.1f} deg  "
                  f"centroid displacement ({cs[-1][0]-cs[0][0]:+.2f},{cs[-1][1]-cs[0][1]:+.2f}) mm", flush=True)


if __name__ == "__main__":
    main()
