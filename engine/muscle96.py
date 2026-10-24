# muscle96.py -- original physiological-grouped muscle layer (96 body-wall muscles, each a first-order integrator; the Boyle bang-bang layer is dropped)
# Design document: v81 original muscle-layer proposal (2026-10-21, project archive) §3
# Constant table: data/muscle96_constants.csv (all literature-anchored)
#
# Architecture (spec §3; nothing invented in-house):
#   rel_n = clip((V_n - vrel_n)/V_SCALE, 0, S_MAX)     network-wide release table (same rule as jax_circuit.py)
#   I_i   = Σ_n sign_n · (wout[n,i]/n_inn[i]) · rel_n   ACh edges positive; GABA edges from DD/VD negative
#   dmA_i/dt = (clip(I_i·g, 0, Amax) - mA_i)/tau_m      each of the 96 muscles integrates in first order
#   A_D[r] = mean(mA[DR_r], mA[DL_r]), same for A_V[r]   (quadrant-averaged to 24 rows, same dimensions as the old path)
#   κ0[r]  = kappa_max · (A_D - A_V)                     (reuses the original preferred_curvature function)
#
# Design discipline:
#   - tau_m starts at 80 ms (Liu 2011 AP half-width ~20 ms + Gao&Zhen contraction timing + calibration),
#     and is never again used as a frequency knob; frequency comes out of the neural loop + proprioceptive loop.
#   - GABA v1 is subtractive (consistent with the existing dmndrv path); shunting inhibition is left for v2 evaluation (spec §3).
#   - Amax=1.5 matches the old path's clip ceiling; smooth linear saturation, no hard clip eating differences.
#   - aldicarb/peptidergic g_tot amplifies only ACh edges (ACh-specific, consistent with existing v62 comments).
#   - This module is instantiated only when --muscle96 != 0; default off, old path bit-identical.
import re
import numpy as np

V_SCALE, S_MAX = 25.0, 1.5   # same values as the network-wide release rule in jax_circuit.py


class Muscle96:
    """96-muscle graded activation layer. Input: neuron voltages (mV, in situ); output: 24 rows of dorsal/ventral activation.

    Direction gating (forced by v81 P3 measurements; same semantics as the old Boyle layer's g_f/g_b):
    the forward state passes only the forward group (DB/VB excitation + DD cross-inhibition), the backward state passes only the backward group (DA/VA + VD),
    head classes (RME/RMD/SMD/SMB) always pass. Biological counterpart: the command interneurons AVB/AVA mutually exclusively gate
    class-A/B motor neurons (Kawano 2011; Kato 2015); the old Boyle path already had g_f/g_b gating,
    here the same command gating is moved to the anatomical drive interface — not a new mechanism.
    P3 measurement basis: offline replay of good ablA trajectories shows class A still carrying a backward travelling wave during forward crawling
    (lag=-0.75 s, NCC=0.72, amp 42% of class B); without gating, wave direction destabilises at high gain."""

    def __init__(self, muscle_maps, tau_m_ms=80.0, amax=1.5, neuron_names=None,
                 ei_nmj=0.6, tau_ca_ms=0.0, gj_cc=0.0):
        muscles, wout, inhib, n_inn = muscle_maps
        self.names = list(muscles)                     # 96 of them, e.g. DR01/VL07
        sign = 1.0 - 2.0 * inhib                       # ACh +1; DD/VD (GABA) -1
        W = (wout.T / n_inn[:, None]) * sign[None, :]  # (96, N) normalised signed innervation matrix
        exc = sign > 0
        self.Wexc = np.where(exc[None, :], W, 0.0)     # ACh edges (target of g_tot)
        # P7 (2026-10-21 constant anchoring): NMJ measured E:I=1.6:1 → GABA edges multiplied by ei_nmj=0.6
        #   Anchor: Liu/Hollopeter/Jorgensen PNAS 2009 (PMC2705609) — field stimulation of the whole cholinergic system in the same preparation
        #   evoked 2.01±0.11 nA vs ChR2 whole-GABAergic evoked 1.23±0.12 nA → I/E = 0.61.
        #   The old value 1.0 (sign-symmetric) has no measurement basis. Drive and baseline terms share self.Winh, scaled consistently.
        #   Note: this ratio is an NMJ aggregate-current ratio; P1 falsified its global extrapolation to neuron-neuron edges.
        self.ei_nmj = float(ei_nmj)
        self.Winh = self.ei_nmj * np.where(exc[None, :], 0.0, W)  # GABA edges (negative, already multiplied by E:I)
        # P7 absolute-current reporting scale: full-amplitude cholinergic drive (mean drive when all ACh motor neurons release at S_MAX)
        #   ↔ measured 2.01 nA (PMC2705609) → g_abs nA per unit of dimensionless drive.
        #   The current→contraction mapping has no worm measurement (field blank, honestly declared): mA stays dimensionless on the kmax calibration path;
        #   g_abs only translates drive into nA for audit reporting; it does not enter the dynamical numerical path (the scalar cancels in the gain).
        _dmax = self.Wexc @ (S_MAX * np.ones(wout.shape[0]))
        self.g_abs = 2.01 / float(_dmax.mean())        # nA per unit dimensionless drive
        row = np.zeros(96, dtype=int)
        quad = np.zeros(96, dtype=int)                 # 0=dorsal (DR/DL), 1=ventral (VR/VL)
        side = np.zeros(96, dtype=int)                 # 0=L, 1=R (reserved)
        for i, m in enumerate(self.names):
            mm = re.fullmatch(r"([DV])([LR])(\d\d)", m)
            if mm is None:
                raise ValueError(f"muscle name not [DV][LR]dd: {m}")
            quad[i] = 0 if mm.group(1) == "D" else 1
            side[i] = 0 if mm.group(2) == "L" else 1
            row[i] = int(mm.group(3)) - 1
        self.row, self.quad, self.side = row, quad, side
        # P11 (2026-10-23): muscle-muscle gap-junction layer (Liu/Chen/Wang 2011 JBC 286:44285 PMC3243499:
        #   neighbouring-muscle AP/Ca synchrony is carried by GJs and does not depend on neural input; longitudinal same-quadrant neighbour coupling coefficient measured 0.25 (unc-9 knockout -> 0.1);
        #   same-row left-right inter-quadrant synchrony 55% vs longitudinal 75% -> transverse weight = longitudinal x0.73 (bounded estimate, declared);
        #   Liu 2006 JBC 281:7881: coupling both between same-quadrant muscle bellies and between muscle arms across quadrants; White 1986 same-quadrant belly GJs).
        # Mapping: two-compartment steady state cc=g/(1+g) -> g=cc/(1-cc)=1/3; per-muscle-row chain tridiagonal steady-state solution
        #   (1+c*deg)E - c*E_nbr = A_inf, inverse matrix precomputed; at each step E_ss=Minv@A_inf replaces the fast-layer target;
        #   exponential-Euler toward E_ss -> unconditionally stable and the steady state reproduces the measured coupling coefficient exactly. 0=off=old behaviour bit-identical.
        self.gj = float(gj_cc)
        self._Minv = None
        if self.gj > 0.0:
            c_lon = (0.25 / 0.75) * self.gj
            c_tra = c_lon * 0.73
            M = np.zeros((96, 96))
            for i in range(96):
                deg = 0.0
                # longitudinal: adjacent rows in the same quadrant (same quad+side)
                for j in range(96):
                    if j == i:
                        continue
                    if self.quad[j] == self.quad[i] and self.side[j] == self.side[i] \
                            and abs(self.row[j] - self.row[i]) == 1:
                        M[i, j] -= c_lon; deg += c_lon
                    elif self.quad[j] == self.quad[i] and self.side[j] != self.side[i] \
                            and self.row[j] == self.row[i]:
                        M[i, j] -= c_tra; deg += c_tra
                M[i, i] = 1.0 + deg
            self._Minv = np.linalg.inv(M)
            print(f"[muscle96] P11 muscle gap-junction layer ON: longitudinal c={c_lon:.3f} (cc=0.25 anchor) "
                  f"transverse c={c_tra:.3f} (x0.73 bounded estimate), steady-state tridiagonal analytic solution", flush=True)
        self.tau = float(tau_m_ms)                     # ms (two-layer mode: fast layer = electrical excitation/AP layer)
        self.tau_ca = float(tau_ca_ms)                 # ms; >0 enables two layers (P10, 2026-10-22):
        # fast-layer tau = AP half-width anchor (Gao&Zhen 2011 PNAS 108:2557 measured 15.5±0.9 ms;
        #   corroborated by Liu 2011 JPhysiol 589:101 ~20 ms, PMC3039263);
        # slow-layer tau_ca = calcium/contraction-layer anchor (Butler/Branicky et al. J R Soc Interface 2015;12:20140963
        #   measured GCaMP3 muscle fluorescence decay 1.57 s; the phase-corrected intermediate-filter assumption 1.0 s — apparent calcium-transient decay,
        #   includes indicator kinetics, the true contraction tau is smaller -> 1.0 s taken as the lower edge of a bounded estimate, sensitivity band declared).
        # DC gain unchanged (steady state mA=mE=A_inf); only the dynamic filtering and phase change; 0 = old single layer bit-identical.
        self.mE = np.zeros(96)                         # electrical-excitation integration state of the two-layer mode
        self.amax = float(amax)
        self.mA = np.zeros(96)                         # activation integration state
        # row-aggregation mask: (24, 96) L/R mean
        M = np.zeros((2, 24, 96))
        for i in range(96):
            M[self.quad[i], self.row[i], i] = 0.5
        self.agg = M
        # tonic baseline sig0: tonic release at rest (v59 design rel_rest≈0.4).
        # subtracting it = "activation represents deviation from resting tension", isomorphic to the codebase-wide dV=V-V_rest convention;
        # dorsal/ventral anatomical contact counts are asymmetric (DA9+DB7+DD6 vs VB11+VD13); without baseline subtraction the animal bends at rest.
        # under aldicarb (g_tot>1) I=(g_tot-1)*exc_rest>0 -> tonic contracture, consistent with pharmacology.
        self.sig0 = np.full(wout.shape[0], 0.4)
        # direction-group masks (disjoint): forward=DB/VB+DD, backward=DA/VA+VD, the rest (head classes etc.) always pass
        N = wout.shape[0]
        self.m_fwd = np.zeros(N); self.m_bwd = np.zeros(N); self.m_always = np.ones(N)
        if neuron_names is not None:
            for i, n in enumerate(neuron_names):
                n = str(n)
                if re.fullmatch(r"(DB\d\d|VB\d\d|DD\d\d)", n):
                    self.m_fwd[i] = 1.0; self.m_always[i] = 0.0
                elif re.fullmatch(r"(DA\d\d|VA\d\d|VD\d\d)", n):
                    self.m_bwd[i] = 1.0; self.m_always[i] = 0.0
            print(f"[muscle96] direction gating: forward group {int(self.m_fwd.sum())} / "
                  f"backward group {int(self.m_bwd.sum())} / always-on {int(self.m_always.sum())}", flush=True)

    def calibrate_rest(self, V_rest, vrel, mode="rel"):
        """Store the true resting drive sig0 (called once after the v62 warm-up).
        with mode='sp' sig0 is identically 0 (softplus(dV)-ln2 is 0 at rest, no table needed)."""
        if mode == "rel":
            self.sig0 = self.release(V_rest, vrel)
        else:
            self.sig0 = np.zeros_like(self.sig0)
        self.sig0_h = self.sig0.copy()   # v86: homeostatic-calibration tracking state (used when m96homeo>0)
        self.reset()
        return self.sig0

    def reset(self):
        self.mA[:] = 0.0
        self.mE[:] = 0.0

    def release(self, V, vrel):
        """Network-wide graded release: s≈0.4 at rest (v59 design: vrel = in-situ rest -10)."""
        return np.clip((V - vrel) / V_SCALE, 0.0, S_MAX)

    @staticmethod
    def sp_signal(dV):
        """softplus(dV)-ln2 deviation drive (same form as v66 mndrive): 0 at rest, approximately linear on depolarisation,
        independent of the vrel table — immune to vrel mismatch (v81 P3 control arm)."""
        return np.logaddexp(0.0, dV) - 0.6931471805599453

    def step(self, sig, motor_dt_ms, g_tot=1.0, gate_f=1.0, gate_b=0.0, gain=1.0,
             homeo_tau_s=0.0):
        """Advance one motor-layer step, return (A_D, A_V), each (24,).
        Activation = deviation from the resting-tension baseline; the resting state is by convention forward-gated (the real animal's default state is forward crawling).
        gate_f/gate_b: direction command gates (=v62's g_f/g_b, the interface of AVB/AVA mutual-exclusion gating).
        gain: applied BEFORE per-muscle saturation (P3 measurement: multiplying after aggregation destroys the proportional signal,
              the loop fails to bootstrap; the ablA control mndrive=0.10 is likewise a pre-clip structure).
        homeo_tau_s>0: v86 homeostatic tension calibration — baseline sig0_h slowly tracks each neuron's
              tonic release with time constant tau (seconds), passing only wave-frequency (phase) components; tonic asymmetry (vrel saturation / connectome eccentricity)
              is absorbed -> the static bend bias (left-drifting crawl) disappears. 0=off=old behaviour bit-identical."""
        mult = self.m_always + gate_f * self.m_fwd + gate_b * self.m_bwd
        mult0 = self.m_always + self.m_fwd          # rest = forward-state baseline
        if homeo_tau_s > 0.0:
            self.sig0_h += (motor_dt_ms / (homeo_tau_s * 1000.0)) * (sig - self.sig0_h)
        base = self.sig0_h if homeo_tau_s > 0.0 else self.sig0
        I = ((self.Wexc @ (sig * mult)) * g_tot + (self.Winh @ (sig * mult))
             - (self.Wexc + self.Winh) @ (base * mult0))
        A_inf = np.clip(gain * I, 0.0, self.amax)
        if self._Minv is not None:
            # P11: muscle gap-junction steady-state redistribution (steady-state spread of electrical excitation over coupled sheets)
            A_inf = self._Minv @ A_inf
        if self.tau_ca > 0.0:
            # P10 two-layer cascade: fast electrical-excitation layer -> slow calcium/contraction layer (DC gain unchanged, only dynamics change)
            # fast-layer tau=15.5 ms < motor step 50 ms, explicit Euler would explode -> exponential Euler (first-order closed form, unconditionally stable)
            a1 = np.exp(-motor_dt_ms / self.tau)
            self.mE[:] = self.mE * a1 + A_inf * (1.0 - a1)
            a2 = np.exp(-motor_dt_ms / self.tau_ca)
            self.mA[:] = self.mA * a2 + self.mE * (1.0 - a2)
        else:
            self.mA += (motor_dt_ms / self.tau) * (A_inf - self.mA)
        A_D = self.agg[0] @ self.mA
        A_V = self.agg[1] @ self.mA
        return A_D, A_V

    def head_body_split(self):
        """Head-group (rows 01-06) / body-group (rows 07-24) masks, for grouped-calibration probes."""
        head = self.row < 6
        return head, ~head
