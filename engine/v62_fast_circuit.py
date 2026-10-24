# v62_fast_circuit.py  (accelerated v62, 2026-10-10)
# Mathematically identical term-by-term to v62_worm_circuit.py, with three engineering accelerations:
#   1) merged-group integration: the 8 cell classes have only 2 gating layouts (AWC / RMD),
#      the 7 classes on the RMD layout (VA/VB/VD/DA/DB/AVA/RMD) are merged into one (78, n_states) matrix,
#      parameters broadcast by row -> rhs calls per step drop from 8 to 2.
#   2) np.add.at scatter-adds replaced by scipy.sparse CSR matvecs (chemical exc/inh, wireless exc/inh, gap-junction Laplacian).
#   3) clamp cell->within-group row owner table prebuilt; no per-step np.nonzero search.
# Numerics note: element-wise ops inside merged groups run in the same order as per-class calls, bit-for-bit; sparse matvec sums in a different order than add.at,
# giving ulp-level differences (~1e-16 relative), logged by the verify_v62_fast.py parity run.
# Interface identical to Worm62: __init__(g_syn, g_gap, ...), step(clamp, I_extra), _V(), idx/neurons/N/t.
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
import sys
sys.path.insert(0, str(Path(__file__).parent))
import anchored_engine_v56 as eng
from anchored_engine_v55 import EK, ECA, ENA, GATES_AWC, GATES_RMD

HERE = Path(__file__).parent
DT = 0.01
V_REL, V_SCALE, S_MAX = -55.0, 25.0, 1.5
E_EX, TAU_EX = 0.0, 2.0
E_INH, TAU_INH = -90.0, 5.0
TAU_WL = 500.0
NET_NPZ = HERE.parent / "data" / "net_gapfix.npz"  # run data/make_net_gapfix_npz.py once
WL_NPZ = HERE.parent / "data" / "wireless_payload.csv"  # g_wl=0 in all paper experiments
EQ_CACHE = HERE.parent / "data" / "eq_states.json"

_RMD_LAYOUT = {"VA", "VB", "VD", "DA", "DB", "AVA", "RMD"}
_GATES = {"AWC": GATES_AWC, "RMD": GATES_RMD}
_IDX = {ct: {n: i + 1 for i, n in enumerate(g)} for ct, g in _GATES.items()}
# all parameter keys used by rhs
_PKEYS = ("gshal", "gshak", "gkir", "gkvs1", "gkqt3", "gegl2", "gegl36",
          "gcca1", "gunc2", "gegl19", "gbk", "gslo1", "gbk2", "gslo2", "gsk",
          "gleak", "ELEAK", "gnca", "vol", "C")


def _rhs_merged(t, Y, Pv, layout, Iext):
    """Identical term-by-term to anchored_engine_v55_batch.rhs_batch; Pv is a per-row parameter (n,) array."""
    IX = _IDX[layout]
    v = Y[:, 0]
    dY = np.zeros_like(Y)
    exp = np.exp

    def S(name):
        return Y[:, IX[name]]

    def D(name):
        return IX[name]

    # ---- SHL-1 (Kv4) ----
    minf = 1 / (1 + exp(-(v - 11.2 + 18) / 14.1))
    tm = (13.8 / (exp(-(v + 17.5165) / 12.9213) + exp((v + 3.7082) / 6.4876)) + 1.8849) * 0.1
    hinf = 1 / (1 + exp((v + 33.1 + 18) / 8.3))
    ths = (8422.0 / (1 + exp((v + 37.7391) / 6.3785)) + 118.8983) * 0.1
    thf = (539.1584 / (1 + exp((v + 28.1990) / 4.9199)) + 27.2811) * 0.1
    m, hf, hs = S("m_shal"), S("hf_shal"), S("hs_shal")
    dY[:, D("m_shal")] = (minf - m) / tm
    dY[:, D("hf_shal")] = (hinf - hf) / thf
    dY[:, D("hs_shal")] = (hinf - hs) / ths
    I_shal = Pv["gshal"] * m**3 * (0.7 * hf + 0.3 * hs) * (v - EK)

    # ---- SHK-1 (Kv1) ----
    minf = 1 / (1 + exp(-(v - 20.4) / 7.7))
    tm = 26.571450568169027 / (exp(-(v + 33.741611800716130) / 15.364937728953288)
                               + exp((v + 33.741611800716130) / 15.757936311607475)) + 1.990037272604829
    hinf = 1 / (1 + exp((v + 6.95) / 5.8))
    m, h = S("m_shak"), S("h_shak")
    dY[:, D("m_shak")] = (minf - m) / tm
    dY[:, D("h_shak")] = (hinf - h) / 1400.0
    I_shak = Pv["gshak"] * m * h * (v - EK)

    # ---- IRK ----
    minf = 1 / (1 + exp((v + 52.0 + 30) / 13.0))
    tm = 17.0752 / (exp(-(v + 17.8258) / 20.3154) + exp((v + 43.4414) / 11.1691)) + 3.8329
    m = S("m_kir")
    dY[:, D("m_kir")] = (minf - m) / tm
    I_kir = Pv["gkir"] * m * (v - EK)

    I_kvs1 = I_kqt3 = I_egl2 = I_egl36 = 0.0
    if layout == "AWC":
        # ---- KVS-1 ----
        minf = 1 / (1 + exp(-(v - 57.1 + 30) / 25.0))
        tm = (30.0 / (1 + exp(-(v - 18.1232) / -20.0)) + 1.0) / 10.0
        hinf = 1 / (1 + exp((v - 47.3 + 30) / 11.1))
        th = (88.4715 / (1 + exp(-(v - 50.0) / -15.0)) + 53.4060) * 0.1
        m, h = S("m_kvs1"), S("h_kvs1")
        dY[:, D("m_kvs1")] = (minf - m) / tm
        dY[:, D("h_kvs1")] = (hinf - h) / th
        I_kvs1 = Pv["gkvs1"] * m * h * (v - EK)
        # ---- KQT-3 (Kv7) ----
        minf = 1 / (1 + exp(-(v + 12.6726 + 10) / 15.8008))
        tmf = (395.3 / (1 + ((v + 38.1) / 33.59)**2)) * 0.1
        tms = (5503.0 - 5345.4 / (1 + 10**(-0.02827 * (-23.9 - v)))
               - 4590.6 / (1 + 10**(-0.0357 * (v + 14.15)))) * 0.1
        winf = 0.49 + 0.51 / (1 + exp((v + 1.084) / 28.78))
        tw = (5.44 + 29.2 / (1 + ((v + 48.09) / 48.83)**2)) * 0.1
        sinf = 0.34 + 0.66 / (1 + exp((v + 45.3) / 12.3))
        mf, ms, w, s = S("mf_kqt3"), S("ms_kqt3"), S("w_kqt3"), S("s_kqt3")
        dY[:, D("mf_kqt3")] = (minf - mf) / tmf
        dY[:, D("ms_kqt3")] = (minf - ms) / tms
        dY[:, D("w_kqt3")] = (winf - w) / tw
        dY[:, D("s_kqt3")] = (sinf - s) / 500.0
        I_kqt3 = Pv["gkqt3"] * (0.3 * mf + 0.7 * ms) * w * s * (v - EK)
        # ---- EGL-2 (Kv11/eag) ----
        minf = 1 / (1 + exp(-(v + 6.8594) / 14.9131))
        tm = ((16.78 / (1 + exp((v + 122.5682) / 13.7976))) + 8.0969) * 0.5
        m = S("m_egl2")
        dY[:, D("m_egl2")] = (minf - m) / tm
        I_egl2 = Pv["gegl2"] * m * (v - EK)
    else:
        # ---- EGL-36 (Kv3) ----
        minf = 1 / (1 + exp(-(v - 63.0) / 28.5))
        for nm, tau, w_ in (("m1_egl36", 355.0, 0.31), ("m2_egl36", 63.0, 0.36), ("m3_egl36", 13.0, 0.39)):
            dY[:, D(nm)] = (minf - S(nm)) / tau
        I_egl36 = Pv["gegl36"] * (0.31 * S("m1_egl36") + 0.36 * S("m2_egl36") + 0.39 * S("m3_egl36")) * (v - EK)

    # ---- CCA-1 (CaV3/T) ----
    minf = 1 / (1 + exp(-(v + 42.65 + 15) / (1.7 * 1.4)))
    tm = ((40.0 / (1 + exp(-(v + 62.5393 + 30) / (-12.4758 * 1.7)))) + 0.6947) * 0.5
    hinf = 1 / (1 + exp((v + 58.0 + 15) / (7.0 * 1.15)))
    th = ((280.0 / (1 + exp((v + 60.7312 + 15) / (8.5224 * 1.1)))) + 19.7456) * 0.08
    m, h = S("m_cca1"), S("h_cca1")
    dY[:, D("m_cca1")] = (minf - m) / tm
    dY[:, D("h_cca1")] = (hinf - h) / th
    I_cca1 = Pv["gcca1"] * m**2 * h * (v - ECA)

    # ---- UNC-2 (CaV2 P/Q) ----
    minf_u = 1 / (1 + exp(-(v + 12.17 + 25) / 3.97))
    tm_u = (1.4969 / (exp(-(v + 8.1761 + 30) / 9.0753) + exp((v + 8.1761 + 30) / 15.3456)) + 0.1029) * 3.0
    hinf_u = 1 / (1 + exp((v + 52.47 + 25) / 5.6))
    th_u = (83.8037 / (1 + exp((v - 52.8997 + 30) / 3.4557))
            + 72.0995 / (1 + exp(-(v - 23.9009 + 30) / 3.5903))) * 1.7
    m, h = S("m_unc2"), S("h_unc2")
    dY[:, D("m_unc2")] = (minf_u - m) / tm_u
    dY[:, D("h_unc2")] = (hinf_u - h) / th_u
    I_unc2 = Pv["gunc2"] * m * h * (v - ECA)
    alpha_u = minf_u / tm_u; beta_u = 1.0 / tm_u - alpha_u

    # ---- EGL-19 (CaV1 L) ----
    minf_e = 1 / (1 + exp(-(v - 5.6 + 10) / 7.5))
    tm_e = (2.3359 + 2.9324 * exp(-((v - 5.2357 + 10)**2) / 36.0)
            + 1.8739 * exp(-((v - 1.3930 + 10)**2) / 900.0))
    hinf_e = ((1.4314 / (1 + exp(-(v - 24.8573 + 10) / 11.9541)) + 0.1427)
              * (5.9589 / (1 + exp((v + 10.5428 + 10) / 8.0552)) + 0.6038))
    ths_e = 0.4 * ((0.55 * 81.1179) / (1 + exp((v + 22.9723 + 10) / 5.0)) + 43.0937
                   + (0.9 * 40.4885) / (1 + exp((v - 28.7251 + 10) / 3.7125)) + 0.0)
    m, h = S("m_egl19"), S("hs_egl19")
    dY[:, D("m_egl19")] = (minf_e - m) / tm_e
    dY[:, D("hs_egl19")] = (hinf_e - h) / ths_e
    I_egl19 = Pv["gegl19"] * m * h * (v - ECA)
    alpha_e = minf_e / tm_e; beta_e = 1.0 / tm_e - alpha_e

    # ---- Ca nanodomain + SLO1/SLO2 complex ----
    r_, d_, F_, kb_, b_, gsc_ = 13e-9, 250e-12, 96485.0, 500e6, 30e-6, 40e-12
    cao = ((np.abs(gsc_ * (v - ECA) * 1e-3) / (8 * np.pi * r_ * d_ * F_))
           * np.exp(-r_ / np.sqrt(d_ / (kb_ * b_))) * 1e6 * 1e-3) + 0.05
    cac = 0.05
    wom, wyx, kyx, nyx = 3.152961, 0.012643, 34.338784, 0.000100
    wop, wxy, kxy, nxy = 0.156217, -0.027527, 55.726816, 1.299198
    kcm = wom * exp(-wyx * v) / (1 + (cac / kyx)**nyx)
    kom = wom * exp(-wyx * v) / (1 + (cao / kyx)**nyx)
    kop = wop * exp(-wxy * v) / (1 + (kxy / cao)**nxy)
    den1 = (kop + kom) * (kcm + alpha_u) + beta_u * kcm
    minf_bk = (S("m_unc2") * kop * (alpha_u + beta_u + kcm)) / den1
    tm_bk = (alpha_u + beta_u + kcm) / den1
    dY[:, D("mbk")] = (minf_bk - S("mbk")) / tm_bk
    I_bk = Pv["gbk"] * S("mbk") * S("h_unc2") * (v - EK)
    den2 = (kop + kom) * (kcm + alpha_e) + beta_e * kcm
    minf_s1 = (S("m_egl19") * kop * (alpha_e + beta_e + kcm)) / den2
    tm_s1 = (alpha_e + beta_e + kcm) / den2
    dY[:, D("mslo1")] = (minf_s1 - S("mslo1")) / tm_s1
    I_slo1 = Pv["gslo1"] * S("mslo1") * S("hs_egl19") * (v - EK)
    wom1, wyx1, kyx1, nyx1 = 0.896395, 0.019405, 3294.553404, 0.000010
    wop1, wxy1, kxy1, nxy1 = 0.026719, -0.024123, 93.449423, 1.835067
    kcm1 = wom1 * exp(-wyx1 * v) / (1 + (cac / kyx1)**nyx1)
    kom1 = wom1 * exp(-wyx1 * v) / (1 + (cao / kyx1)**nyx1)
    kop1 = wop1 * exp(-wxy1 * v) / (1 + (kxy1 / cao)**nxy1)
    den3 = (kop1 + kom1) * (kcm1 + alpha_e) + beta_e * kcm1
    minf_b2 = (S("m_egl19") * kop1 * (alpha_e + beta_e + kcm1)) / den3
    tm_b2 = (alpha_e + beta_e + kcm1) / den3
    dY[:, D("mbk2")] = (minf_b2 - S("mbk2")) / tm_b2
    I_bk2 = Pv["gbk2"] * S("mbk2") * S("hs_egl19") * (v - EK)
    den4 = (kop1 + kom1) * (kcm1 + alpha_u) + beta_u * kcm1
    minf_s2 = (S("m_unc2") * kop1 * (alpha_u + beta_u + kcm1)) / den4
    tm_s2 = (alpha_u + beta_u + kcm1) / den4
    dY[:, D("mslo2")] = (minf_s2 - S("mslo2")) / tm_s2
    I_slo2 = Pv["gslo2"] * S("mslo2") * S("h_unc2") * (v - EK)

    # ---- Intracellular calcium (microdomain) + KCNL ----
    I_ca = I_unc2 + I_egl19 + I_cca1
    alpha_ca = 1 / (2 * Pv["vol"] * 96.485)
    ca = S("ca")
    rs = np.where(I_ca < 0, -0.001 * alpha_ca * I_ca - (ca - 5e-5) / 50.0, (5e-5 - ca) / 50.0)
    dY[:, D("ca")] = rs
    minf_sk = ca / (0.00033 + ca)
    dY[:, D("m_sk")] = (minf_sk - S("m_sk")) / 6.3
    I_sk = Pv["gsk"] * S("m_sk") * (v - EK)

    I_leak = Pv["gleak"] * (v - Pv["ELEAK"])
    I_nca = Pv["gnca"] * (v - ENA)

    I_tot = (I_shal + I_shak + I_kir + I_kvs1 + I_kqt3 + I_egl2 + I_egl36
             + I_cca1 + I_unc2 + I_egl19 + I_bk + I_slo1 + I_bk2 + I_slo2 + I_sk + I_leak + I_nca)
    dY[:, 0] = (Iext - I_tot) / Pv["C"]
    return dY


class Worm62Fast:
    def __init__(self, g_syn, g_gap, g_wl=None, tau_wl=None, syn_delay_ms=0.0,
                 g_inh_mult=5.0, vclip=None, netnpz=None):
        # P1 adjudication (2026-10-21 constant anchoring): g_inh_mult is a "bounded estimate", not a measured constant.
        #   NMJ measured E:I=1.6:1 (ACh 2.01 nA / GABA 1.23 nA, Liu 2009 PNAS PMC2705609)
        #   applies strictly to the neuromuscular junction and has been handed to the muscle96 input-stage anchoring (see P7).
        #   This parameter multiplies neuron-neuron inhibitory edges — a synapse type with no worm measurement (field blank).
        #   Global extrapolation to 0.6 experiment: crawling collapses (v 0.017 mm/s, f 0.111 Hz, κ_amp 16.1),
        #   falsifying "NMJ-ratio globalisation". 5.0 is retained as an estimate and must not be repackaged as a physiological constant.
        self.G_SYN = g_syn; self.G_INH = g_inh_mult * g_syn; self.G_GAP = g_gap
        self.g_inh_mult = g_inh_mult
        self.vclip = vclip
        self.G_WL = g_syn if g_wl is None else g_wl
        self.G_WL_INH = g_inh_mult * self.G_WL   # same source as the main inhibition (frozen config g_wl=0, zero behavioural impact)
        self.TAU_WL = TAU_WL if tau_wl is None else tau_wl
        z = np.load(netnpz if netnpz is not None else NET_NPZ, allow_pickle=False)
        self.neurons = [str(x) for x in z["neurons"]]
        self.N = len(self.neurons)
        self.idx = {n: i for i, n in enumerate(self.neurons)}
        ctypes = [str(x) for x in z["ctypes"]]
        self.vrel = np.array([eng.RELEASE_REF.get(ct, V_REL) for ct in ctypes])
        import json as _json
        with open(EQ_CACHE, encoding='utf-8') as _fh:
            eq = {k: np.asarray(v, dtype=float) for k, v in _json.load(_fh).items()}
        # merged groups: (layout, Pv, Y, rows_global)
        self.groups = []
        self._owner = {}
        for lay in ("AWC", "RMD"):
            rows_lay = []
            Ys = []
            Ps = []
            for ct in sorted(set(ctypes)):
                lay_ct = "RMD" if ct in _RMD_LAYOUT else ct
                if lay_ct != lay:
                    continue
                rows = [i for i, c in enumerate(ctypes) if c == ct]
                y0 = eq[ct].copy() if ct in eq else eng.init_state(ct)
                Ys.append(np.tile(y0, (len(rows), 1)))
                for r in rows:
                    Ps.append(eng.cell_params(ct))
                rows_lay.extend(rows)
            if not rows_lay:
                continue
            Y = np.vstack(Ys)
            Pv = {k: np.array([p.get(k, 0.0) for p in Ps], dtype=float) for k in _PKEYS}
            gi = len(self.groups)
            self.groups.append([lay, Pv, Y, np.array(rows_lay, dtype=int)])
            for r_local, r_global in enumerate(rows_lay):
                self._owner[r_global] = (gi, r_local)
        # sparse synaptic matrices
        chem = z["chem"]; sign = z["sign"]; gap = z["gap"]
        ex = sign > 0
        N = self.N
        self.M_ex = csr_matrix((self.G_SYN * chem[ex, 2],
                                (chem[ex, 1].astype(int), chem[ex, 0].astype(int))), shape=(N, N))
        self.M_in = csr_matrix((self.G_INH * chem[~ex, 2],
                                (chem[~ex, 1].astype(int), chem[~ex, 0].astype(int))), shape=(N, N))
        # W54 (2026-10-23): split of the A-MN(VA/DA)->B-MN(VB/DB) excitatory chemical crosstalk edge set
        # Kato et al. 2015 Cell 163:656 subcircuit mutual-inhibition analog: in the backward state the forward module should be suppressed;
        # anatomical audit: VA->VB w=5.508 locks VA into the forward program (DA->DB only 1.65 -> DA is free and carries a wave,
        # measured phase gradient DA 0.31 rad/seg = λ20 segs vs VA 0.02 = synchronous); set_abx enables block-wise gear shifting
        import re as _re
        _nm = [str(n) for n in self.neurons]
        def _cls(n, pfx):
            return n.startswith(pfx) and any(ch.isdigit() for ch in n)
        _ab = np.array([_cls(_nm[int(c[0])], "VA") or _cls(_nm[int(c[0])], "DA")
                        for c in chem]) & ex & \
              np.array([_cls(_nm[int(c[1])], "VB") or _cls(_nm[int(c[1])], "DB")
                        for c in chem])
        self.M_ex_ab = csr_matrix((self.G_SYN * chem[_ab, 2],
                                   (chem[_ab, 1].astype(int), chem[_ab, 0].astype(int))), shape=(N, N))
        self.M_ex_rest = self.M_ex - self.M_ex_ab
        self._abx_gain = 1.0
        _wlt = np.genfromtxt(WL_NPZ, delimiter=',', skip_header=1)
        wlp, wls, wlw = _wlt[:, 0].astype(int), _wlt[:, 1].astype(int), _wlt[:, 2]
        wl_ex = wlw > 0
        self.M_wlex = csr_matrix((self.G_WL * wlw[wl_ex],
                                  (wls[wl_ex].astype(int), wlp[wl_ex].astype(int))), shape=(N, N))
        self.M_wlin = csr_matrix((self.G_WL_INH * (-wlw[~wl_ex]),
                                  (wls[~wl_ex].astype(int), wlp[~wl_ex].astype(int))), shape=(N, N))
        # gap-junction Laplacian: I_gap[a]+=g w(Va-Vb), I_gap[b]-=g w(Va-Vb)
        a = gap[:, 0].astype(int); b = gap[:, 1].astype(int); w = self.G_GAP * gap[:, 2]
        self.A_g = a; self.B_g = b; self.W_g = gap[:, 2]   # W33: rectification edge set needs the endpoint table (pure attribute mount, behaviour unchanged)
        rr = np.concatenate([a, a, b, b]); cc = np.concatenate([a, b, a, b])
        vv = np.concatenate([w, -w, -w, w])
        self.L_gap = csr_matrix((vv, (rr, cc)), shape=(N, N))
        self.gA = np.zeros(N); self.gI = np.zeros(N)
        self.gWA = np.zeros(N); self.gWI = np.zeros(N)
        # synaptic-delay ring buffer (same as v62)
        self.syn_delay_ms = float(syn_delay_ms)
        self._dsteps = int(round(self.syn_delay_ms / DT))
        if self._dsteps > 0:
            size = self._dsteps + 1
            s0 = self._Srel(self._V())
            self._sbuf = np.repeat(s0[:, None], size, axis=1)
            self._spos = 0
        else:
            self._sbuf = None
        self.t = 0.0

    def set_abx(self, gain):
        """W54: A->B crosstalk gain gear shift (gain=1.0 restores M_ex = bit-identical old behaviour)"""
        if gain != self._abx_gain:
            self._abx_gain = gain
            self.M_ex = self.M_ex_rest + gain * self.M_ex_ab
            self._refresh_mex()

    def _refresh_mex(self):
        pass  # fast_circuit reads self.M_ex directly; the numba subclass overrides this to refresh CSR arrays

    def _V(self):
        V = np.empty(self.N)
        for lay, Pv, Y, rows in self.groups:
            V[rows] = Y[:, 0]
        return V

    def _Srel(self, V):
        return np.clip((V - self.vrel) / V_SCALE, 0.0, S_MAX)

    def step(self, clamp=None, I_extra=None):
        V = self._V()
        sV = self._Srel(V)
        if self._sbuf is not None:
            size = self._sbuf.shape[1]
            self._sbuf[:, self._spos] = sV
            sC = self._sbuf[:, (self._spos + 1) % size]
            self._spos = (self._spos + 1) % size
        else:
            sC = sV
        dA = self.M_ex @ sC
        dI = self.M_in @ sC
        self.gA += DT * (-self.gA / TAU_EX + dA)
        self.gI += DT * (-self.gI / TAU_INH + dI)
        dWA = self.M_wlex @ sV
        dWI = self.M_wlin @ sV
        self.gWA += DT * (-self.gWA / self.TAU_WL + dWA)
        self.gWI += DT * (-self.gWI / self.TAU_WL + dWI)
        I_syn = (self.gA + self.gWA) * (V - E_EX) + (self.gI + self.gWI) * (V - E_INH)
        I_gap = self.L_gap @ V
        I_ext_all = -I_syn - I_gap
        if I_extra is not None:
            I_ext_all = I_ext_all + I_extra
        for lay, Pv, Y, rows in self.groups:
            dY = _rhs_merged(self.t, Y, Pv, lay, I_ext_all[rows])
            Y += DT * dY
            if self.vclip is not None:
                np.clip(Y[:, 0], self.vclip[0], self.vclip[1], out=Y[:, 0])
        if clamp is not None:
            for i, v in clamp:
                gi, r = self._owner[i]
                self.groups[gi][2][r, 0] = v
        self.t += DT

    def run(self, T, clamp=None, clamp_win=None, rec_dt=2.0):
        n = int(round(T / DT)); rec = int(round(rec_dt / DT))
        ts, Vs = [], []
        for k in range(n):
            cl = None
            if clamp is not None and clamp_win[0] <= self.t < clamp_win[1]:
                cl = clamp
            self.step(cl)
            if k % rec == 0:
                ts.append(self.t); Vs.append(self._V())
        return np.array(ts), np.array(Vs).T
