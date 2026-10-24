# v62_numba_circuit.py  (numba JIT neural kernel, 2026-10-10)
# Mathematically identical term-by-term to v62_worm_circuit.py; on top of v62_fast_circuit it compiles
# the whole step (synaptic scatter-add + gating ODE + Euler + guardrails + clamp) into one @njit kernel.
# Limitation: only syn_delay_ms=0 is supported (delay buffer not ported; asserted at construction).
# Parity: verify_v62_numba.py vs the old engine for 35K steps, criterion max|dV| < 1e-6 mV.
import numpy as np
import math
from pathlib import Path
from numba import njit
import sys
sys.path.insert(0, str(Path(__file__).parent))
import anchored_engine_v56 as eng
from anchored_engine_v55 import EK, ECA, ENA, GATES_AWC, GATES_RMD
from v62_fast_circuit import Worm62Fast, _PKEYS

DT = 0.01
V_SCALE, S_MAX = 25.0, 1.5
E_EX, TAU_EX = 0.0, 2.0
E_INH, TAU_INH = -90.0, 5.0

# ---- Canonical gating column order (positional constants into the IX array) ----
CANON = ["m_shal", "hf_shal", "hs_shal", "m_shak", "h_shak", "m_kir",
         "m_kvs1", "h_kvs1", "mf_kqt3", "ms_kqt3", "w_kqt3", "s_kqt3", "m_egl2",
         "m1_egl36", "m2_egl36", "m3_egl36", "m_cca1", "h_cca1",
         "m_unc2", "h_unc2", "m_egl19", "hs_egl19",
         "mbk", "mslo1", "mbk2", "mslo2", "ca", "m_sk"]
C_SHAL, C_HFSHAL, C_HSSHAL, C_SHAK, C_HSHAK, C_KIR = 0, 1, 2, 3, 4, 5
C_KVS1, C_HKVS1, C_MFKQT3, C_MSKQT3, C_WKQT3, C_SKQT3, C_EGL2 = 6, 7, 8, 9, 10, 11, 12
C_M1E36, C_M2E36, C_M3E36 = 13, 14, 15
C_CCA1, C_HCCA1, C_UNC2, C_HUNC2, C_EGL19, C_HSEGL19 = 16, 17, 18, 19, 20, 21
C_MBK, C_MSLO1, C_MBK2, C_MSLO2, C_CA, C_MSK = 22, 23, 24, 25, 26, 27

# ---- Parameter column-order constants (aligned with _PKEYS) ----
for _i, _k in enumerate(_PKEYS):
    globals()["PK_" + _k.upper()] = _i


@njit(cache=True)
def _rhs_nb(Y, dY, P, Iext, IX, is_awc):
    """Scalar per-cell translation of anchored_engine_v55_batch.rhs_batch; formulas identical term by term."""
    n = Y.shape[0]
    for r in range(n):
        v = Y[r, 0]
        # ---- SHL-1 (Kv4) ----
        minf = 1.0 / (1.0 + np.exp(-(v - 11.2 + 18) / 14.1))
        tm = (13.8 / (np.exp(-(v + 17.5165) / 12.9213) + np.exp((v + 3.7082) / 6.4876)) + 1.8849) * 0.1
        hinf = 1.0 / (1.0 + np.exp((v + 33.1 + 18) / 8.3))
        ths = (8422.0 / (1.0 + np.exp((v + 37.7391) / 6.3785)) + 118.8983) * 0.1
        thf = (539.1584 / (1.0 + np.exp((v + 28.1990) / 4.9199)) + 27.2811) * 0.1
        m = Y[r, IX[C_SHAL]]; hf = Y[r, IX[C_HFSHAL]]; hs = Y[r, IX[C_HSSHAL]]
        dY[r, IX[C_SHAL]] = (minf - m) / tm
        dY[r, IX[C_HFSHAL]] = (hinf - hf) / thf
        dY[r, IX[C_HSSHAL]] = (hinf - hs) / ths
        I_shal = P[r, PK_GSHAL] * m**3 * (0.7 * hf + 0.3 * hs) * (v - EK)
        # ---- SHK-1 (Kv1) ----
        minf = 1.0 / (1.0 + np.exp(-(v - 20.4) / 7.7))
        tm = 26.571450568169027 / (np.exp(-(v + 33.741611800716130) / 15.364937728953288)
                                   + np.exp((v + 33.741611800716130) / 15.757936311607475)) + 1.990037272604829
        hinf = 1.0 / (1.0 + np.exp((v + 6.95) / 5.8))
        m = Y[r, IX[C_SHAK]]; h = Y[r, IX[C_HSHAK]]
        dY[r, IX[C_SHAK]] = (minf - m) / tm
        dY[r, IX[C_HSHAK]] = (hinf - h) / 1400.0
        I_shak = P[r, PK_GSHAK] * m * h * (v - EK)
        # ---- IRK ----
        minf = 1.0 / (1.0 + np.exp((v + 52.0 + 30) / 13.0))
        tm = 17.0752 / (np.exp(-(v + 17.8258) / 20.3154) + np.exp((v + 43.4414) / 11.1691)) + 3.8329
        m = Y[r, IX[C_KIR]]
        dY[r, IX[C_KIR]] = (minf - m) / tm
        I_kir = P[r, PK_GKIR] * m * (v - EK)
        I_kvs1 = 0.0; I_kqt3 = 0.0; I_egl2 = 0.0; I_egl36 = 0.0
        if is_awc:
            # ---- KVS-1 ----
            minf = 1.0 / (1.0 + np.exp(-(v - 57.1 + 30) / 25.0))
            tm = (30.0 / (1.0 + np.exp(-(v - 18.1232) / -20.0)) + 1.0) / 10.0
            hinf = 1.0 / (1.0 + np.exp((v - 47.3 + 30) / 11.1))
            th = (88.4715 / (1.0 + np.exp(-(v - 50.0) / -15.0)) + 53.4060) * 0.1
            m = Y[r, IX[C_KVS1]]; h = Y[r, IX[C_HKVS1]]
            dY[r, IX[C_KVS1]] = (minf - m) / tm
            dY[r, IX[C_HKVS1]] = (hinf - h) / th
            I_kvs1 = P[r, PK_GKVS1] * m * h * (v - EK)
            # ---- KQT-3 (Kv7) ----
            minf = 1.0 / (1.0 + np.exp(-(v + 12.6726 + 10) / 15.8008))
            tmf = (395.3 / (1.0 + ((v + 38.1) / 33.59)**2)) * 0.1
            tms = (5503.0 - 5345.4 / (1.0 + 10.0**(-0.02827 * (-23.9 - v)))
                   - 4590.6 / (1.0 + 10.0**(-0.0357 * (v + 14.15)))) * 0.1
            winf = 0.49 + 0.51 / (1.0 + np.exp((v + 1.084) / 28.78))
            tw = (5.44 + 29.2 / (1.0 + ((v + 48.09) / 48.83)**2)) * 0.1
            sinf = 0.34 + 0.66 / (1.0 + np.exp((v + 45.3) / 12.3))
            mf = Y[r, IX[C_MFKQT3]]; ms = Y[r, IX[C_MSKQT3]]
            w = Y[r, IX[C_WKQT3]]; s = Y[r, IX[C_SKQT3]]
            dY[r, IX[C_MFKQT3]] = (minf - mf) / tmf
            dY[r, IX[C_MSKQT3]] = (minf - ms) / tms
            dY[r, IX[C_WKQT3]] = (winf - w) / tw
            dY[r, IX[C_SKQT3]] = (sinf - s) / 500.0
            I_kqt3 = P[r, PK_GKQT3] * (0.3 * mf + 0.7 * ms) * w * s * (v - EK)
            # ---- EGL-2 (Kv11/eag) ----
            minf = 1.0 / (1.0 + np.exp(-(v + 6.8594) / 14.9131))
            tm = ((16.78 / (1.0 + np.exp((v + 122.5682) / 13.7976))) + 8.0969) * 0.5
            m = Y[r, IX[C_EGL2]]
            dY[r, IX[C_EGL2]] = (minf - m) / tm
            I_egl2 = P[r, PK_GEGL2] * m * (v - EK)
        else:
            # ---- EGL-36 (Kv3) ----
            minf = 1.0 / (1.0 + np.exp(-(v - 63.0) / 28.5))
            m1 = Y[r, IX[C_M1E36]]; m2 = Y[r, IX[C_M2E36]]; m3 = Y[r, IX[C_M3E36]]
            dY[r, IX[C_M1E36]] = (minf - m1) / 355.0
            dY[r, IX[C_M2E36]] = (minf - m2) / 63.0
            dY[r, IX[C_M3E36]] = (minf - m3) / 13.0
            I_egl36 = P[r, PK_GEGL36] * (0.31 * m1 + 0.36 * m2 + 0.39 * m3) * (v - EK)
        # ---- CCA-1 (CaV3/T) ----
        minf = 1.0 / (1.0 + np.exp(-(v + 42.65 + 15) / (1.7 * 1.4)))
        tm = ((40.0 / (1.0 + np.exp(-(v + 62.5393 + 30) / (-12.4758 * 1.7)))) + 0.6947) * 0.5
        hinf = 1.0 / (1.0 + np.exp((v + 58.0 + 15) / (7.0 * 1.15)))
        th = ((280.0 / (1.0 + np.exp((v + 60.7312 + 15) / (8.5224 * 1.1)))) + 19.7456) * 0.08
        m = Y[r, IX[C_CCA1]]; h = Y[r, IX[C_HCCA1]]
        dY[r, IX[C_CCA1]] = (minf - m) / tm
        dY[r, IX[C_HCCA1]] = (hinf - h) / th
        I_cca1 = P[r, PK_GCCA1] * m * m * h * (v - ECA)
        # ---- UNC-2 (CaV2 P/Q) ----
        minf_u = 1.0 / (1.0 + np.exp(-(v + 12.17 + 25) / 3.97))
        tm_u = (1.4969 / (np.exp(-(v + 8.1761 + 30) / 9.0753) + np.exp((v + 8.1761 + 30) / 15.3456)) + 0.1029) * 3.0
        hinf_u = 1.0 / (1.0 + np.exp((v + 52.47 + 25) / 5.6))
        th_u = (83.8037 / (1.0 + np.exp((v - 52.8997 + 30) / 3.4557))
                + 72.0995 / (1.0 + np.exp(-(v - 23.9009 + 30) / 3.5903))) * 1.7
        m = Y[r, IX[C_UNC2]]; h = Y[r, IX[C_HUNC2]]
        dY[r, IX[C_UNC2]] = (minf_u - m) / tm_u
        dY[r, IX[C_HUNC2]] = (hinf_u - h) / th_u
        I_unc2 = P[r, PK_GUNC2] * m * h * (v - ECA)
        alpha_u = minf_u / tm_u; beta_u = 1.0 / tm_u - alpha_u
        # ---- EGL-19 (CaV1 L) ----
        minf_e = 1.0 / (1.0 + np.exp(-(v - 5.6 + 10) / 7.5))
        tm_e = (2.3359 + 2.9324 * np.exp(-((v - 5.2357 + 10)**2) / 36.0)
                + 1.8739 * np.exp(-((v - 1.3930 + 10)**2) / 900.0))
        hinf_e = ((1.4314 / (1.0 + np.exp(-(v - 24.8573 + 10) / 11.9541)) + 0.1427)
                  * (5.9589 / (1.0 + np.exp((v + 10.5428 + 10) / 8.0552)) + 0.6038))
        ths_e = 0.4 * ((0.55 * 81.1179) / (1.0 + np.exp((v + 22.9723 + 10) / 5.0)) + 43.0937
                       + (0.9 * 40.4885) / (1.0 + np.exp((v - 28.7251 + 10) / 3.7125)) + 0.0)
        m = Y[r, IX[C_EGL19]]; h = Y[r, IX[C_HSEGL19]]
        dY[r, IX[C_EGL19]] = (minf_e - m) / tm_e
        dY[r, IX[C_HSEGL19]] = (hinf_e - h) / ths_e
        I_egl19 = P[r, PK_GEGL19] * m * h * (v - ECA)
        alpha_e = minf_e / tm_e; beta_e = 1.0 / tm_e - alpha_e
        # ---- Ca nanodomain + SLO1/SLO2 complex ----
        r_ = 13e-9; d_ = 250e-12; F_ = 96485.0; kb_ = 500e6; b_ = 30e-6; gsc_ = 40e-12
        cao = ((abs(gsc_ * (v - ECA) * 1e-3) / (8 * np.pi * r_ * d_ * F_))
               * np.exp(-r_ / np.sqrt(d_ / (kb_ * b_))) * 1e6 * 1e-3) + 0.05
        cac = 0.05
        wom = 3.152961; wyx = 0.012643; kyx = 34.338784; nyx = 0.000100
        wop = 0.156217; wxy = -0.027527; kxy = 55.726816; nxy = 1.299198
        kcm = wom * np.exp(-wyx * v) / (1.0 + (cac / kyx)**nyx)
        kom = wom * np.exp(-wyx * v) / (1.0 + (cao / kyx)**nyx)
        kop = wop * np.exp(-wxy * v) / (1.0 + (kxy / cao)**nxy)
        m_unc2 = Y[r, IX[C_UNC2]]; h_unc2 = Y[r, IX[C_HUNC2]]
        m_egl19 = Y[r, IX[C_EGL19]]; hs_egl19 = Y[r, IX[C_HSEGL19]]
        den1 = (kop + kom) * (kcm + alpha_u) + beta_u * kcm
        minf_bk = (m_unc2 * kop * (alpha_u + beta_u + kcm)) / den1
        tm_bk = (alpha_u + beta_u + kcm) / den1
        dY[r, IX[C_MBK]] = (minf_bk - Y[r, IX[C_MBK]]) / tm_bk
        I_bk = P[r, PK_GBK] * Y[r, IX[C_MBK]] * h_unc2 * (v - EK)
        den2 = (kop + kom) * (kcm + alpha_e) + beta_e * kcm
        minf_s1 = (m_egl19 * kop * (alpha_e + beta_e + kcm)) / den2
        tm_s1 = (alpha_e + beta_e + kcm) / den2
        dY[r, IX[C_MSLO1]] = (minf_s1 - Y[r, IX[C_MSLO1]]) / tm_s1
        I_slo1 = P[r, PK_GSLO1] * Y[r, IX[C_MSLO1]] * hs_egl19 * (v - EK)
        wom1 = 0.896395; wyx1 = 0.019405; kyx1 = 3294.553404; nyx1 = 0.000010
        wop1 = 0.026719; wxy1 = -0.024123; kxy1 = 93.449423; nxy1 = 1.835067
        kcm1 = wom1 * np.exp(-wyx1 * v) / (1.0 + (cac / kyx1)**nyx1)
        kom1 = wom1 * np.exp(-wyx1 * v) / (1.0 + (cao / kyx1)**nyx1)
        kop1 = wop1 * np.exp(-wxy1 * v) / (1.0 + (kxy1 / cao)**nxy1)
        den3 = (kop1 + kom1) * (kcm1 + alpha_e) + beta_e * kcm1
        minf_b2 = (m_egl19 * kop1 * (alpha_e + beta_e + kcm1)) / den3
        tm_b2 = (alpha_e + beta_e + kcm1) / den3
        dY[r, IX[C_MBK2]] = (minf_b2 - Y[r, IX[C_MBK2]]) / tm_b2
        I_bk2 = P[r, PK_GBK2] * Y[r, IX[C_MBK2]] * hs_egl19 * (v - EK)
        den4 = (kop1 + kom1) * (kcm1 + alpha_u) + beta_u * kcm1
        minf_s2 = (m_unc2 * kop1 * (alpha_u + beta_u + kcm1)) / den4
        tm_s2 = (alpha_u + beta_u + kcm1) / den4
        dY[r, IX[C_MSLO2]] = (minf_s2 - Y[r, IX[C_MSLO2]]) / tm_s2
        I_slo2 = P[r, PK_GSLO2] * Y[r, IX[C_MSLO2]] * h_unc2 * (v - EK)
        # ---- Intracellular calcium + KCNL ----
        I_ca = I_unc2 + I_egl19 + I_cca1
        alpha_ca = 1.0 / (2.0 * P[r, PK_VOL] * 96.485)
        ca = Y[r, IX[C_CA]]
        if I_ca < 0:
            rs = -0.001 * alpha_ca * I_ca - (ca - 5e-5) / 50.0
        else:
            rs = (5e-5 - ca) / 50.0
        dY[r, IX[C_CA]] = rs
        minf_sk = ca / (0.00033 + ca)
        dY[r, IX[C_MSK]] = (minf_sk - Y[r, IX[C_MSK]]) / 6.3
        I_sk = P[r, PK_GSK] * Y[r, IX[C_MSK]] * (v - EK)
        # ---- Leak and total ----
        I_leak = P[r, PK_GLEAK] * (v - P[r, PK_ELEAK])
        I_nca = P[r, PK_GNCA] * (v - ENA)
        I_tot = (I_shal + I_shak + I_kir + I_kvs1 + I_kqt3 + I_egl2 + I_egl36
                 + I_cca1 + I_unc2 + I_egl19 + I_bk + I_slo1 + I_bk2 + I_slo2 + I_sk + I_leak + I_nca)
        dY[r, 0] = (Iext[r] - I_tot) / P[r, PK_C]


@njit(cache=True)
def _csr_mv(data, indices, indptr, x, out):
    n = indptr.shape[0] - 1
    for i in range(n):
        acc = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            acc += data[p] * x[indices[p]]
        out[i] = acc


@njit(cache=True)
def _step_nb(Yawc, Pawc, Yrmd, Prmd, rows_awc, rows_rmd, IXawc, IXrmd,
             gA, gI, gWA, gWI,
             Mex_d, Mex_i, Mex_p, Min_d, Min_i, Min_p,
             Mwex_d, Mwex_i, Mwex_p, Mwin_d, Mwin_i, Mwin_p,
             Lg_d, Lg_i, Lg_p,
             vrel, I_extra, has_Iextra,
             clamp_lay, clamp_row, clamp_val, n_clamp,
             vclip_lo, vclip_hi, has_vclip, tau_wl, e_inh,
             V, sV, dA, dI, dWA, dWI, I_syn, I_gap,
             dYawc, dYrmd, Iext_awc, Iext_rmd,
             rect_i, rect_j, rect_w, rect_alpha, rect_vr, has_rect,
             bg_i, bg_j, bg_w, has_bg, dt):
    N = V.shape[0]
    # 1) gather V
    for i in range(rows_awc.shape[0]):
        V[rows_awc[i]] = Yawc[i, 0]
    for i in range(rows_rmd.shape[0]):
        V[rows_rmd[i]] = Yrmd[i, 0]
    # 2) release variable
    for i in range(N):
        s = (V[i] - vrel[i]) / V_SCALE
        if s < 0.0:
            s = 0.0
        elif s > S_MAX:
            s = S_MAX
        sV[i] = s
    # 3) synaptic scatter (CSR)
    _csr_mv(Mex_d, Mex_i, Mex_p, sV, dA)
    _csr_mv(Min_d, Min_i, Min_p, sV, dI)
    _csr_mv(Mwex_d, Mwex_i, Mwex_p, sV, dWA)
    _csr_mv(Mwin_d, Mwin_i, Mwin_p, sV, dWI)
    # 4) conductance relaxation
    for i in range(N):
        gA[i] += dt * (-gA[i] / TAU_EX + dA[i])
        gI[i] += dt * (-gI[i] / TAU_INH + dI[i])
        gWA[i] += dt * (-gWA[i] / tau_wl + dWA[i])
        gWI[i] += dt * (-gWI[i] / tau_wl + dWI[i])
    # 5) currents
    _csr_mv(Lg_d, Lg_i, Lg_p, V, I_gap)
    if has_rect:
        # W33: selective rectification correction (non-ohmic): forward conductance *(1+alpha), backward *(1-alpha), antisymmetry conserved
        for e in range(rect_i.shape[0]):
            ii = rect_i[e]
            jj = rect_j[e]
            dvv = V[ii] - V[jj]
            corr = rect_w[e] * rect_alpha * dvv * math.tanh(dvv / rect_vr)
            I_gap[ii] += corr
            I_gap[jj] -= corr
    if has_bg:
        # W47: bell-shaped Vj gating (Liu 2006 JBC Fig 2B measured: Boltzmann A=0.10, V0=40, Gjmin=0.13)
        # fully open at small |Vj|, conductance closes to 13% at large |Vj| -> auto-decouples when the two ends of a coupled block diverge, guarding against rail avalanche
        # per-edge reduction of the ohmic current i_ohm*(gate-1), antisymmetry conserved (independent of the L_gap CSR sign convention)
        for e in range(bg_i.shape[0]):
            ii = bg_i[e]
            jj = bg_j[e]
            dvv = V[ii] - V[jj]
            i_ohm = bg_w[e] * dvv
            gate = 0.87 / (1.0 + math.exp(0.10 * (math.fabs(dvv) - 40.0))) + 0.13
            corr = i_ohm * (gate - 1.0)
            I_gap[ii] += corr
            I_gap[jj] -= corr
    for i in range(N):
        I_syn[i] = (gA[i] + gWA[i]) * (V[i] - E_EX) + (gI[i] + gWI[i]) * (V[i] - e_inh)
    # 6) per-group external current
    for i in range(rows_awc.shape[0]):
        g = rows_awc[i]
        val = -I_syn[g] - I_gap[g]
        if has_Iextra:
            val += I_extra[g]
        Iext_awc[i] = val
    for i in range(rows_rmd.shape[0]):
        g = rows_rmd[i]
        val = -I_syn[g] - I_gap[g]
        if has_Iextra:
            val += I_extra[g]
        Iext_rmd[i] = val
    # 7) rhs + Euler + vclip
    _rhs_nb(Yawc, dYawc, Pawc, Iext_awc, IXawc, True)
    _rhs_nb(Yrmd, dYrmd, Prmd, Iext_rmd, IXrmd, False)
    for i in range(Yawc.shape[0]):
        for j in range(Yawc.shape[1]):
            Yawc[i, j] += dt * dYawc[i, j]
        if has_vclip:
            vv = Yawc[i, 0]
            if vv < vclip_lo:
                Yawc[i, 0] = vclip_lo
            elif vv > vclip_hi:
                Yawc[i, 0] = vclip_hi
    for i in range(Yrmd.shape[0]):
        for j in range(Yrmd.shape[1]):
            Yrmd[i, j] += dt * dYrmd[i, j]
        if has_vclip:
            vv = Yrmd[i, 0]
            if vv < vclip_lo:
                Yrmd[i, 0] = vclip_lo
            elif vv > vclip_hi:
                Yrmd[i, 0] = vclip_hi
    # 8) clamp (identity: v62 overwrites the voltage column after Euler)
    for j in range(n_clamp):
        if clamp_lay[j] == 0:
            Yawc[clamp_row[j], 0] = clamp_val[j]
        else:
            Yrmd[clamp_row[j], 0] = clamp_val[j]


class Worm62NB(Worm62Fast):
    """numba-kernel version of Worm62Fast; only syn_delay_ms=0 supported. Interface fully identical."""

    def __init__(self, *a, e_inh=-90.0, rect_alpha=0.0, rect_vr=15.0, mem_tau_scale=1.0, bgate=0.0, **k):
        super().__init__(*a, **k)
        self.e_inh = float(e_inh)
        self.rect_alpha = float(rect_alpha)
        self.rect_vr = float(rect_vr)
        assert self._sbuf is None, "Worm62NB supports only syn_delay_ms=0"
        assert len(self.groups) == 2 and self.groups[0][0] == "AWC" and self.groups[1][0] == "RMD"
        # group data
        self._Yawc = self.groups[0][2]
        self._Yrmd = self.groups[1][2]
        self._rows_awc = self.groups[0][3].astype(np.int64)
        self._rows_rmd = self.groups[1][3].astype(np.int64)
        self._Pawc = np.stack([self.groups[0][1][k] for k in _PKEYS], axis=1)
        self._Prmd = np.stack([self.groups[1][1][k] for k in _PKEYS], axis=1)
        # W36: membrane integration speed scaling (2026-10-12) - C/scale => integrates scale-times faster, steady-state I-V unchanged
        # Physical reading: in-vivo NMJ/sensorimotor synapse effective currents far exceed those under the distilled gain
        if mem_tau_scale != 1.0:
            self._Pawc[:, PK_C] /= float(mem_tau_scale)
            self._Prmd[:, PK_C] /= float(mem_tau_scale)
            print(f"[W36] membrane tau scale x{mem_tau_scale}: C/{mem_tau_scale}, steady-state I-V unchanged", flush=True)
        # gating column indices (canonical order -> actual columns)
        idx_awc = {n: i + 1 for i, n in enumerate(GATES_AWC)}
        idx_rmd = {n: i + 1 for i, n in enumerate(GATES_RMD)}
        self._IXawc = np.array([idx_awc.get(n, -1) for n in CANON], dtype=np.int64)
        self._IXrmd = np.array([idx_rmd.get(n, -1) for n in CANON], dtype=np.int64)
        # CSR arrays
        def _csr(M):
            return M.data.astype(float), M.indices.astype(np.int64), M.indptr.astype(np.int64)
        self._Mex = _csr(self.M_ex); self._Min = _csr(self.M_in)
        self._csr = _csr  # W54: refreshed when set_abx shifts gears
        self._Mwex = _csr(self.M_wlex); self._Mwin = _csr(self.M_wlin)
        self._Lg = _csr(self.L_gap)
        # scratch
        N = self.N
        self._Vs = np.zeros(N); self._sV = np.zeros(N)
        self._dA = np.zeros(N); self._dI = np.zeros(N)
        self._dWA = np.zeros(N); self._dWI = np.zeros(N)
        self._Isyn = np.zeros(N); self._Igap = np.zeros(N)
        self._dYawc = np.zeros_like(self._Yawc); self._dYrmd = np.zeros_like(self._Yrmd)
        self._Iext_awc = np.zeros(self._Yawc.shape[0]); self._Iext_rmd = np.zeros(self._Yrmd.shape[0])
        self._Iex0 = np.zeros(N)

    def _refresh_mex(self):
        # W54: refresh the CSR arrays read by the numba kernel after an A->B crosstalk gear shift
        self._Mex = self._csr(self.M_ex)
        # clamp buffer (up to 64 clamped cells)
        self._cl_lay = np.zeros(64, np.int64); self._cl_row = np.zeros(64, np.int64)
        self._cl_val = np.zeros(64)
        # W33: selective-rectification edge set (AVB->B-class and B-class chain anterior->posterior); rect_alpha=0 gives has_rect=False, old behaviour bit-identical
        self._rect_i, self._rect_j, self._rect_w = self._build_rect()
        self._has_rect = bool(self.rect_alpha != 0.0 and self._rect_i.shape[0] > 0)
        if self.rect_alpha != 0.0:
            print(f"[W33] gap rectification ON: {self._rect_i.shape[0]} edges, alpha={self.rect_alpha}, Vr={self.rect_vr} mV", flush=True)
        # W47: bell-gated gap-junction edge set (all gap edges, Liu 2006 anchor); bgate=0 gives has_bg=False, old behaviour bit-identical
        self._bgate = float(bgate)
        if self._bgate != 0.0:
            self._bg_i = self.A_g.astype(np.int64)
            self._bg_j = self.B_g.astype(np.int64)
            self._bg_w = (self.G_GAP * self.W_g).astype(float)
            self._has_bg = bool(self._bg_i.shape[0] > 0)
            print(f"[W47] bell-gated gap ON: {self._bg_i.shape[0]} edges (Liu2006 A=0.10 V0=40 Gmin=0.13)", flush=True)
        else:
            self._bg_i = np.zeros(0, np.int64); self._bg_j = np.zeros(0, np.int64)
            self._bg_w = np.zeros(0)
            self._has_bg = False
        self._warmup()

    def _build_rect(self):
        import re
        names = [str(n) for n in self.neurons]
        def cls(n):
            for p in ("AVB", "DB", "VB"):
                if n.startswith(p):
                    return p
            return None
        def num(n):
            m = re.search(r"(\d+)", n)
            return int(m.group(1)) if m else -1
        ri = []; rj = []; rw = []
        for e in range(self.A_g.shape[0]):
            a = int(self.A_g[e]); b = int(self.B_g[e])
            ca = cls(names[a]); cb = cls(names[b])
            src = -1; dst = -1
            pair = set((ca, cb))
            if "AVB" in pair and ("DB" in pair or "VB" in pair):
                src, dst = (a, b) if ca == "AVB" else (b, a)
            elif ca is not None and ca == cb and ca in ("DB", "VB"):
                if 0 <= num(names[a]) < num(names[b]):
                    src, dst = a, b
                elif 0 <= num(names[b]) < num(names[a]):
                    src, dst = b, a
            if src >= 0:
                ri.append(src); rj.append(dst)
                rw.append(self.G_GAP * float(self.W_g[e]))
        return np.array(ri, np.int64), np.array(rj, np.int64), np.array(rw, float)

    def _warmup(self):
        # compile the kernel on a copy of the state, without touching the real state
        _step_nb(self._Yawc.copy(), self._Pawc, self._Yrmd.copy(), self._Prmd,
                 self._rows_awc, self._rows_rmd, self._IXawc, self._IXrmd,
                 self.gA.copy(), self.gI.copy(), self.gWA.copy(), self.gWI.copy(),
                 self._Mex[0], self._Mex[1], self._Mex[2], self._Min[0], self._Min[1], self._Min[2],
                 self._Mwex[0], self._Mwex[1], self._Mwex[2], self._Mwin[0], self._Mwin[1], self._Mwin[2],
                 self._Lg[0], self._Lg[1], self._Lg[2],
                 self.vrel, self._Iex0, False,
                 self._cl_lay, self._cl_row, self._cl_val, 0,
                 -100.0, 60.0, True, self.TAU_WL, self.e_inh,
                 self._Vs, self._sV, self._dA, self._dI, self._dWA, self._dWI,
                 self._Isyn, self._Igap, self._dYawc.copy(), self._dYrmd.copy(),
                 self._Iext_awc, self._Iext_rmd,
                 self._rect_i, self._rect_j, self._rect_w, self.rect_alpha, self.rect_vr, self._has_rect,
                 self._bg_i, self._bg_j, self._bg_w, self._has_bg, DT)

    def step(self, clamp=None, I_extra=None):
        n_cl = 0
        if clamp is not None:
            for i, v in clamp:
                gi, r = self._owner[i]
                self._cl_lay[n_cl] = gi; self._cl_row[n_cl] = r; self._cl_val[n_cl] = v
                n_cl += 1
        has_ie = I_extra is not None
        if has_ie:
            ie = I_extra
        else:
            ie = self._Iex0
        has_vc = self.vclip is not None
        lo, hi = (self.vclip if has_vc else (0.0, 0.0))
        _step_nb(self._Yawc, self._Pawc, self._Yrmd, self._Prmd,
                 self._rows_awc, self._rows_rmd, self._IXawc, self._IXrmd,
                 self.gA, self.gI, self.gWA, self.gWI,
                 self._Mex[0], self._Mex[1], self._Mex[2], self._Min[0], self._Min[1], self._Min[2],
                 self._Mwex[0], self._Mwex[1], self._Mwex[2], self._Mwin[0], self._Mwin[1], self._Mwin[2],
                 self._Lg[0], self._Lg[1], self._Lg[2],
                 self.vrel, ie, has_ie,
                 self._cl_lay, self._cl_row, self._cl_val, n_cl,
                 lo, hi, has_vc, self.TAU_WL, self.e_inh,
                 self._Vs, self._sV, self._dA, self._dI, self._dWA, self._dWI,
                 self._Isyn, self._Igap, self._dYawc, self._dYrmd,
                 self._Iext_awc, self._Iext_rmd,
                 self._rect_i, self._rect_j, self._rect_w, self.rect_alpha, self.rect_vr, self._has_rect,
                 self._bg_i, self._bg_j, self._bg_w, self._has_bg, DT)
        self.t += DT
