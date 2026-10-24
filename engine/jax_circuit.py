# jax_circuit.py  (W60-ACC3: JAX 批量神经内核, 2026-10-12)
# 数学逐项镜像 v62_numba_circuit._step_nb (显式 Euler, 同 DT, 同更新顺序)。
# 架构参照 kairess/worm-whisperer 的 jaxsim/batch 模式 (PolyForm Noncommercial, 引用致谢):
#   dense 矩阵 + lax.scan 时间积分 + vmap 种群批评。
# 差异: 通道方程/参数全为本项目 v62 (重锚常数), 非 c302 C2。
# 对拍: test_jax_parity.py vs numba 引擎, 判据见文件末尾。
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

# ---- 与 v62_numba_circuit 相同的常数 ----
V_SCALE, S_MAX = 25.0, 1.5
EK, ECA, ENA = -80.0, 60.0, 30.0
E_EX, TAU_EX = 0.0, 2.0
TAU_INH = 5.0

# 规范门控列序 (与 numba CANON 一致); Y 列: 0=V, 1..28 门控
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
NG = len(CANON)  # 28

_PKEYS = ("gshal", "gshak", "gkir", "gkvs1", "gkqt3", "gegl2", "gegl36",
          "gcca1", "gunc2", "gegl19", "gbk", "gslo1", "gbk2", "gslo2", "gsk",
          "gleak", "ELEAK", "gnca", "vol", "C")
PK = {k: i for i, k in enumerate(_PKEYS)}


def extract(net, W, muscle_maps, dt_ms):
    """从建好的 Worm62NB 实例提取 JAX 参数包 (宿主侧一次性)。
    net: build_net() 产物; W: w54_redistill.W (需 ensure_param_struct 已跑);
    muscle_maps: load_muscle_maps(net) 产物。"""
    from scipy.sparse import csr_matrix
    N = net.N
    chem, sign, gap = W["chem"], W["sign"], W["gap"]
    cmasks, gmasks = W["cmasks"], W["gmasks"]

    # ---- 细胞组 → 规范 29 列布局 ----
    def canon_group(gi):
        lay, Pv, Y, rows = net.groups[gi]
        IX = net._IXawc if lay == "AWC" else net._IXrmd
        Pmat = net._Pawc if lay == "AWC" else net._Prmd
        Yc = np.zeros((Y.shape[0], 1 + NG))
        Yc[:, 0] = Y[:, 0]
        for ci in range(NG):
            col = int(IX[ci])
            if col >= 0:
                Yc[:, 1 + ci] = Y[:, col]
        return Pmat.copy(), Yc, np.asarray(rows, np.int64), lay

    Pawc, Y0awc, rows_awc, lay0 = canon_group(0)
    Prmd, Y0rmd, rows_rmd, lay1 = canon_group(1)
    assert lay0 == "AWC" and lay1 == "RMD"

    # ---- 单位 dense 矩阵 (乘子=1 时的模式矩阵) ----
    ex = sign > 0
    def chem_dense(mask_rows, take_ex):
        M = np.zeros((N, N))
        sel = take_ex & mask_rows
        e = np.nonzero(sel)[0]
        M[chem[e, 1].astype(int), chem[e, 0].astype(int)] += chem[e, 2]
        return M
    restc = np.ones(chem.shape[0], bool)
    for m in cmasks:
        restc &= ~m
    Mex_u = [chem_dense(m, ex) for m in cmasks] + [chem_dense(restc, ex)]
    Min_u = [chem_dense(m, ~ex) for m in cmasks] + [chem_dense(restc, ~ex)]

    restg = np.ones(gap.shape[0], bool)
    for m in gmasks:
        restg &= ~m
    def gap_dense(mask):
        L = np.zeros((N, N))
        e = np.nonzero(mask)[0]
        a = gap[e, 0].astype(int); b = gap[e, 1].astype(int); w = gap[e, 2]
        np.add.at(L, (a, a), w); np.add.at(L, (a, b), -w)
        np.add.at(L, (b, a), -w); np.add.at(L, (b, b), w)
        return L
    Lg_u = [gap_dense(m) for m in gmasks] + [gap_dense(restg)]

    muscles, wout, inhib, n_inn = muscle_maps
    P = dict(
        N=N, dt=float(dt_ms),
        Pawc=Pawc, Prmd=Prmd, Y0awc=Y0awc, Y0rmd=Y0rmd,
        rows_awc=rows_awc, rows_rmd=rows_rmd,
        vrel=np.asarray(net.vrel, float),
        Mex_u=Mex_u, Min_u=Min_u, Lg_u=Lg_u,
        G_SYN=net.G_SYN, G_INH=net.G_INH, G_GAP=net.G_GAP,
        e_inh=float(net.e_inh),
        vclip=(float(net.vclip[0]), float(net.vclip[1])) if net.vclip else None,
        wout=np.asarray(wout, float), inhib=np.asarray(inhib, float),
        n_inn=np.asarray(n_inn, float),
    )
    return P


def _rhs(v, Y, Pm, is_awc):
    """逐行向量化 rhs; Pm (n,20) 参数; 返回 dY (n,29)。逐项镜像 _rhs_nb。"""
    exp = jnp.exp
    dY = jnp.zeros_like(Y)
    def S(ci): return Y[:, 1 + ci]
    def D(ci, val): return dY.at[:, 1 + ci].set(val)

    # SHL-1 (Kv4)
    minf = 1.0 / (1.0 + exp(-(v - 11.2 + 18) / 14.1))
    tm = (13.8 / (exp(-(v + 17.5165) / 12.9213) + exp((v + 3.7082) / 6.4876)) + 1.8849) * 0.1
    hinf = 1.0 / (1.0 + exp((v + 33.1 + 18) / 8.3))
    ths = (8422.0 / (1.0 + exp((v + 37.7391) / 6.3785)) + 118.8983) * 0.1
    thf = (539.1584 / (1.0 + exp((v + 28.1990) / 4.9199)) + 27.2811) * 0.1
    dY = D(C_SHAL, (minf - S(C_SHAL)) / tm)
    dY = D(C_HFSHAL, (hinf - S(C_HFSHAL)) / thf)
    dY = D(C_HSSHAL, (hinf - S(C_HSSHAL)) / ths)
    I_shal = Pm[:, PK["gshal"]] * S(C_SHAL)**3 * (0.7 * S(C_HFSHAL) + 0.3 * S(C_HSSHAL)) * (v - EK)
    # SHK-1 (Kv1)
    minf = 1.0 / (1.0 + exp(-(v - 20.4) / 7.7))
    tm = 26.571450568169027 / (exp(-(v + 33.741611800716130) / 15.364937728953288)
                               + exp((v + 33.741611800716130) / 15.757936311607475)) + 1.990037272604829
    hinf = 1.0 / (1.0 + exp((v + 6.95) / 5.8))
    dY = D(C_SHAK, (minf - S(C_SHAK)) / tm)
    dY = D(C_HSHAK, (hinf - S(C_HSHAK)) / 1400.0)
    I_shak = Pm[:, PK["gshak"]] * S(C_SHAK) * S(C_HSHAK) * (v - EK)
    # IRK
    minf = 1.0 / (1.0 + exp((v + 52.0 + 30) / 13.0))
    tm = 17.0752 / (exp(-(v + 17.8258) / 20.3154) + exp((v + 43.4414) / 11.1691)) + 3.8329
    dY = D(C_KIR, (minf - S(C_KIR)) / tm)
    I_kir = Pm[:, PK["gkir"]] * S(C_KIR) * (v - EK)

    I_kvs1 = I_kqt3 = I_egl2 = I_egl36 = jnp.zeros_like(v)
    if is_awc:
        # KVS-1
        minf = 1.0 / (1.0 + exp(-(v - 57.1 + 30) / 25.0))
        tm = (30.0 / (1.0 + exp(-(v - 18.1232) / -20.0)) + 1.0) / 10.0
        hinf = 1.0 / (1.0 + exp((v - 47.3 + 30) / 11.1))
        th = (88.4715 / (1.0 + exp(-(v - 50.0) / -15.0)) + 53.4060) * 0.1
        dY = D(C_KVS1, (minf - S(C_KVS1)) / tm)
        dY = D(C_HKVS1, (hinf - S(C_HKVS1)) / th)
        I_kvs1 = Pm[:, PK["gkvs1"]] * S(C_KVS1) * S(C_HKVS1) * (v - EK)
        # KQT-3 (Kv7)
        minf = 1.0 / (1.0 + exp(-(v + 12.6726 + 10) / 15.8008))
        tmf = (395.3 / (1.0 + ((v + 38.1) / 33.59)**2)) * 0.1
        tms = (5503.0 - 5345.4 / (1.0 + 10.0**(-0.02827 * (-23.9 - v)))
               - 4590.6 / (1.0 + 10.0**(-0.0357 * (v + 14.15)))) * 0.1
        winf = 0.49 + 0.51 / (1.0 + exp((v + 1.084) / 28.78))
        tw = (5.44 + 29.2 / (1.0 + ((v + 48.09) / 48.83)**2)) * 0.1
        sinf = 0.34 + 0.66 / (1.0 + exp((v + 45.3) / 12.3))
        dY = D(C_MFKQT3, (minf - S(C_MFKQT3)) / tmf)
        dY = D(C_MSKQT3, (minf - S(C_MSKQT3)) / tms)
        dY = D(C_WKQT3, (winf - S(C_WKQT3)) / tw)
        dY = D(C_SKQT3, (sinf - S(C_SKQT3)) / 500.0)
        I_kqt3 = Pm[:, PK["gkqt3"]] * (0.3 * S(C_MFKQT3) + 0.7 * S(C_MSKQT3)) * S(C_WKQT3) * S(C_SKQT3) * (v - EK)
        # EGL-2 (Kv11/eag)
        minf = 1.0 / (1.0 + exp(-(v + 6.8594) / 14.9131))
        tm = ((16.78 / (1.0 + exp((v + 122.5682) / 13.7976))) + 8.0969) * 0.5
        dY = D(C_EGL2, (minf - S(C_EGL2)) / tm)
        I_egl2 = Pm[:, PK["gegl2"]] * S(C_EGL2) * (v - EK)
    else:
        # EGL-36 (Kv3)
        minf = 1.0 / (1.0 + exp(-(v - 63.0) / 28.5))
        dY = D(C_M1E36, (minf - S(C_M1E36)) / 355.0)
        dY = D(C_M2E36, (minf - S(C_M2E36)) / 63.0)
        dY = D(C_M3E36, (minf - S(C_M3E36)) / 13.0)
        I_egl36 = Pm[:, PK["gegl36"]] * (0.31 * S(C_M1E36) + 0.36 * S(C_M2E36) + 0.39 * S(C_M3E36)) * (v - EK)

    # CCA-1 (CaV3/T)
    minf = 1.0 / (1.0 + exp(-(v + 42.65 + 15) / (1.7 * 1.4)))
    tm = ((40.0 / (1.0 + exp(-(v + 62.5393 + 30) / (-12.4758 * 1.7)))) + 0.6947) * 0.5
    hinf = 1.0 / (1.0 + exp((v + 58.0 + 15) / (7.0 * 1.15)))
    th = ((280.0 / (1.0 + exp((v + 60.7312 + 15) / (8.5224 * 1.1)))) + 19.7456) * 0.08
    dY = D(C_CCA1, (minf - S(C_CCA1)) / tm)
    dY = D(C_HCCA1, (hinf - S(C_HCCA1)) / th)
    I_cca1 = Pm[:, PK["gcca1"]] * S(C_CCA1)**2 * S(C_HCCA1) * (v - ECA)
    # UNC-2 (CaV2 P/Q)
    minf_u = 1.0 / (1.0 + exp(-(v + 12.17 + 25) / 3.97))
    tm_u = (1.4969 / (exp(-(v + 8.1761 + 30) / 9.0753) + exp((v + 8.1761 + 30) / 15.3456)) + 0.1029) * 3.0
    hinf_u = 1.0 / (1.0 + exp((v + 52.47 + 25) / 5.6))
    th_u = (83.8037 / (1.0 + exp((v - 52.8997 + 30) / 3.4557))
            + 72.0995 / (1.0 + exp(-(v - 23.9009 + 30) / 3.5903))) * 1.7
    dY = D(C_UNC2, (minf_u - S(C_UNC2)) / tm_u)
    dY = D(C_HUNC2, (hinf_u - S(C_HUNC2)) / th_u)
    I_unc2 = Pm[:, PK["gunc2"]] * S(C_UNC2) * S(C_HUNC2) * (v - ECA)
    alpha_u = minf_u / tm_u; beta_u = 1.0 / tm_u - alpha_u
    # EGL-19 (CaV1 L)
    minf_e = 1.0 / (1.0 + exp(-(v - 5.6 + 10) / 7.5))
    tm_e = (2.3359 + 2.9324 * exp(-((v - 5.2357 + 10)**2) / 36.0)
            + 1.8739 * exp(-((v - 1.3930 + 10)**2) / 900.0))
    hinf_e = ((1.4314 / (1.0 + exp(-(v - 24.8573 + 10) / 11.9541)) + 0.1427)
              * (5.9589 / (1.0 + exp((v + 10.5428 + 10) / 8.0552)) + 0.6038))
    ths_e = 0.4 * ((0.55 * 81.1179) / (1.0 + exp((v + 22.9723 + 10) / 5.0)) + 43.0937
                   + (0.9 * 40.4885) / (1.0 + exp((v - 28.7251 + 10) / 3.7125)) + 0.0)
    dY = D(C_EGL19, (minf_e - S(C_EGL19)) / tm_e)
    dY = D(C_HSEGL19, (hinf_e - S(C_HSEGL19)) / ths_e)
    I_egl19 = Pm[:, PK["gegl19"]] * S(C_EGL19) * S(C_HSEGL19) * (v - ECA)
    alpha_e = minf_e / tm_e; beta_e = 1.0 / tm_e - alpha_e

    # Ca 纳米域 + SLO1/SLO2
    r_, d_, F_, kb_, b_, gsc_ = 13e-9, 250e-12, 96485.0, 500e6, 30e-6, 40e-12
    cao = ((jnp.abs(gsc_ * (v - ECA) * 1e-3) / (8 * jnp.pi * r_ * d_ * F_))
           * exp(-r_ / jnp.sqrt(d_ / (kb_ * b_))) * 1e6 * 1e-3) + 0.05
    cac = 0.05
    wom, wyx, kyx, nyx = 3.152961, 0.012643, 34.338784, 0.000100
    wop, wxy, kxy, nxy = 0.156217, -0.027527, 55.726816, 1.299198
    kcm = wom * exp(-wyx * v) / (1.0 + (cac / kyx)**nyx)
    kom = wom * exp(-wyx * v) / (1.0 + (cao / kyx)**nyx)
    kop = wop * exp(-wxy * v) / (1.0 + (kxy / cao)**nxy)
    den1 = (kop + kom) * (kcm + alpha_u) + beta_u * kcm
    minf_bk = (S(C_UNC2) * kop * (alpha_u + beta_u + kcm)) / den1
    tm_bk = (alpha_u + beta_u + kcm) / den1
    dY = D(C_MBK, (minf_bk - S(C_MBK)) / tm_bk)
    I_bk = Pm[:, PK["gbk"]] * S(C_MBK) * S(C_HUNC2) * (v - EK)
    den2 = (kop + kom) * (kcm + alpha_e) + beta_e * kcm
    minf_s1 = (S(C_EGL19) * kop * (alpha_e + beta_e + kcm)) / den2
    tm_s1 = (alpha_e + beta_e + kcm) / den2
    dY = D(C_MSLO1, (minf_s1 - S(C_MSLO1)) / tm_s1)
    I_slo1 = Pm[:, PK["gslo1"]] * S(C_MSLO1) * S(C_HSEGL19) * (v - EK)
    wom1, wyx1, kyx1, nyx1 = 0.896395, 0.019405, 3294.553404, 0.000010
    wop1, wxy1, kxy1, nxy1 = 0.026719, -0.024123, 93.449423, 1.835067
    kcm1 = wom1 * exp(-wyx1 * v) / (1.0 + (cac / kyx1)**nyx1)
    kom1 = wom1 * exp(-wyx1 * v) / (1.0 + (cao / kyx1)**nyx1)
    kop1 = wop1 * exp(-wxy1 * v) / (1.0 + (kxy1 / cao)**nxy1)
    den3 = (kop1 + kom1) * (kcm1 + alpha_e) + beta_e * kcm1
    minf_b2 = (S(C_EGL19) * kop1 * (alpha_e + beta_e + kcm1)) / den3
    tm_b2 = (alpha_e + beta_e + kcm1) / den3
    dY = D(C_MBK2, (minf_b2 - S(C_MBK2)) / tm_b2)
    I_bk2 = Pm[:, PK["gbk2"]] * S(C_MBK2) * S(C_HSEGL19) * (v - EK)
    den4 = (kop1 + kom1) * (kcm1 + alpha_u) + beta_u * kcm1
    minf_s2 = (S(C_UNC2) * kop1 * (alpha_u + beta_u + kcm1)) / den4
    tm_s2 = (alpha_u + beta_u + kcm1) / den4
    dY = D(C_MSLO2, (minf_s2 - S(C_MSLO2)) / tm_s2)
    I_slo2 = Pm[:, PK["gslo2"]] * S(C_MSLO2) * S(C_HUNC2) * (v - EK)

    # 胞内钙 + KCNL
    I_ca = I_unc2 + I_egl19 + I_cca1
    alpha_ca = 1.0 / (2.0 * Pm[:, PK["vol"]] * 96.485)
    ca = S(C_CA)
    rs = jnp.where(I_ca < 0, -0.001 * alpha_ca * I_ca - (ca - 5e-5) / 50.0, (5e-5 - ca) / 50.0)
    dY = D(C_CA, rs)
    minf_sk = ca / (0.00033 + ca)
    dY = D(C_MSK, (minf_sk - S(C_MSK)) / 6.3)
    I_sk = Pm[:, PK["gsk"]] * S(C_MSK) * (v - EK)

    I_leak = Pm[:, PK["gleak"]] * (v - Pm[:, PK["ELEAK"]])
    I_nca = Pm[:, PK["gnca"]] * (v - ENA)
    I_tot = (I_shal + I_shak + I_kir + I_kvs1 + I_kqt3 + I_egl2 + I_egl36
             + I_cca1 + I_unc2 + I_egl19 + I_bk + I_slo1 + I_bk2 + I_slo2 + I_sk + I_leak + I_nca)
    dY = dY.at[:, 0].set((0.0 - I_tot) / Pm[:, PK["C"]])   # Iext 在步进函数中补
    return dY



def build_eval(P, T_ticks, ctrl_hz=30, fwd_idx=(), n2w_idx=()):
    """构造个体评估函数: theta(11,) -> (acts, dm) 各 (T_ticks-excwarm, 96)。
    驱动/钳制/肌肉逻辑逐项镜像 w54_redistill.eval_theta (CTRL_HZ=30 配置)。
    返回的函数可被 jit / vmap。"""
    N = P["N"]; dt = P["dt"]
    NSTEPS = int(round(1000.0 / ctrl_hz / dt))
    warm_ticks = int(round(3.0 * ctrl_hz)); excwarm = int(round(0.5 * ctrl_hz))
    ref_ticks = float(int(round(300 * 1.5 * ctrl_hz / 1000.0)))
    exc_decay = (1000.0 / ctrl_hz) / 300.0
    Pawc = jnp.asarray(P["Pawc"]); Prmd = jnp.asarray(P["Prmd"])
    Y0awc = jnp.asarray(P["Y0awc"]); Y0rmd = jnp.asarray(P["Y0rmd"])
    rows_awc = jnp.asarray(P["rows_awc"]); rows_rmd = jnp.asarray(P["rows_rmd"])
    vrel = jnp.asarray(P["vrel"])
    Mex_u = [jnp.asarray(M) for M in P["Mex_u"]]
    Min_u = [jnp.asarray(M) for M in P["Min_u"]]
    Lg_u = [jnp.asarray(M) for M in P["Lg_u"]]
    wout = jnp.asarray(P["wout"]); inhib = jnp.asarray(P["inhib"]); n_inn = jnp.asarray(P["n_inn"])
    signed = jnp.where(inhib > 0, -1.0, 1.0)
    fwd = jnp.asarray(list(fwd_idx), jnp.int64) if len(fwd_idx) else None
    vclip = P["vclip"]; e_inh = P["e_inh"]

    # n2w 钳制细胞的组属与组内行号
    n2w = list(n2w_idx)
    has_n2w = len(n2w) > 0
    if has_n2w:
        owner_lay = np.zeros(N, np.int64); owner_row = np.zeros(N, np.int64)
        owner_lay[np.asarray(P["rows_awc"])] = 0
        owner_row[np.asarray(P["rows_awc"])] = np.arange(len(P["rows_awc"]))
        owner_lay[np.asarray(P["rows_rmd"])] = 1
        owner_row[np.asarray(P["rows_rmd"])] = np.arange(len(P["rows_rmd"]))
        n2w_lay = jnp.asarray(owner_lay[n2w], jnp.int64)
        n2w_row = jnp.asarray(owner_row[n2w], jnp.int64)
        kn2w = len(n2w)

    def step_group(Y, Pm, is_awc, Iext_rows):
        v = Y[:, 0]
        dY = _rhs(v, Y, Pm, is_awc)
        dY = dY.at[:, 0].add(Iext_rows / Pm[:, PK["C"]])
        Y2 = Y + dt * dY
        if vclip is not None:
            Y2 = Y2.at[:, 0].set(jnp.clip(Y2[:, 0], vclip[0], vclip[1]))
        return Y2

    def eval_one(theta):
        th = jnp.asarray(theta, jnp.float64)
        e = jnp.exp(th)
        Mex = P["G_SYN"] * (sum(e[i] * Mex_u[i] for i in range(len(Mex_u) - 1)) + e[6] * Mex_u[-1])
        Min = P["G_INH"] * (sum(e[i] * Min_u[i] for i in range(len(Min_u) - 1)) + e[7] * Min_u[-1])
        Lg = P["G_GAP"] * (sum(e[4 + j] * Lg_u[j] for j in range(len(Lg_u) - 1)) + e[8] * Lg_u[-1])
        fwdi = 0.1517 * e[9]
        excth = 0.0815 + 0.03 * th[10]

        def one_substep(c, _):
            Yawc_, Yrmd_, gA_, gI_, Iex, do_clamp, clamp_vals = c
            V = jnp.zeros(N, jnp.float64).at[rows_awc].set(Yawc_[:, 0]).at[rows_rmd].set(Yrmd_[:, 0])
            sV = jnp.clip((V - vrel) / V_SCALE, 0.0, S_MAX)
            dA = Mex @ sV
            dI = Min @ sV
            gA_ = gA_ + dt * (-gA_ / TAU_EX + dA)
            gI_ = gI_ + dt * (-gI_ / TAU_INH + dI)
            I_gap = Lg @ V
            I_syn = gA_ * (V - E_EX) + gI_ * (V - e_inh)
            I_tot = -I_syn - I_gap + Iex
            Yawc_ = step_group(Yawc_, Pawc, True, I_tot[rows_awc])
            Yrmd_ = step_group(Yrmd_, Prmd, False, I_tot[rows_rmd])
            if has_n2w:
                Yawc_ = Yawc_.at[n2w_row, 0].set(
                    jnp.where((n2w_lay == 0) & do_clamp, clamp_vals, Yawc_[n2w_row, 0]))
                Yrmd_ = Yrmd_.at[n2w_row, 0].set(
                    jnp.where((n2w_lay == 1) & do_clamp, clamp_vals, Yrmd_[n2w_row, 0]))
            return (Yawc_, Yrmd_, gA_, gI_, Iex, do_clamp, clamp_vals), None

        def one_tick(c, tick):
            Yawc_, Yrmd_, gA_, gI_, musc_a, musc_ref = c
            V0 = jnp.zeros(N, jnp.float64).at[rows_awc].set(Yawc_[:, 0]).at[rows_rmd].set(Yrmd_[:, 0])
            Iex = jnp.zeros(N, jnp.float64)
            if fwd is not None:
                Iex = Iex.at[fwd].set(fwdi * (-50.0 - V0[fwd]))
            v_dr = -45.0 + 25.0 * jnp.sin(2.0 * jnp.pi * tick / ctrl_hz)
            do_clamp = tick < warm_ticks
            if has_n2w:
                clamp_vals = jnp.full((kn2w,), v_dr, jnp.float64)
            else:
                clamp_vals = jnp.zeros(0, jnp.float64)
            c0 = (Yawc_, Yrmd_, gA_, gI_, Iex, do_clamp, clamp_vals)
            (Yawc_, Yrmd_, gA_, gI_, _, _, _), _ = lax.scan(one_substep, c0, None, length=NSTEPS)
            V = jnp.zeros(N, jnp.float64).at[rows_awc].set(Yawc_[:, 0]).at[rows_rmd].set(Yrmd_[:, 0])
            drive = (V + 80.0) / 100.0
            dm = (signed * drive) @ wout / n_inn
            trig = (musc_ref <= 0) & (dm > excth)
            new_a = jnp.where(trig, 1.0, musc_a)
            new_ref = jnp.where(trig, ref_ticks, musc_ref)
            new_a = new_a + exc_decay * (0.0 - new_a)
            new_ref = jnp.where(new_ref > 0, new_ref - 1.0, new_ref)
            do_musc = tick >= excwarm                       # eval_theta: tick<excwarm 时肌肉逻辑冻结, 记录 0
            musc_a = jnp.where(do_musc, new_a, musc_a)
            musc_ref = jnp.where(do_musc, new_ref, musc_ref)
            act_row = jnp.where(do_musc, jnp.clip(new_a, 0.0, 0.8), jnp.zeros(96, jnp.float64))
            dm_row = jnp.where(do_musc, dm, jnp.zeros(96, jnp.float64))
            return (Yawc_, Yrmd_, gA_, gI_, musc_a, musc_ref), (act_row, dm_row)

        c0 = (Y0awc, Y0rmd, jnp.zeros(N, jnp.float64), jnp.zeros(N, jnp.float64),
              jnp.zeros(96, jnp.float64), jnp.zeros(96, jnp.float64))
        _, (acts, dms) = lax.scan(one_tick, c0, jnp.arange(T_ticks))
        return acts[excwarm:], dms[excwarm:]

    return eval_one


def build_batch(P, T_ticks, ctrl_hz=30, fwd_idx=(), n2w_idx=()):
    """jit(vmap(eval_one)): thetas (B,11) -> acts (B,T,96), dm (B,T,96)"""
    ev = build_eval(P, T_ticks, ctrl_hz, fwd_idx, n2w_idx)
    return jax.jit(jax.vmap(ev))


def build_step(P, fwd_idx=(), n2w_idx=(), ctrl_hz=30):
    """暴露单 dt 步进 (诊断/闭环用): 固定 Mex/Min/Lg 由 theta 组装一次传入。
    返回 jit step(state, Mex, Min, Lg, fwdi, do_clamp, clamp_v) -> state。
    state = (Yawc, Yrmd, gA, gI)。数学与 one_substep 逐项相同。"""
    N = P["N"]; dt = P["dt"]
    Pawc = jnp.asarray(P["Pawc"]); Prmd = jnp.asarray(P["Prmd"])
    rows_awc = jnp.asarray(P["rows_awc"]); rows_rmd = jnp.asarray(P["rows_rmd"])
    vrel = jnp.asarray(P["vrel"])
    vclip = P["vclip"]; e_inh = P["e_inh"]
    fwd = jnp.asarray(list(fwd_idx), jnp.int64) if len(fwd_idx) else None
    n2w = list(n2w_idx)
    has_n2w = len(n2w) > 0
    if has_n2w:
        owner_lay = np.zeros(N, np.int64); owner_row = np.zeros(N, np.int64)
        owner_lay[np.asarray(P["rows_awc"])] = 0
        owner_row[np.asarray(P["rows_awc"])] = np.arange(len(P["rows_awc"]))
        owner_lay[np.asarray(P["rows_rmd"])] = 1
        owner_row[np.asarray(P["rows_rmd"])] = np.arange(len(P["rows_rmd"]))
        n2w_lay = jnp.asarray(owner_lay[n2w], jnp.int64)
        n2w_row = jnp.asarray(owner_row[n2w], jnp.int64)
        kn2w = len(n2w)

    def step_group(Y, Pm, is_awc, Iext_rows):
        v = Y[:, 0]
        dY = _rhs(v, Y, Pm, is_awc)
        dY = dY.at[:, 0].add(Iext_rows / Pm[:, PK["C"]])
        Y2 = Y + dt * dY
        if vclip is not None:
            Y2 = Y2.at[:, 0].set(jnp.clip(Y2[:, 0], vclip[0], vclip[1]))
        return Y2

    def step(state, Mex, Min, Lg, Iex, do_clamp, clamp_vals):
        Yawc_, Yrmd_, gA_, gI_ = state
        V = jnp.zeros(N, jnp.float64).at[rows_awc].set(Yawc_[:, 0]).at[rows_rmd].set(Yrmd_[:, 0])
        sV = jnp.clip((V - vrel) / V_SCALE, 0.0, S_MAX)
        dA = Mex @ sV
        dI = Min @ sV
        gA_ = gA_ + dt * (-gA_ / TAU_EX + dA)
        gI_ = gI_ + dt * (-gI_ / TAU_INH + dI)
        I_gap = Lg @ V
        I_syn = gA_ * (V - E_EX) + gI_ * (V - e_inh)
        I_tot = -I_syn - I_gap + Iex
        Yawc_ = step_group(Yawc_, Pawc, True, I_tot[rows_awc])
        Yrmd_ = step_group(Yrmd_, Prmd, False, I_tot[rows_rmd])
        if has_n2w:
            Yawc_ = Yawc_.at[n2w_row, 0].set(
                jnp.where((n2w_lay == 0) & do_clamp, clamp_vals, Yawc_[n2w_row, 0]))
            Yrmd_ = Yrmd_.at[n2w_row, 0].set(
                jnp.where((n2w_lay == 1) & do_clamp, clamp_vals, Yrmd_[n2w_row, 0]))
        return (Yawc_, Yrmd_, gA_, gI_)

    def assemble_V(Yawc_, Yrmd_):
        return jnp.zeros(N, jnp.float64).at[rows_awc].set(Yawc_[:, 0]).at[rows_rmd].set(Yrmd_[:, 0])

    return step, assemble_V
