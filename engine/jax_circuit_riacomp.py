# -*- coding: utf-8 -*-
# jax_circuit_riacomp.py — RIA 三区室扩展模块 v2 再生性树突版 (2026-10-19, 漏洞2攻关)
# 机制规格书: 06_蠕虫连线组线/RIA三区室原型_机制规格书_2026-10-19.md
# v2 修订 (本会话实测驱动):
#   v1 被动区室+钙库定量封死(整流效应 0.03 vs 执行基准 0.92, 且真实载体还小20×)。
#   v2 区室 = 冻结 RMD 型 HH 方程按面积比 c_ratio 等比缩放的再生性树突
#   (Liu 2018: RIA 神经突大幅 Ca 信号, UNC-2/EGL-19/CCA-1 在表达谱内)。
#   通道动力学零改动, 零新通道参数; 仅剩几何/分流参数 g_ax, s_acc2 (全文献锚)。
# 结构:
#   RIAL/RIAR 各加 nrD/nrV 区室(扩展状态, 非网络新节点);
#   RIA->SMDD*/SMDV* 输出改由 nrD/nrV 电压驱动; SMDD*/SMDV*->RIA 反馈改投 nrD/nrV;
#   AIY->RIA 抑制留胞体, 经 ACC-2 等效分流(s_acc2)门控区室;
#   交叉路由默认(route=-1): nrD->SMDV, nrV->SMDD (拓扑判决见规格书§3)。
# 纪律: 本模块不被调用时, 冻结引擎逐位不变。
import numpy as np
import jax.numpy as jnp

from jax_circuit import V_SCALE, S_MAX, ECA, E_EX, TAU_EX, TAU_INH, PK, _rhs

# 区室固定锚 (规格书 §4): 区室体积/全细胞 ≈ 0.1 (细神经突解剖估计)
C_RATIO = 0.1

SMDD_SET = ("SMDDL", "SMDDR")
SMDV_SET = ("SMDVL", "SMDVR")
RIA_PAIR = ("RIAL", "RIAR")
AIY_PAIR = ("AIYL", "AIYR")


def build_comp_spec(net_npz):
    """从冻结连接组识别重路由边。返回边索引字典 (chem 行号)。"""
    z = np.load(net_npz, allow_pickle=False)
    names = [str(s) for s in z["neurons"]]
    idx = {n: i for i, n in enumerate(names)}
    chem, sign = z["chem"], z["sign"]
    pre, post = chem[:, 0].astype(int), chem[:, 1].astype(int)
    spec = {"names": names, "idx": idx}
    for ria in RIA_PAIR:
        j = idx[ria]
        for tag, posts, pres in (
                ("OUT_D", SMDD_SET, None), ("OUT_V", SMDV_SET, None),
                ("FB_D", None, SMDD_SET), ("FB_V", None, SMDV_SET)):
            if posts:
                m = (pre == j) & np.isin(post, [idx[p] for p in posts])
            else:
                m = (post == j) & np.isin(pre, [idx[p] for p in pres])
            spec[f"{ria}_{tag}"] = np.nonzero(m)[0]
        m = (post == j) & np.isin(pre, [idx[p] for p in AIY_PAIR]) & (sign < 0)
        spec[f"{ria}_AIY"] = np.nonzero(m)[0]
    for ria in RIA_PAIR:
        for tag in ("OUT_D", "OUT_V", "FB_D", "FB_V", "AIY"):
            e = spec[f"{ria}_{tag}"]
            w = chem[e, 2] if len(e) else []
            print(f"  [spec] {ria} {tag}: {len(e)}条 w={np.round(w, 3).tolist()}")
    return spec


def _dense(N, chem, rows):
    M = np.zeros((N, N))
    if len(rows):
        M[chem[rows, 1].astype(int), chem[rows, 0].astype(int)] += chem[rows, 2]
    return M


def build_riacomp_mats(P, spec, net_npz):
    """构造重路由矩阵包。Mex_m 已剔除重路由边, 调用方用它替换原 Mex。"""
    z = np.load(net_npz, allow_pickle=False)
    chem = z["chem"]
    N = P["N"]
    g_ex = float(P["G_SYN"])
    idx = spec["idx"]
    ria_rows = np.array([idx[r] for r in RIA_PAIR], np.int64)

    out_d = sum((_dense(N, chem, spec[f"{r}_OUT_D"]) for r in RIA_PAIR), np.zeros((N, N)))
    out_v = sum((_dense(N, chem, spec[f"{r}_OUT_V"]) for r in RIA_PAIR), np.zeros((N, N)))
    fb_d = sum((_dense(N, chem, spec[f"{r}_FB_D"]) for r in RIA_PAIR), np.zeros((N, N)))
    fb_v = sum((_dense(N, chem, spec[f"{r}_FB_V"]) for r in RIA_PAIR), np.zeros((N, N)))
    aiy = sum((_dense(N, chem, spec[f"{r}_AIY"]) for r in RIA_PAIR), np.zeros((N, N)))

    Mex_full = P["Mex_u"][0]
    Mex_m = Mex_full - out_d - out_v - fb_d - fb_v
    # 区室 HH 参数: RMD 行模板, 全部电导/C/vol 按 c_ratio 缩放 (动力学逐位不变)
    Prmd0 = np.asarray(P["Prmd"][0], float).copy()
    for k in ("gshal", "gshak", "gkir", "gkvs1", "gkqt3", "gegl2", "gegl36",
              "gcca1", "gunc2", "gegl19", "gbk", "gslo1", "gbk2", "gslo2",
              "gsk", "gleak", "gnca", "vol", "C"):
        Prmd0[PK[k]] *= C_RATIO
    Pc = np.tile(Prmd0, (4, 1))
    Y0c = np.tile(np.asarray(P["Y0rmd"][0], float), (4, 1))
    return dict(
        Mex_m=jnp.asarray(g_ex * Mex_m),
        OUT_D=jnp.asarray(g_ex * out_d), OUT_V=jnp.asarray(g_ex * out_v),
        FB_D=jnp.asarray(g_ex * fb_d), FB_V=jnp.asarray(g_ex * fb_v),
        AIY_M=jnp.asarray(float(P["G_INH"]) * aiy),
        ria_rows=jnp.asarray(ria_rows),
        vrel_ria=jnp.asarray(P["vrel"][ria_rows]),
        Pc=jnp.asarray(Pc), Y0c=jnp.asarray(Y0c),
    )


# 区室参数(全部文献锚, 规格书 §4):
#   g_ax(nS) 轴突耦合电导: 细神经突电缆估计 0.005-0.5nS, 默认 0.05
#   s_acc2(无量纲) ACC-2 感觉分流增益 (Liu 2018 感觉抑制运动编码)
#   route: +1 同侧路由 / -1 交叉路由(默认, 拓扑判决)
CP_DEFAULT = dict(g_ax=0.05, s_acc2=5.0, route=-1.0)

# ---------------------------------------------------------------------------
# v3 钙库局部释放门控版 (2026-10-19, 账本§64)
# 机制转向依据 (Hendricks 2012 Nature 487:99, PMC3393794 本会话全文核对):
#   - 线虫神经元等电位(isopotential), 区室化钙信号非电压机制;
#   - SMD(ACh) -> RIA 区室 GAR-3(mAChR, Gαq) -> EGL-8(PLCβ) -> ER 钙库动员,
#     局部 Ca 直接调制局部 dyadic 突触释放, 胞体电压几乎不动;
#   - 区室(运动反馈)与同步化(感觉)钙信号可加性叠加;
#   - gar-3 敲除 -> 头摆加深(负反馈限幅功能) -> 区室输出=负反馈。
# v3 与 v2 的区别: 区室不再携带 HH 电压; 活性变量 = 慢钙库 cstore(4,),
#   释放 = 胞体电压驱动基线 + k_mod*cstore (加性, Hendricks Fig1d 两类信号幅度可比
#   -> k_mod 按 O(1) 锚定: k_mod = rel_typ/c_typ, rig 预热期实测, 敏感性 ×{0.3,1,3} 披露);
#   ACC-2 感觉分流改为对 GAR-3 驱动的除法性门控 fbc/(1+s_acc2*gIaiy) (Liu 2018
#   感觉门控运动编码的氯分流等效)。
#   tau_store: GPCR->ER 钙库慢时间尺度, 文献量级锚 (GCaMP3 瞬变秒级, 5Hz 成像可见,
#   默认 1.0s, 扫描 0.3-3s 披露)。
# v3b 修订 (§65): 区室边输出律改为局部钙库主导 (soma_add=0):
#   Hendricks 2012 原文 "We did not detect any prominent calcium response in the RIA
#   cell body" —— 胞体钙静默而区室钙满幅摆动 -> 区室 dyadic 突触释放由局部钙库主导;
#   v3a 的 soma+store 加性合成在 S_MAX=1.5 硬饱和下永久顶死 (基线2.39>1.5),
#   调制传不到 SMD (实现缺陷, 非生物结论)。soma_add=1 保留 v3a 供对照。
#   k_mod 锚定改为满幅标度: k_mod=S_MAX/c_typ (区室瞬变占满释放量程, Hendricks
#   Fig1d 0-100% 归一化摆动); s_acc2 锚定 Liu 2018 Fig4I/K+Fig5F 感觉压制深度
#   ~55-65%: s_acc2=1.5/gIaiy_peak (rig 实测 12.7)。
# 纪律: 本模块不被调用时, 冻结引擎逐位不变。
CP3_DEFAULT = dict(tau_store=1.0, k_mod=1.0, s_acc2=5.0, route=-1.0, soma_add=0.0)


def build_step_riacomp(P, spec, mats, cp=None, ctrl_hz=30):
    """RIA 三区室(再生性树突)版单 dt 步进。
    state=(Yawc,Yrmd,gA,gI,Yc(4,29),gAc(4),gIaiy(2))。
    Yc 行序: [RIAL_nD, RIAL_nV, RIAR_nD, RIAR_nV]; Yc[:,0]=Vc。"""
    cp = dict(CP_DEFAULT if cp is None else cp)
    N = P["N"]; dt = P["dt"]
    rows_awc = jnp.asarray(P["rows_awc"]); rows_rmd = jnp.asarray(P["rows_rmd"])
    vrel = jnp.asarray(P["vrel"])
    Pawc = jnp.asarray(P["Pawc"]); Prmd = jnp.asarray(P["Prmd"])
    vclip = P["vclip"]; e_inh = P["e_inh"]

    OUT_D, OUT_V = mats["OUT_D"], mats["OUT_V"]
    FB_D, FB_V = mats["FB_D"], mats["FB_V"]
    AIY_M = mats["AIY_M"]
    ria_rows = mats["ria_rows"]
    vrel_ria = mats["vrel_ria"]
    Pc = mats["Pc"]

    g_ax = cp["g_ax"]; s_acc2 = cp["s_acc2"]
    soma_of = jnp.asarray([ria_rows[0], ria_rows[0], ria_rows[1], ria_rows[1]], jnp.int64)
    rd_rows = jnp.asarray([ria_rows[0], ria_rows[1]], jnp.int64)

    def step(state, Mex_m, Min, Lg, Iex, do_clamp, clamp_vals):
        Yawc_, Yrmd_, gA_, gI_, Yc_, gAc_, gIaiy_ = state
        Vc_ = Yc_[:, 0]
        V = jnp.zeros(N, jnp.float64).at[rows_awc].set(Yawc_[:, 0]).at[rows_rmd].set(Yrmd_[:, 0])
        sV = jnp.clip((V - vrel) / V_SCALE, 0.0, S_MAX)
        relc = jnp.clip((Vc_ - jnp.repeat(vrel_ria, 2)) / V_SCALE, 0.0, S_MAX)
        relD = jnp.zeros(N, jnp.float64).at[rd_rows].set(relc[jnp.asarray([0, 2])])
        relV = jnp.zeros(N, jnp.float64).at[rd_rows].set(relc[jnp.asarray([1, 3])])
        # route=+1 同侧(nrD 驱 SMDD), route=-1 交叉(nrD 驱 SMDV)
        r_out_d = jnp.where(cp["route"] > 0, OUT_D @ relD, OUT_D @ relV)
        r_out_v = jnp.where(cp["route"] > 0, OUT_V @ relV, OUT_V @ relD)
        # 主网络 (与 build_step 逐项相同)
        dA = Mex_m @ sV + r_out_d + r_out_v
        dI = Min @ sV
        gA_ = gA_ + dt * (-gA_ / TAU_EX + dA)
        gI_ = gI_ + dt * (-gI_ / TAU_INH + dI)
        I_gap = Lg @ V
        I_syn = gA_ * (V - E_EX) + gI_ * (V - e_inh)
        I_tot = -I_syn - I_gap + Iex
        dYa = _rhs(Yawc_[:, 0], Yawc_, Pawc, True)
        dYa = dYa.at[:, 0].add(I_tot[rows_awc] / Pawc[:, PK["C"]])
        Yawc2 = Yawc_ + dt * dYa
        dYr = _rhs(Yrmd_[:, 0], Yrmd_, Prmd, False)
        dYr = dYr.at[:, 0].add(I_tot[rows_rmd] / Prmd[:, PK["C"]])
        Yrmd2 = Yrmd_ + dt * dYr
        if vclip is not None:
            Yawc2 = Yawc2.at[:, 0].set(jnp.clip(Yawc2[:, 0], vclip[0], vclip[1]))
            Yrmd2 = Yrmd2.at[:, 0].set(jnp.clip(Yrmd2[:, 0], vclip[0], vclip[1]))
        # ---- 区室 (主动 HH, RMD 型缩放) ----
        fb = jnp.concatenate([(FB_D @ sV)[ria_rows], (FB_V @ sV)[ria_rows]])
        fbc = fb[jnp.asarray([0, 2, 1, 3])]      # 匹配 Yc 行序 [L_D,L_V,R_D,R_V]
        gAc2 = gAc_ + dt * (-gAc_ / TAU_EX + fbc)
        aiy_in = (AIY_M @ sV)[ria_rows]
        gIaiy2 = gIaiy_ + dt * (-gIaiy_ / TAU_INH + aiy_in)
        gsh = s_acc2 * jnp.repeat(gIaiy2, 2)      # ACC-2 等效分流 (E=ELEAK=-80)
        Vsoma_c = V[soma_of]
        I_syn_c = gAc2 * (Vc_ - E_EX)
        I_ax = g_ax * (Vc_ - Vsoma_c)
        I_shunt = gsh * (Vc_ - (-80.0))
        Iext_c = -I_syn_c - I_ax - I_shunt        # 内流为正 (引擎约定)
        dYc = _rhs(Vc_, Yc_, Pc, False)
        dYc = dYc.at[:, 0].add(Iext_c / Pc[:, PK["C"]])
        Yc2 = Yc_ + dt * dYc
        if vclip is not None:
            Yc2 = Yc2.at[:, 0].set(jnp.clip(Yc2[:, 0], vclip[0], vclip[1]))
        return (Yawc2, Yrmd2, gA_, gI_, Yc2, gAc2, gIaiy2)

    def assemble_V2(Yawc_, Yrmd_):
        return jnp.zeros(N, jnp.float64).at[rows_awc].set(Yawc_[:, 0]).at[rows_rmd].set(Yrmd_[:, 0])

    def init_state2():
        return (jnp.asarray(P["Y0awc"]), jnp.asarray(P["Y0rmd"]),
                jnp.zeros(N, jnp.float64), jnp.zeros(N, jnp.float64),
                mats["Y0c"], jnp.zeros(4, jnp.float64), jnp.zeros(2, jnp.float64))

    return step, assemble_V2, init_state2


def build_step_riacomp3(P, spec, mats, cp=None, ctrl_hz=30):
    """RIA 三区室 v3: 钙库局部释放门控 (Hendricks 2012 机制, 账本§64)。
    state=(Yawc,Yrmd,gA,gI,cstore(4,),gAc(4),gIaiy(2))。
    cstore 行序: [RIAL_nD, RIAL_nV, RIAR_nD, RIAR_nV] (无量纲释放当量)。
    输出律: rel_soma(胞体电压基线) + k_mod*cstore, 逐 RIA 加性, 路由同 v2。"""
    cp = dict(CP3_DEFAULT if cp is None else cp)
    N = P["N"]; dt = P["dt"]
    rows_awc = jnp.asarray(P["rows_awc"]); rows_rmd = jnp.asarray(P["rows_rmd"])
    vrel = jnp.asarray(P["vrel"])
    Pawc = jnp.asarray(P["Pawc"]); Prmd = jnp.asarray(P["Prmd"])
    vclip = P["vclip"]; e_inh = P["e_inh"]

    OUT_D, OUT_V = mats["OUT_D"], mats["OUT_V"]
    FB_D, FB_V = mats["FB_D"], mats["FB_V"]
    AIY_M = mats["AIY_M"]
    ria_rows = mats["ria_rows"]
    vrel_ria = mats["vrel_ria"]

    tau = cp["tau_store"]; k_mod = cp["k_mod"]; s_acc2 = cp["s_acc2"]
    rd_rows = jnp.asarray([ria_rows[0], ria_rows[1]], jnp.int64)

    def step(state, Mex_m, Min, Lg, Iex, do_clamp, clamp_vals):
        Yawc_, Yrmd_, gA_, gI_, c_, gAc_, gIaiy_ = state
        V = jnp.zeros(N, jnp.float64).at[rows_awc].set(Yawc_[:, 0]).at[rows_rmd].set(Yrmd_[:, 0])
        sV = jnp.clip((V - vrel) / V_SCALE, 0.0, S_MAX)
        # 胞体电压基线释放 (等电位 -> 两区室共享胞体基线, Hendricks 可加性)
        rel_soma = jnp.clip((V[ria_rows] - vrel_ria) / V_SCALE, 0.0, S_MAX)
        # v3b: soma_add=0 时区室边由局部钙库主导 (Hendricks 胞体钙静默), 消除 S_MAX 永久饱和
        rel_base = rel_soma * cp["soma_add"]
        cD = c_[jnp.asarray([0, 2])]   # 各 RIA 的 nrD
        cV = c_[jnp.asarray([1, 3])]   # 各 RIA 的 nrV
        relD = jnp.zeros(N, jnp.float64).at[rd_rows].set(jnp.clip(rel_base + k_mod * cD, 0.0, S_MAX))
        relV = jnp.zeros(N, jnp.float64).at[rd_rows].set(jnp.clip(rel_base + k_mod * cV, 0.0, S_MAX))
        # route=+1 同侧(nrD 驱 SMDD), route=-1 交叉(nrD 驱 SMDV)
        r_out_d = jnp.where(cp["route"] > 0, OUT_D @ relD, OUT_D @ relV)
        r_out_v = jnp.where(cp["route"] > 0, OUT_V @ relV, OUT_V @ relD)
        # 主网络 (与 build_step 逐项相同)
        dA = Mex_m @ sV + r_out_d + r_out_v
        dI = Min @ sV
        gA_ = gA_ + dt * (-gA_ / TAU_EX + dA)
        gI_ = gI_ + dt * (-gI_ / TAU_INH + dI)
        I_gap = Lg @ V
        I_syn = gA_ * (V - E_EX) + gI_ * (V - e_inh)
        I_tot = -I_syn - I_gap + Iex
        dYa = _rhs(Yawc_[:, 0], Yawc_, Pawc, True)
        dYa = dYa.at[:, 0].add(I_tot[rows_awc] / Pawc[:, PK["C"]])
        Yawc2 = Yawc_ + dt * dYa
        dYr = _rhs(Yrmd_[:, 0], Yrmd_, Prmd, False)
        dYr = dYr.at[:, 0].add(I_tot[rows_rmd] / Prmd[:, PK["C"]])
        Yrmd2 = Yrmd_ + dt * dYr
        if vclip is not None:
            Yawc2 = Yawc2.at[:, 0].set(jnp.clip(Yawc2[:, 0], vclip[0], vclip[1]))
            Yrmd2 = Yrmd2.at[:, 0].set(jnp.clip(Yrmd2[:, 0], vclip[0], vclip[1]))
        # ---- 区室钙库慢变量 ----
        fb = jnp.concatenate([(FB_D @ sV)[ria_rows], (FB_V @ sV)[ria_rows]])
        fbc = fb[jnp.asarray([0, 2, 1, 3])]      # 行序 [L_D,L_V,R_D,R_V]
        gAc2 = gAc_ + dt * (-gAc_ / TAU_EX + fbc)
        aiy_in = (AIY_M @ sV)[ria_rows]
        gIaiy2 = gIaiy_ + dt * (-gIaiy_ / TAU_INH + aiy_in)
        # ACC-2 除法性分流门控 GAR-3 驱动 (Liu 2018 氯分流等效)
        sh = 1.0 / (1.0 + s_acc2 * jnp.repeat(gIaiy2, 2))
        c2 = c_ + dt * (-c_ / tau + gAc2 * sh)
        c2 = jnp.clip(c2, 0.0, S_MAX)
        return (Yawc2, Yrmd2, gA_, gI_, c2, gAc2, gIaiy2)

    def assemble_V3(Yawc_, Yrmd_):
        return jnp.zeros(N, jnp.float64).at[rows_awc].set(Yawc_[:, 0]).at[rows_rmd].set(Yrmd_[:, 0])

    def init_state3():
        return (jnp.asarray(P["Y0awc"]), jnp.asarray(P["Y0rmd"]),
                jnp.zeros(N, jnp.float64), jnp.zeros(N, jnp.float64),
                jnp.zeros(4, jnp.float64), jnp.zeros(4, jnp.float64),
                jnp.zeros(2, jnp.float64))

    return step, assemble_V3, init_state3
