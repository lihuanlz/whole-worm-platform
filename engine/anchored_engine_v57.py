# anchored_engine_v57.py
# 【v5.7 = v5.6 + W09 UNC-2 固有振荡臂（2026-10-10）：
#   A/B 类兴奋性运动神经元 (VA/VB/DA/DB) 的 gunc2 由模块全局 GUNC2_AB 覆盖。
#   依据：Gao et al. 2018 eLife e29915 —— UNC-2 (P/Q/N 型 HVA 钙通道) 是
#   A 类运动神经元固有钙振荡的必要成分（unc-2(lf) 振幅频率双降、unc-2(gf) 双升、
#   unc-13 背景下振荡仍在=细胞自主）；Xu et al. 2018 PNAS —— AVB→B 电突触把
#   B 类推过振荡阈值，缺之则弯曲幅度向尾单调衰减（=我们 W08 的"波不传"症状）。
#   Nicoletti 电压钳拟合里 VA/DA 的 gunc2=0 是钳制态欠定方向，W09 加回。
#   负对照设计：VD/AVA/AWC/RMD 不覆盖；GUNC2_AB=0 时与 v56 逐值相同。
#   v56 文件逐字节不动。】
import anchored_engine_v56 as _v56

GUNC2_AB = 0.0   # nS; 调用方在构造网络前设置 (w01_closed_loop_w09.py 读环境变量 W09_GUNC2AB)
GNCA_A = -1.0    # nS; >=0 时覆盖 A 类 (VA/DA) 的 gnca (v84: Gao 2018 + Frontiers 2022
                 # NCA Na+ leak 是 A-MN rPSC/固有振荡的必要去极化引擎; 拟合 VA/DA gnca=0.01
                 # 全身最低 -> 孤立平衡点 -79mV 远在 UNC-2 HVA 激活域之下 -> 无自振。
                 # Nicoletti 钳制数据对 leak 分量欠定, 与 GUNC2_AB 同性质补回)

_AB_CLASSES = {"VA", "VB", "DA", "DB"}
_A_CLASSES = {"VA", "DA"}
_GABA_CLASSES = {"DD", "VD"}

GLEAK_GABA = 1.0   # 倍数; !=1.0 时缩放 DD/VD 的 gleak (v86: C2.6 eLife 2021 PMC7986531
                   # 锚定 VD/DD 钙信号随爬行节律振荡——实测模型 DD 被强直性饱和释放
                   # (vrel 原位表 tonic 饱和区) 钉在 -18mV 高电导区, 电压钳死不振荡,
                   # 交叉抑制相位分量死亡 -> 腹侧过驱动 -> 爬行静态腹弯左拐。
                   # Nicoletti 孤立钳制拟合的 gleak=0.4/0.2 未含在位强直驱动,
                   # 钳制态欠定方向, 按 GUNC2_AB 同纪律补回; 1.0=逐位不变)

GEGL36_GABA = -1.0  # nS; >=0 时覆盖 DD/VD (ctype=VD) 的 gegl36 (v87: C2.6 同锚。
                    # VD 配方 gegl36=0 是 Nicoletti 2024 钳制拟合的欠定方向——
                    # 单细胞探针实测 (probe_dd_vd_load/egl36_2026-10-21): 缺 EGL-36 时
                    # EGL-19 L型平台 (-25..-5mV 负斜率再生区) 无高压 K 复极 ->
                    # 强直驱动下泊上 ~-4mV 平台, 波频调制死亡 (std<1mV)。
                    # EGL-36 高压激活: 静息关闭不伤 VD5 -45.8+-1.9 锚,
                    # >-20mV 激活复极平台。0.6/1.0 两候选均过锚 (iso -46.5/-47.5),
                    # 0=逐位不变负对照)

UNC2PED_A = -1.0   # [0,1); >=0 时设 A 类 (VA/DA) UNC-2 失活门非失活分量 (2026-10-23, W13)
                   # 依据: Gao 2018 eLife e29915 —— A-MN 内源振荡必需 UNC-2 窗流,
                   # unc-2(gf)=减弱失活动力学→振荡增强; CaV2 家族普遍存在非失活成分。
                   # Nicoletti 钳制拟合 hinf 半失活 -77.47mV 使 -38mV 处 h∞≈0.001,
                   # 窗流=0 → 细胞结构上不可能振荡 (探针 p1-p5 判决: 单稳态无磁滞)。
                   # ped=0.20 恢复双稳窗 I∈[0,1.8]pA (probe_amn_refit_s2/s4c 验证:
                   # 静息锚 -72.1 守 / up-state 点得燃 / AVA 分流门控三相全过)
GSK_A = -1.0       # nS; >=0 时设 A 类 KCNL/SK 钙激活钾电导 (2026-10-23, W13)
                   # KCNL 解剖表达但拟合贡献=0=欠定方向; 恢复 0.05 作慢刹车
CATAUSCALE_A = -1.0  # >=0 时设 A 类胞内钙清除时间常数倍数 (2026-10-23, W13)
                     # 50ms=微域清除; 体钙清除秒级 (Gao 2018 DA9 钙振荡周期~50s); 取 40(=2s)
GUNC2_A = -1.0       # nS; >=0 时覆盖 A 类 (VA/DA) gunc2 (2026-10-23, W13)
                     # E2d 实测: 网络内 A-MN 被 37 gap+40 chem 边钳制, GUNC2_AB=0.4 的
                     # 窗流(~10pA)相对网络电导(<1%)可忽略 => 需放到网络尺度 (1.0+)。
                     # Nicoletti 拟合 gunc2=0=欠定方向, 幅值本属自由选择;
                     # 负对照: VB/DB 不覆盖仍走 GUNC2_AB=0.4, -1=逐位不变

GUNC2_AVA = -1.0     # nS; >=0 时覆盖 AVA 的 gunc2 (2026-10-23, W52 AVA再生平台)
                     # 依据: Meng 2024 Sci Adv adk0002 (PMC11006214) — AVA 多稳态去极化
                     # (-30~-10, 两个稳定区制 -30/-20 与 -15/-10), 基线近 UNC-2 开阈
                     # (紧张性 ACh 释放); Nicoletti 拟合 AVA gunc2=0 = 钳制态欠定方向,
                     # 与 GUNC2_AB/GUNC2_A 同纪律补回; minf 半激活 -37mV 使 AVA 基线(-25)
                     # 已在激活区, ped 非失活分量即再生窗流; -1=逐位不变
UNC2PED_AVA = -1.0   # [0,1); >=0 时设 AVA UNC-2 非失活分量 (W52, Gao 2018 窗流锚同 W13)
GSK_AVA = -1.0       # nS; >=0 时设 AVA KCNL/SK 钙激活钾电导 (W52: 秒级自终结刹车;
                     # AVA SK 拟合贡献=0=欠定方向; 体钙2s清除 -> 平台数秒自终结)
CATAUSCALE_AVA = -1.0  # >=0 时设 AVA 胞内钙清除时间常数倍数 (W52: 微域50ms->体钙2s)

_AVA_CLASSES = {"AVA"}

def cell_params(ctype):
    P = dict(_v56.cell_params(ctype))
    if GUNC2_AB > 0.0 and ctype in _AB_CLASSES:
        P["gunc2"] = GUNC2_AB
    if GNCA_A >= 0.0 and ctype in _A_CLASSES:
        P["gnca"] = GNCA_A
    if GLEAK_GABA != 1.0 and ctype in _GABA_CLASSES and "gleak" in P:
        P["gleak"] = P["gleak"] * GLEAK_GABA
    if GEGL36_GABA >= 0.0 and ctype in _GABA_CLASSES and "gegl36" in P:
        P["gegl36"] = GEGL36_GABA
    if ctype in _A_CLASSES:
        if UNC2PED_A >= 0.0:
            P["unc2_hinf_ped"] = UNC2PED_A
        if GSK_A >= 0.0:
            P["gsk"] = GSK_A
        if CATAUSCALE_A >= 0.0:
            P["ca_tau_scale"] = CATAUSCALE_A
        if GUNC2_A >= 0.0:
            P["gunc2"] = GUNC2_A
    if ctype in _AVA_CLASSES:
        if GUNC2_AVA >= 0.0:
            P["gunc2"] = GUNC2_AVA
        if UNC2PED_AVA >= 0.0:
            P["unc2_hinf_ped"] = UNC2PED_AVA
        if GSK_AVA >= 0.0:
            P["gsk"] = GSK_AVA
        if CATAUSCALE_AVA >= 0.0:
            P["ca_tau_scale"] = CATAUSCALE_AVA
    return P

RELEASE_REF = _v56.RELEASE_REF
ANCHOR_REST = _v56.ANCHOR_REST
layout = _v56.layout
init_state = _v56.init_state
init_state56 = _v56.init_state56
