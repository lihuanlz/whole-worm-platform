# anchored_engine_v56.py
# 【v5.6 = v5.5 + 运动神经元分类配方（W08 锚定臂，2026-10-10）：
#   新增 VA/VB/VD/DA/DB/AVA 六类细胞配方，电导与 E_leak 逐值取自
#   Nicoletti et al. 2024 PLoS Comput Biol (PMC10980225) Table 3
#   ——该表拟合自 Liu et al. 2014 Nat Commun 5:5155 (PMC4197135) 的
#   VA5/VB6/VD5 全细胞实测。静息锚点（实测）：
#     VA5 -71.7±2.4 | VB6 -53.2±2.5 | VD5 -45.8±1.9 mV
#   DA=VA 配方外推（A 类），DB=VB 配方外推（B 类）——Tier-2 外推，卷宗登记。
#   AVA 用 AVAL 配方（E_rev -39, C 9.66）。
#   近似登记（Tier-2/3）：
#   - slo-2 孤立+耦合两部分合并进 gbk2（SLO2-EGL19 复合体动力学）；
#   - 门控动力学沿用 v55（Nicoletti 2019 .ode 校准位移），未重校准；
#   - exp-2（VB6 表达）v55 无此通道机器，缺省=0，登记为缺失；
#   - kcnl 在 VA5/VB6 表达但 Table 3 拟合贡献为零 -> gsk=0；
#   - vol（钙微域体积）沿用 RMD 值 5.65。
#   v55 文件逐字节不动；新类配方只在显式调用 v56 时生效。】
import numpy as np
import anchored_engine_v55 as v55

MN_RECIPES = {
    #            K 通道                                                     Ca 通道                         漏/其他
    "VA": dict(gshal=0.0, gshak=0.1, gkir=1.0, gegl36=0.0,
               gcca1=0.0, gunc2=0.0, gegl19=0.15,
               gbk=0.0, gslo1=0.0, gbk2=6.0, gslo2=0.0, gsk=0.0,
               gnca=0.01, gleak=0.10, ELEAK=-56.5, C=5.84),
               # ELEAK -56.5: 校准至实测静息 -71.7（孤立静息 -71.87）。
               # 偏离 Nicoletti 拟合 -70 的原因：本引擎 IRK/SLO2 动力学
               # 移植自 2019 .ode，与 2024 版不同；E_leak 本就是逐细胞
               # 拟合的"残余反转电位"（该文各细胞 -37~-89.57），登记 Tier-2。
    "VB": dict(gshal=0.0, gshak=0.4, gkir=1.0, gegl36=0.0,
               gcca1=0.0, gunc2=0.1, gegl19=0.10,
               gbk=0.0, gslo1=0.2, gbk2=3.75, gslo2=0.0, gsk=0.0,
               gnca=0.03, gleak=0.13, ELEAK=-52.0, C=7.87),
    "VD": dict(gshal=0.0, gshak=1.2, gkir=0.7, gegl36=0.0,
               gcca1=0.1, gunc2=0.0, gegl19=0.90,
               gbk=0.0, gslo1=0.0, gbk2=3.4, gslo2=0.0, gsk=0.0,
               gnca=0.09, gleak=0.20, ELEAK=-75.0, C=3.52),
    "AVA": dict(gshal=0.0, gshak=0.0, gkir=0.1, gegl36=0.0,
                gcca1=0.0, gunc2=0.0, gegl19=0.10,
                gbk=0.0, gslo1=0.0, gbk2=0.0, gslo2=0.0, gsk=0.0,
                gnca=0.03, gleak=0.15, ELEAK=-52.0, C=9.66),
                # ELEAK -52: 校准至锚 -39（孤立静息 -38.58）；原拟合 -39。
}
MN_RECIPES["DA"] = dict(MN_RECIPES["VA"])   # A 类外推（Tier-2）
MN_RECIPES["DB"] = dict(MN_RECIPES["VB"])   # B 类外推（Tier-2）

# 实测锚点（mV），冒烟判决用
ANCHOR_REST = {"VA": -71.7, "VB": -53.2, "VD": -45.8, "DA": -71.7, "DB": -53.2,
               "AVA": -39.0}  # AVA 无直接静息均值, 取 E_rev 拟合值作近似锚

def cell_params(ctype):
    if ctype in MN_RECIPES:
        P = v55.cell_params("RMD")       # 先拿 RMD 结构（门控布局一致）
        P.update(MN_RECIPES[ctype])      # 再覆盖电导/E_leak/C
        return P
    return v55.cell_params(ctype)

def layout(ctype):
    return "RMD" if ctype in MN_RECIPES else ctype

def rhs56(t, y, ctype, Iext):
    return v55.rhs(t, y, cell_params(ctype), layout(ctype), Iext)

def init_state56(ctype):
    return v55.init_state(layout(ctype))

# v62 接口别名（v62 调 eng.init_state）
init_state = init_state56

# 分类释放参考电位 (mV): 该类锚定静息 + 14.5（与旧体系 RMD 静息-69.5→V_REL-55 同裕度）。
# 仅作用于新类细胞；AWC/RMD 不在表内 -> v62 用全局 V_REL=-55，旧行为不变。
RELEASE_REF = {
    "VA": -71.87 + 14.5, "DA": -71.87 + 14.5,
    "VB": -52.76 + 14.5, "DB": -52.76 + 14.5,
    "VD": -44.52 + 14.5,
    "AVA": -38.58 + 14.5,
}
