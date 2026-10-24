# anchored_engine_v58.py
# 【v5.8 = v5.7 + G3 重锚手术（2026-10-10）：
#   1) AVA 重锚：孤立静息 -38.58 -> -25.000 mV（ELEAK -52 -> -38.425，
#      二分法 LSODA 解，solve_v58_ava_eleak_2026-10-10.py）。
#      依据：AVA 实测 -20~-30 mV 中位 -23（Mellem 2008 J Neurosci PMC2697921；
#      Meng 2024 Sci Adv adk0002：AVA 紧张性活跃、多稳态 -30~-10）。
#      旧锚 -39 系当年取 E_rev 拟合近似，文献冲突登记为 G3，本次裁决。
#   2) RELEASE_REF 全部六类由"静息+14.5"改为"孤立平衡+VREL_OFF"（默认 -10，
#      静息时 s=0.4，紧张性 graded release）。依据：Meng 2024（AVA 基线近 UNC-2 阈）、
#      蠕虫 NMJ graded release 围绕静息调制。rest+14.5 无实测支持，登记 Tier-3 约定变更。
#   MN 静息锚不动：VA5 -70 / VB6 -50 / VD5 -45 与 Liu 2014 实测一致（卷宗二十一节勘误）。
#   负对照：AWC/RMD 不在 RELEASE_REF 表内 -> v62 用全局 V_REL=-55，旧行为不变；
#   VREL_OFF=+14.5 时 RELEASE_REF 与 v56 相差 0.5 mV 内（旧体系用锚点值,本版用孤立平衡值）。
#   v56/v57 文件逐字节不动。】
import anchored_engine_v57 as _v57
import anchored_engine_v56 as _v56

VREL_OFF = -10.0   # mV; 调用方在构造网络前设置 (w01_closed_loop_w13.py 读环境变量 W13_VRELOFF)

# 孤立平衡值 (v56 实测, mV; 冒烟 smoke_v56_rest 复核): VA/DA -71.87, VB/DB -52.76, VD -44.52, AVA(v58) -25.00
_EQ_REST = {"VA": -71.87, "DA": -71.87, "VB": -52.76, "DB": -52.76, "VD": -44.52, "AVA": -25.00}

def cell_params(ctype):
    P = dict(_v57.cell_params(ctype))
    if ctype == "AVA":
        P["ELEAK"] = -38.425   # G3 重锚: 孤立静息 -25.000
    return P

ANCHOR_REST = dict(_v56.ANCHOR_REST)
ANCHOR_REST["AVA"] = -25.0

class _ReleaseRef(dict):
    """RELEASE_REF 随 VREL_OFF 动态计算 (读取时)。"""
    def get(self, k, default=None):
        if k in _EQ_REST:
            return _EQ_REST[k] + VREL_OFF
        return default
    def __contains__(self, k):
        return k in _EQ_REST

RELEASE_REF = _ReleaseRef()

layout = _v56.layout
init_state = _v56.init_state
init_state56 = _v56.init_state56
GUNC2_AB = _v57.GUNC2_AB
