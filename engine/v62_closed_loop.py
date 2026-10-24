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


GSYN = 0.005   # 全局化学突触电导增益 (可用 --gsyn 覆盖)
GGAP = 0.3     # 全局电突触增益 (可用 --ggap 覆盖)
GNCA_A_ARG = -1.0  # --gncaA 传入的 A 类 NCA 电导覆盖 (v84)
GLEAK_GABA_ARG = 1.0  # --gabaleak 传入的 DD/VD gleak 倍数 (v86, C2.6 交叉抑制振荡修复)
GEGL36_GABA_ARG = -1.0  # --gabaegl36 传入的 DD/VD gegl36 覆盖 (v87, C2.6 平台复极修复)
UNC2PED_A_ARG = -1.0   # --unc2pedA 传入的 A 类 UNC-2 非失活分量 (W13, Gao 2018 振荡窗流)
GSK_A_ARG = -1.0       # --gskA 传入的 A 类 KCNL/SK 电导 (W13, 慢刹车, 欠定方向恢复)
CATAUSCALE_A_ARG = -1.0  # --catauscaleA 传入的 A 类体钙清除倍数 (W13, 微域50ms->体钙2s)
GUNC2_A_ARG = -1.0     # --gunc2A 传入的 A 类 gunc2 覆盖 (W13, 网络尺度窗流)
GUNC2_AVA_ARG = -1.0   # --gunc2ava 传入的 AVA gunc2 覆盖 (W52, Meng 2024 UNC-2 基线开阈)
UNC2PED_AVA_ARG = -1.0 # --unc2pedava 传入的 AVA UNC-2 非失活分量 (W52, 再生窗流)
GSK_AVA_ARG = -1.0     # --gskava 传入的 AVA KCNL/SK 电导 (W52, 平台秒级自终结)
CATAUSCALE_AVA_ARG = -1.0  # --catauscaleava 传入的 AVA 体钙清除倍数 (W52)


# ---- 趋化世界层 (架构借鉴 worm-whisperer worm/env/chem.py, Lee 2026, MIT) ----
# 关键差异: 他们的 ASE 电流由外部 RL agent 环路转灌命令层 (其网络感觉->命令静默);
# 我们这里 ASEL/ASER 电流直接注入全通道网络, 走真实解剖通路 ASEL/ASER->AIY/AIZ->命令层。
class ChemField:
    """2D 高斯浓度场 (诱饵源)"""
    def __init__(self, source_xy, sigma=2.0, peak=1.0):
        self.src = np.asarray(source_xy, float); self.sigma = sigma; self.peak = peak
    def conc(self, xy):
        d2 = ((np.asarray(xy) - self.src) ** 2).sum(-1)
        return self.peak * np.exp(-d2 / (2 * self.sigma ** 2))


class ASEInput:
    """头节点浓度的 dC/dt 低通(τ)后: 升->ASEL(ON), 降->ASER(OFF)
    生理: Suzuki et al. 2008 (ASEL/ASER ON-OFF 不对称);
          Pierce-Shimomura 1999; Iino & Yoshida 2009 (趋化依赖 dC/dt)"""
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
        print(f"[gncaA] A类(VA/DA) NCA Na+ leak 覆盖为 {GNCA_A_ARG:g} nS "
              f"(Gao 2018 + Frontiers 2022: NCA 是 A-MN 固有振荡必要引擎)", flush=True)
    if GLEAK_GABA_ARG != 1.0:
        v57.GLEAK_GABA = GLEAK_GABA_ARG
        print(f"[gabaleak] DD/VD gleak x{GLEAK_GABA_ARG:g} "
              f"(C2.6: VD/DD 钙信号应随爬行节律振荡; 修复 tonic 饱和钳死)", flush=True)
    if GEGL36_GABA_ARG >= 0.0:
        v57.GEGL36_GABA = GEGL36_GABA_ARG
        print(f"[gabaegl36] DD/VD gegl36 = {GEGL36_GABA_ARG:g} nS "
              f"(C2.6: EGL-36 复极 EGL-19 平台, 恢复 VD/DD 波频振荡; VD5 锚守)", flush=True)
    if UNC2PED_A_ARG >= 0.0:
        v57.UNC2PED_A = UNC2PED_A_ARG
        print(f"[unc2pedA] A类(VA/DA) UNC-2 非失活分量 = {UNC2PED_A_ARG:g} "
              f"(W13: Gao 2018 振荡窗流; 恢复 A-MN 双稳再生性兴奋)", flush=True)
    if GSK_A_ARG >= 0.0:
        v57.GSK_A = GSK_A_ARG
        print(f"[gskA] A类(VA/DA) KCNL/SK = {GSK_A_ARG:g} nS (W13: 慢刹车)", flush=True)
    if CATAUSCALE_A_ARG >= 0.0:
        v57.CATAUSCALE_A = CATAUSCALE_A_ARG
        print(f"[catauscaleA] A类(VA/DA) 钙清除 x{CATAUSCALE_A_ARG:g} (W13: 体钙2s)", flush=True)
    if GUNC2_A_ARG >= 0.0:
        v57.GUNC2_A = GUNC2_A_ARG
        print(f"[gunc2A] A类(VA/DA) gunc2 = {GUNC2_A_ARG:g} nS (W13: 网络尺度窗流)", flush=True)
    if GUNC2_AVA_ARG >= 0.0:
        v57.GUNC2_AVA = GUNC2_AVA_ARG
        print(f"[gunc2ava] AVA gunc2 = {GUNC2_AVA_ARG:g} nS "
              f"(W52: Meng 2024 PMC11006214 AVA基线近UNC-2开阈/多稳态-30~-10)", flush=True)
    if UNC2PED_AVA_ARG >= 0.0:
        v57.UNC2PED_AVA = UNC2PED_AVA_ARG
        print(f"[unc2pedava] AVA UNC-2 非失活分量 = {UNC2PED_AVA_ARG:g} "
              f"(W52: 再生窗流, Gao 2018 同锚)", flush=True)
    if GSK_AVA_ARG >= 0.0:
        v57.GSK_AVA = GSK_AVA_ARG
        print(f"[gskava] AVA KCNL/SK = {GSK_AVA_ARG:g} nS (W52: 平台秒级自终结)", flush=True)
    if CATAUSCALE_AVA_ARG >= 0.0:
        v57.CATAUSCALE_AVA = CATAUSCALE_AVA_ARG
        print(f"[catauscaleava] AVA 钙清除 x{CATAUSCALE_AVA_ARG:g} (W52: 体钙2s)", flush=True)
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
    ap.add_argument("--stim", type=float, default=10.0, help="AVB 刺激幅度 pA")
    ap.add_argument("--initdir", type=float, default=180.0,
                    help="v83: 初始朝向 deg (虫子面向的方位角, 0=+x 90=+y; 默认180=旧行为逐位不变, "
                         "Rod2D.initial 的 heading 杆延伸方向=面向的反方向, 故内部+180)")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--smoke", action="store_true", help="只跑 2s 基线 + 3s 刺激")
    ap.add_argument("--tag", default="")
    ap.add_argument("--block", action="append", default=[], help="通道阻断 名:乘子, 如 gunc2:0")
    ap.add_argument("--kmax", type=float, default=6.0, help="运动层最大曲率 1/mm")
    ap.add_argument("--gsr", type=float, default=2.0, help="牵张感受器增益")
    ap.add_argument("--span", type=float, default=0.25, help="牵张感受器空间跨度分数")
    ap.add_argument("--pep", type=float, default=0.0,
                    help="肽能增益强度 alpha (DVA->NLP-12->CKR-2 轴, 0=虚拟ckr-2 KO)")
    ap.add_argument("--peptau", type=float, default=3.0, help="肽能调制时间常数 s")
    ap.add_argument("--ckr1", type=float, default=0.0,
                    help="二期 CKR-1 肽能增益 G0 (pA), 0=关(旧行为逐位不变). "
                         "Chen 2022 Cell Rep 38:110330 (PMID 35139370) + Chen 2024 (PMC10952047, "
                         "Mendeley 10.17632/jm9yt2hdnk.1 原始数据): 同一 Galpha-q 受体 CKR-1 在 SMD 集成两运动状态 — "
                         "NLP-12 臂(DVA牵张->SMD, 体曲率增益; nlp-12 缺失 -21.6%% / ckr-1 -17.9%%) + "
                         "NLP-18 臂(ASI->SMD+AIB, Omega转向增益; nlp-18 缺失 -11.6%% / ckr-1 -12.3%%). "
                         "Fig3: NMJ minis/EPSC 不受 CKR-1 影响 -> 增益作用于 SMD/AIB 兴奋性, 不动突触传递")
    ap.add_argument("--ckr1tau12", type=float, default=12.9,
                    help="NLP-12 臂时间常数 s (Chen 2024 Fig5E 实测 Ca2+ 10-90%% 上升时间 12.93+-0.98s)")
    ap.add_argument("--ckr1tau18", type=float, default=22.1,
                    help="NLP-18 臂时间常数 s (同上 Fig5E 实测 22.08+-3.25s)")
    ap.add_argument("--ckr1m0", type=float, default=0.0,
                    help="CKR-1 双臂调制态初值 M0 in [0,1] (预热同档, 同 ggapmod 例; 0=旧行为)")
    ap.add_argument("--ggapmod", type=float, default=0.0,
                    help="二期机制化 (2026-10-21): 缝隙连接电导状态变量调制. !=0 时 Lg 逐块按 "
                         "g_eff=(1-k(1-M))xGGAP 缩放; M=前部机械感觉层(ALM/AVM轻触觉 + 多巴胺能 "
                         "CEP/ADE)活动经 DOP-2 GPCR 慢变量. 锚: Chen 2007 unc-1 门控 UNC-9 耦合降 "
                         "~70%% (PMC1976340); LeBoeuf 2015 DOP-2->UNC-7 (PMC4495247); Schwarz 2018 "
                         "EGL-4/PKG->UNC-7/9 (PMC6283151); Kawano 2011 UNC-7/9 shunt (PMID 22099460). "
                         "逃逸/爬行 3.3x 电导比恰落 unc-1 实测调制范围两端. 须配合 --ggap 1.0 "
                         "(锚定满刻度): M~0->0.3 爬行档, M~1->1.0 逃逸档. 0=关闭=旧行为逐位不变")
    ap.add_argument("--ggapmodtau", type=float, default=3.0,
                    help="ggapmod 状态变量 M 时间常数 s (GPCR 信号量级锚, 与 peptau 同级 1-10s)")
    ap.add_argument("--ggapmodk", type=float, default=0.7,
                    help="ggapmod 调制深度 (Chen 2007 实测: unc-1;unc-9 通路破坏耦合降 ~70%%)")
    ap.add_argument("--ggapmodm0", type=float, default=0.0,
                    help="ggapmod 初始/设定状态 M0 in [0,1] (慢调制状态语义: 0=爬行档 1=逃逸档; "
                         "配 --ggapmodtau 600 (Schwarz 2018 EGL-4/PKG 睡眠-唤醒分钟级) 即状态锁定. "
                         "急性感觉驱动切换已于 2026-10-21 实验证伪 (剂量响应 0.6/0.75/0.9 三档 "
                         "frac_bwd=0.02/0.05/0.00, 耦合态须秒级以上预建立): 逃逸/爬行=两个 "
                         "慢调制状态, 非急性切换; 急性触觉逃逸由化学通路执行 (Chalfie 1985)")
    ap.add_argument("--ggapmodash", type=float, default=0.0,
                    help="B5: 0=驱动池不含 ASH=旧行为逐位不变; !=0 时 ASHL/R 加入 ggapmod 驱动池 "
                         "(Chaubey 2023 PMC10319955: 状态依赖 innexin/INX-20 调制定位于 ASH 本身; "
                         "2026-10-21 急性切换证伪实验的驱动池为 ALM/AVM/CEP/ADE, 未含 ASH — "
                         "化学逃逸时 M 无从升起, 本旗标补上该结构缺口)")
    ap.add_argument("--avarect", type=float, default=0.0,
                    help="W49 (2026-10-23): AVA<->A-MN 整流缝隙连接强度 in [0,1], 0=关=旧行为逐位不变. "
                         "锚 Shui 2020 Sci Adv 6:eabb3076 (PMC7455501): UNC-7b(AVA侧)/UNC-9(A-MN侧) "
                         "异型通道球-受体门控 — Vj=V_AVA-V_AMN>0 (AVA更去极化) 球堵孔=关, "
                         "Vj<0 球脱落=开 (电流只走 A->AVA 反方向); Liu 2017 Nat Commun 8:14818 "
                         "在体双电极单向整流; Kawano 2011 Neuron 72:572 shunt 静息抑制; "
                         "Gao 2018 eLife e29915 状态依赖抑制+易化; Starich 2009 Neural Dev 4:16 组成. "
                         "实现: 逐块按上一块电压(50ms延迟, 同ggapmod例)算 Boltzmann 门控重建 Lg")
    ap.add_argument("--avarectv0", type=float, default=17.0,
                    help="W49 Boltzmann 半活 Vj 中点 mV (Tier-2 定标 2026-10-23: 爬行态Vj=+9.2须高通/"
                         "bout态Vj=20-77须关断, 分离解 V0=17; 生物映射~3x电压压缩 -> ~+50mV; "
                         "Shui 2020 表S3拟合值待补; V0=0实测崩爬行 v=0.064 在案)")
    ap.add_argument("--avarecta", type=float, default=0.22,
                    help="W49 Boltzmann 斜率 /mV (Tier-2 定标: =引擎~3x电压压缩 x Liu2006 innexin "
                         "门控斜率 0.10 的映射; 表S3待补)")
    ap.add_argument("--avarectmin", type=float, default=0.05,
                    help="W49 Gjmin 残余电导比 (Tier-2: Liu 2017 反向耦合~0, '最强整流')")
    ap.add_argument("--cellablmode", type=str, default="clamp", choices=["clamp", "edge"],
                    help="cellabl 消融协议: clamp=旧行为(-300pA恒流超极化, 边仍在); "
                         "edge=真激光消融语义(删除触及该细胞的全部化学+电突触边, 不注入). "
                         "W13 墙二: 钳压协议下被钳细胞经残余 gap 边反向拖拽邻居=假象, "
                         "Gao 2018 消融制备复现必须用 edge 模式")
    ap.add_argument("--mnprop", type=float, default=0.0,
                    help="B类运动神经元本体感觉增益 pA/(1/mm) (Wen et al. 2012, Neuron 76:750: "
                         "B类MN无分支神经突感受相邻体节弯曲, 活动被身体波相位锁定驱动); "
                         "0=关闭=旧行为逐位不变; D/V反对称注入, 极性经 preferred_curvature 源码核定: "
                         "κ>0=背侧收缩 -> DB +κ, VB -κ (v66 修正, v65 指派相反, 锁定统计不受影响), 每块(50ms)注入")
    ap.add_argument("--mnoffset", type=int, default=0,
                    help="B类MN本体感觉感受位置的前方偏移量 (体节数; v67 R1, Wen et al. 2012 实验锚点: "
                         "VB/DB 去极化与胞体前方 ~200um 处同侧弯曲相关 -> 22节段~50um/节 -> ~4节; "
                         "Bryden & Cohen 2008 理论: 前方感受+时间延迟 -> 行波色散, 最小B回路即可振荡). "
                         "0=同体节=旧行为逐位不变")
    ap.add_argument("--mnproptau", type=float, default=0.0,
                    help="第一通道(Wen前方正反馈)曲率信号的EMA时间常数 ms (v68 R7: Bryden & Cohen 2008 "
                         "Biol Cybern 98:339 行波色散机制——前方感受+时间延迟->头尾行波; "
                         "瞬时正反馈只同步不传播, 延迟提供节间相位梯度; "
                         "0=关闭=直通=旧行为逐位不变; 仅 mnprop!=0 时生效)")
    ap.add_argument("--mnprop2", type=float, default=0.0,
                    help="第二本体感觉通道增益 pA/(1/mm) (v67 R4 双符号共存检验: Wen 2012 实测=同侧收缩正反馈"
                         "(本参数取正, 前方感受); Cohen组建模假设=同侧牵张负反馈(取负, 后方感受); "
                         "两通道并存时按各自增益线性叠加, 判据=Wen Fig7: B-MN活性须与同侧弯曲正相关; "
                         "0=关闭=旧行为逐位不变)")
    ap.add_argument("--mnoffset2", type=int, default=0,
                    help="第二本体感觉通道感受位置偏移 (体节数, 负值=后方; 仅 mnprop2!=0 时生效)")
    ap.add_argument("--mnprop2tau", type=float, default=0.0,
                    help="第二通道曲率信号的一阶低通时间常数 ms (v67 R5, Zhao et al. 2025 PLOS Comput Biol "
                         "21(6):e1013171: 负反馈来自肌肉收缩到指定长度后的延迟电流, 肌肉收缩/舒张本身有 "
                         "百ms级动力学 -> 用EMA代理肌肉的机械延迟, 消除1块(50ms)纯延迟造成的10Hz伪影; "
                         "0=关闭=直通=旧行为逐位不变; 仅 mnprop2!=0 时生效)")
    ap.add_argument("--mnprop2thr", type=float, default=0.0,
                    help="第二通道负反馈收缩阈值 1/mm (v67 R5, Zhao 2025: 肌肉收缩到指定长度后负反馈电流"
                         "才回到运动神经元 -> 半波整流 max(同侧收缩-阈值,0), 避免与正反馈通道线性相消; "
                         "0=关闭=纯线性=旧行为逐位不变; 仅 mnprop2!=0 时生效)")
    ap.add_argument("--mnback", type=int, default=0,
                    help="倒退时第二本体感觉通道感受野镜像反转 (v70 R14, 根因A修复: "
                         "mndrv 不被方向门控 -> 倒退门控开了 B-MN 环仍按尾->头调谐往前推, 两力打架只减速; "
                         "生物对应: 倒退时波从头发起(尾->头=前进, 头->尾=倒退), "
                         "环路镜像 = 感受偏移符号取反 (后方感受<->前方感受); "
                         "1=倒退门控(relg/rel dir_state<0)生效时 mnoffset2 取反, 0=关闭=旧行为逐位不变; "
                         "仅 mnprop2!=0 时生效)")
    ap.add_argument("--bmnclip", type=str, default="clip0", choices=["clip0", "reflect"],
                    help="B类感受位边界语义 (W57c 2026-10-23: clip0=旧行为 越界感受位钳到边界节 "
                         "=尾侧B类(VB09-11/DB06-07)全部钳到节23且尾尖κ~0 -> 自洽死区(爬行尾波消失), "
                         "倒退镜像时头侧B类钳到节0=强制同相 -> 头部同步块=倒退波驻波化(+2°/节); "
                         "reflect=越界感受位镜像回体内(同W57 A类修复), 边界细胞仍获行波信号; "
                         "默认clip0=旧行为逐位不变)")
    ap.add_argument("--mnanat", type=int, default=0,
                    help="B类MN胞体-体节映射方式 (v69 R10a, 开放问题#3尾部死区修复): "
                         "1=按解剖支配表定位 (neuron_muscle.csv: DB01->肌行07-09->体节~6, "
                         "DB07->肌行22-24->体节~21; 本体感受位置与肌肉支配位置对齐, 局部反馈环闭拢), "
                         "0=旧的按序号均匀撒nseg节段=旧行为逐位不变")
    ap.add_argument("--rivabl", type=float, default=0.0,
                    help="RIVL/R 恒定注入 pA (v69 R10b, 负值=功能性消融: Gray et al. 2005 "
                         "PNAS 102:3184 doi:10.1073/pnas.0409009101, RIV消融->反转后腹侧转向偏置消失; "
                         "0=关闭=旧行为逐位不变)")
    ap.add_argument("--cellabl", type=str, default="",
                    help="功能性消融细胞名单, 逗号分隔, 支持前缀通配 (如 'AVA*,AVB*,DB*,VB*'): "
                         "恒定强超极化=HisCl式沉默 (Gao 2018 eLife 7:e29915 premotor/B-MN 消融协议); "
                         "空=关闭=旧行为逐位不变")
    ap.add_argument("--cellablp", type=float, default=-300.0,
                    help="cellabl 注入幅度 pA (默认-300 强超极化)")
    ap.add_argument("--gncaA", type=float, default=-1.0,
                    help="A类(VA/DA) NCA Na+ leak 电导覆盖 nS (v84, Gao 2018 振荡引擎; "
                         "-1=关闭=旧行为逐位不变)")
    ap.add_argument("--unc2pedA", type=float, default=-1.0,
                    help="A类(VA/DA) UNC-2 失活门非失活分量 [0,1) (W13 2026-10-23, Gao 2018: "
                         "A-MN 内源振荡必需 UNC-2 窗流, unc-2(gf)=减弱失活; 单细胞探针判决 "
                         "原细胞单稳态无磁滞不可能振荡, ped0.20 恢复双稳窗+AVA分流门控三相全过; "
                         "-1=关闭=旧行为逐位不变)")
    ap.add_argument("--gskA", type=float, default=-1.0,
                    help="A类(VA/DA) KCNL/SK 钙激活钾电导 nS (W13: 解剖表达欠定方向恢复, "
                         "慢刹车; 0.05 过锚, >=0.1 钳死点火; -1=关闭=旧行为逐位不变)")
    ap.add_argument("--catauscaleA", type=float, default=-1.0,
                    help="A类(VA/DA) 胞内钙清除时间常数倍数 (W13: 50ms微域->体钙秒级, "
                         "Gao 2018 DA9 钙振荡~50s; 取40=2s; -1=关闭=旧行为逐位不变)")
    ap.add_argument("--gunc2A", type=float, default=-1.0,
                    help="A类(VA/DA) gunc2 覆盖 nS (W13: E2d 实测 GUNC2_AB=0.4 窗流在网络电导下"
                         "<1%% 被淹没, 需网络尺度 1.0+; VB/DB 不动; -1=关闭=旧行为逐位不变)")
    ap.add_argument("--gunc2ava", type=float, default=-1.0,
                    help="AVA gunc2 覆盖 nS (W52, Meng 2024 Sci Adv 10:adk0002: AVA 基线近 UNC-2 开阈, "
                         "双稳区制 -30/-20 与 -15/-10; 本地引擎 AVA gunc2=0=钳制拟合欠定方向, "
                         "命令层缺再生载体; -1=关闭=旧行为逐位不变)")
    ap.add_argument("--unc2pedava", type=float, default=-1.0,
                    help="AVA UNC-2 失活门非失活分量 [0,1) (W52: -25mV 处 hinf~0, "
                         "再生窗流必需 ped; -1=关闭=旧行为逐位不变)")
    ap.add_argument("--gskava", type=float, default=-1.0,
                    help="AVA KCNL/SK 钙激活钾电导 nS (W52: 活动依赖终结, 同 W13 A-MN 纪律; "
                         "-1=关闭=旧行为逐位不变)")
    ap.add_argument("--catauscaleava", type=float, default=-1.0,
                    help="AVA 胞内钙清除时间常数倍数 (W52: 体钙秒级终结; -1=关闭=旧行为逐位不变)")
    ap.add_argument("--gabaleak", type=float, default=1.0,
                    help="DD/VD gleak 倍数 (v86, C2.6=eLife 2021 PMC7986531: VD/DD 钙信号应随爬行节律振荡; "
                         "实测 tonic 饱和钳死不振荡->腹侧过驱动左拐; 1.0=旧行为逐位不变)")
    ap.add_argument("--gabaegl36", type=float, default=-1.0,
                    help="DD/VD gegl36 覆盖 nS (v87, C2.6 同锚: VD 配方 gegl36=0 是钳制拟合欠定方向, "
                         "EGL-19 L型平台 (-25..-5mV 负斜率) 无高压K复极 -> 泊 ~-4mV 平台波频死亡; "
                         "EGL-36 静息关闭不伤 VD5 锚, 单细胞探针 0.6/1.0 过锚; -1=旧行为逐位不变)")
    ap.add_argument("--amnprop", type=float, default=0.0,
                    help="A类MN(VA/DA)本体感觉通道增益 pA/(1/mm) (v85, Gao et al. 2018 eLife 7:e29915: "
                         "A类为倒退内生振荡器且接受本体感觉entrainment; 环路镜像B链倒退形态=前方感受; "
                         "0=关闭=旧行为逐位不变)")
    ap.add_argument("--amnoffset", type=int, default=6,
                    help="A类通道感受位置向头侧偏移 体节数 (正值=前方/头侧感受 -> 头->尾传播=倒退波, "
                         "与 mnback 镜像后 B 链同构; 仅 amnprop!=0 时生效)")
    ap.add_argument("--amngate", type=int, default=1,
                    help="A类通道方向门控 (Kato et al. 2015 Cell 163:656 前进/倒退亚环路分工): "
                         "1=仅倒退态(dir_state<0)生效, 0=全程生效; 仅 amnprop!=0 时生效)")
    ap.add_argument("--amnclip", type=str, default="clip0", choices=["clip0", "edge0", "reflect"],
                    help="A类感受边界语义 (2026-10-23 W55: clip0=旧行为 头侧胞体感受位clip到节0 "
                         "=头侧8节强制同相(数值伪影); edge0=吻端边界外零注入=头侧A类纯内生振荡器 "
                         "(Gao 2018 eLife 7:e29915 A类内生振荡属性), 成波种子; 默认clip0=旧行为逐位不变; "
                         "W57 reflect=越界感受位镜像回体内 (edge0断粮致尾VA09-12被VD GABA按死的修复, "
                         "无同相伪影: 反射后各节感受位仍分散)")
    ap.add_argument("--amndir", type=str, default="anterior", choices=["anterior", "posterior"],
                    help="A类本体感受方向 (W56 2026-10-23 翻案: Gao 2018 图1B/2B实锤倒退波=posterior->anterior "
                         "(尾->头)传播, 图5 VA11领先VA10 -> 接力须感受尾侧(后方); v85b的前方感受产生头->尾波"
                         "=前进方向, 方向装反=J2全败的结构根因; anterior=旧行为(v85b)逐位不变, "
                         "posterior=尾侧感受(v85a方向, 尾->头倒退波)")
    ap.add_argument("--amnhp", type=float, default=0.0,
                    help="A类本体环高通时间常数 秒 (2026-10-22 倒退锯齿锁修复: 牵张感受器相位性/"
                         "自适应 — 只对曲率变化响应, 静态弯DC被扣除, 破强tonic驱动+负反馈的"
                         "Turing式静态锁定; 信号=kappa_EMA80ms-kappa_EMA(amnhp); "
                         "0=关闭=旧行为逐位不变; 证据等级: 机制类比有界估计)")
    ap.add_argument("--bpropbwd", type=float, default=0.0,
                    help="倒退期B链(mnprop2)增益衰减比例 (v85, Kato 2015 亚环路互抑analog: "
                         "倒退门控生效时 B链增益 x(1-bpropbwd); 0=不衰减=旧行为逐位不变)")
    ap.add_argument("--bmnhp", type=float, default=0.0,
                    help="W58: 倒退态 mnprop2(B链本体环)高通时间常数 秒 (牵张感受器相位-tonic混合/"
                         "tonic适应衰减, 肌梭适应教科书机制 Matthews 1972 + 线虫DVA/TRP-4相位偏好 "
                         "Li 2006 Nature; 切断 bout自持环DC通道: 静态S弯->mnprop2 DC->B链tonic反转"
                         "->AVA平台供养->bout不退; 仅dir_state<0生效, 爬行回归逐位不动; "
                         "复用amnhp慢EMA参考, 0=关闭=旧行为逐位不变)")
    ap.add_argument("--abx", type=float, default=1.0,
                    help="倒退期 A-MN(VA/DA)->B-MN(VB/DB) 兴奋性化学串扰增益 (W54 2026-10-23, "
                         "Kato 2015 亚环路互抑analog 同 bpropbwd 族: 倒退态前进模块应被抑制; "
                         "解剖审计: VA->VB w=5.508 把VA锁入前进程序=VA同步无波 (DA->DB仅1.65, "
                         "DA自由->相位梯度0.31rad/节=lambda20节行波); 倒退门控生效时 A->B化学边增益x=abx; "
                         "1.0=不衰减=旧行为逐位不变)")
    ap.add_argument("--avdx", type=float, default=1.0,
                    help="倒退期 A-MN(VA/DA)<->VD/DD 抑制性运动神经元电突触增益 (W54b 2026-10-23: "
                         "abx0.3 实弹后 VA 仍同步 -> 次候选锁 VA<->VD gap1.188/DD->VA 0.532 "
                         "(DA侧仅0.192); 存档实测 VD/DD/VB 全同步块, VA 被拖入; "
                         "机制类比=状态依赖 innexin 调制 (肽能紧张性标定候选, 诚实标注现象学); "
                         "倒退门控生效时该边集增益x=avdx; 1.0=不衰减=旧行为逐位不变)")
    ap.add_argument("--sloa", type=float, default=0.0,
                    help="A类MN SLO-1/SLO-2型活动依赖钾电导 nS (v92, Yu/Wang/Gao 2022 Front Mol Neurosci "
                         "15:845733: slo-1(lf)/slo-2(lf) -> tonic/long-lasting rPSC增大+倒退增速 = A类振荡的"
                         "活动依赖负反馈刹车; UNC-2钙内流->钙/钠激活钾外流终结爆发, 与七鳃鳗VGCC->KCa->AHP同源; "
                         "引擎实现: ca=胞体活动轨迹((softplus(dV)-ln2)/25内流, tau清除), I_K=-G*ca*(V-E_K)外向; "
                         "持续静态驱动被适应(DC压制), 0.6Hz节律成分通过 = amnhp高通在胞体级的机制化实现; "
                         "爬行期A类静默 dV≈0 -> ca→0 -> 零电流; 0=关闭=旧行为逐位不变)")
    ap.add_argument("--sloatau", type=float, default=1.0,
                    help="sloa 钙轨迹清除时间常数 s (Yu 2022: rPSC tonic上升0.82s/时程3.9-9.9s 秒级适应; "
                         "取同 amnhp 验证窗口 1.0s; 敏感带在案)")
    ap.add_argument("--sloath", type=float, default=15.0,
                    help="sloa 钙内流激活阈 dV mV (UNC-2=高压激活VGCC, 强去极化才开: Yu 2022/Gao 2018; "
                         "实测分布: 爬行A类dV≤+11 / 倒退bout峰值中位+56, 阈取间隙15 -> 爬行内流恒零, "
                         "回归逐位不动; 硬整流=HVA陡Boltzmann的约化)")
    ap.add_argument("--amnosc", type=float, default=0.0,
                    help="A类MN UNC-2 再生内流臂增益 pA (v92 稳态倒退结构件: Yu 2022 同一 UNC-2 通道的"
                         "去极化臂 — HVA Ca2+ 内流本身再生去极化, 与 sloa SLO 刹车臂构成松弛振荡器 "
                         "(Gao 2018: A类=倒退内生振荡器); 振荡驱动替代 tonic 平台供养 -> 静态 Turing 模"
                         "无持续能量不立, 本体链 entrain 相位成温和行波; 复用 _infl=max(dV-sloath,0)/25 "
                         "-> 爬行期恒零, 回归天然逐位不动; 需配合 --sloa 使用; 0=关闭=旧行为逐位不变)")
    ap.add_argument("--amngrad", type=float, default=0.0,
                    help="A类 UNC-2 驱动尾高头低乘性梯度 (W56, 2026-10-23; Gao 2018 eLife 7:e29915: "
                         "DA9 最尾单元活性最高=领跑振荡器, 七鳃鳗最快振荡器entrain机制 -> 尾->头行波; "
                         "因子=1+g*(2*seg/(nseg-1)-1): 头(1-g)尾(1+g)线性; 梯度幅值=欠定参数(补二十在案), "
                         "0=均匀=旧行为逐位不变; 仅 amnosc!=0 时生效)")
    ap.add_argument("--dvaprop", type=float, default=0.0,
                    help="DVA/TRP-4 幅度限制刹车增益 pA/(1/mm) (v92, Li/Feng/Sternberg/Xu 2006 Nature 440:684 "
                         "PMC2865900: trp-4(lf)=过度弯曲表型 -> DVA 牵张感受负向抑制腹索MN防止过弯; "
                         "引擎实现: |kappa慢EMA| 超极化注入A类MN胞体节段, 行波在dvatau尺度平均~0=AC放行, "
                         "静态弯DC全通过=选择性拆除Turing静态锁; 仅倒退态(dir_state<0)生效 -> 爬行回归不动; "
                         "0=关闭=旧行为逐位不变; 证据等级: 机制实测(表型方向)+增益有界估计)")
    ap.add_argument("--dvatau", type=float, default=1.0,
                    help="dvaprop 慢EMA时间常数 s (肽能/多巴胺幅度调控为秒级慢信号: Tao 2019 PVD/NLP-12, "
                         "PDE-dopamine-AVK-FLP-1 环路; 取1.0s>>行波周期~2s同级窗口, 敏感带在案)")
    ap.add_argument("--dvatarget", default="mn", choices=["mn", "muscle"],
                    help="dvaprop 刹车作用点: mn=超极化A类MN(实测反效果, 削弱行波解锁力); "
                         "muscle=按 1/(1+G*|DC|) 缩放肌肉行激活 (Li 2006: DVA抑制体壁肌收缩; "
                         "G 单位 (1/mm)^-1, 推荐0.1-0.4) — 神经侧行波不受损, 直接松开静态弯的收缩")
    ap.add_argument("--avamaster", type=float, default=0.0,
                    help="AVA 主神经元慢紧张兴奋 AVB 增益 pA/mV (v93, Meng 2024 PMC11006214: "
                         "AVA 紧张性去极化经突触外ACh慢兴奋AVB, tau为分钟量级, 使前进为默认态并提供"
                         "倒退->前进的内生退出时钟; 引擎实现: EMA(max(0,dV_AVA-floor)) 兴奋注入AVB, "
                         "dV取上一块50ms延迟同pep2/ggapmod例; 0=关闭=旧行为逐位不变; "
                         "证据等级: 机制实测(双时间尺度模型+TWK-40扰动验证)+增益有界估计)")
    ap.add_argument("--avamastertau", type=float, default=30.0,
                    help="avamaster 慢EMA时间常数 s (Meng 2024: 慢兴奋时间尺度为分钟量级; 取30s保守下界)")
    ap.add_argument("--avamasterfloor", type=float, default=5.0,
                    help="avamaster 计数地板 mV: 只累积 dV_AVA 超此地板的去极化活动 (静息抖动不计)")
    ap.add_argument("--sloe", type=float, default=-80.0,
                    help="sloa 钾平衡电位 mV (生理 K+ 反转 -70~-90, 取标准 -80)")
    ap.add_argument("--mnspan", type=int, default=0,
                    help="B类MN本体感受窗半宽 体节数 (v69 R10a2, Wen et al. 2012 Neuron 76:750: "
                         "感受器是沿神经突分布的, 不是胞体点 -> 感受=胞体位置±span节段曲率均值, "
                         "解剖映射下点感受只剩6个离散位点会冻死振荡, 跨节段窗模拟神经突跨度; "
                         "0=点感受=旧行为逐位不变)")
    ap.add_argument("--mndrive", type=float, default=0.0,
                    help="B类MN->肌肉驱动增益 (v66 负载回路): 解剖支配表 neuron_muscle.csv, "
                         "DB->背侧肌行, VB->腹侧肌行, 驱动 = G * (W @ (softplus(dV)-ln2)); "
                         "0=关闭=旧行为逐位不变")
    ap.add_argument("--dmndrive", type=float, default=0.0,
                    help="D类MN->肌肉交叉抑制增益 (v67 R2, White 1986 成人接线: VD->腹侧肌, "
                         "DD->背侧肌, GABA 抑制性, 从同侧激活中减去; 解剖定位: DD/VD 无长神经突 "
                         "(Cohen 组综述), 故局部无偏移; Ji 2021 eLife: 交叉抑制维持高频率与波形; "
                         "神经级 VD->VB 重置已在 connectome 内 (sign=-1), 此参数补肌肉级半边; "
                         "0=关闭=旧行为逐位不变)")
    ap.add_argument("--muscletau", type=float, default=0.0,
                    help="体壁肌激活的一阶积分时间常数 ms (v68 R7 肌肉层动态, 路线图最高优先: "
                         "Zhao et al. 2025 PLOS Comput Biol 21(6):e1013171 证明爬行发生器="
                         "MN+其支配肌肉的局部负反馈环, 延迟来自肌肉收缩/舒张动力学; "
                         "Shen et al. 2016 Neuron 90:869 DINOs 实测肌肉钙节律). "
                         "肌肉激活 m 以 tau 指数趋近瞬时驱动 drv: dm/dt=(drv-m)/tau, "
                         "在 motor_dt(5ms) 子步上积分; 0=关闭=直通=旧行为逐位不变")
    ap.add_argument("--smddrive", type=float, default=0.0,
                    help="SMD->头肌驱动增益 (v68 R6, Yeon et al. 2018 PLoS Biol 16(6):e2004929 "
                         "doi:10.1371/journal.pbio.2004929: SMDD/V是感觉-运动双角色神经元, "
                         "支配头背/腹肌; Zhao et al. 2025 前部振荡环=SMD+头肌; "
                         "解剖表 neuron_muscle.csv: SMDDL/R->DR/DL01-06, SMDVL/R->VR/VL01-06, "
                         "与B类 mndrive 同公式 softplus(dV)-ln2; 0=关闭=旧行为逐位不变)")
    ap.add_argument("--smdprop", type=float, default=0.0,
                    help="SMD本体感觉反馈增益 pA/(1/mm), 感受信号=头6节段平均曲率 "
                         "(Yeon 2018: SMDD经TRP-1/TRP2感受头部肌肉牵张=牵张反射=负反馈形式, "
                         "取负值: 背弯(κ>0)抑制SMDD/腹弯抑制SMDV; 与Zhao 2025局部负反馈环同构; "
                         "0=关闭=旧行为逐位不变)")
    ap.add_argument("--smdpropseg", type=str, default="0:6",
                    help="v82: SMD本体感觉感受野节段范围 'a:b' (默认 '0:6'=旧行为逐位不变; "
                         "Yeon 2018 原文=颈部肌肉牵张 neck muscle stretch, 颈部=节3:9, "
                         "该段由行波环DB01/02支配节7-9直接驱动=环路的天然点火源)")
    ap.add_argument("--smdabl", type=str, default="",
                    help="SMD反馈消融 (trp-1,2双突变模拟, Yeon 2018: 突变/消融->腹侧绕圈): "
                         "D=关闭SMDD反馈, V=关闭SMDV反馈, 空=不消融; 仅 smdprop!=0 时生效")
    ap.add_argument("--smdpropbw", type=int, default=0,
                    help="W57: 倒退态自动关闭SMD反射 (1=开, dir_state<0时smdprop增益置0; "
                         "w57d/e判决: 常开反射是倒退打转主因; 0=关闭=旧行为逐位不变)")
    ap.add_argument("--muscle96", type=float, default=0.0,
                    help="v81 自有生理分组肌肉层总增益 (src/muscle96.py): 96 体壁肌各自一阶积分, "
                         "驱动=全网统一释放表 rel=clip((V-vrel)/25,0,1.5) 经解剖支配矩阵 "
                         "(neuron_muscle.csv, ACh正/DD-VD GABA负), 常数全部文献锚定 "
                         "(data/muscle96_constants.csv); !=0 时取代 Boyle 砰-砰层+mndrive/"
                         "dmndrv/smddrive 驱动半边 (kappa0 由 A_D-A_V 出, koff 转向偏置保留); "
                         "0=关闭=旧行为逐位不变")
    ap.add_argument("--m96tau", type=float, default=80.0,
                    help="muscle96 肌肉激活时间常数 ms (C1.8: Liu 2011 AP半宽~20ms + "
                         "Gao&Zhen 2011 收缩时序 + P3 校准定死; 永不作频率旋钮)")
    ap.add_argument("--m96tau2", type=float, default=0.0,
                    help="muscle96 两层级联慢层(钙/收缩)时间常数 ms (P10, 2026-10-22: "
                         ">0 时快层=m96tau 取 Gao&Zhen 2011 AP半宽锚 15.5ms, 慢层=钙瞬变衰减锚 "
                         "~1.0-1.57s (J R Soc Interface 2015;12:20140963 实测); "
                         "直流增益不变只改动态/相位; 0=关闭=旧单层逐位不变)")
    ap.add_argument("--m96amax", type=float, default=1.5,
                    help="muscle96 单肌激活饱和上限 (线性平滑饱和, 与旧路径 clip 上限一致; "
                         "submaximal 设计, Liu 2021 Sci Adv)")
    ap.add_argument("--m96sig", default="rel", choices=["rel", "sp"],
                    help="muscle96 驱动信号: rel=vrel 释放表 clip((V-vrel)/25,0,1.5) (默认, 生理形式); "
                         "sp=softplus(dV)-ln2 偏差驱动 (mndrive 同式, 对 vrel 失配免疫; "
                         "v81 P3 实测: rel 因 vrel 表失配大面积饱和, 动态区间被压扁)")
    ap.add_argument("--m96gate", type=float, default=1.0,
                    help="muscle96 方向门控开关: 1=前进态只放行B/DD类,倒退态只放行A/VD类 "
                         "(命令级门控, 旧 Boyle g_f/g_b 同语义); 0=全类常通 "
                         "(v81 P3 对照: mnback 架构下倒退波也由 B 环反向携带, "
                         "门控会断倒退执行端, 默认开仅在纯前进场景验证过)")
    ap.add_argument("--m96homeo", type=float, default=0.0,
                    help="muscle96 稳态张力校准时间常数 秒 (v86: 基线 sig0_h 慢跟随各神经元 "
                         "tonic 释放, 吸收 tonic 背腹不对称 -> 消除静态弯偏置(爬行左拐), "
                         "只透过波频分量; 0=关闭=旧行为逐位不变)")
    ap.add_argument("--m96ei", type=float, default=0.6,
                    help="P7 常数锚定 (2026-10-21): NMJ 抑制/兴奋电流比 E:I。实测 0.61 "
                         "(Liu/Hollopeter/Jorgensen PNAS 2009 PMC2705609: 全ACh诱发2.01nA "
                         "vs 全GABA诱发1.23nA) → 默认 0.6; 1.0=旧行为(无锚, 仅对照用)")
    ap.add_argument("--m96gj", type=float, default=0.0,
                    help="P11 肌肉-肌肉缝隙连接层强度乘子 (2026-10-23, Liu/Chen/Wang 2011 JBC "
                         "286:44285 PMC3243499: 相邻肌AP/钙同步由GJ负责且不依赖神经输入, 纵向同象限"
                         "耦合系数实测0.25(unc-9敲除->0.1失同步+运动缺陷); Liu 2006 JBC 281:7881 "
                         "肌腹间+肌臂间耦合; White 1986 同象限肌腹GJ)。1.0=锚定强度(纵向c=1/3, "
                         "横向x0.73有界估计), 稳态三对角解析解; 0=关闭=旧行为逐位不变)")
    ap.add_argument("--turnsmd", type=float, default=0.0,
                    help="转向经SMD执行的脉冲电流 pA (v68 R9: Gray et al. 2005 PNAS 102:3184 "
                         "doi:10.1073/pnas.0409009101: SMD编码omega转向的陡峭幅度[消融后转角"
                         "137->72度]; event脉冲触发时改为 SMDV兴奋+SMDD抑制的差分注入, "
                         "经 smddrive 解剖驱动头肌完成转向, 替代人工koff曲率偏置; "
                         "0=关闭=旧行为逐位不变[仍用koff])")
    ap.add_argument("--fc1", type=float, default=0.0,
                    help="FC1 转向执行端内生增益 (v74 R18, 全闭环战役): 事件触发时不注人工脉冲, "
                         "改用 RIA->SMD 原位突触预算的 fc1 倍作为差分驱动幅度 (预算由上一块电压"
                         "在线重建, 状态依赖); 机制锚 Kramer 2026 RIM酪胺门控头部运动链 + "
                         "Hendricks 2012 RIA转向编码 + Gray 2005 SMD编码转角; "
                         "0=关闭=旧行为逐位不变 (仍用 turnsmd 人工脉冲)")
    ap.add_argument("--fc1tau2", type=float, default=3.0,
                    help="FC1 门控衰减时间常数 s (pirouette bout 时标, Pierce-Shimomura 1999)")
    ap.add_argument("--fc1dir", default="steer", choices=["steer", "random", "fixed", "weath"],
                    help="FC1 方向来源: steer=AIY(R-L)符号(R19判决: 只编时间导数不编空间方向, 镜像证伪) | "
                         "random=pirrand重掷 | fixed=+1腹向 | weath=v77 R22: w_weath相关积分符号 "
                         "(kap载波一致率0.92, 唯一实测携带空间方向的信号; pulse_dir=-sign(w), "
                         "w<0=腹侧上梯度->腹弯+1; 连续注入仍由weath增益控制, 决策层独立于增益)")
    ap.add_argument("--wrst", type=float, default=0.0,
                    help="W68 (2026-10-24): bout末沿 w_weath 重置+冻结窗 s (efference-copy 延伸: "
                         "pirouette bout 的 omega 深弯改变朝向且曲率残形需~1行波周期耗散, "
                         "旧 w 基于旧朝向的空间认知作废, 残形期采样被污染 -> 末沿清零并冻结累积 "
                         "wrst 秒; 锚 Pierce-Shimomura 1999 bout后直行run + v77 R22b 机动期冻结同思想; "
                         "0=关闭=旧行为逐位不变)")
    ap.add_argument("--fc2", type=float, default=0.0,
                    help="FC2 前进tonic内生增益 pA (v75 R20): AVA突触外tonic ACh基调->AVB "
                         "(Meng et al. Sci Adv 2024, 10.1126/sciadv.adk0002: AVA静息~-23mV构成性"
                         "释放, TWK-40 K2P维持, 双时相门控AVB); 电压门控: AVA>-45mV时上线, "
                         "钳离(HisCl模拟)则关闭 -> 可检验预测; 0=关闭=旧行为逐位不变(用stim)")
    ap.add_argument("--fc2v0", type=float, default=-45.0,
                    help="FC2 电压门控阈值 mV (主细胞论文Fig1: 超极化区-75..-30 / 去极化区-30..-10 分界)")
    ap.add_argument("--fc2vs", type=float, default=15.0, help="FC2 门控斜坡宽度 mV")
    ap.add_argument("--fc2cap", type=float, default=1.3, help="FC2 门控上限 (AVA反转期进一步去极化时)")
    ap.add_argument("--avaclamp", type=float, default=0.0,
                    help="AVA 恒定钳制电流 pA (负=超极化, HisCl/Fig6A 剂量实验模拟; 0=不钳)")
    ap.add_argument("--fc2mn", type=float, default=0.0,
                    help="FC2 第二臂: 同一AVA电压门控的 tonic -> B类MN池 (主细胞论文: 突触外胆碱能"
                         "传输紧张性强化腹侧运动亚环路[L1幼虫证据]; 替代 mntonic 人工强直; 0=关闭)")
    ap.add_argument("--avaplat", type=float, default=0.0,
                    help="FC3 AVA 再生性平台电流电导 pA/mV (v76 R21): VGCC式电压门控正反馈 "
                         "(Mellem 2008 Nat Neurosci 11:865 UNC-2/EGL-19平台电位; Meng 2024 Sci Adv "
                         "Fig1C: AVA双稳区 -30..-20 / -15..-10 mV); 快激活m+慢失活n(bout自终结); "
                         "替代 avahold 人工自持; 0=关闭=旧行为逐位不变")
    ap.add_argument("--avaplatth", type=float, default=-16.0,
                    help="平台电流激活半阈 mV (双稳区分界, Meng 2024 相图)")
    ap.add_argument("--avaplatk", type=float, default=1.5, help="激活斜坡陡度 mV")
    ap.add_argument("--avaplattau", type=float, default=0.2, help="m 激活时间常数 s")
    ap.add_argument("--avaplattau2", type=float, default=4.0, help="n 失活时间常数 s (bout时长)")
    ap.add_argument("--avaplate", type=float, default=60.0, help="Ca 反转电位 mV (驱动势)")
    ap.add_argument("--avaplat2", type=float, default=0.0,
                    help="v91: 门控沿点火 AVA 平台电流电导 pA/mV — 电压阈路径在引擎压缩电压尺下 "
                         "不可用 (引擎 AVA 行程 ~2-5mV, 全程峰值 -30.6 vs 阈 -16); 改用倒退门控翻转 "
                         "(dir_state<0) 为点燃信号 = v77 R23 pirrevp 事件点火的内生化 (事件=网络自身 "
                         "门控翻转而非外部协议). 结构/τ/E 同 avaplat (Meng 2024 上稳态区制 + "
                         "UNC-2/EGL-19 样 VGCC 再生电流); 0=关闭=旧行为逐位不变")
    ap.add_argument("--avaplat2tau", type=float, default=0.2, help="avaplat2 m 激活时间常数 s")
    ap.add_argument("--avaplat2tau2", type=float, default=4.0,
                    help="avaplat2 n 失活时间常数 s (=bout 自终结时长, Hilliard 2005 倒退 bout 1-3s 量级)")
    ap.add_argument("--bwrefr", type=float, default=0.0,
                    help="v91: 倒退 bout 后再进入不应期 秒 (Wicks & Rankin 1995: 倒退后反应抑制); "
                         "bout 末沿起计时, 窗内封锁 relg 前进->倒退门控翻转, 让 AVB 前进程序重建、"
                         "身体解开深弯、切断本体感觉-倒退态混响 (v91 短脉冲实验: 无此则 bout-omega "
                         "极限环 50s 不落回爬行态); 0=关闭=旧行为逐位不变")
    ap.add_argument("--risbrk", type=float, default=0.0,
                    help="W59: RIS刹车计入relg倒退池的耦合系数 (Kato 2015 Cell 163:656 bout终止器; "
                         "dA池 -= risbrk*max(dRIS-risbrkth,0); 0=关闭=旧行为逐位不变)")
    ap.add_argument("--risdrive", type=float, default=0.0,
                    help="W59b: RIS bout终止积分点火电流 pA (Kato 2015: RIS在reversal结束时兴奋; "
                         "倒退态充电/前进态放电, 经网内GABA突触自然终结bout; 0=关闭=旧行为逐位不变)")
    ap.add_argument("--riskill", type=float, default=0.0,
                    help="W69 (2026-10-24): RIS满充自限保险丝 s — 倒退态 ris_m 满充(>0.99)持续超过 "
                         "riskill 秒后强制走放电路径. Kato 2015: RIS兴奋是bout末沿瞬态峰(Ca峰宽~1-2s), "
                         "非持续态; 修复AVB消融态僵死(补四十六: AVB死->前进态永不达->RIS卡满充+"
                         "risbrk持续压AVA池->双门全关僵死17s); 兼作bout时长上限保险丝 "
                         "(Zhao 2003: bout罕见>6s). 0=关闭=旧行为逐位不变)")
    ap.add_argument("--ristau", type=float, default=1.5,
                    help="W59b: RIS积分充电时间常数 s (锚 Zhao 2003 bout中位2-3s, 取1.5)")
    ap.add_argument("--ristau2", type=float, default=0.3,
                    help="W59b: RIS积分放电时间常数 s (bout结束后RIS快速撤, Kato2015瞬变短)")
    ap.add_argument("--risbrkth", type=float, default=0.0,
                    help="W59: RIS刹车起算阈 dV mV (RIS爬行tonic≈-6.6,  bout峰+50~86, 阈取0)")
    ap.add_argument("--mntonic", type=float, default=0.0,
                    help="B类MN恒定去极化注入 pA (v67 R1 诊断: 腹索MN电压弱耦合|dV|<~0.1mV "
                         "趴在静息点, softplus(0)-ln2=0 -> mndrive环在原点增益为零无法点燃; "
                         "真虫B类MN爬行时有AVB/PVC tonic驱动叠加, 此参数模拟缺失的强直工作点; "
                         "0=关闭=旧行为逐位不变)")
    ap.add_argument("--vdsp", type=float, default=0.0,
                    help="W62 (2026-10-24): A类MN->D类MN 的 sp 释放通道增益 (绕 vrel 失配). "
                         "病灶: v62 基座 A-MN 全程(爬行+bout)运行于释放阈下 (dV~-35), rel 表释放=0 "
                         "-> VD/DD 全程静默 -> 肌肉抑制 sculpting 缺失 -> bout 静态净弯 coil 2.0. "
                         "实现: Iex[D] += vdsp * (W_ad @ (softplus(dV_prev)-ln2)), W_ad=A->D 化学边"
                         "解剖分布 (White 1986/Cook 2019 在网); sp 形式同 muscle96/mndrive 例 "
                         "(NMJ 释放跟随突触前去极化=生理第一性). 0=关闭=旧行为逐位不变)")
    ap.add_argument("--vdspx", type=float, default=0.0,
                    help="W64 (2026-10-24, 预注册补三十七): vdsp 亚类分档 — 非0时从 A->D sp 通道"
                         "剔除靶为 VD12/VD13 的边 (bout期活动的VD12/13经真实解剖边 VD12/13->VA12"
                         "(GABA ~19节EM, sign=-1) 压制尾段腹侧驱动元VA12, VD13另直接抑制腹侧尾肌 -> "
                         "vdsp剂量越高尾段背弯越重=0.6反噬机制; 保留VD1-11中身段sculpting收益. "
                         "0=关=旧行为逐位不变)")
    ap.add_argument("--headosc", type=float, default=0.0,
                    help="头部B类MN正弦种子注入振幅 pA (v67 R1b, Wen 2012 chain-of-reflexes 检验: "
                         "波由头部节律发起, B-MN反射链传播; DB/VB反相, 作用于胞体在前2节段的B类; "
                         "0=关闭=旧行为逐位不变)")
    ap.add_argument("--headfreq", type=float, default=0.3,
                    help="headosc 频率 Hz (真虫爬行波频 0.2-0.5 Hz)")
    ap.add_argument("--boylescale", type=float, default=1.0,
                    help="Boyle 运动层输出乘子 (默认1=里程碑行为; <1 削弱肌肉层牵张反馈, "
                         "用于检验 MN 驱动是否真正负重)")
    ap.add_argument("--kb", type=float, default=5.0,
                    help="Rod2D 弯曲刚度 k_b = EI/ell (材料常数, 能量项 (k_b/2)(dphi-ell*kappa0)^2; "
                         "默认5.0=逐位不变. 锚: Fang-Yen 2010 PNAS 内弹性动态实测 -> k_b~0.7; "
                         "Backholm 2013 PNAS 三点弯 -> 0.04-50; 爬行regime约束(弹性~阻力)->O(1); 5.0在窗内)")
    ap.add_argument("--cn", type=float, default=40.0,
                    help="Rod2D 法向阻力比 Cn/Ct (Ct=1归一化). 锚: Boyle 2012 K_agar=30-40取40, "
                         "Berri 2009 运动模拟器拟合; 水=1.5 (Gray&Lissmann 1964)")
    ap.add_argument("--rod2d", default="borrowed", choices=["borrowed", "own"],
                    help="身体层来源: borrowed=worm-whisperer 原版件(默认, 旧行为逐位不变); "
                         "own=自有解析梯度件 src/rod2d_own.py (补三十三, 对齐验证 5 项 PASS, 差~1e-7mm)")
    ap.add_argument("--ald", type=float, default=0.0,
                    help="aldicarb 强度: AChE 阻断导致的 ACh 累积增益上限 (0=不给药)")
    ap.add_argument("--aldtau", type=float, default=5.0, help="aldicarb 起效时间常数 s")
    ap.add_argument("--proto", default="none",
                    help="光遗传协议: switch=AVB10s->AVA12s->AVB8s | ris=AVB全程+RIS12-18s | region=静默分区8-16s | ashchem=ASH化学换能前端8-16s")
    ap.add_argument("--ashchem", type=float, default=0.0,
                    help="ASH 化学前端满幅电流 pA (P11, 2026-10-22; proto=ashchem 时生效; "
                         "Hilliard 2005 EMBO J 24:63 PMC544906: ASH 对驱避剂'存在'持续响应(<10s不降), "
                         "换能链 GPCR->ODR-3/GPA-3->OSM-9(TRPV)->EGL-19 钙; 0=关闭)")
    ap.add_argument("--ashchemtau", type=float, default=1.0,
                    help="ASH 换能 onset 时间常数 秒 (Hilliard 2005 Fig1D/3A: 3s 窗内充分上升, "
                         "上升判据>=2%%/s -> tau~1s 有界估计)")
    ap.add_argument("--ashchemtaua", type=float, default=20.0,
                    help="ASH 适应充电时间常数 秒 (Hilliard 2005 Fig3B: 60s 暴露 ~30s 后衰减 "
                         "-> tau_adapt~20-30s; GPC-1(Ggamma)介导, gpc-1 突变体重复刺激不降)")
    ap.add_argument("--ashchemtaur", type=float, default=100.0,
                    help="ASH 适应恢复时间常数 秒 (Hilliard 2005 Fig5F: 1min 暴露后 1min 仍抑制, "
                         "5min 基本恢复 -> tau_recov~100s 量级)")
    ap.add_argument("--ashchemwin", default="8:16",
                    help="ASH 浓度方波窗口 'start:end' 秒 (默认 8:16 = region 协议同时序; "
                         "v91 短脉冲实验用 8:10 等 — Hilliard 2005 范式: 短促厌刺激->单个倒退bout")
    ap.add_argument("--ashmod", type=float, default=0.0,
                    help="P13: ASH 音量旋钮深度 K (Chaubey 2023 PMC10319955: INX-20/cGMP/EGL-4 下调层; "
                         "增益=1/(1+K*ash_mod); 0=关闭=旧行为逐位不变)")
    ap.add_argument("--ashstarve", type=float, default=0.0,
                    help="P13: 饥饿驱动 0-1 (Chaubey 2023: food deprivation->ODR-1 产 cGMP->INX 网入 ASH "
                         "->敏感度下调; ash_mod 以 tau=60s 升至该值, 复食 tau=300s 回落)")
    ap.add_argument("--region", default="",
                    help="proto=region 时的分区: head_sensory|chemo_sensory|head_motor|tail_sensory|mod_ala_hsn")
    ap.add_argument("--pamp", type=float, default=50.0, help="光遗传刺激幅度 pA")
    ap.add_argument("--phasic", type=float, default=0.0,
                    help="region 协议的相位脉冲宽度 ms (O'Hagan et al. 2005 Nat Neurosci 8:43: TRN 快适应, "
                         "只对力的 onset/offset 响应, 持续力不响应 -> 生理刺激是脉冲不是强直直流; "
                         ">0 时 region 电流只在窗口起点 [8.0, 8.0+w] 和终点 [16.0-w, 16.0] 施加; "
                         "0=全程强直=旧行为逐位不变)")
    ap.add_argument("--avahold", type=float, default=0.0,
                    help="AVA 平台自持电流幅度 pA (v70 R16, Meng et al. 2024 Sci Adv 10:adk0002: "
                         "AVA 是紧张性去极化的主神经元, 平台电位性质 -> 倒退 bout 在刺激结束后自持; "
                         "倒退触发沿充电到该幅度, 之后无条件按 avaholdt 指数衰减 -> bout 自发终结; "
                         "0=关闭=旧行为逐位不变)")
    ap.add_argument("--avaholdt", type=float, default=1500.0,
                    help="AVA 平台自持电流的衰减时间常数 ms (bout 时长的控制旋钮; "
                         "真虫触觉倒退 bout 约 2-3s, Wicks & Rankin 1995)")
    ap.add_argument("--avatonc", type=float, default=0.0,
                    help="AVA 恒定紧张性去极化电流 pA (v92 稳态倒退协议: Kato 2015 Cell 163:656 "
                         "稳态倒退=AVA 中度tonic占优驱动的持续行为, 非逃逸bout; 全程常数注入AVA, "
                         "不经触觉触发/不要avaplat2强平台 -> 温和满全身反波; 0=关闭=旧行为逐位不变)")
    ap.add_argument("--pep2", type=float, default=0.0,
                    help="倒退环路肽能正反馈增益 pA (v73 R17, 漏洞1修复实验: "
                         "FLP-14/FRPR-19 正反馈环 Marques 2021 PLoS Genet 17:e1009880; "
                         "MGL-2/mGluR5 突触前增益 Katz 2019 Nat Commun 10:1701; "
                         "RIM->AVA 倒退态强化 Sordillo 2021 eLife 10:e67723; "
                         "m2 由倒退组池(AVA+AVD+AVE)去极化慢积累驱动, 电流回注组池; "
                         "0=关闭=旧行为逐位不变)")
    ap.add_argument("--pep2tau", type=float, default=3.0,
                    help="肽能正反馈慢变量时间常数 s (肽能信号为秒级, 同 peptau 量级)")
    ap.add_argument("--pep2th", type=float, default=0.8,
                    help="反馈点火阈值 mV: 组池去极化低于此值不积累 "
                         "(设计: 高于爬行波动 ~0.3, 低于 ASH 感觉种子 ~1.2)")
    ap.add_argument("--pep2ref", type=float, default=1.0,
                    help="反馈饱和标度 mV: 组池去极化超出阈值此幅度时 m2 到顶")
    ap.add_argument("--pep2tau2", type=float, default=8.0,
                    help="负臂 n2 时间常数 s (v73 R17b: 模拟 AVR-14 氯通道延迟抑制, "
                         "Sordillo 2021 eLife; n2 慢跟踪 m2, 净驱动=m2-n2 -> bout 自发终结)")
    ap.add_argument("--pep2inh", type=float, default=1.0,
                    help="负臂权重: 净回注 = pep2*(m2 - pep2inh*n2); 1=完全对消")
    ap.add_argument("--chemsrc", default="",
                    help="proto=chem 浓度源坐标 'x,y' mm (留空=无源对照)")
    ap.add_argument("--chemsigma", type=float, default=2.0,
                    help="v72: 浓度场高斯半径 mm (默认2.0=旧行为; 真实趋化板点源过夜扩散"
                         "梯度有效范围为厘米级, Pierce-Shimomura 1999 / Iino & Yoshida 2009)")
    ap.add_argument("--chemgain", type=float, default=50.0,
                    help="ASE 电流增益 pA per (浓度/s)")
    ap.add_argument("--net", default="", help="网络 npz 路径 (默认 data/net_gapfix.npz)")
    ap.add_argument("--gsyn", type=float, default=0.0, help="全局化学突触电导增益 (0=默认0.005)")
    ap.add_argument("--tauex", type=float, default=0.0,
                    help="兴奋性突触电导时间常数 ms (P12 敏感性测试; 0=基线2.0, NMJ括号快沿锚)")
    ap.add_argument("--tauinh", type=float, default=0.0,
                    help="抑制性突触电导时间常数 ms (P12 敏感性测试; 0=基线5.0)")
    ap.add_argument("--ggap", type=float, default=0.0, help="全局电突触增益 (0=默认0.3)")
    ap.add_argument("--out", default="", help="输出目录覆盖 (默认 runs/v62_closed_loop; 纯路径基础设施, 数值零影响)")
    ap.add_argument("--vrel", default="", help="原位释放阈表 .npy (v59: vrel=V_rest原位-10; 留空=旧行为孤立平衡)")
    ap.add_argument("--vrelset", default="",
                    help="逐细胞 vrel 覆盖 'AVAL:-43.3,AVAR:-44.7' (W50: AVA vrel=-83 致释放饱和修复, "
                         "账本补十二; 需配合 --vrel; 缺省=逐比特不变)")
    ap.add_argument("--kturn", type=float, default=0.0, help="转向增益覆盖 (0=默认0.03; v62 趋化校准候选 30-100)")
    ap.add_argument("--gate", default="abs", choices=["abs", "rel", "relg"], help="门控制式: abs=旧绝对阈值3mV; rel=v60相对(AVA vs AVB); relg=v61组池相对(倒退侧AVA+AVD+AVE, Chalfie 1985)")
    ap.add_argument("--turnmode", default="linear", choices=["linear", "pirouette", "event"],
                    help="转向映射: linear=旧行为 head_bias=k_turn*dSMD 双相固定转向; "
                         "pirouette=v63 连续门控, 升梯度直行降梯度转向 (有-90°轨道不动点); "
                         "event=v64 事件化 pirouette, dSMD 上穿阈触发定长转向脉冲+不应期, "
                         "打破轨道不动点 -> 螺旋逼近 (Pierce-Shimomura 1999 离散 pirouette)")
    ap.add_argument("--smdthresh", type=float, default=0.0005, help="pirouette/event 模式 dSMD 触发阈 mV")
    ap.add_argument("--steerthresh", type=float, default=1.0, help="turnsrc=aiy 模式 AIY(R-L) 触发阈 mV (v64c 定案值)")
    ap.add_argument("--pulsebias", type=float, default=0.6, help="event 模式转向脉冲头部曲率偏置")
    ap.add_argument("--pulsedur", type=float, default=3.0, help="event 模式脉冲时长 s")
    ap.add_argument("--refract", type=float, default=2.0, help="event 模式不应期 s")
    ap.add_argument("--turnsrc", default="smd", choices=["smd", "aiy", "dwn"],
                    help="event 转向信号源: smd=SMD(D-V)(旧, 被头摆运动程序污染, 一致率0.70); "
                         "aiy=AIY(R-L)(v64c, 与 ASER-ASEL 一致率 1.00, 分离度19.8mV, "
                         "AIY 即生物趋化比较器, Iino & Yoshida 2009); "
                         "dwn=恶化速率驱动(v83, Pierce-Shimomura 1999: pirouette 率随 dC/dt<0 升高; "
                         "steer=-ase.dcdt, 稳态远离时持续超阈可重触发, 修复 AIY 差分信号稳态回零不触发)")
    ap.add_argument("--dwnthresh", type=float, default=0.005,
                    help="turnsrc=dwn 恶化触发阈 (单位=-dcdt, 浓度/s; 仅在 dC/dt 恶化超阈时触发转向)")
    ap.add_argument("--escomega", type=float, default=0.0,
                    help="B4 逃逸-omega 串联开关: 0=关=旧行为逐位不变; !=0 时倒退 bout 末沿 "
                         "(方向态 -1->+1 跳变) 额外触发一次 event 转向脉冲 (真虫 ASH 回避="
                         "倒退 bout+omega 串联, Hilliard 2002/2005; 耦合载体文献未定论 "
                         "(候选: bout 后 RIV/SMD 去抑制反弹), 本实现以内生方向态边沿为触发, "
                         "执行沿用 v64 event 脉冲层 — 半内生, 已在限制节披露)")
    ap.add_argument("--dwntau", type=float, default=3.0,
                    help="v83b: dwn 触发前对 -dcdt 再做慢低通的时间常数 s (杀灭 0.7Hz 头摆泄漏:"
                         "头摆侧向速度对瞬时 dcdt 的调制幅 ~±0.03 > 净趋化信号 ±0.025, "
                         "不过滤则转向触发锁相头摆相位; PS99 转向率调制本身是秒级慢变量)")
    ap.add_argument("--steerbi", type=float, default=0.0,
                    help="R19 决策层双向触发 (2026-10-20): 0=单向 steer>+thr=旧行为逐位不变; "
                         "1=|steer|>thr 双向触发, 配合 fc1dir=steer 时 dir=sign(steer); "
                         "判决结果=负: steer符号编码时间导数(远离/接近)而非空间方向, "
                         "镜像几何证伪(接近-2.55mm vs -0.57mm), 且负支在两几何下均未点燃 "
                         "(steer从不<-thr) — 旋钮保留仅供复核, 判决详见 全闭环战役路线图 R19")
    ap.add_argument("--weath", type=float, default=0.0,
                    help="v71: weathervane 双向转向增益 pA (0=关闭; klinotaxis, Iino & Yoshida 2009)")
    ap.add_argument("--weathtau", type=float, default=2.0, help="weathervane 头摆×dC/dt 相关低通时间常数 s")
    ap.add_argument("--weathmode", default="dkap", choices=["dkap", "kap"],
                    help="v77 R22: weathervane 载波: dkap=dκ0/dt代理(旧=旧行为逐位不变; 符号审计"
                         "证明其与头摆真实相位失锁, w一致率仅0.06=翻号垃圾) | "
                         "kap=头关节弯曲位置tanh(κ0/3)(载波离线判决: 一致率0.92, w方向正确; "
                         "鼻尖朝向采样浓度场, 本体感觉关节角直接可得, Yeon 2018; "
                         "世界系摆速载波vlat已淘汰: 体架投影消掉摆动, 一致率0.58量级可忽略)")
    ap.add_argument("--pirrevp", type=float, default=0.0,
                    help="v77 R23: pirouette-倒退平台耦合 (0=关闭=旧行为逐位不变): 转向事件对AVA平台门 "
                         "m_plat 点火充电 -> FC3再生性VGCC平台驱动倒退bout杀前进动量 "
                         "(Pierce-Shimomura 1999: pirouette=倒退+omega转向)。"
                         "判决结果=负(2026-10-20): 0.5/1.0两点均致 bout 占空~75%%+前行 regime 崩坏 "
                         "(v_fwd~0.005, freq 0.18Hz, κ~8), 且SMD驱动τ3s先于~4s bout衰减, "
                         "重定向时机错配 — 旋钮保留仅供长窗口run-and-tumble复核, 不入默认栈")
    ap.add_argument("--pirrand", type=int, default=0,
                    help="v71: 随机方向 pirouette 种子 (0=固定腹向=旧行为; >0=每次转向背/腹向随机, Pierce-Shimomura 1999)")
    ap.add_argument("--pirrev", type=float, default=0.0,
                    help="v72: pirouette 倒退耦合 AVA 充电幅度 pA (0=关闭=旧行为逐位不变; "
                         "Pierce-Shimomura 1999 J Neurosci 19:9557: pirouette=倒退bout+omega转向, "
                         "倒退消除前进动量使头弯在小半径完成; 复用 avaholdt 衰减, bout~2-3s, "
                         "Wicks & Rankin 1995)")
    ap.add_argument("--pirrevang", type=float, default=0.0,
                    help="v76: pirrev 方位误差门控阈值 deg (0=关闭门控=旧行为逐位不变; "
                         ">0 时仅当 |朝向-源方位误差| >= 阈值才允许 pirouette 点火 AVA, "
                         "协议块用 90=背向源半球才点火, Pierce-Shimomura 1999 pirouette 处置大误差 "
                         "重定向 / Iino 2009 小误差 weathervane 分工; 999=永不点火; "
                         "W75 确诊: d090 中等误差 pirrev 误点火致 dAVA 57.5%% 时间点燃、接近 "
                         "+3.46mm->+0.91mm 退化; 仅 proto chem 有源 (ase 非 None) 时生效)")
    ap.add_argument("--refrmod", type=float, default=0.0,
                    help="v73: 转向频率调制参考电平 mV (0=固定不应期=旧行为; >0 时 "
                         "refract_eff=refract*clip(refrmod/steer,0.3,1.0), 恶化越快转向越密, "
                         "Pierce-Shimomura 1999 pirouette 频率调制)")
    ap.add_argument("--pirsup", type=float, default=999.0,
                    help="v78 R25: pirouette 接近抑制阈 (默认 999=关闭=旧行为逐位不变; "
                         "否则仅当低通 dC/dt < pirsup (浓度/s, 峰归一) 才允许转向事件再触发, "
                         "Pierce-Shimomura 1999 核心: 浓度升(接近源)抑制 pirouette 直行, "
                         "浓度降(恶化)才转向; 近源驻留/arrest 的时间门)")
    ap.add_argument("--pirc", type=float, default=999.0,
                    help="v78 R25b: 绝对浓度驻留门 (默认 999=关闭=旧行为逐位不变; 否则当 "
                         "即时浓度 >= pirc (峰归一, 0~1) 抑制转向触发——环内梯度归零时靠浓度水平 "
                         "停驻, Sawin 2000 基础减速 / Hills 2004 局域搜索 / Chalasani 2007 "
                         "AWC 局域搜索的绝对浓度通道)")
    ap.add_argument("--pirmag", type=float, default=0.0,
                    help="v78 R25b: 转向锐度误差标定参考 mV (0=定长脉冲=旧行为; >0 时 "
                         "pulsedur_eff=pulsedur*clip(|steer|/pirmag,1,3), 大误差->深长 omega 弯, "
                         "Pierce-Shimomura 1999 转角-方位误差相关)")
    ap.add_argument("--arrc", type=float, default=999.0,
                    help="v78 R25c: 绝对浓度减速阈 (默认 999=关闭=旧行为逐位不变; 低通浓度 "
                         ">= arrc (峰归一) 时 mndrive 乘 arrg 减速——Sawin 2000 Neuron 基础减速 "
                         "反应 (多巴胺 CEP/ADE/PDE, 高浓度/食物->减速驻留), Hills 2004 局域搜索)")
    ap.add_argument("--arrg", type=float, default=0.35,
                    help="v78 R25c: 减速激活时 mndrive 增益 (生物基础减速幅度 ~降至1/3-1/2)")
    ap.add_argument("--arrtau", type=float, default=2.0,
                    help="v78 R25c: 浓度低通时间常数 s (多巴胺紧张性信号时标, 防抖)")
    ap.add_argument("--asad", type=float, default=0.0,
                    help="v79 R26: GPCR级感觉适应强度 (默认0=关闭=旧行为逐位不变; 持续高浓度区适应态"
                         "上升 -> ASEL/ASER注入与weathervane相关信号同步衰减 -> 峰上tether消退 -> "
                         "自发播散——ARR-1/beta-arrestin早期脱敏 (Palmitessa 2005 JBC 280:24649, "
                         "Merritt 2022 PNAS 119:e2119950119), AWA快脱敏保导数表征 (Larsch 2013), "
                         "行为级嗅觉适应 (Colbert & Bargmann 1995 Neuron 14:803)")
    ap.add_argument("--asadtau", type=float, default=30.0,
                    help="v79 R26: 适应态时间常数 s (ARR-1早期+中间时标的有效复合, 如实标注为现象学时标)")
    ap.add_argument("--asadth", type=float, default=0.5,
                    help="v79 R26: 适应激活阈 (峰归一低通浓度; 仅持续高浓度区推动适应, 远场/接近段不受影响)")
    ap.add_argument("--offg", type=float, default=0.0,
                    help="v79 R27: OFF-response/局域搜索转向增强增益 (默认0=关闭=旧行为逐位不变; 近源"
                         "(低通浓度>offth)触发转向时: 脉冲时长 x=ars 且再触发间隔 /=ars, ars=1+K*归一浓度 "
                         "-> 高转向率打转 milling 几何——Hills 2004 Neuron 43:349 离食转向率升(多巴胺依赖), "
                         "Chalasani 2007 Nature 450:63 AWC OFF-resp->omega, Gray 2005 PNAS 102:3184, "
                         "Lopez-Cruz 2019 Neuron 离食局部搜索; dwelling几何=低速+高转向 (Flavell 2013)")
    ap.add_argument("--offth", type=float, default=0.5,
                    help="v79 R27: 转向增强激活阈 (峰归一低通浓度, 语义同 asadth)")
    ap.add_argument("--savestate", default="", help="跑完保存完整闭环状态 pickle (断点续跑)")
    ap.add_argument("--loadstate", default="", help="从 pickle 恢复状态续跑 (跳过预热, t/身体/门控/ASE 全连续)")
    # ---- RIA 三区室 (漏洞2根治, 规格书 RIA三区室原型_机制规格书_2026-10-19.md; 账本§62) ----
    ap.add_argument("--riacomp", type=int, default=0,
                    help="RIA 三区室再生性树突 (0=关闭=旧行为逐位不变; 1=开启: "
                         "RIA->SMD D/V 输出与 SMD->RIA 反馈重路由进 nrD/nrV 区室, "
                         "AIY 抑制经 ACC-2 等效分流门控, Liu 2018/Hendricks 2012 机制)")
    ap.add_argument("--riagax", type=float, default=0.05, help="区室轴突耦合电导 nS (电缆锚 0.005-0.5)")
    ap.add_argument("--riaacc2", type=float, default=5.0, help="ACC-2 等效分流增益 (Liu 2018 感觉抑制)")
    ap.add_argument("--riaroute", type=float, default=-1.0,
                    help="区室输出路由: -1=交叉(nrV->SMDD, miniSOG 实锤, 默认); +1=同侧(对照)")
    # ---- v3 钙库局部释放门控 (Hendricks 2012 机制, 账本§64; 需 riacomp=1) ----
    ap.add_argument("--riaca3", type=int, default=0,
                    help="v3 钙库版: 区室活性=GAR-3慢钙变量, 加性门控局部释放 (默认0=v2电压区室)")
    ap.add_argument("--riakmod", type=float, default=1.0,
                    help="v3 释放门控增益 (O(1)锚: rig batch20 实测 rel_typ/c_typ=48.7)")
    ap.add_argument("--riatau", type=float, default=1.0,
                    help="v3 钙库时间常数 s (GPCR->ER 慢尺度, 文献量级 0.3-3)")
    ap.add_argument("--riasomaadd", type=float, default=0.0,
                    help="v3b=0 区室边局部钙库主导(默认, 消除S_MAX永久饱和); v3a=1 胞体+钙库加性(对照)")
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
        print(f"[net] 使用网络: {NET_NPZ}", flush=True)

    import jax
    import jax.numpy as jnp
    from jax import lax
    jax.config.update("jax_enable_x64", True)

    t0 = time.time()
    net = build_v62()
    if args.vrel:
        net.vrel = np.load(args.vrel)
        print(f"[vrel] 原位释放阈表 {args.vrel} (n={len(net.vrel)}, v59 设计意图 s_rest=0.4)", flush=True)
    z = np.load(NET_NPZ, allow_pickle=False)
    chem, sign, gap = z["chem"], z["sign"], z["gap"]
    names = [str(s) for s in z["neurons"]]
    if args.vrelset:
        assert args.vrel, "--vrelset 需配合 --vrel 使用"
        _idx0 = {n: i for i, n in enumerate(names)}
        for _kv in args.vrelset.split(","):
            _kn, _vv = _kv.split(":")
            net.vrel[_idx0[_kn.strip()]] = float(_vv)
        print(f"[vrelset] 逐细胞释放阈覆盖: {args.vrelset} "
              f"(W50: AVA vrel=-83 释放饱和修复, sV@rest 2.32->分级区, 账本补十二)", flush=True)
    W = {"chem": chem, "sign": sign, "gap": gap, "cmasks": [], "gmasks": []}

    from muscle_maps import load_muscle_maps
    muscle_maps = load_muscle_maps(net)

    # ---- v81: 自有生理分组肌肉层 (muscle96!=0 时实例化; 默认关, 旧路径逐位不变) ----
    m96 = None
    if args.muscle96 != 0.0:
        from muscle96 import Muscle96
        m96 = Muscle96(muscle_maps, tau_m_ms=args.m96tau, amax=args.m96amax,
                       neuron_names=[str(n) for n in names], ei_nmj=args.m96ei,
                       tau_ca_ms=args.m96tau2, gj_cc=args.m96gj)
        print(f"[muscle96] v81 自有肌肉层 ON: G={args.muscle96:g}, tau_m={args.m96tau:g}ms, "
              f"Amax={args.m96amax:g}, 96肌分级积分 (常数表 data/muscle96_constants.csv); "
              f"Boyle层/mndrive/dmndrv/smddrive 驱动半边被取代", flush=True)
        if args.m96tau2 > 0.0:
            print(f"[muscle96] P10 两层级联 ON: 快层(电兴奋) tau={args.m96tau:g}ms "
                  f"(Gao&Zhen 2011 AP半宽锚15.5) -> 慢层(钙/收缩) tau={args.m96tau2:g}ms "
                  f"(RSI2015 钙瞬变衰减锚1.0-1.57s); 直流增益不变只改动态", flush=True)
        print(f"[muscle96] P7锚定: E:I={m96.ei_nmj:g} (NMJ实测0.61, PMC2705609); "
              f"绝对尺 g_abs={m96.g_abs:.3f} nA/单位驱动 — ACh满幅≈2.01 nA, "
              f"GABA满幅≈{m96.ei_nmj * 2.01:.2f} nA (实测1.23); "
              f"电流→收缩映射无实测(领域空白), mA无量纲走kmax标定(有界估计)", flush=True)

    import jax_circuit as jc
    if args.tauex > 0.0 or args.tauinh > 0.0:
        # P12 (2026-10-22): 突触电导时间常数敏感性测试 — 模块常量须在 jit 首次编译前覆盖
        # (jax trace 期折叠), 位置在 import 后 / build_step 首调前, 与 GSYN 覆盖同纪律
        if args.tauex > 0.0: jc.TAU_EX = args.tauex
        if args.tauinh > 0.0: jc.TAU_INH = args.tauinh
        print(f"[tau] 突触时间常数覆盖: TAU_EX={jc.TAU_EX:g}ms TAU_INH={jc.TAU_INH:g}ms "
              f"(基线 2.0/5.0, NMJ 括号快沿锚)", flush=True)
    P = jc.extract(net, W, muscle_maps, DT_MS)

    # ---- 在硅通道阻断: --block gunc2:0 (可重复); 乘子作用于 P 参数矩阵 ----
    for spec_ in args.block:
        cname, cval = spec_.split(":")
        cval = float(cval)
        if cname not in jc.PK:
            raise SystemExit(f"未知通道参数 {cname}; 可选 {list(jc.PK)}")
        ci = jc.PK[cname]
        P["Pawc"] = P["Pawc"].copy(); P["Prmd"] = P["Prmd"].copy()
        P["Pawc"][:, ci] *= cval; P["Prmd"][:, ci] *= cval
        print(f"[block] {cname} x {cval}  (全网络)", flush=True)

    # ---- RIA 三区室扩展 (riacomp=1 时启用; 默认关闭, 旧行为逐位不变) ----
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
            print(f"[riacomp-v3] 钙库释放门控 ON: tau={args.riatau:g}s k_mod={args.riakmod:g} "
                  f"s_acc2={args.riaacc2:g} soma_add={args.riasomaadd:g} "
                  f"route={'交叉' if args.riaroute < 0 else '同侧'} "
                  f"(Hendricks2012机制, 账本§64/§65)", flush=True)
        else:
            ria_cp = dict(g_ax=args.riagax, s_acc2=args.riaacc2, route=args.riaroute)
            step, assemble_V, init_state2 = jrc.build_step_riacomp(P, ria_spec, ria_mats, ria_cp,
                                                                   ctrl_hz=CTRL_HZ)
            print(f"[riacomp] RIA三区室 ON: g_ax={args.riagax:g}nS s_acc2={args.riaacc2:g} "
                  f"route={'交叉' if args.riaroute < 0 else '同侧'} (账本§62)", flush=True)
        riacomp = True
    else:
        step, assemble_V = jc.build_step(P, fwd_idx=(), n2w_idx=(), ctrl_hz=CTRL_HZ)

    N = P["N"]
    if riacomp:
        Mex = ria_mats["Mex_m"]   # 已剔除重路由边 (含 G_SYN 乘子)
    else:
        Mex = jnp.asarray(P["G_SYN"] * P["Mex_u"][0])   # theta=0: 全单位乘子 → 解剖权重
    Min = jnp.asarray(P["G_INH"] * P["Min_u"][0])
    Lg = jnp.asarray(P["G_GAP"] * P["Lg_u"][0])
    # W54 (2026-10-23): A->B 化学串扰边拆分 (倒退态衰减用; abx=1.0 时全量=逐位不变)
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
        print(f"[W54] A->B 串扰倒退衰减 ON: abx={args.abx:g} (Kato2015 亚环路互抑analog; "
              f"边数={int(_abmask.sum())}, VA->VB 5.508 锁链在列)", flush=True)
    # W54b: A-MN<->VD/DD gap 边集 Laplacian (倒退态衰减用; avdx=1.0 时不启用)
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
        print(f"[W54b] A<->VD/DD gap 倒退衰减 ON: avdx={args.avdx:g} "
              f"(边数={len(_rows)}; 现象学状态门控, innexin 调制候选)", flush=True)
    EMPTY = jnp.zeros(0, jnp.float64)
    NSTEPS = int(round(BLOCK_MS / DT_MS))

    def _blk(c, _):
        return step(c, Mex, Min, Lg, _IEX, False, EMPTY), None

    # scan 闭包里 Iex 用 tracer 传入
    def make_run_block():
        def run_block(state, Iex):
            def f(c, _):
                return step(c, Mex, Min, Lg, Iex, False, EMPTY), None
            return lax.scan(f, state, None, length=NSTEPS)[0]
        return jax.jit(run_block)
    run_block = make_run_block()

    # 二期 ggapmod: Lg 运行时缩放通道 (默认关, run_block_lv=None -> 旧路径逐位不变)
    run_block_lv = None
    run_block_lvm = None
    if args.ggapmod != 0.0 or args.avarect != 0.0 or args.abx != 1.0 or args.avdx != 1.0:
        Lg_full_np = np.asarray(P["G_GAP"] * P["Lg_u"][0])   # 满刻度基矩阵 (含 G_GAP 乘子)
        def make_run_block_lv():
            def run_block(state, Iex, Lg_t):
                def f(c, _):
                    return step(c, Mex, Min, Lg_t, Iex, False, EMPTY), None
                return lax.scan(f, state, None, length=NSTEPS)[0]
            return jax.jit(run_block)
        run_block_lv = make_run_block_lv()
        if Mex_abx_bwd_np is not None:
            # W54: Lg+Mex 双逐块通道 (A->B 串扰倒退衰减)
            def make_run_block_lvm():
                def run_block(state, Iex, Lg_t, Mex_t):
                    def f(c, _):
                        return step(c, Mex_t, Min, Lg_t, Iex, False, EMPTY), None
                    return lax.scan(f, state, None, length=NSTEPS)[0]
                return jax.jit(run_block)
            run_block_lvm = make_run_block_lvm()
        if args.ggapmod != 0.0:
            print(f"[ggapmod] 二期缝隙调制 ON: g_eff=(1-{args.ggapmodk:g}(1-M))xGGAP, "
                  f"tau={args.ggapmodtau:g}s; 锚 Chen2007 unc-1(0.7)/LeBoeuf2015 DOP-2/"
                  f"Schwarz2018 EGL-4/Kawano2011 shunt; 配 --ggap 1.0: M~0->0.3爬行 M~1->1.0逃逸",
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
    # ggapmod 驱动集: 前部机械感觉层 (ALM/AVM 轻触觉) + 多巴胺能神经元 (CEP/ADE, DA 信号代理;
    # DA 为容积传输而引擎无线层 g_wl=0, 故以这些细胞去极化活动代理 DA->DOP-2 输入;
    # CEP/ADE/PDE 为秀丽隐杆线虫全部多巴胺能神经元, Sulston 1975/WormAtlas 定性锚)
    DAT = [idx[n] for n in ("ALML", "ALMR", "AVM", "CEPDL", "CEPDR",
                            "CEPVL", "CEPVR", "ADEL", "ADER") if n in idx]
    if args.ggapmodash != 0.0:
        # B5: ASH 入驱动池 — Chaubey 2023 (PMC10319955): 状态依赖的 innexin(INX-20) 调制
        # 定位于 ASH 本身; 化学逃逸时 ASH 去极化 -> M->1 -> g_eff->GGAP (逃逸档),
        # 爬行/趋化时 ASH 静息 -> M~0 -> g_eff~0.3xGGAP (爬行档). 双手工档由此统一为
        # 单一基线 GGAP + 内生状态调制 (深度 k=0.7 锚 Chen 2007 unc-1;unc-9 耦合降70%)
        DAT = DAT + [idx[n] for n in ("ASHL", "ASHR") if n in idx]
    AVD = [idx["AVDL"], idx["AVDR"]]; AVE = [idx["AVEL"], idx["AVER"]]
    SMDD = [idx["SMDDL"], idx["SMDDR"]]; SMDV = [idx["SMDVL"], idx["SMDVR"]]
    RIV = [idx["RIVL"], idx["RIVR"]]; RIS = idx["RIS"]
    ASEL = idx["ASEL"]; ASER = idx["ASER"]
    AIYL_i = idx["AIYL"]; AIYR_i = idx["AIYR"]   # v64c 转向比较器读出 (AIY L/R 对)
    ASH_i = [idx["ASHL"], idx["ASHR"]]; ASI_i = [idx["ASIL"], idx["ASIR"]]  # 二期 CKR-1: NLP-18 感觉驱动池
    AIB_i = [idx["AIBL"], idx["AIBR"]]   # 二期 CKR-1: NLP-18 臂靶 (转向发起中间神经元, Chen 2022)
    # ---- W62: A->D sp 释放通道预计算 (chem 表筛 VA/DA->VD/DD 边, 解剖分布) ----
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
        print(f"[W62] vdsp A->D sp 释放通道 ON: {len(W_ad_t)} 边 (VA/DA->VD/DD), G={args.vdsp:g}", flush=True)
        if args.vdspx != 0.0:
            _keep = np.array([names[_b] not in ("VD12", "VD13") for _b in W_ad_t])
            _drop = int((~_keep).sum())
            W_ad_t = W_ad_t[_keep]; W_ad_s = W_ad_s[_keep]; W_ad_w = W_ad_w[_keep]
            print(f"[W64] vdspx 亚类分档 ON: 剔除 VD12/VD13 靶边 {_drop} 条, 保留 {len(W_ad_t)} 条", flush=True)
    if args.rivabl != 0.0:
        print(f"[rivabl] RIVL/R 恒定注入 {args.rivabl:g} pA (Gray 2005 功能性消融协议)", flush=True)

    # ---- 通用功能性消融 (Gao 2018 协议): 前缀通配 -> 恒定强超极化 ----
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
        print(f"[cellabl] 功能性消融 {len(abl_idx)} 细胞 ({args.cellablp:g} pA 恒定): "
              f"{','.join(names[i] for i in abl_idx)}", flush=True)

    # ---- 删边式真消融 (cellablmode=edge): 删除触及被消融细胞的全部化学+电突触边 ----
    abl_set = set(abl_idx) if (args.cellablmode == "edge" and abl_idx) else set()
    if abl_set:
        Mex_np = np.asarray(Mex).copy(); Min_np = np.asarray(Min).copy(); Lg_np = np.asarray(Lg).copy()
        for _M in (Mex_np, Min_np, Lg_np):
            _M[abl_idx, :] = 0.0; _M[:, abl_idx] = 0.0
        Mex = jnp.asarray(Mex_np); Min = jnp.asarray(Min_np); Lg = jnp.asarray(Lg_np)
        if "Lg_full_np" in dir():
            Lg_full_np[abl_idx, :] = 0.0; Lg_full_np[:, abl_idx] = 0.0
        print(f"[cellabl-edge] 真激光消融语义 {len(abl_idx)} 细胞: 化学+电突触边全删, 无电流注入 "
              f"(Gao 2018 消融制备; W13墙二: 钳压协议残余gap边拖拽邻居=假象, 本模式替代)",
              flush=True)

    # ---- W49: AVA<->A-MN 整流缝隙连接边集预计算 (Shui 2020 球-受体门控) ----
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
                continue   # edge 消融时剔除被删边
            if a in ava_ids and b in amn_ids:
                ar_a.append(a); ar_b.append(b); ar_ava.append(a); ar_amn.append(b); ar_w.append(w)
            elif b in ava_ids and a in amn_ids:
                ar_a.append(a); ar_b.append(b); ar_ava.append(b); ar_amn.append(a); ar_w.append(w)
        if ar_a:
            ar_a = np.array(ar_a); ar_b = np.array(ar_b)
            ar_ava = np.array(ar_ava); ar_amn = np.array(ar_amn)
            ar_gw = np.array(ar_w) * float(P["G_GAP"])
            _base = np.asarray(Lg)   # 已含 edge 消融处理 (若开启)
            Lg_ar_ohm = np.zeros_like(_base)
            np.add.at(Lg_ar_ohm, (ar_a, ar_a), ar_gw); np.add.at(Lg_ar_ohm, (ar_a, ar_b), -ar_gw)
            np.add.at(Lg_ar_ohm, (ar_b, ar_a), -ar_gw); np.add.at(Lg_ar_ohm, (ar_b, ar_b), ar_gw)
            AR = dict(a=ar_a, b=ar_b, ava=ar_ava, amn=ar_amn, gw=ar_gw,
                      Lg_rest=_base - Lg_ar_ohm)
            print(f"[W49] AVA<->A-MN 整流 ON: {len(ar_a)} 边, 强度={args.avarect:g}, "
                  f"Boltzmann A={args.avarecta:g} V0={args.avarectv0:g} Gmin={args.avarectmin:g} "
                  f"(Shui2020 PMC7455501 球-受体: Vj=V_AVA-V_AMN>0堵孔=关 <0开=A->AVA; "
                  f"Liu2017 在体单向; 电压取上一块50ms延迟同ggapmod例)", flush=True)
        else:
            print("[W49] 警告: 未找到 AVA<->A-MN gap 边 (或全部被 edge 消融), 整流不生效", flush=True)

    def _Lg_w49(Vabs):
        """W49: 由绝对电压向量组装整流 Lg. gate=Boltzmann(Vj), Vj=V_AVA-V_AMN:
        Vj>0 (AVA更去极化=倒退态平台) -> 球堵孔 gate->Gmin (A-MN 解放);
        Vj<0 (A-MN 更去极化) -> gate->1 (A->AVA 导通, Kawano 分流/反向放大)."""
        Vj = Vabs[AR["ava"]] - Vabs[AR["amn"]]
        gate = args.avarectmin + (1.0 - args.avarectmin) / (
            1.0 + np.exp(args.avarecta * (Vj - args.avarectv0)))
        gw = ((1.0 - args.avarect) + args.avarect * gate) * AR["gw"]
        Lt = AR["Lg_rest"].copy()
        np.add.at(Lt, (AR["a"], AR["a"]), gw); np.add.at(Lt, (AR["a"], AR["b"]), -gw)
        np.add.at(Lt, (AR["b"], AR["a"]), -gw); np.add.at(Lt, (AR["b"], AR["b"]), gw)
        return Lt

    # ---- 静默分区光遗传：类群前缀 -> 细胞索引 ----
    REGION_PREFIX = {
        "head_sensory":  ["IL", "CEP", "OLL", "OLQ"],
        "chemo_sensory": ["ASE", "ASG", "ASH", "ASI", "ASJ", "ASK", "ADF",
                          "AWA", "AWB", "AWC", "AUA", "BAG", "AFD", "FLP", "SDQ"],
        "head_motor":    ["RIV", "RMD", "RME", "RMH", "SMB", "SMD"],
        "tail_sensory":  ["PVD", "PLN", "PVN", "PVQ"],
        "mod_ala_hsn":   ["ALA", "HSN"],
        # --- 逃逸反射回路 (net323 找回缺失细胞后) ---
        "touch_ant":     ["ALM", "AVM"],          # 前触觉 (文献: -> AVD -> 倒退)
        "touch_post":    ["PLM"],                  # 后触觉 (文献: -> PVC/AVA -> 加速前进)
        "ash":           ["ASH"],                  # 伤害感觉 (文献: -> AVA -> 倒退)
        "afd":           ["AFD"],                  # 温度感觉 (Narayan & Sternberg 2011 PMC3111291: AFD->AIY 增益探针)
        "ava":           ["AVA"],                  # 命令层直激 (Roberts 2016 式: ChR2::AVA -> 倒退+AVB抑制)
        "avb":           ["AVB"],                  # 命令层直激对照 (Roberts 2016 式)
        "asel":          ["ASEL"],                 # 趋化 ON 通路 (文献: 浓度升响应, 维持前行)
        "aser":          ["ASER"],                 # 趋化 OFF 通路 (文献: 浓度降响应, 促转向/倒车)
        "pvd":           ["PVD"],                  #  harsh touch (文献: -> 逃逸)
        "phb":           ["PHB"],                  # 尾感觉 (文献: -> AVA -> 倒退)
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
    x = body.initial(heading=np.radians(args.initdir + 180.0))   # v83: initdir=面向方位角
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
    # v60 相对门控参数 (仅 --gate rel 时生效): floor=静默地板(低于此双方都不开门, 保持静息态),
    # hyst=迟滞余量(翻转要求赢家净超此值, 防抖动)
    gate_floor = 0.5; gate_hyst = 1.0; dir_state = +1
    last_bout_end_ms = -1.0e18   # v91 bwrefr: 上一 bout 末沿时刻 (初值=远古)

    # ---- v65 本体感觉环: B 类运动神经元 ← 相邻体节曲率 (Wen et al. 2012, Neuron 76:750) ----
    # 解剖: B 类运动神经元 (DB/VB) 神经突无分支沿腹索延伸, 被提出直接感受体壁弯曲;
    # Wen 2012 钙成像+光遗传证实 B 类 MN 活动与身体弯曲波相位锁定, 是前进波传播的必要反馈。
    # 构造: 每类细胞按序号均匀映射到 body 体节; D/V 反对称;
    # 每块注入一次, κ 取上一块末值 -> 固定 1 块 (50ms) 反馈延迟 (约占波周期 5-10%, 如实声明)。
    # 极性 (v66 修正): preferred_curvature 源码 κ>0=背侧收缩 -> DB +κ (背弯兴奋背侧B类),
    # VB -κ; v65 曾相反 (VB +κ/DB -κ)——锁定幅度/相关不受影响 (分析按类折算符号),
    # 仅类间指派互换; 接输出侧 (mndrive) 时绝对极性必须如此才构成正反馈。
    # 默认 --mnprop 0: 本段完全不执行, 旧行为逐位不变。
    BMN_idx = np.zeros(0, dtype=int); BMN_sgn = np.zeros(0); BMN_seg = np.zeros(0, dtype=int)
    BMN_seg_eff2 = np.zeros(0, dtype=int)
    BMN_seg_eff2_rev = BMN_seg_eff2
    AMN_idx = np.zeros(0, dtype=int); AMN_sgn = np.zeros(0); AMN_seg_eff = np.zeros(0, dtype=int)
    if args.mnprop != 0.0 or args.mntonic != 0.0 or args.headosc != 0.0 or args.mnprop2 != 0.0 or args.mnanat != 0 or args.amnprop != 0.0 or args.dvaprop != 0.0:
        nseg = body.n - 1
        _ci, _sg, _se = [], [], []
        if args.mnanat != 0:
            # v69 R10a: 解剖映射 — 胞体体节 = 该细胞支配肌肉行的加权平均位置 (neuron_muscle.csv);
            # preferred_curvature: 体节j曲率由肌行j,j+1 (0基) 驱动 -> 肌行r作用于体节r-1,r,
            # 支配中心(肌行单位) r_mean 对应体节 r_mean-0.5; 感受位置与支配位置对齐,
            # 使 Zhao 2025 局部负反馈环在真实解剖位置闭拢 (修复尾部死区=开放问题#3);
            # 无肌肉支配记录的细胞回退到均匀映射
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
            print(f"[mnanat] B类MN胞体->体节解剖映射 ON: 按 neuron_muscle.csv 支配行定位, "
                  f"seg范围 {min(_se)}..{max(_se)}", flush=True)
        else:
            for cls, s in (("DB", +1.0), ("VB", -1.0)):
                ks = [k for k in range(1, 30) if f"{cls}{k:02d}" in idx]
                for rank, k in enumerate(ks):
                    _ci.append(idx[f"{cls}{k:02d}"]); _sg.append(s)
                    _se.append(min(int((rank + 0.5) / len(ks) * nseg), nseg - 1))
        BMN_idx = np.array(_ci, dtype=int); BMN_sgn = np.array(_sg); BMN_seg = np.array(_se, dtype=int)
        # v67 R1: 感受位置 = 胞体体节向前方偏移 mnoffset 节 (前方=头部=节段索引减小; Rod2D xs[0]=头);
        # 头部 MN 前方无节段时 clip 到 0。mnoffset=0 时 BMN_seg_eff==BMN_seg, 逐位不变。
        BMN_seg_eff = np.clip(BMN_seg - args.mnoffset, 0, nseg - 1)
        # v67 R4: 第二通道感受位置 (负偏移=后方, Cohen 牵张符号配对)
        BMN_seg_eff2 = np.clip(BMN_seg - args.mnoffset2, 0, nseg - 1)
        # v70 R14: 倒退镜像 — 倒退门控生效时偏移取反 (后方感受<->前方感受 = 波方向反转);
        # 默认 mnback=0 时 BMN_seg_eff2_rev 与 BMN_seg_eff2 相同, 逐位不变
        if args.mnback != 0:
            BMN_seg_eff2_rev = np.clip(BMN_seg + args.mnoffset2, 0, nseg - 1)
        else:
            BMN_seg_eff2_rev = BMN_seg_eff2
        if args.bmnclip == "reflect":
            # W57c (2026-10-23): 越界感受位镜像回体内 —— clip0 时尾侧B类全部钳到节23
            # (尾尖κ~0=自洽死区: 尾尖不弯->无信号->尾B死->尾尖更不弯), 倒退镜像时头侧
            # 钳到节0=强制同相=头部同步块驻波化; 反射后边界细胞获得分散的行波信号
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
            print(f"[bmnclip] reflect ON: 前进态越界 {int((_raw2 > nseg - 1).sum() + (_raw2 < 0).sum())} / "
                  f"倒退态越界 {int(((BMN_seg + args.mnoffset2) > nseg - 1).sum() + ((BMN_seg + args.mnoffset2) < 0).sum())} "
                  f"B类感受位镜像回体内", flush=True)
        # v67 R1b: 头部 B 类掩码 (胞体映射到前 2 节段), 供 --headosc 正弦种子注入
        BMN_head = BMN_seg <= 1
        print(f"[mnprop] B类MN本体感觉环 ON: G={args.mnprop:g} pA/(1/mm), "
              f"{len(_ci)} cells (DB/VB) -> {nseg} segs, 1-block(50ms)反馈延迟, "
              f"前方偏移={args.mnoffset}节, 头部细胞={int(BMN_head.sum())}", flush=True)
        if args.mnprop2 != 0.0:
            print(f"[mnprop2] 第二本体感觉通道 ON: G2={args.mnprop2:g} pA/(1/mm), "
                  f"偏移2={args.mnoffset2}节 (负=后方)", flush=True)
            if args.mnback != 0:
                print(f"[mnback] 倒退镜像 ON: 倒退门控生效时偏移2取反 "
                      f"({args.mnoffset2}->{-args.mnoffset2}节, 波方向随方向态反转)", flush=True)
        # ---- v85: A类MN(VA/DA)本体感觉环 (Gao et al. 2018 eLife 7:e29915) ----
        # A类为倒退内生振荡器并接受本体感觉entrainment; 环路镜像B链倒退形态: 前方感受(-amnoffset节,
        # 头侧) -> 头->尾传播 = 倒退波 (v85b修正: v85a曾误取后方感受=尾->头=前进波, 与B链前进态同向打架);
        # 胞体->体节映射复用B链同一套解剖/均匀逻辑 (DA->背侧肌行, VA->腹侧)
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
            AMN_seg = np.array(_seA, dtype=int)   # v92: 胞体节段 (dvaprop 刹车作用点)
            # W56 (2026-10-23): amndir=posterior -> 感受尾侧(+amnoffset) -> 尾->头接力=倒退波 (Gao 2018);
            # anterior=旧行为(v85b): 感受头侧 -> 头->尾波=前进方向
            _off = args.amnoffset if args.amndir == "anterior" else -args.amnoffset
            _rawA = np.array(_seA, dtype=int) - _off
            # W55 (2026-10-23): amnclip=edge0 时边界外注入置零 —— 边界外A类
            # 不受 clip 同相伪影约束, 成为纯内生振荡器群 (Gao 2018), 成波种子
            if args.amnclip == "reflect":
                # W57 (2026-10-23): 越界感受位镜像回体内 —— edge0 断粮实验证明尾部VA09-12
                # 失去注入后被 bout 期激活的 VD(GABA) 按死 (bout期超极化-11mV);
                # 反射后感受位仍分散(无clip同相伪影), 尾部A类骑B链载波恢复参与
                _refl = _rawA.copy()
                _over = _refl > nseg - 1
                _refl[_over] = 2 * (nseg - 1) - _refl[_over]
                _under = _refl < 0
                _refl[_under] = -_refl[_under]
                AMN_seg_eff = np.clip(_refl, 0, nseg - 1)
                AMN_emask = np.ones(len(_seA), dtype=float)
                print(f"[amnclip] reflect ON: 越界 {int((_over | _under).sum())}/{len(_seA)} A类感受位 "
                      f"镜像回体内 (dir={args.amndir})", flush=True)
            else:
                AMN_seg_eff = np.clip(_rawA, 0, nseg - 1)
                if args.amnclip == "edge0":
                    AMN_emask = ((_rawA >= 0) & (_rawA <= nseg - 1)).astype(float)
                    print(f"[amnclip] edge0 ON: 边界外 {int((AMN_emask == 0).sum())}/{AMN_emask.size} A类胞体 "
                          f"零注入 (纯内生振荡端, dir={args.amndir})", flush=True)
                else:
                    AMN_emask = np.ones(len(_seA), dtype=float)
            # W56 (2026-10-23): UNC-2 驱动尾高头低梯度 (Gao 2018: DA9 最尾=领跑振荡器)
            if args.amngrad != 0.0:
                AMN_gfac = 1.0 + args.amngrad * (2.0 * np.array(_seA, dtype=float) / (nseg - 1) - 1.0)
                print(f"[amngrad] UNC-2尾高头低梯度 ON: g={args.amngrad:g} 头x{AMN_gfac.min():.2f} "
                      f"尾x{AMN_gfac.max():.2f} (Gao2018 DA9领跑)", flush=True)
            else:
                AMN_gfac = np.ones(len(_seA), dtype=float)
            if args.amnprop != 0.0:
                print(f"[amnprop] A类MN本体感觉环 ON: G={args.amnprop:g} pA/(1/mm), "
                      f"{len(_ciA)} cells (DA/VA) -> {nseg} segs, 前方感受偏移=-{args.amnoffset}节 (头侧, 头->尾倒退波), "
                      f"门控={'仅倒退态' if args.amngate else '全程'} (Gao2018 A类振荡器entrainment)", flush=True)
            if args.dvaprop != 0.0:
                print(f"[dvaprop] DVA/TRP-4 幅度刹车 ON: G={args.dvaprop:g} pA/(1/mm) tau={args.dvatau:g}s, "
                      f"|kappa慢EMA| 超极化注入 {len(_ciA)} A类MN胞体节段, 仅倒退态生效 "
                      f"(Li 2006 Nature PMC2865900: trp-4(lf)=过度弯曲, DVA负向限幅)", flush=True)

    # ---- v66 负载回路: B 类 MN -> 肌肉解剖驱动 (Wen 2012 闭环的下半边) ----
    # 支配权重 = 解剖表 neuron_muscle.csv (BAAIWorm 导出, 96 体壁肌 x MN, 单位=接触数);
    # DB->背侧肌行, VB->腹侧肌行 (B类胆碱能兴奋同侧体壁肌); L/R 两肌平均到单行 (x0.5);
    # 每肌按总支配归一 (n_inn); 驱动函数 softplus(dV)-ln2: dV=0 -> 0 (静息无贡献),
    # dV=+10mV -> ~9.3, 负值微负 (分级释放低于静息, 生理合理)。
    # 默认 --mndrive 0: 矩阵不建, 旧行为逐位不变。
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
            print(f"[mndrive] B类MN->肌肉驱动 ON: G={args.mndrive:g}, 解剖表支配 "
                  f"D侧非零 {int((Wdrv_D > 0).sum())} / V侧 {int((Wdrv_V > 0).sum())} 项, "
                  f"boylescale={args.boylescale:g}", flush=True)
    # v67 R2: D 类 MN -> 肌肉交叉抑制矩阵 (White 1986 成人: DD->背侧肌, VD->腹侧肌)
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
        print(f"[dmndrive] D类MN->肌肉交叉抑制 ON: G={args.dmndrive:g}, 解剖表支配 "
              f"DD->D侧非零 {int((Wdm_D > 0).sum())} / VD->V侧 {int((Wdm_V > 0).sum())} 项, "
              f"局部无偏移 (D类无长神经突)", flush=True)

    # ---- v68 R6: SMD -> 头肌解剖驱动 + SMD 本体感觉环 (Yeon 2018 / Zhao 2025 前部环) ----
    # 解剖: SMDDL/R->DR/DL01-06 (头背肌), SMDVL/R->VR/VL01-06 (头腹肌), 表已核实;
    # 驱动公式与 B 类 mndrive 相同; 反馈=头6节段平均曲率, 负增益=牵张反射(Zhao负反馈环同构)。
    # 默认 --smddrive/--smdprop 0: 不建矩阵不注入, 旧行为逐位不变。
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
        SMDD_idx = SMD_idx[SMD_sgn > 0]; SMDV_idx = SMD_idx[SMD_sgn < 0]  # R9 转向用类掩码
        if args.smdabl.upper() == "D":    # trp-1,2 双突变模拟: SMDD 反馈断 (Yeon 2018)
            SMD_sgn[SMD_sgn > 0] = 0.0
        elif args.smdabl.upper() == "V":
            SMD_sgn[SMD_sgn < 0] = 0.0
        print(f"[smd] SMD环 ON: drive G={args.smddrive:g} (D侧非零 {int((Wsmd_D > 0).sum())} / "
              f"V侧 {int((Wsmd_V > 0).sum())} 项), prop G={args.smdprop:g} (头6节段均值), "
              f"abl='{args.smdabl}', 细胞={len(SMD_idx)}", flush=True)

    # ---- v74 R18 (FC1): RIA->SMD 原位突触预算边表 (事件门控内生转向驱动的幅度基准) ----
    # 摸底 2026-10-20: RIA->SMD 强直预算 ~20pA (8边), turnsmd=40 脉冲缺口仅 2x;
    # D/V 不对称由解剖给出 (RIAL偏SMDV 4.68v4.00 / RIAR偏SMDD 6.19v5.10)
    FC1_pre = np.zeros(0, dtype=int); FC1_post = np.zeros(0, dtype=int); FC1_w = np.zeros(0)
    if args.fc1 != 0.0:
        _ria = [idx[c] for c in ("RIAL", "RIAR") if c in idx]
        _smdset = set(SMDD_idx.tolist()) | set(SMDV_idx.tolist())
        _e = [(int(c[0]), int(c[1]), c[2]) for c, sg_ in zip(chem, sign)
              if int(c[0]) in _ria and int(c[1]) in _smdset and sg_ > 0]
        FC1_pre = np.array([e[0] for e in _e], dtype=int)
        FC1_post = np.array([e[1] for e in _e], dtype=int)
        FC1_w = np.array([e[2] for e in _e])
        print(f"[fc1] 内生转向驱动 ON: G={args.fc1:g} x RIA->SMD预算 ({len(_e)}边), "
              f"dir={args.fc1dir}, tau2={args.fc1tau2:g}s", flush=True)

    # ---- 断点续跑 (v63b): 载入存档则跳过预热, 神经/身体/马达/门控/ASE 状态全恢复 ----
    if args.ckr1 != 0.0:
        print(f"[ckr1] 二期 CKR-1 肽能增益 ON: G0={args.ckr1:g} pA, "
              f"tau12={args.ckr1tau12:g}s tau18={args.ckr1tau18:g}s M0={args.ckr1m0:g}; "
              f"锚 Chen2022(PMID 35139370)/Chen2024(PMC10952047): NLP-12臂->SMD 体曲率 / "
              f"NLP-18臂->SMD+AIB Omega转向; w18/w12=1.42 (Ca2+峰 0.502/0.353)", flush=True)
    ckpt = None
    if args.loadstate:
        import pickle
        with open(args.loadstate, "rb") as fh:
            ckpt = pickle.load(fh)
        print(f"[ckpt] 载入 {args.loadstate} (存档 t={ckpt['t_ms']/1000:.1f}s)", flush=True)

    # ---- 基线预热 2 s (续跑时跳过, 直接恢复存档状态) ----
    if ckpt is None:
        print(f"[init] build {time.time()-t0:.1f}s; warmup 2s baseline...", flush=True)
        Iex0 = jnp.zeros(N, jnp.float64)
        if args.ckr1 != 0.0 and args.ckr1m0 != 0.0:
            # CKR-1 预热同档: M=M0 常数注入 (同 ggapmod 预热例; 否则 V_rest 捕于无肽态, 换挡瞬态污染)
            Iex0 = Iex0.at[np.asarray(SMDD + SMDV)].add(
                jnp.asarray(args.ckr1 * (1.0 + 1.42) * float(args.ckr1m0), jnp.float64))
            Iex0 = Iex0.at[np.asarray(AIB_i)].add(
                jnp.asarray(args.ckr1 * 1.42 * float(args.ckr1m0), jnp.float64))
        n_warm = int(round(2000 / BLOCK_MS))
        for b in range(n_warm):
            if run_block_lv is not None:
                # ggapmod: 预热也走调制档 (M=M0 -> g_eff=(1-k(1-M0))xGGAP);
                # 否则预热在错误耦合态平衡, V_rest 捕于错误工作点, 换挡瞬态致痉挛 (二期首日实测)
                _gmod0 = 1.0 - args.ggapmodk * (1.0 - float(args.ggapmodm0)) \
                    if args.ggapmod != 0.0 else 1.0
                if AR is not None:
                    # W49: 预热同例走整流档 (上一块电压, 首块=Y0 孤立锚: AVA-25/AMN-72 -> 近关断)
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
            # 旧 4 元存档 + riacomp=1: 补区室初始态 (Yc/gAc/gIaiy)
            _Y0 = init_state2()
            state = state + _Y0[4:]
            print(f"[ckpt] 旧存档补区室初始态 (riacomp)", flush=True)
        V_rest = ckpt["V_rest"]
        x = jnp.asarray(ckpt["x"])
        fwd_m = ckpt["fwd_m"]; bwd_m = ckpt["bwd_m"]
        print(f"[ckpt] 状态已恢复, 跳过预热 (V_rest 沿用存档)", flush=True)

    if m96 is not None:
        # v81: 用真实静息电压+实际 vrel 表重算紧张性基线 (背腹解剖不对称, 见 muscle96.py)
        _rel0 = m96.calibrate_rest(V_rest, net.vrel, mode=args.m96sig)
        print(f"[muscle96] 静息基线 I0 已标定 (mode={args.m96sig}): rel0 mean={_rel0.mean():.3f} "
              f"min={_rel0.min():.3f} max={_rel0.max():.3f}", flush=True)

    # ---- 刺激阶段 ----
    secs = 3.0 if args.smoke else args.secs
    n_blocks = int(round(secs * 1000 / BLOCK_MS))

    # ---- 趋化协议初始化 ----
    ase = None
    chem_c = [0.0]          # v78 R25b: 每块即时浓度 (供 pirc 绝对浓度驻留门)
    arr_clp = [0.0]         # v78 R25c: 低通浓度/峰 (Sawin 基础减速的多巴胺紧张性信号)
    asad = [0.0]            # v79 R26: GPCR 级感觉适应态 0-1 (ARR-1 式脱敏; 仅 asad!=0 时更新)
    asad_gain = [1.0]       # v79 R26: 适应增益 1/(1+K*a), 供 weathervane 相关信号同步衰减
    # P11 (2026-10-22): ASH 化学换能前端状态 (Hilliard 2005 PMC544906 锚)
    #   m = 换能驱动态 (tau_on~1s); a = GPC-1 式适应态 (充电 tau_a~20s, 恢复 tau_r~100s)
    #   I_ASH = ashchem * m * max(1-a, 0); 浓度方波 c=1 窗 8-16s (与 region 协议同时序对照)
    ash_m = [0.0]; ash_a = [0.0]
    ash_mod = [0.0]     # P13: INX-20/cGMP 调制态 0-1 (Chaubey 2023; 饥饿驱动, tau 60s 升/300s 复)
    if args.proto == "chem":
        if args.chemsrc:
            sx, sy = (float(v) for v in args.chemsrc.split(","))
            ase = ASEInput(ChemField((sx, sy), sigma=args.chemsigma), gain=args.chemgain)
            print(f"[chem] 源=({sx:g},{sy:g}) sigma={args.chemsigma:g} gain={args.chemgain:g} pA/(C/s) "
                  f"tau=0.5s | ASEL/ASER 走解剖通路", flush=True)
        else:
            print("[chem] 无源对照 (仅 AVB 工作点)", flush=True)

    def make_Iex(t_s):
        """按光遗传协议返回当前时刻的外加电流向量"""
        I = np.zeros(N)
        if args.proto == "switch":        # 前进->倒退->前进 状态切换
            if t_s < 10.0:      I[AVB] = args.pamp
            elif t_s < 22.0:    I[AVA] = args.pamp
            else:               I[AVB] = args.pamp
        elif args.proto == "avapulse":    # v93: AVA 短脉冲 (触觉模拟, Wicks 1995 bout 2-3s),
            if 6.0 <= t_s < 8.0:  I[AVA] = args.pamp   # 之后无外加驱动 -> 考验内生退出时钟
        elif args.proto == "ris":         # 前进中 RIS 定点兴奋 -> 骤停
            I[AVB] = args.pamp
            if 12.0 <= t_s < 18.0: I[RIS] = args.pamp
        elif args.proto == "region":      # 前进中静默分区定点兴奋 8-16s
            I[AVB] = args.stim
            if args.phasic > 0.0:
                # v70 R15: 快适应相位脉冲 (onset+offset 双脉冲, O'Hagan 2005)
                _w = args.phasic / 1000.0
                if (8.0 <= t_s < 8.0 + _w) or (16.0 - _w <= t_s < 16.0):
                    I[RC] = args.pamp
            elif 8.0 <= t_s < 16.0:
                I[RC] = args.pamp
        elif args.proto == "ashchem":     # P11: ASH 化学换能前端 (Hilliard 2005 锚)
            I[AVB] = args.stim
            _dt = BLOCK_MS / 1000.0
            _w0, _w1 = (float(x) for x in args.ashchemwin.split(":"))
            _c = 1.0 if (_w0 <= t_s < _w1) else 0.0   # 浓度方波 (默认 8-16s, region 同时序)
            ash_m[0] += (_dt / args.ashchemtau) * (_c - ash_m[0])          # 换能 onset tau~1s
            if ash_m[0] > ash_a[0]:
                ash_a[0] += (_dt / args.ashchemtaua) * (ash_m[0] - ash_a[0])  # 适应充电 20s
            else:
                ash_a[0] += (_dt / args.ashchemtaur) * (ash_m[0] - ash_a[0])  # 恢复 100s
            _i = args.ashchem * ash_m[0] * max(1.0 - ash_a[0], 0.0)
            # P13 (2026-10-22): INX-20/cGMP/EGL-4 音量旋钮 (Chaubey 2023 PMC10319955)
            #   饥饿 -> ODR-1 产 cGMP (AWB/AWC/ASI) -> 经 INX 网(ADL->RMG/ASK->ASH)入 ASH
            #   -> EGL-4(PKG)->RGS-2/3 -> 下调 GPCR 信号 -> ASH 敏感度降 (inx-20ko=超敏)
            #   tau: 分钟级 (cGMP 缝隙扩散+PKG 磷酸化; Hilliard 恢复 5min 同量级)
            if args.ashstarve > 0.0:
                ash_mod[0] += (_dt / 60.0) * (args.ashstarve - ash_mod[0])
            else:
                ash_mod[0] += (_dt / 300.0) * (0.0 - ash_mod[0])
            _i *= 1.0 / (1.0 + args.ashmod * ash_mod[0])
            I[ASH_i[0]] += _i; I[ASH_i[1]] += _i
        elif args.proto == "chem":        # 趋化: AVB 工作点 + ASE 感觉闭环
            I[AVB] = args.stim
            if ase is not None:
                on_, off_, _c = ase.currents(np.asarray(x)[0], BLOCK_MS / 1000.0)
                I[ASEL] += on_; I[ASER] += off_
                chem_c[0] = _c                 # v78 R25b: 即时浓度供绝对浓度驻留门
                # v78 R25c: 低通浓度/峰 (多巴胺紧张性时标; MGL-1 型 GPCR 持续状态转导)
                arr_clp[0] += (BLOCK_MS / 1000.0 / args.arrtau) * (_c / max(ase.f.peak, 1e-12) - arr_clp[0])
                if args.asad != 0.0:
                    # v79 R26: GPCR 级感觉适应 (ARR-1 式脱敏) — 持续高浓度(>asadth)区适应态上升,
                    # ASEL/ASER 注入增益 1/(1+K*a) 衰减 -> 峰上 tether 消退 -> 自发播散;
                    # asad=0 时本段不执行, 旧行为逐位不变
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
    kap2_f = np.zeros(body.n - 1)   # v67 R5: 第二通道EMA滤波态 (肌肉机械延迟代理)
    kap1_f = np.zeros(body.n - 1)   # v68 R7: 第一通道EMA滤波态 (Bryden-Cohen行波色散延迟)
    kap2_hp = np.zeros(body.n - 1)  # v91: A类环高通参考态 (仅 --amnhp>0 时使用)
    kap2_dva = np.zeros(body.n - 1)  # v92: DVA刹车慢EMA态 (仅 --dvaprop!=0 时使用)
    am_ema = 0.0                     # v93: AVA主神经元慢EMA态 (仅 --avamaster!=0 时使用)
    mA_D = np.zeros(24); mA_V = np.zeros(24)   # v68 R7: 肌肉激活积分态 (D/V 各24肌行)
    t_ms = 0.0; omega_t0 = None
    hold_I = 0.0   # v70 R16: AVA 平台自持电流状态 (触发沿充电, 无条件指数衰减)
    m2_pep = 0.0          # v73 R17: 倒退环路肽能正反馈慢变量 (pep2=0 时不参与)
    n2_pep = 0.0          # v73 R17b: 负臂慢变量 (AVR-14 式延迟抑制, bout 自终结)
    m_fc1 = 0.0           # v74 R18: 转向执行门控慢变量 (事件充电=1, 指数衰减; fc1=0 不参与)
    m_plat = np.zeros(len(AVA))  # v76 R21: AVA 平台电流快激活门 (L/R 各自, VGCC式)
    n_plat = np.zeros(len(AVA))  # v76 R21: 慢失活门 (跟踪m, bout 自终结)
    mp2 = np.zeros(len(AVA))     # v91: avaplat2 门控沿点火平台 快激活门
    ris_m = 0.0                  # W59b: RIS bout终止积分态 (倒退充电/前进放电)
    ris_hold_ms = 0.0            # W69: ris_m 满充保持计时 (riskill 保险丝用)
    np2 = np.zeros(len(AVA))     # v91: avaplat2 慢失活门
    slo_ca = np.zeros(len(AMN_idx))  # v92: A类SLO钙轨迹 (活动依赖负反馈, --sloa>0 时参与)
    dV_prev = np.zeros(N)  # v73 R17: 上一块电压 (反馈信号, 1 块=50ms 延迟, 如实声明)
    dir_state_prev = +1   # v70 R16c: 触发沿检测用的上一块方向态
    fwd_since = -1.0e9    # v70 R16e: 上次 bout 结束时刻 (2s 再武装计时)
    pulse_until = -1.0; refr_until = 0.0; dsmd_prev = 0.0   # v64 event pirouette 状态
    smd_turn_on = False   # v68 R9: SMD执行转向的脉冲状态 (1块延迟生效)
    w_weath = 0.0; kap0_prev = 0.0   # v71 W1: weathervane 相关积分态 + 头关节曲率前值
    w_frz_until_ms = 0.0             # W68: bout末沿 w 冻结窗终点 (wrst=0 时恒不生效)
    dwn_lp = 0.0          # v83b: 恶化信号慢低通态 (tau=dwntau, 杀头摆泄漏)
    pulse_dir = 1.0   # v71 W2: 当前转向脉冲方向 (+1=腹向, -1=背向; pirrand>0 时每次触发重掷)
    rng_pir = np.random.default_rng(args.pirrand) if args.pirrand > 0 else None
    # ---- 肽能层状态 (DVA->NLP-12->CKR-2): DVA 牵张驱动, CKR-2 突触前 NMJ 增益 ----
    pep_m = 0.0                      # NLP-12/CKR-2 激活水平 m(t) in [0,1]
    gm_m = float(args.ggapmodm0)     # ggapmod: 调制状态 M(t) in [0,1], 初值=M0 (二期, 见 --ggapmod)
    ckr_m12 = float(args.ckr1m0); ckr_m18 = float(args.ckr1m0)   # 二期 CKR-1 双臂调制态 (见 --ckr1)
    KAP_REF = 3.0                    # DVA 饱和牵张参考曲率 1/mm
    pep_dt = BLOCK_MS / 1000.0
    if ckpt is not None:
        kappa = ckpt["kappa"]; t_ms = ckpt["t_ms"]; omega_t0 = ckpt["omega_t0"]
        kap2_f = ckpt.get("kap2_f", np.zeros(body.n - 1))
        kap1_f = ckpt.get("kap1_f", np.zeros(body.n - 1))
        kap2_hp = ckpt.get("kap2_hp", np.zeros(body.n - 1))
        mA_D = ckpt.get("mA_D", np.zeros(24)); mA_V = ckpt.get("mA_V", np.zeros(24))
        pep_m = ckpt["pep_m"]; dir_state = ckpt["dir_state"]
        last_bout_end_ms = ckpt.get("last_bout_end_ms", -1.0e18)
        gm_m = float(ckpt.get("gm_m", 0.0))   # ggapmod 状态续跑 (旧 ckpt 无此键则 0)
        ckr_m12 = float(ckpt.get("ckr_m12", 0.0)); ckr_m18 = float(ckpt.get("ckr_m18", 0.0))   # CKR-1 续跑
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
        slo_ca = ckpt.get("slo_ca", np.zeros(len(AMN_idx)))   # v92: SLO钙轨迹续跑 (旧ckpt缺省=0)
        if m96 is not None and ckpt.get("mA96") is not None:
            m96.mA = np.asarray(ckpt["mA96"], dtype=float).copy()   # v81: 96肌积分态连续
            if ckpt.get("mE96") is not None:
                m96.mE = np.asarray(ckpt["mE96"], dtype=float).copy()  # P10: 快层态连续
            print(f"[ckpt] muscle96 积分态已恢复 (mean mA={m96.mA.mean():.3f})", flush=True)
        ash_m[0] = float(ckpt.get("ash_m", 0.0)); ash_a[0] = float(ckpt.get("ash_a", 0.0))  # P11
        ash_mod[0] = float(ckpt.get("ash_mod", 0.0))  # P13
        if ase is not None and ckpt.get("ase_prev") is not None:
            ase.prev = ckpt["ase_prev"]; ase.dcdt = ckpt["ase_dcdt"]
            arr_clp[0] = ckpt.get("arr_clp", 0.0)
            asad[0] = ckpt.get("asad", 0.0)         # v79 R26: 适应态连续 (旧ckpt缺省=0)
            if args.asad != 0.0:
                asad_gain[0] = 1.0 / (1.0 + args.asad * asad[0])
    print(f"[run] stim AVB {args.stim} pA x {secs}s ({n_blocks} blocks) pep={args.pep}", flush=True)
    ris_off = False  # v70 R18n: RIS 骤停规则门控第二本体感觉通道 (仅 RIS 协议触发)
    for b in range(n_blocks):
        Iex = make_Iex(t_ms / 1000.0)
        if args.mnprop != 0.0:
            # v65 本体感觉注入: 上块末曲率 -> B 类 MN 外加电流 (D/V 反对称)
            # v67 R1: 感受位置取胞体前方 mnoffset 节 (Wen 2012: ~200um anterior)
            # v68 R7: 可选EMA延迟 (Bryden-Cohen 2008 行波色散: 前方感受+时间延迟->传播)
            if args.mnproptau > 0.0:
                kap1_f = kap1_f + (BLOCK_MS / args.mnproptau) * (kappa - kap1_f)
            else:
                kap1_f = kappa
            if args.mnspan > 0:
                # v69 R10a2: 神经突跨度窗 (Wen 2012: 感受沿神经突分布, 非胞体点)
                _w1 = 2 * args.mnspan + 1
                kap1_s = np.convolve(kap1_f, np.ones(_w1) / _w1, mode="same")
            else:
                kap1_s = kap1_f
            Iex = Iex.at[BMN_idx].add(
                jnp.asarray(args.mnprop * BMN_sgn * kap1_s[BMN_seg_eff], jnp.float64))
        if args.mnprop2 != 0.0 or args.amnprop != 0.0:
            # v67 R4: 第二本体感觉通道 (Cohen 牵张符号=负增益, 后方感受=负偏移)
            # v67 R5: 可选EMA低通 (Zhao 2025 肌肉机械延迟代理), 消除局部负反馈的10Hz块延迟伪影;
            #         可选收缩阈值半波整流 (Zhao 2025: 收缩到指定长度才触发负反馈), 防线性相消
            if args.mnprop2tau > 0.0:
                kap2_f = kap2_f + (BLOCK_MS / args.mnprop2tau) * (kappa - kap2_f)
            else:
                kap2_f = kappa
            if args.mnspan > 0:
                _w2 = 2 * args.mnspan + 1
                kap2_s = np.convolve(kap2_f, np.ones(_w2) / _w2, mode="same")
            else:
                kap2_s = kap2_f
            # v70 R14: 倒退镜像 — dir_state 持上一块末值, 倒退态时感受位置取镜像阵列
            # (波方向随方向态反转; 切换延迟=1块50ms, 与既有反馈延迟同级)
            _eff2 = BMN_seg_eff2_rev if (args.mnback != 0 and dir_state < 0) else BMN_seg_eff2
            # v91/W58 高通参考: 慢EMA(=静态弯DC) — A环(amnhp)与B环倒退态(bmnhp)共用同一状态;
            # tau 优先取 amnhp (A环已在用), amnhp=0 时取 bmnhp; 时间常数语义相同
            _hp_s = None
            _hp_tau = args.amnhp if args.amnhp > 0.0 else args.bmnhp
            if _hp_tau > 0.0:
                kap2_hp = kap2_hp + (BLOCK_MS / (_hp_tau * 1000.0)) * (kappa - kap2_hp)
                if args.mnspan > 0:
                    _hp_s = np.convolve(kap2_hp, np.ones(2 * args.mnspan + 1) / (2 * args.mnspan + 1), mode="same")
                else:
                    _hp_s = kap2_hp
            if args.bmnhp > 0.0 and dir_state < 0:
                # W58: B链倒退态高通 — 切断 bout 自持环 DC 通道
                # (静态S弯->mnprop2 DC->B链tonic反转->AVA平台供养->bout不退; 相位性entrainment保留)
                d2 = BMN_sgn * (kap2_s - _hp_s)[_eff2]
            else:
                d2 = BMN_sgn * kap2_s[_eff2]
            if args.mnprop2thr > 0.0:
                d2 = np.maximum(d2 - args.mnprop2thr, 0.0)
            if not ris_off:
                # v70 R18n: RIS 规则触发时本通道随 Boyle 环一并关断 —— 否则统一栈里
                # 波由 mnprop2 维持, RIS 门控清零只关 boylescale 半边, 骤停变加速 (r18m)
                if args.mnprop2 != 0.0:
                    _g2 = args.mnprop2
                    if args.bpropbwd != 0.0 and dir_state < 0:
                        # v85: 倒退期B链增益衰减 (Kato et al. 2015 Cell 163:656 前后亚环路互抑analog)
                        _g2 = args.mnprop2 * (1.0 - args.bpropbwd)
                    Iex = Iex.at[BMN_idx].add(jnp.asarray(_g2 * d2, jnp.float64))
                if args.amnprop != 0.0 and (args.amngate == 0 or dir_state < 0):
                    # v85: A类MN本体感觉注入 (Gao 2018: A类倒退振荡器接受本体感觉entrainment);
                    # 镜像B链倒退形态=前方感受(AMN_seg_eff=胞体-amnoffset), 复用kap2_s同一滤波/span产物;
                    # amngate=1时仅倒退态生效 —— 倒退期波驱动由A链接管
                    if args.amnhp > 0.0:
                        # v91: 高通相位性反馈 — 扣除慢EMA参考(=静态弯DC), 只传曲率变化;
                        # 破强tonic下负反馈静态锁定 (倒退锯齿锁修复, 2026-10-22)
                        # W58起: kap2_hp/_hp_s 统一在 d2 计算前更新 (B环bmnhp共用), 此处直接用
                        _sigA = kap2_s - _hp_s
                    else:
                        _sigA = kap2_s
                    Iex = Iex.at[AMN_idx].add(
                        jnp.asarray(args.amnprop * AMN_sgn * AMN_emask * _sigA[AMN_seg_eff], jnp.float64))
                if args.dvaprop != 0.0 and dir_state < 0:
                    # v92: DVA/TRP-4 幅度限制刹车 (Li 2006 Nature PMC2865900: trp-4(lf)=过度弯曲 ->
                    # DVA 牵张感受负向抑制腹索MN/体壁肌 防过弯). 慢EMA曲率=静态弯DC读出 (行波平均~0=AC放行).
                    # 仅倒退态生效 (dir_state<0) -> 爬行回归逐位不动
                    kap2_dva = kap2_dva + (BLOCK_MS / (args.dvatau * 1000.0)) * (kappa - kap2_dva)
                    if args.dvatarget == "mn":
                        # |DC| 超极化注入A类MN胞体节段 (实测: 削弱行波解锁力, 反效果在案)
                        Iex = Iex.at[AMN_idx].add(
                            jnp.asarray(-args.dvaprop * np.abs(kap2_dva[AMN_seg]), jnp.float64))
        if args.mntonic != 0.0:
            # v67 R1 诊断: B 类 MN 恒定去极化 (模拟真虫 tonic 驱动, 提供环路工作点)
            Iex = Iex.at[BMN_idx].add(jnp.asarray(args.mntonic, jnp.float64))
        if args.vdsp != 0.0 and len(W_ad_t) > 0:
            # W62: A->D sp 释放通道 — Iex[D] += G * W_ad @ (softplus(dV_prev)-ln2)
            # dV 取上一块 (50ms 延迟同 ggapmod 例); sp 形式同 muscle96/mndrive (NMJ 释放跟随
            # 突触前去极化=生理第一性, 绕 vrel 在 A->D 组边上的失配; 解剖分布=chem 边权重)
            _spd = np.logaddexp(0.0, np.asarray(dV_prev)[W_ad_s]) - 0.6931471805599453
            _id = np.zeros(N); np.add.at(_id, W_ad_t, W_ad_w * _spd)
            _tgt = np.unique(W_ad_t)
            Iex = Iex.at[jnp.asarray(_tgt)].add(jnp.asarray(args.vdsp * _id[_tgt], jnp.float64))
        if args.avatonc != 0.0:
            # v92: AVA 恒定紧张性注入 (Kato 2015: 稳态倒退=AVA tonic 占优; 无触发无平台)
            Iex = Iex.at[np.asarray(AVA)].add(jnp.asarray(args.avatonc, jnp.float64))
        if args.avamaster != 0.0:
            # v93: AVA 主神经元慢紧张兴奋 AVB (Meng 2024 PMC11006214 双时间尺度模型:
            # 快相位抑制经 RIM/LGC-55 已由 M6 网承载; 此处实现慢兴奋臂 = 倒退退出的内生时钟,
            # 目标=取代 AVB 消融探针, 让前进/倒退态切换全内生). dV 取上一块 (50ms 延迟同 pep2 例)
            am_ema += (max(0.0, float(dV_prev[AVA].mean()) - args.avamasterfloor) - am_ema) \
                      * (BLOCK_MS / 1000.0 / args.avamastertau)
            Iex = Iex.at[np.asarray(AVB)].add(jnp.asarray(args.avamaster * am_ema, jnp.float64))
        if (args.avahold != 0.0 or args.pirrev != 0.0) and hold_I != 0.0:
            # v70 R16: AVA 平台自持注入 (Meng 2024: AVA 紧张性去极化 -> 刺激撤后 bout 自持)
            # v72: pirrev 充电也走此注入
            Iex = Iex.at[np.asarray(AVA)].add(jnp.asarray(hold_I, jnp.float64))
        if args.rivabl != 0.0:
            # v69 R10b: RIV 功能性消融 (Gray et al. 2005 PNAS 102:3184: RIV 消融 ->
            # 反转后转向的腹侧偏置消失); 恒定超极化把 RIVL/R 钳离工作点
            Iex = Iex.at[np.asarray(RIV)].add(jnp.asarray(args.rivabl, jnp.float64))
        if abl_idx and args.cellablmode == "clamp":
            # v84: 通用功能性消融 (Gao 2018 eLife 7:e29915 复现探针)
            # cellablmode=edge 时边已删, 不注入 (真激光消融语义)
            Iex = Iex.at[np.asarray(abl_idx)].add(jnp.asarray(args.cellablp, jnp.float64))
        if args.smdprop != 0.0:
            # v68 R6: SMD 本体感觉注入: 头6节段平均曲率, 负增益=牵张反射
            # (Yeon 2018: TRP-1/2 介导; 背弯κ>0 -> 抑制SMDD / 腹弯 -> 抑制SMDV)
            # v82: 感受野由 --smdpropseg 可调 (默认0:6旧行为; 颈部=3:9, Yeon 2018 原文 neck)
            _sa, _sb = (int(x) for x in args.smdpropseg.split(":"))
            _smd_g = args.smdprop
            if args.smdpropbw != 0 and dir_state < 0:
                # W57: 倒退态自动关闭 SMD 颈部反射 (w57d/e 判决: 常开反射在倒退态
                # 无行波约束头部 -> 打转主因; 生物对应=方向态依赖的反射增益调制,
                # 类比起步时牵张反射增益被下行调制, Lamont 2018 综述)
                _smd_g = 0.0
            Iex = Iex.at[SMD_idx].add(
                jnp.asarray(_smd_g * SMD_sgn * kappa[_sa:_sb].mean(), jnp.float64))
        if args.turnsmd != 0.0 and smd_turn_on:
            # v68 R9: 转向脉冲经SMD执行 (Gray 2005: SMD编码omega转角幅度);
            # SMDV兴奋(腹侧收缩)+SMDD抑制 = 腹向深弯; 经 smddrive 解剖通路作用于头肌
            Iex = Iex.at[SMDV_idx].add(jnp.asarray(args.turnsmd * pulse_dir, jnp.float64))
            Iex = Iex.at[SMDD_idx].add(jnp.asarray(-args.turnsmd * pulse_dir, jnp.float64))
        if args.fc1 != 0.0:
            # v74 R18: 内生预算转向驱动 — 幅度 = fc1 x m_fc1 x RIA->SMD 原位突触预算
            # (预算由上一块电压在线重建: 状态依赖, RIA静默则无驱动; 1块=50ms延迟, 与pep2同例)
            if m_fc1 > 1.0e-3:
                _Vp = dV_prev + V_rest
                _sV = np.clip((_Vp[FC1_pre] - net.vrel[FC1_pre]) / 25.0, 0.0, 1.5)
                _bud = float(args.gsyn * np.sum(FC1_w * _sV * np.abs(_Vp[FC1_post])))
                _amp = args.fc1 * m_fc1 * _bud
                if args.offg != 0.0:
                    # v79 R27b: 近源转向执行增益 (首轮诊断: 时长放大不治本, 执行驱动太弱
                    # ~24°/事件; _amp 同乘 ars -> 近源转向真正 reorient, 远场 ars=1 逐位不变)
                    _amp = _amp * (1.0 + args.offg * max(arr_clp[0] - args.offth, 0.0) / max(1.0 - args.offth, 1e-12))
                Iex = Iex.at[SMDV_idx].add(jnp.asarray(_amp * pulse_dir, jnp.float64))
                Iex = Iex.at[SMDD_idx].add(jnp.asarray(-_amp * pulse_dir, jnp.float64))
            m_fc1 *= float(np.exp(-(BLOCK_MS / 1000.0) / args.fc1tau2))
        if args.weath != 0.0 and ase is not None:
            # v71 W1: weathervane 双向转向 (klinotaxis, Iino & Yoshida 2009 J Neurosci 29:5370)
            # 单点 ASE 只有时间信号, 但蠕动让头在背腹(=侧向)扫过梯度:
            # 头向某侧摆且 dC/dt 升 => 该侧为上梯度 => 持续向该侧弯。
            # w=lowpass(s·g, τw): w>0=背侧上梯度 -> SMDD兴奋/SMDV抑制 -> 背弯 (κ>0=背弯);
            # w<0 反之 -> 腹弯。方向由 s·g 符号决定 => 双向。g=ase.dcdt (带符号滤波值)。
            if args.weathmode == "kap":
                # v77 R22 载波判决胜者: s=tanh(κ0/3) 头关节弯曲位置 — 鼻尖朝向即浓度采样相位,
                # 与 dC/dt 相关得上下梯度侧 (离线判决: 一致率0.92, w方向正确, |w|p95~0.0034;
                # 关节角本体感觉直接可得, Yeon 2018 Nat Commun 9:?? 躯体感觉门控)
                _s = float(np.tanh(kappa[0] / 3.0))
            else:
                # dkap(旧): s=tanh(dκ0/dt/8)=头摆动方向代理 — 符号审计证明与头摆相位失锁,
                # w一致率0.06=翻号垃圾; 保留仅为旧行为逐位不变
                _dk0 = (kappa[0] - kap0_prev) / (BLOCK_MS / 1000.0)
                kap0_prev = float(kappa[0])
                _s = float(np.tanh(_dk0 / 8.0))
            w_update = (BLOCK_MS / 1000.0 / args.weathtau) * (_s * ase.dcdt * asad_gain[0] - w_weath)
            if m_fc1 < 0.5 and t_ms >= w_frz_until_ms:
                # v77 R22b efference-copy式门控: 转向机动主峰期(m_fc1>0.5, ~2s)κ0/dcdt被机动
                # 本身污染(腹弯执行+仍在远离 -> s·g假"背上梯度"信号, 实测致dir=−1误转 |err|放大);
                # 主峰期冻结相关积分, w主要累积正常爬行相的自采样
                # W68: bout末沿后 wrst 秒内同样冻结 (曲率残形耗散期, 同思想)
                w_weath += w_update
            Iex = Iex.at[SMDD_idx].add(jnp.asarray(args.weath * w_weath, jnp.float64))
            Iex = Iex.at[SMDV_idx].add(jnp.asarray(-args.weath * w_weath, jnp.float64))
        if args.headosc != 0.0:
            # v67 R1b: 头部 B 类正弦种子 (D/V 反相 -> 头颈背腹交替弯曲 -> 检验反射链传播)
            _hs = args.headosc * np.sin(2.0 * np.pi * args.headfreq * (t_ms / 1000.0))
            Iex = Iex.at[BMN_idx[BMN_head]].add(
                jnp.asarray(_hs * BMN_sgn[BMN_head], jnp.float64))
        if args.pep2 != 0.0:
            # v73 R17: 倒退环路肽能正反馈 (FLP-14/FRPR-19 环路, Marques 2021 PLoS Genet;
            # MGL-2/mGluR5 突触前增益, Katz 2019 Nat Commun; RIM->AVA 倒退态强化, Sordillo 2021 eLife)。
            # 组池(AVA+AVD+AVE)去极化 -> m2 慢积累 -> 电流回注组池: 感觉种子(ASH ~1.2mV)越过
            # pep2th 即可能点燃; 池电压回落则 m2 衰减 -> 自限。信号取上一块 (1 块=50ms 延迟, 如实声明)。
            _pool = float(dV_prev[AVA].mean() + dV_prev[AVD].mean() + dV_prev[AVE].mean())
            _M2inf = min(max((_pool - args.pep2th) / args.pep2ref, 0.0), 1.0)
            m2_pep += (_M2inf - m2_pep) * (BLOCK_MS / 1000.0 / args.pep2tau)
            n2_pep += (m2_pep - n2_pep) * (BLOCK_MS / 1000.0 / args.pep2tau2)
            Iex = Iex.at[np.asarray(AVA + AVD + AVE)].add(
                jnp.asarray(args.pep2 * (m2_pep - args.pep2inh * n2_pep), jnp.float64))
        if args.ckr1 != 0.0:
            # 二期 CKR-1 肽能增益层 (Chen 2022 Cell Rep 38:110330 PMID 35139370;
            # Chen 2024 PMC10952047 + Mendeley 10.17632/jm9yt2hdnk.1 原始数据锚):
            # 同一 Galpha-q 受体 CKR-1 在 SMD 集成两个运动状态:
            #  NLP-12 臂: DVA(牵张激活机械感觉) -> NLP-12 -> CKR-1 on SMD -> 体曲率增益
            #    突变体: 体曲率 WT 4.380 -> nlp-12 3.435 (-21.6%) / ckr-1 3.596 (-17.9%);
            #    tau12=12.9s (Fig5E 实测 Ca2+ 10-90% 上升); 驱动 = DVA 牵张代理 mean|kappa|/KAP_REF
            #    (同 pep 层, KAP_REF=3.0 已有锚); 效应 = SMD 四细胞 Gαq 去极化注入
            #  NLP-18 臂: ASI(多模态感觉) -> NLP-18 -> CKR-1 on SMD + AIB -> Omega 转向增益
            #    突变体: Omega 曲率 WT 7.072 -> nlp-18 6.250 (-11.6%) / ckr-1 6.203 (-12.3%);
            #    tau18=22.1s; 驱动 = ASH+ASI 感觉池去极化 (softplus(dV)-ln2)/25mV 同 ggapmod 例,
            #    信号取上一块 (50ms 延迟同 pep2 例); 爬行期 dV~0 -> M18~0, 逃逸刺激期升起
            #  臂权重比 w18/w12 = 1.42 = Ca2+ 峰值比 0.502/0.353 (Fig5D 实测激活容量);
            #  EC50 NLP-18 13.6 nM vs NLP-12 26.5 nM (Fig5B 剂量响应, Hill 拟合)
            #  Fig3: NMJ minis/EPSC 不受 CKR-1 影响 -> 增益只加在 SMD/AIB 兴奋性, 不动突触传递
            #  G0 幅度 = 有界估计: 以 in silico 突变体对比 (on/off 曲率差) 校验突变体表型分数,
            #  敏感区间入账 (常数锚定总表二期 CKR-1 节)
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
            # v75 R20 (FC2): AVA 突触外 tonic ACh 基调 -> AVB (Meng et al. Sci Adv 2024 master
            # neuron: AVA 去极化静息~-23mV -> 构成性 ACh 释放 tonic 兴奋前进环路; TWK-40 维持)。
            # 电压门控斜坡: AVA 在去极化区(fc2v0=-45以上) tonic 上线, 钳到超极化区(<-50)自动关闭,
            # 对应 HisCl 剂量实验的前进速度进行性下降 (该文 Fig6A) — 可检验预测。
            # 注: vrel 表 AVA=-83mV 使化学释放饱和, 无法承载 tonic 状态依赖, 故按电压门控实现 (审计备案)。
            _Va = float((dV_prev[AVA] + V_rest[AVA]).mean())
            _g2 = min(max((_Va - args.fc2v0) / args.fc2vs, 0.0), args.fc2cap)
            Iex = Iex.at[np.asarray(AVB)].add(jnp.asarray(args.fc2 * _g2, jnp.float64))
            if args.fc2mn != 0.0:
                # v75 R20c (FC2 第二臂): 同一 AVA 电压门控 tonic -> B类MN池 (突触外 ACh 紧张性
                # 强化腹侧运动亚环路; 倒退期 AVA 去极化 -> 门控保持 -> mnback 镜像波同获工作点)
                Iex = Iex.at[BMN_idx].add(jnp.asarray(args.fc2mn * _g2, jnp.float64))
        if args.avaclamp != 0.0:
            # v75 R20b: AVA 恒定钳制 (HisCl 模拟; 负电流=超极化)
            Iex = Iex.at[np.asarray(AVA)].add(jnp.asarray(args.avaclamp, jnp.float64))
        if args.risdrive != 0.0:
            # W59b: RIS bout 终止积分点火 (Kato 2015 Cell 163:656: RIS 在 reversal 结束/
            # forward 开始时特异性兴奋; 本网RIS全程静默=终结者未激活, 实测 bout 不退根因).
            # 对偶 avaplat2: 倒退态充电(ristau) 前进态放电(ristau2) -> RIS真实去极化 ->
            # 经网内GABA突触自然抑制AVA/A类 -> bout终结; tau锚Zhao2003 bout中位2-3s。
            # 诚实标注: 真虫RIS bout活动的上游驱动机制未明, 此为功能等价实现(有界估计)。
            # W69 riskill: 满充超时强制放电 (RIS瞬态峰自限, Kato 2015; AVB消融僵死修复)
            _ris_charging = (dir_state < 0) and (
                args.riskill == 0.0 or ris_hold_ms < args.riskill * 1000.0)
            if _ris_charging:
                ris_m += (BLOCK_MS / 1000.0 / args.ristau) * (1.0 - ris_m)
                if args.riskill != 0.0 and ris_m > 0.99:
                    ris_hold_ms += BLOCK_MS
            else:
                ris_m += (BLOCK_MS / 1000.0 / args.ristau2) * (0.0 - ris_m)
                if dir_state >= 0:
                    ris_hold_ms = 0.0   # 前进态重置保险丝计时
            Iex = Iex.at[np.asarray(RIS)].add(jnp.asarray(args.risdrive * ris_m, jnp.float64))
        if args.avaplat != 0.0:
            # v76 R21 (FC3): AVA 再生性平台电流 — 快激活 m(sigmoid电压门控, 阈-16mV) +
            # 慢失活 n(跟踪m, tau2=4s) + Ca驱动势自限; 正反馈维持 bout, n 追上 m 即自终结。
            # 逐细胞 (AVAL/AVAR 独立); 信号取上一块 (50ms 延迟同 pep2 例声明)。
            _Va3 = dV_prev[AVA] + V_rest[AVA]
            _minf = 1.0 / (1.0 + np.exp(-(_Va3 - args.avaplatth) / args.avaplatk))
            m_plat = m_plat + (BLOCK_MS / 1000.0 / args.avaplattau) * (_minf - m_plat)
            n_plat = n_plat + (BLOCK_MS / 1000.0 / args.avaplattau2) * (m_plat - n_plat)
            _Iplat = -args.avaplat * m_plat * np.clip(1.0 - n_plat, 0.0, None) * (_Va3 - args.avaplate)
            Iex = Iex.at[np.asarray(AVA)].add(jnp.asarray(_Iplat, jnp.float64))
        if args.avaplat2 != 0.0:
            # v91 (ASH 完整化): 门控沿点火的 AVA 平台电流 — 与 avaplat 同结构 (VGCC 样再生
            # Ca 电流 + 快激活 m + 慢失活 n 自终结), 但点燃信号取倒退门控态 (dir_state<0,
            # 上一块, 50ms 延迟同 pep2/ggapmod 例): 引擎电压尺压缩 (~3x, AVA 行程 2-5mV)
            # 使胞体电压阈路径不可达, 门控翻转才是网络级"AVA 已激活"签名 (v77 R23 pirrevp
            # 事件点火的内生化). 爬行期门控恒前进 -> m=0 -> 零电流 -> 基线逐位不变;
            # bout 期 m 快充 (0.2s), n 慢追 (4s) -> 平台自持 -> n 追上即自终结 (Hilliard
            # 2005 bout 1-3s 量级); 结构/τ/E 锚 Meng 2024 + Mellem 2008, 电导 G 为
            # 有界估计 (无直接测量, 标定目标=bout 连贯且自终结, 账本备案)
            _on2 = 1.0 if dir_state < 0 else 0.0
            mp2 += (BLOCK_MS / 1000.0 / args.avaplat2tau) * (_on2 - mp2)
            np2 += (BLOCK_MS / 1000.0 / args.avaplat2tau2) * (mp2 - np2)
            _Va4 = dV_prev[AVA] + V_rest[AVA]
            _Iplat2 = -args.avaplat2 * mp2 * np.clip(1.0 - np2, 0.0, None) * (_Va4 - args.avaplate)
            Iex = Iex.at[np.asarray(AVA)].add(jnp.asarray(_Iplat2, jnp.float64))
        if args.sloa != 0.0 and AMN_idx.size > 0:
            # v92: SLO-1/SLO-2 型活动依赖钾电流 (Yu/Wang/Gao 2022 Front Mol Neurosci 15:845733
            # 实测: slo-1(lf) tonic rPSC频率+电荷增大/倒退增速, slo-2(lf) tonic~4x/long-lasting~10x,
            # = A类振荡器上的活动依赖负反馈刹车; UNC-2钙内流->KCa外流终结爆发, 七鳃鳗同源保守);
            # 实现: ca=胞体活动轨迹, 内流=max(dV-sloath,0)/25 (UNC-2高压激活阈: 爬行dV≤+11恒零,
            # bout峰值+56才开 = HVA陡激活的约化, 引擎电压尺/25同ggapmod例不添新旋钮),
            # tau=sloatau 清除; I_K=-G*ca*(V-E_K) 外向超极化;
            # 持续静态驱动 -> ca 积累 -> DC 被适应压制; 0.6Hz 节律成分 tau 跟不上 -> 通过
            # = amnhp 信号级高通在胞体级的机制化对应物; 信号取上一块 (50ms 延迟同 pep2 例);
            # 爬行期 A类静默 dV≈0 -> ca→0 -> 零电流 -> 前进零影响 (实测回归闸门验证)
            _dVA = dV_prev[AMN_idx]
            _infl = np.maximum(_dVA - args.sloath, 0.0) / 25.0
            slo_ca += (BLOCK_MS / 1000.0 / args.sloatau) * (_infl - slo_ca)
            _VaA = _dVA + V_rest[AMN_idx]
            _Islo = -args.sloa * np.clip(slo_ca, 0.0, None) * (_VaA - args.sloe)
            Iex = Iex.at[AMN_idx].add(jnp.asarray(_Islo, jnp.float64))
            if args.amnosc != 0.0:
                # v92: UNC-2 再生臂 (Yu 2022 同一通道的去极化臂; Gao 2018 A类内生振荡器):
                # 再生内流点火 -> 电压上冲 -> ca 积累 -> SLO 刹车熄灭 -> 复归 = 松弛振荡;
                # 振荡驱动替代 tonic 平台供养静态 Turing 模; _infl 爬行期恒零 -> 回归不动
                Iex = Iex.at[AMN_idx].add(jnp.asarray(args.amnosc * AMN_gfac * _infl, jnp.float64))
        if run_block_lv is not None:
            # 二期 ggapmod: M <- 前感觉/多巴胺能层活动 (softplus(dV)-ln2, dV 取上一块 50ms 延迟
            # 同 pep2 例; V_SCALE=25 引擎自身电压尺不添新旋钮; 静息 dV≈0 (V_rest=预热末态),
            # softplus 凸性致爬行期 M~0.03 微 jitter, 电导影响 <2%, 忽略)
            if args.ggapmod != 0.0:
                _spda = float((np.logaddexp(0.0, dV_prev[DAT]) - 0.6931471805599453).mean())
                gm_inf = min(max(_spda / 25.0, 0.0), 1.0)
                gm_m += (gm_inf - gm_m) * (BLOCK_MS / 1000.0 / args.ggapmodtau)
                _gmod = 1.0 - args.ggapmodk * (1.0 - gm_m)
            else:
                _gmod = 1.0
            if AR is not None:
                # W49: 整流档 — 逐块按上一块绝对电压 (dV_prev+V_rest, 50ms 延迟同 ggapmod 例)
                # 组装 Boltzmann 门控 Lg; 与 ggapmod 可叠加 (先整流边替换, 再 _gmod 全尺度)
                _Vp = dV_prev + V_rest
                _Lgt = jnp.asarray(_gmod * _Lg_w49(_Vp), jnp.float64)
            else:
                _Lgt = jnp.asarray(_gmod * Lg_full_np, jnp.float64)
            if Lg_avd_np is not None:
                # W54b: A<->VD/DD gap 按方向态换档 (倒退衰减, 前进复原=加零)
                _fvd = args.avdx if dir_state < 0 else 1.0
                _Lgt = _Lgt + jnp.asarray(_gmod * (_fvd - 1.0) * Lg_avd_np, jnp.float64)
            if run_block_lvm is not None:
                # W54: A->B 串扰按方向态换档 (倒退衰减, 前进复原)
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
            # v60 相对门控: 方向 = AVA/AVB 相对比较 + 迟滞; 双方低于 floor 时闭门(静息)
            if dir_state > 0 and dA > dB + gate_hyst and dA > gate_floor:
                dir_state = -1
            elif dir_state < 0 and dB > dA + gate_hyst and dB > gate_floor:
                dir_state = +1
            act = max(dB, dA)
            g_f = 1.0 if (dir_state > 0 and act > gate_floor) else 0.0
            g_b = 1.0 if (dir_state < 0 and act > gate_floor) else 0.0
        elif args.gate == "relg":
            # v61 组池相对门控: 倒退侧 = AVA+AVD+AVE 组池 (Chalfie 1985 命令元组解剖),
            # 前进侧 = AVB; 同样的迟滞/地板规则
            dA = dV[AVA].mean() + dV[AVD].mean() + dV[AVE].mean()
            if args.risbrk != 0.0:
                # W59: RIS 刹车计入倒退池 (Kato 2015 Cell 163:656: RIS 在 reversal 结束时
                # 特异性兴奋, GABA能抑制AVA/A类 = bout终止器; 网内RIS->AVA突触单细胞
                # 压不过3细胞池的同相共振 — 门控层如实计入RIS对倒退命令的功能性抑制;
                # 0=关闭=旧行为逐位不变)
                # W75 (2026-10-24, 七行为回归修复): switch 协议 AVA 相 (10-22s) 是实验者
                # 持续钳制 (Gao 2018 AVA::ChR2 持续光照=持续倒退), 应覆盖内生 RIS 终结
                # (risbrk 生理语义=自发 bout 终结器, Kato 2015 自发 reversal->forward 瞬态);
                # 其余协议/时窗逐位不变 (region/avapulse 刺激后 bout 2-3s 终结是对的, Wicks 1995)
                _risbrk_on = not (args.proto == "switch" and 10.0 <= (t_ms / 1000.0) < 22.0)
                if _risbrk_on:
                    dA = dA - args.risbrk * max(float(dRIS) - args.risbrkth, 0.0)
            if dir_state > 0 and dA > dB + gate_hyst and dA > gate_floor \
                    and (t_ms - last_bout_end_ms) > args.bwrefr * 1000.0:
                # v91 bwrefr: bout 后不应期窗内封锁前进->倒退翻转 (Wicks & Rankin 1995)
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
            # v70 R16e: 触发沿充电 + 无条件衰减 + 2s 再武装 — bout 结束后门控若在
            # 残余电流上抖动会在1块内"前进->倒退"闪翻, 无再武装则无限再触发;
            # 要求连续前进 >=2s 才允许下一次充电 (Wicks & Rankin 1995: bout 后反应不应期)
            # v72: pirrev 在事件脉冲处充电; 此处仅负责衰减与再武装 bookkeeping
            if args.avahold != 0.0 and dir_state_prev > 0 and dir_state < 0 and (t_ms - fwd_since) > 2000.0:
                hold_I = args.avahold
            else:
                hold_I *= np.exp(-BLOCK_MS / args.avaholdt)
            if dir_state > 0 and dir_state_prev < 0:
                fwd_since = t_ms
        _bout_end = (dir_state_prev < 0 and dir_state > 0)   # B4: 倒退 bout 末沿 (逃逸-omega 串联触发信号)
        if _bout_end:
            last_bout_end_ms = t_ms   # v91 bwrefr: bout 末沿计时 (不应期起点)
            if args.wrst != 0.0:
                # W68: bout 已改变朝向, 旧 w 作废清零; 曲率残形耗散期内冻结累积
                w_weath = 0.0
                w_frz_until_ms = t_ms + args.wrst * 1000.0
                # W68b: bout 结束=转向机动结束, fc1 驱动清零不拖尾 (Pierce-Shimomura 1999:
                # pirouette 后接直行 run; m_fc1 tau2=3s 长尾会把 bout 后爬行钉在深弯打转态,
                # 实测 w68_wrst2_d180 bout 后 |κ|max 17-24 持续 6s 不降)
                m_fc1 = 0.0
        dir_state_prev = dir_state
        if args.turnmode == "event":
            # v64 事件化 pirouette: dSMD 上穿阈 -> 定长转向脉冲 + 不应期 (离散 pirouette,
            # Pierce-Shimomura 1999); 直行段径向回逼, 打破连续门控的 -90° 轨道不动点
            t_s = t_ms / 1000.0
            if args.turnsrc == "aiy":
                steer = (dV[AIYR_i] - dV[AIYL_i])
                thr = args.steerthresh
            elif args.turnsrc == "dwn":
                # v83: 恶化速率触发 (Pierce-Shimomura 1999): steer=恶化水平(-dcdt),
                # 感觉层自身的 dcdt 滤波值(tau=0.5s), 非外部信号; 稳态远离时持续超阈 ->
                # 不应期一过即重触发 (生物 pirouette bout 节拍), 方向仍由 w_weath 网络相关给出
                # v83b: 再过一道 dwntau 慢低通 — 头摆侧向速度对瞬时 dcdt 的调制(±0.03)
                # 大于净趋化信号(±0.025), 不过滤则触发锁相头摆 (d000 方位实测连环误触发)
                _raw = float(-ase.dcdt) if ase is not None else 0.0
                dwn_lp += (BLOCK_MS / 1000.0 / args.dwntau) * (_raw - dwn_lp)
                steer = dwn_lp
                thr = args.dwnthresh
            else:
                steer = dSMD
                thr = args.smdthresh
            if t_s < pulse_until:
                # v68 R9: turnsmd!=0 时转向由 SMD->肌肉通路执行, 人工 koff 偏置关闭
                head_bias = (args.pulsebias * pulse_dir) if (args.turnsmd == 0.0 and args.fc1 == 0.0) else 0.0
            elif (steer > thr or (args.steerbi != 0.0 and steer < -thr)
                    or (args.escomega != 0.0 and _bout_end)) and t_s >= refr_until \
                    and (ase is None or (ase.dcdt < args.pirsup
                                         and (args.pirc >= 999.0
                                              or chem_c[0] / max(ase.f.peak, 1e-12) < args.pirc))):
                # v64b 修正: 电平+不应期再触发 (不用上穿沿——持续降梯度时 dSMD 不下阈,
                # 边沿检测只发一次脉冲后径直跑离); 生物重复 pirouette 正是节拍式重触发
                # v78 R25: 接近抑制门 (pirsup<999 时) — 低通 dC/dt >= 阈 = 正在接近源,
                # 禁止转向再触发 (Pierce-Shimomura 1999: pirouette 只在恶化时发动);
                # 直线滑向峰, 越过峰 dC/dt 翻负 -> 立即放行矫正转向 -> 近源驻留
                # v78 R25b: 绝对浓度驻留门 (pirc<999 时) — 即时浓度 >= pirc(峰归一) 抑制触发,
                # 环内梯度归零后靠浓度水平停驻 (Sawin 2000/Hills 2004/Chalasani 2007)
                _pdur = args.pulsedur
                if args.pirmag > 0.0:
                    # v78 R25b: 转角随方位误差标定 (Pierce-Shimomura 1999 转角-误差相关,
                    # 大误差->深长 omega 弯; 上限 3x 防 regime 崩坏)
                    _pdur = args.pulsedur * float(np.clip(abs(steer) / args.pirmag, 1.0, 3.0))
                if args.offg != 0.0:
                    # v79 R27: OFF-response/局域搜索转向增强 — 近源(低通浓度>offth)时
                    # 不应期 /=ars -> 转向再触发加密 (首轮教训: 放大脉冲时长不治本且过冲,
                    # 执行增益增强改在 fc1 _amp 处同 ars 施加); offg=0 时本段不执行,
                    # 远场/接近段 ars=1 逐位不变 (Hills 2004/Chalasani 2007/Gray 2005/
                    # Lopez-Cruz 2019; dwelling几何=低速+高转向 Flavell 2013)
                    _ars = 1.0 + args.offg * max(arr_clp[0] - args.offth, 0.0) / max(1.0 - args.offth, 1e-12)
                pulse_until = t_s + _pdur
                refr_eff = args.refract
                if args.refrmod != 0.0 and (steer > 0.0 or (args.steerbi != 0.0 and steer < 0.0)):
                    # v73: 转向频率调制 (Pierce-Shimomura 1999 核心发现: pirouette 频率随
                    # 恶化速率 dC/dt<0 增加); steer 幅度 ≈ v·sin(误差角)·|∇C|, 大误差掉头期
                    # 缩短不应期 -> 脉冲密集 (生物 pirouette bout, 40%为连续多转), 锁定后恢复
                    # R19: 双向触发时用 |steer| 标定 (负支=快速接近期的大摆幅)
                    _sref = abs(steer) if args.steerbi != 0.0 else steer
                    refr_eff = args.refract * float(np.clip(args.refrmod / _sref, 0.3, 1.0))
                if args.offg != 0.0:
                    refr_eff = refr_eff / _ars          # v79 R27: 近源再触发加密
                refr_until = pulse_until + refr_eff
                if rng_pir is not None:
                    # v71 W2: 随机方向 pirouette (Pierce-Shimomura 1999: 时机编码梯度, 方向重随机)
                    pulse_dir = 1.0 if rng_pir.random() < 0.5 else -1.0
                if args.pirrev != 0.0:
                    # v72: pirouette 倒退耦合 (Pierce-Shimomura 1999: pirouette=倒退bout+omega
                    # 转向耦合; 倒退先消除前进动量, 头弯随后在小半径完成; 复用 hold_I 衰减,
                    # bout 时长由 avaholdt 控制, 脉冲触发即充电, 无条件衰减 -> bout 自发终结)
                    # v76: 方位误差门控 (pirrevang>0 时) — 仅 |朝向-源方位| >= pirrevang 才点火,
                    # 小误差交给 weathervane/SMD 臂 (Pierce-Shimomura 1999 / Iino 2009 分工;
                    # W75 d090 误点火 57.5% dAVA 占空 = 退化实锤, 对照 pirrev=0 接近 +3.46mm);
                    # 朝向/源方位取上一块身体态 (50ms 延迟同 ggapmod/amnprop 例); ase=None 放行
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
                    # v77 R23: 事件点火AVA平台门 (m_plat 充电 -> FC3再生性VGCC平台驱动倒退bout,
                    # 杀前进动量 -> omega小半径转向; m_plat自身0.2s衰减, 需平台再生点燃才持续,
                    # bout由n_plat慢失活自终结 — 执行端全内生, 仅"事件->点火"耦合为工程规则)
                    m_plat = np.maximum(m_plat, args.pirrevp)
                if args.fc1 != 0.0:
                    # v74 R18: 方向内生 + 门控充电 (fc1dir=steer: AIY(R-L)符号即转向方向;
                    # random: 沿用上方 rng_pir 重掷; fixed: 恒 +1 腹向;
                    # weath: v77 R22, pulse_dir=-sign(w_weath), 空间方向唯一实测载体)
                    if args.fc1dir == "steer":
                        pulse_dir = 1.0 if steer >= 0.0 else -1.0
                    elif args.fc1dir == "fixed":
                        pulse_dir = 1.0
                    elif args.fc1dir == "weath":
                        pulse_dir = -1.0 if w_weath > 0.0 else 1.0
                    m_fc1 = 1.0
                    print(f"[fc1] t={t_s:.1f}s 事件充电 dir={pulse_dir:+.0f} steer={steer:+.2f}mV",
                          flush=True)
                if rng_pir is not None:
                    print(f"[pir] t={t_s:.1f}s pulse_dir={pulse_dir:+.0f} steer={steer:+.1f}mV"
                          + (f" rev={args.pirrev:g}pA" if args.pirrev != 0.0 else "")
                          + (f" refr={refr_eff:.1f}s" if args.refrmod != 0.0 else ""), flush=True)
                if args.escomega != 0.0 and _bout_end:
                    print(f"[escomega] t={t_s:.1f}s bout末沿触发 omega 转向脉冲 "
                          f"dir={pulse_dir:+.0f} dur={_pdur:.1f}s", flush=True)
                head_bias = (args.pulsebias * pulse_dir) if (args.turnsmd == 0.0 and args.fc1 == 0.0) else 0.0
            else:
                head_bias = 0.0
            smd_turn_on = args.turnsmd != 0.0 and t_s < pulse_until
            dsmd_prev = dSMD
        elif args.turnmode == "pirouette":
            # v63 pirouette 门控: 单点化学感受器无法测梯度左右 (Pierce-Shimomura 1999),
            # 生物策略 = 升梯度抑制转向直行, 降梯度触发转向; dSMD<0=ASEL相, >0=ASER相 (实测符号)
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
        # 肽能慢变量更新: DVA 读取全身平均|曲率| -> NLP-12 水平 (秒级时间常数)
        M_inf = min(np.abs(kappa).mean() / KAP_REF, 1.0)
        pep_m += (M_inf - pep_m) * (pep_dt / args.peptau)
        g_pep = 1.0 + args.pep * pep_m    # CKR-2 突触前增益 (Hu 2011: ACh 释放增强)
        # aldicarb: AChE 阻断 -> ACh 随时间累积 (药效学压缩时标)
        g_ald = 1.0 + args.ald * (1.0 - np.exp(-(t_ms / 1000.0) / args.aldtau))
        g_tot = g_pep * g_ald
        # v66 负载回路驱动 (每块一次, dV 块内恒定): softplus(dV)-ln2 -> 解剖支配矩阵 -> 肌行
        if args.mndrive != 0.0 or args.dmndrive != 0.0 or args.smddrive != 0.0:
            sp = np.logaddexp(0.0, dV) - 0.6931471805599453
        if args.mndrive != 0.0:
            # v78 R25c: 绝对浓度减速 (Sawin 2000 基础减速; Hills 2004 ARS; 多巴胺
            # DOP-2/3 抑制 RIC/SIA -> 降 ACh/章胺; DA->DVA 促驻留) — 低通浓度超阈时
            # 前进驱动乘 arrg; smddrive 不减速 (局域搜索需要转向力保留)
            _mnd = args.mndrive
            if ase is not None and args.arrc < 999.0 and arr_clp[0] >= args.arrc:
                _mnd = args.mndrive * args.arrg
            mndrv_D = _mnd * (Wdrv_D @ sp)
            mndrv_V = _mnd * (Wdrv_V @ sp)
        else:
            mndrv_D = 0.0; mndrv_V = 0.0
        if args.smddrive != 0.0:
            # v68 R6: SMD->头肌驱动 (Yeon 2018: SMD 双角色=感受+驱动)
            smddrv_D = args.smddrive * (Wsmd_D @ sp)
            smddrv_V = args.smddrive * (Wsmd_V @ sp)
        else:
            smddrv_D = 0.0; smddrv_V = 0.0
        if args.dmndrive != 0.0:
            # v67 R2: D 类交叉抑制, 从同侧激活减去 (GABA 放松对侧收缩);
            # 抑制不进 g_tot (aldicarb/pep 均为 ACh 特异性), dmndrv=0 时减 0.0 逐位不变
            dmndrv_D = args.dmndrive * (Wdm_D @ sp)
            dmndrv_V = args.dmndrive * (Wdm_V @ sp)
        else:
            dmndrv_D = 0.0; dmndrv_V = 0.0
        for _ in range(int(round(BLOCK_MS / 1000 / motor_dt))):
            kr = np.concatenate([[kappa[0]], 0.5 * (kappa[1:] + kappa[:-1]), [kappa[-1]]])
            if m96 is not None:
                # v81: 自有肌肉层驱动 — 驱动信号经解剖矩阵, 96肌各自积分;
                # dV 块内恒定 (50ms 块粒度); g_tot 只放大 ACh 边
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
                    # v92: DVA刹车肌肉侧 (Li 2006: DVA抑制体壁肌收缩) — 关节j曲率<->肌行j,j+1
                    # (preferred_curvature 同一映射); |DC| 越深收缩越松, 行波AC分量不受影响
                    _bk = np.abs(kap2_dva)
                    _bkr = 0.5 * (np.concatenate([_bk, _bk[-1:]]) + np.concatenate([_bk[:1], _bk]))[:24]
                    _dg = 1.0 / (1.0 + args.dvaprop * _bkr)
                    A_D = A_D * _dg; A_V = A_V * _dg
            else:
                fD, fV = fwd_m.step(kr, motor_dt, g_f, 0.0, kappa_offset=koff)
                bD, bV = bwd_m.step(kr, motor_dt, g_b, 0.0, kappa_offset=koff)
                # boylescale=1 且 mndrv=0 且 dmndrv=0 时逐位退化为里程碑形式 (f+b)*g_tot
                drv_D = ((fD + bD) * args.boylescale + mndrv_D + smddrv_D) * g_tot - dmndrv_D
                drv_V = ((fV + bV) * args.boylescale + mndrv_V + smddrv_V) * g_tot - dmndrv_V
                if args.muscletau > 0.0:
                    # v68 R7: 肌肉激活一阶积分 (Zhao 2025: 负反馈延迟来自肌肉收缩/舒张动力学;
                    # 这给了局部 MN-肌肉负反馈环起振所需的相位资源)
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
                        A_D=np.array([r["A_D"] for r in log]),   # v81 P3: 肌行激活轨迹 (校准探针用)
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
        print(f"[ckpt] 已保存 {args.savestate} @ t={t_ms/1000:.1f}s", flush=True)

    # ---- 行为指标 ----
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

    # ---- 趋化判据 (对齐 worm-whisperer phase3_chemotaxis.py 的输出格式) ----
    if args.proto == "chem" and len(log) > 8:
        cs = np.stack([np.asarray(r["x"]).mean(0) for r in log])      # 质心轨迹
        heads = np.stack([np.asarray(r["x"])[0] for r in log])        # 头节点轨迹
        hd0 = xs[0][0] - xs[0][-1]; hd1 = xs[-1][0] - xs[-1][-1]
        cross = float(hd0[0] * hd1[1] - hd0[1] * hd1[0])
        dot = float(hd0 @ hd1) / (np.linalg.norm(hd0) * np.linalg.norm(hd1) + 1e-12)
        ang = float(np.degrees(np.arctan2(cross, np.clip(dot, -1.0, 1.0))))
        if ase is not None:
            src = ase.f.src
            d0 = float(np.linalg.norm(cs[0] - src)); d1 = float(np.linalg.norm(cs[-1] - src))
            rep = ASEInput(ase.f, gain=args.chemgain)                 # 确定性回放取峰值
            imax = 0.0
            for hy in heads:
                o_, f_, _ = rep.currents(hy, BLOCK_MS / 1000.0)
                imax = max(imax, o_, f_)
            print(f"[chem] 源距离 {d0:.2f} -> {d1:.2f} mm (接近 {d0 - d1:+.2f} mm)  "
                  f"净转向 {ang:+.1f} deg  max ASE {imax:.1f} pA"
                  + (f"  asad={asad[0]:.2f} gain={asad_gain[0]:.2f}" if args.asad != 0.0 else ""), flush=True)
        else:
            print(f"[chem] 无源对照  净转向 {ang:+.1f} deg  "
                  f"质心位移 ({cs[-1][0]-cs[0][0]:+.2f},{cs[-1][1]-cs[0][1]:+.2f}) mm", flush=True)


if __name__ == "__main__":
    main()
