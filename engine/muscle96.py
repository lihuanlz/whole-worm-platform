# muscle96.py -- v81 自有生理分组肌肉层 (96 体壁肌各自一阶积分, 抛弃 Boyle 砰-砰层)
# 立项书: 06_蠕虫连线组线/v81_自有肌肉层立项书_2026-10-21.md §3
# 常数表: data/muscle96_constants.csv (P1, 全部文献锚定)
#
# 架构 (立项书§3, 无一处自造):
#   rel_n = clip((V_n - vrel_n)/V_SCALE, 0, S_MAX)     全网统一释放表 (jax_circuit.py 同规则)
#   I_i   = Σ_n sign_n · (wout[n,i]/n_inn[i]) · rel_n   ACh 边正号; DD/VD 的 GABA 边负号
#   dmA_i/dt = (clip(I_i·g, 0, Amax) - mA_i)/tau_m      96 肌各自一阶积分
#   A_D[r] = mean(mA[DR_r], mA[DL_r]), A_V[r] 同理       (象限平均到 24 行, 与旧路径同量纲)
#   κ0[r]  = kappa_max · (A_D - A_V)                     (preferred_curvature 原函数复用)
#
# 设计纪律:
#   - tau_m 起步 80 ms (C1.8: Liu 2011 AP 半宽 ~20ms + Gao&Zhen 收缩时序 + P3 校准定死),
#     永不再当频率旋钮; 频率由神经环+本体感觉环出。
#   - GABA v1 走减法 (与现 dmndrv 路径一致); shunting 抑制留 v2 评估 (立项书§3)。
#   - Amax=1.5 与旧路径 clip 上限一致, 平滑线性饱和, 不用硬 clip 吃差分。
#   - aldicarb/肽能 g_tot 只放大 ACh 边 (ACh 特异性, 与 v62 既有注释一致)。
#   - 本模块只在 --muscle96 != 0 时实例化; 默认关, 旧路径逐位不变。
import re
import numpy as np

V_SCALE, S_MAX = 25.0, 1.5   # 与 jax_circuit.py 全网释放规则同值


class Muscle96:
    """96 体壁肌分级激活层。输入: 神经元电压 (mV, 原位); 输出: 24 行背/腹激活。

    方向门控 (v81 P3 实测倒逼, 与旧 Boyle 层 g_f/g_b 语义一致):
    前进态只放行前进组 (DB/VB 兴奋 + DD 交叉抑制), 倒退态只放行倒退组 (DA/VA + VD),
    头类 (RME/RMD/SMD/SMB) 恒通。生物学对应: 命令中间神经元 AVB/AVA 互斥门控
    A/B 类运动神经元 (Kawano 2011; Kato 2015); 旧 Boyle 路径本就有 g_f/g_b 门控,
    这里把同一命令门控移到解剖驱动界面, 不是新机制。
    P3 实测依据: 离线重放 ablA 好轨迹, A 类在前进爬行中仍带反向行波
    (lag=-0.75s, NCC=0.72, amp 为 B 类的 42%), 不门控时高增益下波方向失稳。"""

    def __init__(self, muscle_maps, tau_m_ms=80.0, amax=1.5, neuron_names=None,
                 ei_nmj=0.6, tau_ca_ms=0.0, gj_cc=0.0):
        muscles, wout, inhib, n_inn = muscle_maps
        self.names = list(muscles)                     # 96 个, 形如 DR01/VL07
        sign = 1.0 - 2.0 * inhib                       # ACh +1; DD/VD (GABA) -1
        W = (wout.T / n_inn[:, None]) * sign[None, :]  # (96, N) 归一带符号支配矩阵
        exc = sign > 0
        self.Wexc = np.where(exc[None, :], W, 0.0)     # ACh 边 (g_tot 作用对象)
        # P7 (2026-10-21 常数锚定): NMJ 实测 E:I=1.6:1 → GABA 边乘 ei_nmj=0.6
        #   锚: Liu/Hollopeter/Jorgensen PNAS 2009 (PMC2705609) — 同制备场刺激全胆碱能
        #   诱发 2.01±0.11 nA vs ChR2 全GABA能诱发 1.23±0.12 nA → I/E = 0.61。
        #   旧值 1.0 (符号对称) 无任何实测依据。驱动项与基线项共用 self.Winh, 缩放一致。
        #   注: 该比值属 NMJ 聚合电流比; P1 已证伪其向神经-神经边的全局外推。
        self.ei_nmj = float(ei_nmj)
        self.Winh = self.ei_nmj * np.where(exc[None, :], 0.0, W)  # GABA 边 (负值, 已乘 E:I)
        # P7 绝对电流报告尺: 满幅胆碱能驱动(全部ACh运动神经元 S_MAX 释放时的平均驱动)
        #   ↔ 实测 2.01 nA (PMC2705609) → g_abs nA/单位无量纲驱动。
        #   电流→收缩映射线虫无实测 (领域空白, 诚实标注): mA 保持无量纲走 kmax 标定路径,
        #   g_abs 只把驱动翻译成 nA 用于审计报告, 不进动力学数值路径 (标量在增益中对消)。
        _dmax = self.Wexc @ (S_MAX * np.ones(wout.shape[0]))
        self.g_abs = 2.01 / float(_dmax.mean())        # nA / 单位无量纲驱动
        row = np.zeros(96, dtype=int)
        quad = np.zeros(96, dtype=int)                 # 0=背(DR/DL), 1=腹(VR/VL)
        side = np.zeros(96, dtype=int)                 # 0=L, 1=R (备用)
        for i, m in enumerate(self.names):
            mm = re.fullmatch(r"([DV])([LR])(\d\d)", m)
            if mm is None:
                raise ValueError(f"muscle name not [DV][LR]dd: {m}")
            quad[i] = 0 if mm.group(1) == "D" else 1
            side[i] = 0 if mm.group(2) == "L" else 1
            row[i] = int(mm.group(3)) - 1
        self.row, self.quad, self.side = row, quad, side
        # P11 (2026-10-23): 肌肉-肌肉缝隙连接层 (Liu/Chen/Wang 2011 JBC 286:44285 PMC3243499:
        #   相邻肌AP/钙同步由GJ负责且不依赖神经输入; 纵向同象限相邻耦合系数实测0.25(unc-9敲除->0.1);
        #   同排左右象限间同步55% vs 纵向75% -> 横向权=纵向x0.73(有界估计, 申报);
        #   Liu 2006 JBC 281:7881: 同象限肌腹间+象限间肌臂间均有耦合; White 1986 同象限肌腹GJ)。
        # 映射: 两室稳态 cc=g/(1+g) -> g=cc/(1-cc)=1/3; 每肌行链三对角稳态解
        #   (1+c*deg)E - c*E_nbr = A_inf, 预求逆矩阵, 步进时 E_ss=Minv@A_inf 替代快层目标;
        #   指数欧拉向 E_ss -> 无条件稳定且稳态严格复现实测耦合系数。0=关闭=旧行为逐位不变。
        self.gj = float(gj_cc)
        self._Minv = None
        if self.gj > 0.0:
            c_lon = (0.25 / 0.75) * self.gj
            c_tra = c_lon * 0.73
            M = np.zeros((96, 96))
            for i in range(96):
                deg = 0.0
                # 纵向: 同象限(same quad+side)相邻行
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
            print(f"[muscle96] P11 肌肉缝隙连接层 ON: 纵向c={c_lon:.3f}(cc=0.25锚) "
                  f"横向c={c_tra:.3f}(x0.73有界估计), 稳态三对角解析解", flush=True)
        self.tau = float(tau_m_ms)                     # ms (两层模式: 快层=电兴奋/AP 层)
        self.tau_ca = float(tau_ca_ms)                 # ms; >0 开两层 (P10, 2026-10-22):
        # 快层 tau = AP 半宽锚 (Gao&Zhen 2011 PNAS 108:2557 实测 15.5±0.9ms;
        #   Liu 2011 JPhysiol 589:101 ~20ms 互证, PMC3039263);
        # 慢层 tau_ca = 钙/收缩层锚 (Butler/Branicky 等 J R Soc Interface 2015;12:20140963
        #   实测 GCaMP3 肌肉荧光衰减 1.57s, 相位校正中间滤波假设 1.0s — 表观钙瞬变衰减,
        #   含指示剂动力学, 真实收缩 tau 更小 -> 取 1.0s 为有界估计下沿, 敏感带申报)。
        # 直流增益不变 (稳态 mA=mE=A_inf), 只改动态滤波与相位; 0=旧单层逐位不变。
        self.mE = np.zeros(96)                         # 两层模式电兴奋积分态
        self.amax = float(amax)
        self.mA = np.zeros(96)                         # 激活积分态
        # 行聚合掩码: (24, 96) L/R 平均
        M = np.zeros((2, 24, 96))
        for i in range(96):
            M[self.quad[i], self.row[i], i] = 0.5
        self.agg = M
        # 紧张性基线 sig0: 静息紧张性释放 (v59 设计 rel_rest≈0.4)。
        # 减去它 = "激活表示对静息张力的偏离", 与全代码库 dV=V-V_rest 约定同构;
        # 背腹解剖接触数本不对称 (DA9+DB7+DD6 vs VB11+VD13), 不减基线则静息弯身。
        # aldicarb 下 (g_tot>1) I=(g_tot-1)*exc_rest>0 -> 紧张性挛缩, 与药理一致。
        self.sig0 = np.full(wout.shape[0], 0.4)
        # 方向分组掩码 (disjoint): 前进组=DB/VB+DD, 倒退组=DA/VA+VD, 其余(头类等)恒通
        N = wout.shape[0]
        self.m_fwd = np.zeros(N); self.m_bwd = np.zeros(N); self.m_always = np.ones(N)
        if neuron_names is not None:
            for i, n in enumerate(neuron_names):
                n = str(n)
                if re.fullmatch(r"(DB\d\d|VB\d\d|DD\d\d)", n):
                    self.m_fwd[i] = 1.0; self.m_always[i] = 0.0
                elif re.fullmatch(r"(DA\d\d|VA\d\d|VD\d\d)", n):
                    self.m_bwd[i] = 1.0; self.m_always[i] = 0.0
            print(f"[muscle96] 方向门控: 前进组 {int(self.m_fwd.sum())} / "
                  f"倒退组 {int(self.m_bwd.sum())} / 恒通 {int(self.m_always.sum())}", flush=True)

    def calibrate_rest(self, V_rest, vrel, mode="rel"):
        """存真实静息驱动 sig0 (v62 预热后调用一次)。
        mode='sp' 时 sig0 恒 0 (softplus(dV)-ln2 静息=0, 无需表)。"""
        if mode == "rel":
            self.sig0 = self.release(V_rest, vrel)
        else:
            self.sig0 = np.zeros_like(self.sig0)
        self.sig0_h = self.sig0.copy()   # v86: 稳态校准跟踪态 (m96homeo>0 时用)
        self.reset()
        return self.sig0

    def reset(self):
        self.mA[:] = 0.0
        self.mE[:] = 0.0

    def release(self, V, vrel):
        """全网统一分级释放: 静息时 s≈0.4 (v59 设计: vrel=原位静息-10)。"""
        return np.clip((V - vrel) / V_SCALE, 0.0, S_MAX)

    @staticmethod
    def sp_signal(dV):
        """softplus(dV)-ln2 偏差驱动 (v66 mndrive 同式): 静息=0, 去极化近似线性,
        不依赖 vrel 表 — 对 vrel 失配免疫 (v81 P3 对照臂)。"""
        return np.logaddexp(0.0, dV) - 0.6931471805599453

    def step(self, sig, motor_dt_ms, g_tot=1.0, gate_f=1.0, gate_b=0.0, gain=1.0,
             homeo_tau_s=0.0):
        """推进一个运动层步长, 返回 (A_D, A_V) 各 (24,)。
        激活 = 对静息张力基线的偏离; 静息态约定为前进门控 (真虫默认态=前进爬行)。
        gate_f/gate_b: 方向命令门控 (=v62 的 g_f/g_b, AVB/AVA 互斥门控的接口)。
        gain: 作用在每肌饱和【之前】 (P3 实测: 乘在聚合后会毁掉比例信号,
              环路 bootstrap 不起来; ablA 对照 mndrive=0.10 同为 pre-clip 结构)。
        homeo_tau_s>0: v86 稳态张力校准 — 基线 sig0_h 以 tau(秒) 慢跟随各神经元
              tonic 释放, 只透过波频(相位)分量; tonic 不对称 (vrel饱和/连接组偏心)
              被吸收 -> 消除静态弯偏置 (爬行左拐)。0=关闭=旧行为逐位不变。"""
        mult = self.m_always + gate_f * self.m_fwd + gate_b * self.m_bwd
        mult0 = self.m_always + self.m_fwd          # 静息=前进态基线
        if homeo_tau_s > 0.0:
            self.sig0_h += (motor_dt_ms / (homeo_tau_s * 1000.0)) * (sig - self.sig0_h)
        base = self.sig0_h if homeo_tau_s > 0.0 else self.sig0
        I = ((self.Wexc @ (sig * mult)) * g_tot + (self.Winh @ (sig * mult))
             - (self.Wexc + self.Winh) @ (base * mult0))
        A_inf = np.clip(gain * I, 0.0, self.amax)
        if self._Minv is not None:
            # P11: 肌肉缝隙连接稳态重分配 (电兴奋在耦合片上的稳态扩散)
            A_inf = self._Minv @ A_inf
        if self.tau_ca > 0.0:
            # P10 两层级联: 电兴奋快层 -> 钙/收缩慢层 (直流增益不变, 只改动态)
            # 快层 tau=15.5ms < 运动步长 50ms, 显式欧拉必炸 -> 指数欧拉(一阶闭式解, 无条件稳定)
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
        """头组(行01-06)/身体组(行07-24)掩码, 供 P3 分组校准探针用。"""
        head = self.row < 6
        return head, ~head
