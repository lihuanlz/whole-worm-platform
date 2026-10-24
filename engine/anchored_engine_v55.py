# anchored_engine_v55.py
# 【v5.5 = 实测动力学引擎（P27，2026-10-06 模型卡冻结）：
#   逐行移植 Nicoletti 2019 PLoS ONE 14(7):e0218738 官方 XPPAUT 源
#   （ModelDB #267187，refs/AWC.ode、refs/RMD.ode），含全部校准位移参数。
#   单位: V=mV, I=pA (nS·mV), t=ms, C=pF。dv/dt=(Iext-I_tot)/C。
#   无我方手设门控参数。确定性。】
import numpy as np

EK, ECA, ENA = -80.0, 60.0, 30.0

# ---------------- 细胞类型组成（.ode 冻结值） ----------------
def cell_params(ctype):
    P = dict(
        ELEAK=-80.0,
        gshal=2.9 if ctype == "AWC" else 2.48,
        gshak=0.1 if ctype == "AWC" else 1.1,
        gkir=0.65 if ctype == "AWC" else 0.2,
        gunc2=1.0 if ctype == "AWC" else 0.9,
        gegl19=1.55 if ctype == "AWC" else 0.99,
        gcca1=0.7 if ctype == "AWC" else 3.1,
        gbk=0.11 if ctype == "AWC" else 0.3,      # SLO1-UNC2
        gslo1=0.11 if ctype == "AWC" else 0.3,    # SLO1-EGL19
        gbk2=0.1 if ctype == "AWC" else 0.3,      # SLO2-EGL19
        gslo2=0.1 if ctype == "AWC" else 0.3,     # SLO2-UNC2
        gsk=0.06, gnca=0.055 if ctype == "AWC" else 0.05,
        gleak=0.27 if ctype == "AWC" else 0.4,
        C=3.1 if ctype == "AWC" else 1.2,
        vol=31.16 if ctype == "AWC" else 5.65,
    )
    if ctype == "AWC":
        P.update(gkvs1=0.8, gkqt3=0.55, gegl2=0.85, gegl36=0.0)
    else:
        P.update(gkvs1=0.0, gkqt3=0.0, gegl2=0.0, gegl36=1.3)
    return P

# 状态向量布局
GATES_AWC = ["m_shal","hf_shal","hs_shal","m_kvs1","h_kvs1","m_shak","h_shak",
             "mf_kqt3","ms_kqt3","w_kqt3","s_kqt3","m_egl2","m_kir",
             "m_cca1","h_cca1","m_unc2","h_unc2","m_egl19","hs_egl19",
             "mbk","mslo1","mbk2","mslo2","ca","m_sk"]
GATES_RMD = ["m_shal","hf_shal","hs_shal","m1_egl36","m2_egl36","m3_egl36",
             "m_shak","h_shak","m_kir","m_cca1","h_cca1","m_unc2","h_unc2",
             "m_egl19","hs_egl19","mbk","mslo1","mbk2","mslo2","ca","m_sk"]

def gates(ctype):
    return GATES_AWC if ctype == "AWC" else GATES_RMD

_IDX = {ct: {n: i + 1 for i, n in enumerate(gates(ct))} for ct in ("AWC", "RMD")}

def init_state(ctype):
    """.ode init 行: v=-70, 激活=0, 失活=1, ca=5e-5, m_sk=0.13563"""
    g = gates(ctype)
    y = np.zeros(1 + len(g))
    y[0] = -70.0
    for i, name in enumerate(g):
        n = i + 1
        if name.startswith(("h",)): y[n] = 1.0
        elif name in ("w_kqt3",): y[n] = 0.0
        elif name == "s_kqt3": y[n] = 0.0
        elif name == "ca": y[n] = 5e-5
        elif name == "m_sk": y[n] = 0.13563
    return y

def _x(x1, x2):  # 安全指数
    return np.exp(np.clip(x1, -500, 500)) if x2 is None else None

# ---------------- 右端函数 ----------------
def rhs(t, y, P, ctype, Iext):
    v = y[0]
    S = dict(zip(gates(ctype), y[1:]))
    d = np.zeros_like(y)
    exp = np.exp

    # ---- SHL-1 (Kv4) ----
    minf = 1/(1+exp(-(v-11.2+18)/14.1))
    tm = (13.8/(exp(-(v+17.5165)/12.9213)+exp((v+3.7082)/6.4876))+1.8849)*0.1
    hinf = 1/(1+exp((v+33.1+18)/8.3))
    ths = (8422.0/(1+exp((v+37.7391)/6.3785))+118.8983)*0.1
    thf = (539.1584/(1+exp((v+28.1990)/4.9199))+27.2811)*0.1
    m, hf, hs = S["m_shal"], S["hf_shal"], S["hs_shal"]
    d[_IDX[ctype]["m_shal"]] = (minf-m)/tm
    d[_IDX[ctype]["hf_shal"]] = (hinf-hf)/thf
    d[_IDX[ctype]["hs_shal"]] = (hinf-hs)/ths
    I_shal = P["gshal"]*m**3*(0.7*hf+0.3*hs)*(v-EK)

    # ---- SHK-1 (Kv1) ----
    minf = 1/(1+exp(-(v-20.4)/7.7))
    tm = 26.571450568169027/(exp(-(v+33.741611800716130)/15.364937728953288)
                             +exp((v+33.741611800716130)/15.757936311607475))+1.990037272604829
    hinf = 1/(1+exp((v+6.95)/5.8))
    m, h = S["m_shak"], S["h_shak"]
    d[_IDX[ctype]["m_shak"]] = (minf-m)/tm
    d[_IDX[ctype]["h_shak"]] = (hinf-h)/1400.0
    I_shak = P["gshak"]*m*h*(v-EK)

    # ---- IRK ----
    minf = 1/(1+exp((v+52.0+30)/13.0))
    tm = 17.0752/(exp(-(v+17.8258)/20.3154)+exp((v+43.4414)/11.1691))+3.8329
    m = S["m_kir"]
    d[_IDX[ctype]["m_kir"]] = (minf-m)/tm
    I_kir = P["gkir"]*m*(v-EK)

    I_kvs1 = I_kqt3 = I_egl2 = I_egl36 = 0.0
    if ctype == "AWC":
        # ---- KVS-1 ----
        minf = 1/(1+exp(-(v-57.1+30)/25.0))
        tm = (30.0/(1+exp(-(v-18.1232)/-20.0))+1.0)/10.0
        hinf = 1/(1+exp((v-47.3+30)/11.1))
        th = (88.4715/(1+exp(-(v-50.0)/-15.0))+53.4060)*0.1
        m, h = S["m_kvs1"], S["h_kvs1"]
        d[_IDX[ctype]["m_kvs1"]] = (minf-m)/tm
        d[_IDX[ctype]["h_kvs1"]] = (hinf-h)/th
        I_kvs1 = P["gkvs1"]*m*h*(v-EK)
        # ---- KQT-3 (Kv7) ----
        minf = 1/(1+exp(-(v+12.6726+10)/15.8008))
        tmf = (395.3/(1+((v+38.1)/33.59)**2))*0.1
        tms = (5503.0-5345.4/(1+10**(-0.02827*(-23.9-v)))
               -4590.6/(1+10**(-0.0357*(v+14.15))))*0.1
        winf = 0.49+0.51/(1+exp((v+1.084)/28.78))
        tw = (5.44+29.2/(1+((v+48.09)/48.83)**2))*0.1
        sinf = 0.34+0.66/(1+exp((v+45.3)/12.3))
        mf, ms, w, s = S["mf_kqt3"], S["ms_kqt3"], S["w_kqt3"], S["s_kqt3"]
        d[_IDX[ctype]["mf_kqt3"]] = (minf-mf)/tmf
        d[_IDX[ctype]["ms_kqt3"]] = (minf-ms)/tms
        d[_IDX[ctype]["w_kqt3"]] = (winf-w)/tw
        d[_IDX[ctype]["s_kqt3"]] = (sinf-s)/500.0
        I_kqt3 = P["gkqt3"]*(0.3*mf+0.7*ms)*w*s*(v-EK)
        # ---- EGL-2 (Kv11/eag) ----
        minf = 1/(1+exp(-(v+6.8594)/14.9131))
        tm = ((16.78/(1+exp((v+122.5682)/13.7976)))+8.0969)*0.5
        m = S["m_egl2"]
        d[_IDX[ctype]["m_egl2"]] = (minf-m)/tm
        I_egl2 = P["gegl2"]*m*(v-EK)
    else:
        # ---- EGL-36 (Kv3) ----
        minf = 1/(1+exp(-(v-63.0)/28.5))
        for nm, tau, w_ in (("m1_egl36",355.0,0.31),("m2_egl36",63.0,0.36),("m3_egl36",13.0,0.39)):
            d[1+gates(ctype).index(nm)] = (minf-S[nm])/tau
        I_egl36 = P["gegl36"]*(0.31*S["m1_egl36"]+0.36*S["m2_egl36"]+0.39*S["m3_egl36"])*(v-EK)

    # ---- CCA-1 (CaV3/T) ----
    minf = 1/(1+exp(-(v+42.65+15)/(1.7*1.4)))
    tm = ((40.0/(1+exp(-(v+62.5393+30)/(-12.4758*1.7))))+0.6947)*0.5
    hinf = 1/(1+exp((v+58.0+15)/(7.0*1.15)))
    th = ((280.0/(1+exp((v+60.7312+15)/(8.5224*1.1))))+19.7456)*0.08
    m, h = S["m_cca1"], S["h_cca1"]
    d[_IDX[ctype]["m_cca1"]] = (minf-m)/tm
    d[_IDX[ctype]["h_cca1"]] = (hinf-h)/th
    I_cca1 = P["gcca1"]*m**2*h*(v-ECA)

    # ---- UNC-2 (CaV2 P/Q) ----
    # unc2_vshift: gf 突变模拟旋钮, >0 = 激活曲线左移(更易开); 缺省 0, 加 0.0 逐比特不变
    sh_u = P.get("unc2_vshift", 0.0)
    minf_u = 1/(1+exp(-(v+12.17+25+sh_u)/3.97))
    tm_u = (1.4969/(exp(-(v+8.1761+30)/9.0753)+exp((v+8.1761+30)/15.3456))+0.1029)*3.0
    hinf_u = 1/(1+exp((v+52.47+25)/5.6))
    # unc2_hinf_ped (2026-10-23 探针旋钮): UNC-2 非失活分量 A∈[0,1)。
    # 依据: Gao 2018 unc-2(gf)=减弱失活动力学→A-MN 振荡增强; CaV2 家族普遍存在
    # 非失活窗流成分。缺省 0 = 与历史版本逐比特不变。
    ped_u = P.get("unc2_hinf_ped", 0.0)
    if ped_u:
        hinf_u = ped_u + (1.0-ped_u)*hinf_u
    th_u = (83.8037/(1+exp((v-52.8997+30)/3.4557))
            +72.0995/(1+exp(-(v-23.9009+30)/3.5903)))*1.7
    m, h = S["m_unc2"], S["h_unc2"]
    d[_IDX[ctype]["m_unc2"]] = (minf_u-m)/tm_u
    d[_IDX[ctype]["h_unc2"]] = (hinf_u-h)/th_u
    I_unc2 = P["gunc2"]*m*h*(v-ECA)
    alpha_u = minf_u/tm_u; beta_u = 1.0/tm_u-alpha_u

    # ---- EGL-19 (CaV1 L) ----
    minf_e = 1/(1+exp(-(v-5.6+10)/7.5))
    tm_e = (2.3359+2.9324*exp(-((v-5.2357+10)**2)/36.0)
            +1.8739*exp(-((v-1.3930+10)**2)/900.0))
    hinf_e = ((1.4314/(1+exp(-(v-24.8573+10)/11.9541))+0.1427)
              *(5.9589/(1+exp((v+10.5428+10)/8.0552))+0.6038))
    ths_e = 0.4*((0.55*81.1179)/(1+exp((v+22.9723+10)/5.0))+43.0937
                 +(0.9*40.4885)/(1+exp((v-28.7251+10)/3.7125))+0.0)
    m, h = S["m_egl19"], S["hs_egl19"]
    d[_IDX[ctype]["m_egl19"]] = (minf_e-m)/tm_e
    d[_IDX[ctype]["hs_egl19"]] = (hinf_e-h)/ths_e
    I_egl19 = P["gegl19"]*m*h*(v-ECA)
    alpha_e = minf_e/tm_e; beta_e = 1.0/tm_e-alpha_e

    # ---- Ca 纳米域 + SLO1/SLO2 复合体 ----
    r_, d_, F_, kb_, b_, gsc_ = 13e-9, 250e-12, 96485.0, 500e6, 30e-6, 40e-12
    cao = ((abs(gsc_*(v-ECA)*1e-3)/(8*np.pi*r_*d_*F_))
           *np.exp(-r_/np.sqrt(d_/(kb_*b_)))*1e6*1e-3)+0.05
    cac = 0.05
    # SLO1 参数
    wom, wyx, kyx, nyx = 3.152961, 0.012643, 34.338784, 0.000100
    wop, wxy, kxy, nxy = 0.156217, -0.027527, 55.726816, 1.299198
    kcm = wom*exp(-wyx*v)/(1+(cac/kyx)**nyx)
    kom = wom*exp(-wyx*v)/(1+(cao/kyx)**nyx)
    kop = wop*exp(-wxy*v)/(1+(kxy/cao)**nxy)
    den1 = (kop+kom)*(kcm+alpha_u)+beta_u*kcm
    minf_bk = (S["m_unc2"]*kop*(alpha_u+beta_u+kcm))/den1
    tm_bk = (alpha_u+beta_u+kcm)/den1
    d[_IDX[ctype]["mbk"]] = (minf_bk-S["mbk"])/tm_bk
    I_bk = P["gbk"]*S["mbk"]*S["h_unc2"]*(v-EK)
    den2 = (kop+kom)*(kcm+alpha_e)+beta_e*kcm
    minf_s1 = (S["m_egl19"]*kop*(alpha_e+beta_e+kcm))/den2
    tm_s1 = (alpha_e+beta_e+kcm)/den2
    d[_IDX[ctype]["mslo1"]] = (minf_s1-S["mslo1"])/tm_s1
    I_slo1 = P["gslo1"]*S["mslo1"]*S["hs_egl19"]*(v-EK)
    # SLO2 参数
    wom1, wyx1, kyx1, nyx1 = 0.896395, 0.019405, 3294.553404, 0.000010
    wop1, wxy1, kxy1, nxy1 = 0.026719, -0.024123, 93.449423, 1.835067
    kcm1 = wom1*exp(-wyx1*v)/(1+(cac/kyx1)**nyx1)
    kom1 = wom1*exp(-wyx1*v)/(1+(cao/kyx1)**nyx1)
    kop1 = wop1*exp(-wxy1*v)/(1+(kxy1/cao)**nxy1)
    den3 = (kop1+kom1)*(kcm1+alpha_e)+beta_e*kcm1
    minf_b2 = (S["m_egl19"]*kop1*(alpha_e+beta_e+kcm1))/den3
    tm_b2 = (alpha_e+beta_e+kcm1)/den3
    d[_IDX[ctype]["mbk2"]] = (minf_b2-S["mbk2"])/tm_b2
    I_bk2 = P["gbk2"]*S["mbk2"]*S["hs_egl19"]*(v-EK)
    den4 = (kop1+kom1)*(kcm1+alpha_u)+beta_u*kcm1
    minf_s2 = (S["m_unc2"]*kop1*(alpha_u+beta_u+kcm1))/den4
    tm_s2 = (alpha_u+beta_u+kcm1)/den4
    d[_IDX[ctype]["mslo2"]] = (minf_s2-S["mslo2"])/tm_s2
    I_slo2 = P["gslo2"]*S["mslo2"]*S["h_unc2"]*(v-EK)

    # ---- 胞内钙（微域）+ KCNL ----
    I_ca = I_unc2+I_egl19+I_cca1
    alpha_ca = 1/(2*P["vol"]*96.485)
    ca = S["ca"]
    # ca_tau_scale (2026-10-23 探针旋钮): 胞内钙清除时间常数缩放。50ms 是微域清除;
    # 体钙清除(泵/缓冲)在秒级——Gao 2018 DA9 钙振荡周期 ~50s 需要慢体钙变量。
    # 缺省 1.0 = 与历史版本逐比特不变。
    ctau = 50.0*P.get("ca_tau_scale", 1.0)
    rs = (-0.001*alpha_ca*I_ca-(ca-5e-5)/ctau) if I_ca < 0 else ((5e-5-ca)/ctau)
    d[_IDX[ctype]["ca"]] = rs
    minf_sk = ca/(0.00033+ca)
    d[_IDX[ctype]["m_sk"]] = (minf_sk-S["m_sk"])/6.3
    I_sk = P["gsk"]*S["m_sk"]*(v-EK)

    I_leak = P["gleak"]*(v-P["ELEAK"])
    I_nca = P["gnca"]*(v-ENA)

    I_tot = (I_shal+I_shak+I_kir+I_kvs1+I_kqt3+I_egl2+I_egl36
             +I_cca1+I_unc2+I_egl19+I_bk+I_slo1+I_bk2+I_slo2+I_sk+I_leak+I_nca)
    d[0] = (Iext-I_tot)/P["C"]
    return d, dict(I_tot=I_tot, I_ca=I_ca, I_shal=I_shal, I_shak=I_shak,
                   I_kir=I_kir, I_kvs1=I_kvs1, I_kqt3=I_kqt3, I_egl2=I_egl2,
                   I_egl36=I_egl36, I_cca1=I_cca1, I_unc2=I_unc2,
                   I_egl19=I_egl19, I_bk=I_bk, I_slo1=I_slo1, I_bk2=I_bk2,
                   I_slo2=I_slo2, I_sk=I_sk, I_leak=I_leak, I_nca=I_nca)

def rhs_ivp(ctype, Iext_fun, g_override=None):
    P = cell_params(ctype)
    if g_override:
        P.update(g_override)
    def f(t, y):
        return rhs(t, y, P, ctype, Iext_fun(t))[0]
    return f
