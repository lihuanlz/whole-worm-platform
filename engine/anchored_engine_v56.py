# anchored_engine_v56.py
# [v5.6 = v5.5 + motor-neuron class recipes (anchored arm, 2026-10-10):
#   adds VA/VB/VD/DA/DB/AVA six-class cell recipes; conductances and E_leak taken value-by-value from
#   Nicoletti et al. 2024 PLoS Comput Biol (PMC10980225) Table 3
#   —— a table fitted to the whole-cell measurements of Liu et al. 2014 Nat Commun 5:5155 (PMC4197135)
#   on VA5/VB6/VD5. Rest anchors (measured):
#     VA5 -71.7±2.4 | VB6 -53.2±2.5 | VD5 -45.8±1.9 mV
#   DA = VA recipe extrapolated (class A), DB = VB recipe extrapolated (class B) — Tier-2 extrapolation, logged.
#   AVA uses the AVAL recipe (E_rev -39, C 9.66).
#   Registered approximations (Tier-2/3):
#   - slo-2 isolated+coupled components merged into gbk2 (SLO2-EGL19 complex dynamics);
#   - gating kinetics carried over from v55 (Nicoletti 2019 .ode calibration shifts), not re-calibrated;
#   - exp-2 (expressed in VB6) has no channel machinery in v55; default=0, logged as missing;
#   - kcnl is expressed in VA5/VB6 but its Table 3 fitted contribution is zero -> gsk=0;
#   - vol (calcium microdomain volume) carries the RMD value 5.65.
#   The v55 file is left byte-identical; the new class recipes take effect only when v56 is called explicitly.]
import numpy as np
import anchored_engine_v55 as v55

MN_RECIPES = {
    #            K channels                                                 Ca channels                     leak/other
    "VA": dict(gshal=0.0, gshak=0.1, gkir=1.0, gegl36=0.0,
               gcca1=0.0, gunc2=0.0, gegl19=0.15,
               gbk=0.0, gslo1=0.0, gbk2=6.0, gslo2=0.0, gsk=0.0,
               gnca=0.01, gleak=0.10, ELEAK=-56.5, C=5.84),
               # ELEAK -56.5: calibrated to the measured rest -71.7 (isolated rest -71.87).
               # Reason for deviating from the Nicoletti fit -70: this engine's IRK/SLO2 kinetics
               # are ported from the 2019 .ode and differ from the 2024 version; E_leak is a per-cell
               # fitted "residual reversal potential" anyway (their cells span -37~-89.57); logged Tier-2.
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
                # ELEAK -52: calibrated to anchor -39 (isolated rest -38.58); original fit -39.
}
MN_RECIPES["DA"] = dict(MN_RECIPES["VA"])   # class-A extrapolation (Tier-2)
MN_RECIPES["DB"] = dict(MN_RECIPES["VB"])   # class-B extrapolation (Tier-2)

# Measured anchors (mV), used for smoke-test adjudication
ANCHOR_REST = {"VA": -71.7, "VB": -53.2, "VD": -45.8, "DA": -71.7, "DB": -53.2,
               "AVA": -39.0}  # AVA has no direct resting mean; use the E_rev fitted value as approximate anchor

def cell_params(ctype):
    if ctype in MN_RECIPES:
        P = v55.cell_params("RMD")       # take the RMD structure first (gating layout is shared)
        P.update(MN_RECIPES[ctype])      # then overwrite conductances/E_leak/C
        return P
    return v55.cell_params(ctype)

def layout(ctype):
    return "RMD" if ctype in MN_RECIPES else ctype

def rhs56(t, y, ctype, Iext):
    return v55.rhs(t, y, cell_params(ctype), layout(ctype), Iext)

def init_state56(ctype):
    return v55.init_state(layout(ctype))

# v62 interface alias (v62 calls eng.init_state)
init_state = init_state56

# Per-class release reference potential (mV): the class anchored rest + 14.5 (same margin as the old RMD rest -69.5 -> V_REL -55).
# Applies only to the new classes; AWC/RMD are not in the table -> v62 uses global V_REL=-55, old behaviour unchanged.
RELEASE_REF = {
    "VA": -71.87 + 14.5, "DA": -71.87 + 14.5,
    "VB": -52.76 + 14.5, "DB": -52.76 + 14.5,
    "VD": -44.52 + 14.5,
    "AVA": -38.58 + 14.5,
}
