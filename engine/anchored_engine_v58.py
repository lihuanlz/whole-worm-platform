# anchored_engine_v58.py
# [v5.8 = v5.7 + G3 re-anchoring surgery (2026-10-10):
#   1) AVA re-anchored: isolated rest -38.58 -> -25.000 mV (ELEAK -52 -> -38.425,
#      bisection LSODA solve, solve_v58_ava_eleak_2026-10-10.py).
#      Basis: AVA measured -20~-30 mV, median -23 (Mellem 2008 J Neurosci PMC2697921;
#      Meng 2024 Sci Adv adk0002: AVA tonically active, multistable -30~-10).
#      The old -39 anchor was an E_rev fit approximation; the conflict was logged as G3 and is adjudicated here.
#   2) RELEASE_REF for all six classes changed from "rest+14.5" to "isolated equilibrium + VREL_OFF" (default -10,
#      s=0.4 at rest, tonic graded release). Basis: Meng 2024 (AVA baseline near the UNC-2 threshold),
#      worm NMJ graded release is modulated around rest. rest+14.5 has no measurement support; logged as a Tier-3 convention change.
#   MN rest anchors unchanged: VA5 -70 / VB6 -50 / VD5 -45, consistent with Liu 2014 measurements.
#   Negative control: AWC/RMD are not in RELEASE_REF -> v62 uses global V_REL=-55, old behaviour unchanged;
#   with VREL_OFF=+14.5 RELEASE_REF matches v56 within 0.5 mV (the old system used anchor values, this version uses isolated-equilibrium values).
#   v56/v57 files left byte-identical.]
import anchored_engine_v57 as _v57
import anchored_engine_v56 as _v56

VREL_OFF = -10.0   # mV; set by the caller before network construction

# Isolated-equilibrium values (measured on v56, mV): VA/DA -71.87, VB/DB -52.76, VD -44.52, AVA(v58) -25.00
_EQ_REST = {"VA": -71.87, "DA": -71.87, "VB": -52.76, "DB": -52.76, "VD": -44.52, "AVA": -25.00}

def cell_params(ctype):
    P = dict(_v57.cell_params(ctype))
    if ctype == "AVA":
        P["ELEAK"] = -38.425   # G3 re-anchor: isolated rest -25.000
    return P

ANCHOR_REST = dict(_v56.ANCHOR_REST)
ANCHOR_REST["AVA"] = -25.0

class _ReleaseRef(dict):
    """RELEASE_REF computed dynamically from VREL_OFF (at read time)."""
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
