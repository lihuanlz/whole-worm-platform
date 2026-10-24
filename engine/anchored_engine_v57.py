# anchored_engine_v57.py
# [v5.7 = v5.6 + UNC-2 intrinsic-oscillation arm (2026-10-10):
#   gunc2 of class-A/B excitatory motor neurons (VA/VB/DA/DB) is overridden by the module-global GUNC2_AB.
#   Basis: Gao et al. 2018 eLife e29915 — UNC-2 (P/Q/N-type HVA calcium channel) is
#   required for the intrinsic calcium oscillation of class-A motor neurons (unc-2(lf) lowers amplitude and frequency, unc-2(gf) raises both,
#   oscillation persists on an unc-13 background = cell-autonomous); Xu et al. 2018 PNAS — the AVB→B gap junction pushes
#   class B across the oscillation threshold; without it, bending amplitude decays monotonically toward the tail (our "wave does not propagate" symptom).
#   gunc2=0 for VA/DA in the Nicoletti voltage-clamp fits is an underdetermined direction of the clamped state; added back here.
#   Negative-control design: VD/AVA/AWC/RMD not overridden; GUNC2_AB=0 is value-identical to v56.
#   The v56 file is left byte-identical.]
import anchored_engine_v56 as _v56

GUNC2_AB = 0.0   # nS; set by the caller before network construction
GNCA_A = -1.0    # nS; >=0 overrides gnca of class A (VA/DA) (v84: Gao 2018 + Frontiers 2022
                 # NCA Na+ leak is the necessary depolarising engine of A-MN rPSC/intrinsic oscillation; fitted VA/DA gnca=0.01
                 # the lowest in the body -> isolated equilibrium -79 mV, far below the UNC-2 HVA activation domain -> no self-oscillation.
                 # The Nicoletti clamp data underdetermine the leak component; restored under the same discipline as GUNC2_AB)

_AB_CLASSES = {"VA", "VB", "DA", "DB"}
_A_CLASSES = {"VA", "DA"}
_GABA_CLASSES = {"DD", "VD"}

GLEAK_GABA = 1.0   # multiplier; !=1.0 scales gleak of DD/VD (v86: eLife 2021 PMC7986531
                   # anchors VD/DD calcium oscillating with the crawl rhythm — in the measured model DD is pinned by tonic saturated release
                   # (tonic saturation zone of the in-situ vrel table) at the -18 mV high-conductance region, voltage-clamped so it cannot oscillate,
                   # the cross-inhibition phase component dies -> ventral overdrive -> crawling bends ventrally and drifts left.
                   # gleak=0.4/0.2 from the Nicoletti isolated-clamp fits does not include in-situ tonic drive,
                   # an underdetermined direction of the clamped state, restored under the same discipline as GUNC2_AB; 1.0 = bit-identical)

GEGL36_GABA = -1.0  # nS; >=0 overrides gegl36 of DD/VD (ctype=VD) (v87: same anchor).
                    # gegl36=0 in the VD recipe is an underdetermined direction of the Nicoletti 2024 clamp fit —
                    # single-cell probe measurements (probe_dd_vd_load/egl36_2026-10-21): without EGL-36,
                    # the EGL-19 L-type plateau (-25..-5 mV negative-slope regenerative region) has no high-voltage K repolarisation ->
                    # under tonic drive it parks at a ~-4 mV plateau and wave-frequency modulation dies (std<1 mV).
                    # EGL-36 high-voltage activation: closed at rest so the VD5 -45.8+-1.9 anchor is unharmed,
                    # activates above -20 mV to repolarise the plateau. Both candidates 0.6/1.0 pass the anchors (iso -46.5/-47.5),
                    # 0 = bit-identical negative control)

UNC2PED_A = -1.0   # [0,1); >=0 sets the non-inactivating fraction of the UNC-2 inactivation gate of class A (VA/DA) (2026-10-23, W13)
                   # Basis: Gao 2018 eLife e29915 — the A-MN endogenous oscillation requires the UNC-2 window current,
                   # unc-2(gf) = weakened inactivation kinetics → stronger oscillation; CaV2-family channels commonly carry a non-inactivating component.
                   # The Nicoletti clamp fit hinf half-inactivation -77.47 mV gives h∞≈0.001 at -38 mV,
                   # window current = 0 → the cell structurally cannot oscillate (probe p1-p5 verdict: monostable, no hysteresis).
                   # ped=0.20 restores the bistable window I∈[0,1.8] pA (probe_amn_refit_s2/s4c verified:
                   # rest anchor -72.1 holds / up-state ignitable / AVA shunt-gating three phases all pass)
GSK_A = -1.0       # nS; >=0 sets the class-A KCNL/SK calcium-activated potassium conductance (2026-10-23, W13)
                   # KCNL is anatomically expressed but its fitted contribution=0 = underdetermined direction; 0.05 restored as a slow brake
CATAUSCALE_A = -1.0  # >=0 sets the class-A intracellular calcium clearance time-constant multiplier (2026-10-23, W13)
                     # 50 ms = microdomain clearance; somatic clearance is seconds-level (Gao 2018 DA9 Ca oscillation period ~50 s); 40 taken (=2 s)
GUNC2_A = -1.0       # nS; >=0 overrides gunc2 of class A (VA/DA) (2026-10-23, W13)
                     # E2d measurement: in-network A-MNs are clamped by 37 gap + 40 chem edges; the GUNC2_AB=0.4
                     # window current (~10 pA) is negligible against network conductances (<1%) => must be raised to network scale (1.0+).
                     # Nicoletti fitted gunc2=0 = underdetermined direction; its magnitude is a free choice anyway;
                     # negative control: VB/DB not overridden, still run GUNC2_AB=0.4; -1 = bit-identical

GUNC2_AVA = -1.0     # nS; >=0 overrides gunc2 of AVA (2026-10-23, W52 AVA regenerative plateau)
                     # Basis: Meng 2024 Sci Adv adk0002 (PMC11006214) — AVA multistable depolarisation
                     # (-30~-10, two stable regimes -30/-20 and -15/-10), baseline near the UNC-2 opening threshold
                     # (tonic ACh release); Nicoletti fitted AVA gunc2=0 = underdetermined clamped-state direction,
                     # restored under the same discipline as GUNC2_AB/GUNC2_A; minf half-activation -37 mV puts the AVA baseline (-25)
                     # already inside the activation region; the ped non-inactivating fraction is the regenerative window current; -1 = bit-identical
UNC2PED_AVA = -1.0   # [0,1); >=0 sets the AVA UNC-2 non-inactivating fraction (W52; Gao 2018 window-current anchor as in W13)
GSK_AVA = -1.0       # nS; >=0 sets the AVA KCNL/SK calcium-activated potassium conductance (W52: seconds-scale self-terminating brake;
                     # AVA SK fitted contribution=0 = underdetermined direction; somatic Ca 2 s clearance -> plateau self-terminates within seconds)
CATAUSCALE_AVA = -1.0  # >=0 sets the AVA intracellular calcium clearance time-constant multiplier (W52: microdomain 50 ms -> somatic 2 s)

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
