# -*- coding: utf-8 -*-
# jax_circuit_riacomp.py — RIA three-compartment extension module, v2 regenerative-dendrite edition (2026-10-19)
# Mechanism specification: RIA three-compartment prototype specification (2026-10-19, project archive).
# v2 revision (driven by this session's measurements):
#   v1 passive compartments + calcium store were quantitatively ruled out (rectification effect 0.03 vs executive benchmark 0.92, and the real carrier is 20x smaller still).
#   A v2 compartment is a regenerative dendrite: the frozen RMD-type HH equations scaled isometrically by the area ratio c_ratio
#   (Liu 2018: large Ca signals in RIA neurites; UNC-2/EGL-19/CCA-1 are in its expression profile).
#   Zero changes to channel kinetics, zero new channel parameters; only the geometry/shunt parameters g_ax, s_acc2 remain (all literature-anchored).
# Structure:
#   RIAL/RIAR each gain nrD/nrV compartments (extended state, not new network nodes);
#   RIA->SMDD*/SMDV* output is now driven by nrD/nrV voltages; SMDD*/SMDV*->RIA feedback is rerouted to nrD/nrV;
#   AIY->RIA inhibition stays at the soma and gates the compartments through an ACC-2-equivalent shunt (s_acc2);
#   crossed routing is default (route=-1): nrD->SMDV, nrV->SMDD (topology adjudication in spec §3).
# Discipline: when this module is not called, the frozen engine is bit-identical.
import numpy as np
import jax.numpy as jnp

from jax_circuit import V_SCALE, S_MAX, ECA, E_EX, TAU_EX, TAU_INH, PK, _rhs

# Fixed compartment anchor (spec §4): compartment volume / whole cell ≈ 0.1 (anatomical estimate for the thin neurite)
C_RATIO = 0.1

SMDD_SET = ("SMDDL", "SMDDR")
SMDV_SET = ("SMDVL", "SMDVR")
RIA_PAIR = ("RIAL", "RIAR")
AIY_PAIR = ("AIYL", "AIYR")


def build_comp_spec(net_npz):
    """Identify the rerouted edges from the frozen connectome. Returns a dict of edge indices (chem row numbers)."""
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
            print(f"  [spec] {ria} {tag}: {len(e)} edges w={np.round(w, 3).tolist()}")
    return spec


def _dense(N, chem, rows):
    M = np.zeros((N, N))
    if len(rows):
        M[chem[rows, 1].astype(int), chem[rows, 0].astype(int)] += chem[rows, 2]
    return M


def build_riacomp_mats(P, spec, net_npz):
    """Build the rerouting matrix pack. Mex_m has the rerouted edges removed; the caller substitutes it for the original Mex."""
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
    # Compartment HH parameters: RMD row template, all conductances/C/vol scaled by c_ratio (kinetics bit-identical)
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


# Compartment parameters (all literature-anchored, spec §4):
#   g_ax (nS) axial coupling conductance: thin-neurite cable estimate 0.005-0.5 nS, default 0.05
#   s_acc2 (dimensionless) ACC-2 sensory shunt gain (Liu 2018 sensory suppression of motor encoding)
#   route: +1 ipsilateral routing / -1 crossed routing (default, topology adjudication)
CP_DEFAULT = dict(g_ax=0.05, s_acc2=5.0, route=-1.0)

# ---------------------------------------------------------------------------
# v3 calcium-store local-release gating edition (2026-10-19)
# Basis for the mechanism turn (Hendricks 2012 Nature 487:99, PMC3393794, full text checked this session):
#   - worm neurons are isopotential; compartmentalised calcium signals are not a voltage mechanism;
#   - SMD(ACh) -> RIA compartment GAR-3 (mAChR, Gαq) -> EGL-8 (PLCβ) -> ER calcium-store mobilisation,
#     local Ca directly modulates local dyadic synaptic release while soma voltage barely moves;
#   - compartment (motor feedback) and synchronising (sensory) calcium signals superpose additively;
#   - gar-3 knockout -> deeper head swings (a negative-feedback limiting function) -> compartment output = negative feedback.
# v3 vs v2: compartments no longer carry HH voltage; the activity variable is a slow calcium store cstore(4,),
#   release = soma-voltage-driven baseline + k_mod*cstore (additive; the two signal classes in Hendricks Fig1d have comparable amplitude
#   -> k_mod anchored at O(1): k_mod = rel_typ/c_typ, measured during rig warm-up; sensitivity ×{0.3,1,3} disclosed);
#   the ACC-2 sensory shunt becomes a divisive gate on the GAR-3 drive, fbc/(1+s_acc2*gIaiy) (Liu 2018
#   chloride-shunt equivalent of sensory gating of motor encoding).
#   tau_store: slow timescale of the GPCR->ER calcium store, literature magnitude anchor (GCaMP3 transients are seconds-level, visible at 5 Hz imaging,
#   default 1.0 s, sweep 0.3-3 s disclosed).
# v3b revision (§65): the compartment-edge output law becomes local-store-dominated (soma_add=0):
#   Hendricks 2012, original text: "We did not detect any prominent calcium response in the RIA
#   cell body" — soma calcium is silent while compartment calcium swings full-range -> compartment dyadic release is dominated by the local store;
#   v3a's additive soma+store synthesis is permanently pegged at the S_MAX=1.5 hard saturation (baseline 2.39>1.5),
#   so modulation never reaches SMD (an implementation defect, not a biological conclusion). soma_add=1 retains v3a for control.
#   k_mod anchoring changed to full-scale calibration: k_mod=S_MAX/c_typ (the compartment transient spans the full release range; Hendricks
#   Fig1d 0-100% normalised swing); s_acc2 anchored to the Liu 2018 Fig4I/K+Fig5F sensory suppression depth
#   ~55-65%: s_acc2=1.5/gIaiy_peak (rig-measured 12.7).
# Discipline: when this module is not called, the frozen engine is bit-identical.
CP3_DEFAULT = dict(tau_store=1.0, k_mod=1.0, s_acc2=5.0, route=-1.0, soma_add=0.0)


def build_step_riacomp(P, spec, mats, cp=None, ctrl_hz=30):
    """RIA three-compartment (regenerative-dendrite) single-dt step.
    state=(Yawc,Yrmd,gA,gI,Yc(4,29),gAc(4),gIaiy(2))。
    Yc row order: [RIAL_nD, RIAL_nV, RIAR_nD, RIAR_nV]; Yc[:,0]=Vc."""
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
        # route=+1 ipsilateral (nrD drives SMDD), route=-1 crossed (nrD drives SMDV)
        r_out_d = jnp.where(cp["route"] > 0, OUT_D @ relD, OUT_D @ relV)
        r_out_v = jnp.where(cp["route"] > 0, OUT_V @ relV, OUT_V @ relD)
        # main network (term-by-term identical to build_step)
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
        # ---- compartments (active HH, RMD-type scaling) ----
        fb = jnp.concatenate([(FB_D @ sV)[ria_rows], (FB_V @ sV)[ria_rows]])
        fbc = fb[jnp.asarray([0, 2, 1, 3])]      # matches Yc row order [L_D,L_V,R_D,R_V]
        gAc2 = gAc_ + dt * (-gAc_ / TAU_EX + fbc)
        aiy_in = (AIY_M @ sV)[ria_rows]
        gIaiy2 = gIaiy_ + dt * (-gIaiy_ / TAU_INH + aiy_in)
        gsh = s_acc2 * jnp.repeat(gIaiy2, 2)      # ACC-2-equivalent shunt (E=ELEAK=-80)
        Vsoma_c = V[soma_of]
        I_syn_c = gAc2 * (Vc_ - E_EX)
        I_ax = g_ax * (Vc_ - Vsoma_c)
        I_shunt = gsh * (Vc_ - (-80.0))
        Iext_c = -I_syn_c - I_ax - I_shunt        # inward current positive (engine convention)
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
    """RIA three-compartment v3: calcium-store local-release gating (Hendricks 2012 mechanism).
    state=(Yawc,Yrmd,gA,gI,cstore(4,),gAc(4),gIaiy(2))。
    cstore row order: [RIAL_nD, RIAL_nV, RIAR_nD, RIAR_nV] (dimensionless release equivalents).
    Output law: rel_soma (soma-voltage baseline) + k_mod*cstore, additive per RIA, routing as in v2."""
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
        # soma-voltage baseline release (isopotential -> both compartments share the soma baseline, Hendricks additivity)
        rel_soma = jnp.clip((V[ria_rows] - vrel_ria) / V_SCALE, 0.0, S_MAX)
        # v3b: with soma_add=0 compartment edges are dominated by the local store (Hendricks soma-Ca silence), eliminating the permanent S_MAX saturation
        rel_base = rel_soma * cp["soma_add"]
        cD = c_[jnp.asarray([0, 2])]   # nrD of each RIA
        cV = c_[jnp.asarray([1, 3])]   # nrV of each RIA
        relD = jnp.zeros(N, jnp.float64).at[rd_rows].set(jnp.clip(rel_base + k_mod * cD, 0.0, S_MAX))
        relV = jnp.zeros(N, jnp.float64).at[rd_rows].set(jnp.clip(rel_base + k_mod * cV, 0.0, S_MAX))
        # route=+1 ipsilateral (nrD drives SMDD), route=-1 crossed (nrD drives SMDV)
        r_out_d = jnp.where(cp["route"] > 0, OUT_D @ relD, OUT_D @ relV)
        r_out_v = jnp.where(cp["route"] > 0, OUT_V @ relV, OUT_V @ relD)
        # main network (term-by-term identical to build_step)
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
        # ---- compartment calcium-store slow variable ----
        fb = jnp.concatenate([(FB_D @ sV)[ria_rows], (FB_V @ sV)[ria_rows]])
        fbc = fb[jnp.asarray([0, 2, 1, 3])]      # row order [L_D,L_V,R_D,R_V]
        gAc2 = gAc_ + dt * (-gAc_ / TAU_EX + fbc)
        aiy_in = (AIY_M @ sV)[ria_rows]
        gIaiy2 = gIaiy_ + dt * (-gIaiy_ / TAU_INH + aiy_in)
        # ACC-2 divisive shunt gating of the GAR-3 drive (Liu 2018 chloride-shunt equivalent)
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
