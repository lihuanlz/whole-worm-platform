# -*- coding: utf-8 -*-
"""Original 2D viscoelastic-rod body layer (2026-10-23) — replaces the borrowed worm-whisperer rod2d/muscle_map modules.

Equations (standard published forms, no copied code):
- Energy: stretch (k_s/2)(|Δx|−ℓ)² + bending (k_b/2)(Δφ − ℓκ0)²   (Boyle et al. 2012, J R Soc Interface 9:2765)
- Overdamped low-Re dynamics: v = Ft/Ct + Fn/Cn  (resistive force theory, Gray & Lissmann 1964, J Exp Biol 41:135)
- Cn/Ct ≈ 40 for agar crawling (Boyle 2012); k_s=5e4 near-inextensible stiff spring; k_b=5 bending stiffness
- Curvature map: κ0_j = kmax·mean(A_D−A_V over adjacent rows)  (dorsoventral difference → joint preferred curvature, linear geometry)

Difference from the borrowed modules: forces use an analytic gradient (hand-written closed-form ∂E/∂x), not jax.grad autodiff;
interface aligned bit-for-bit (initial/run/curvature + preferred_curvature); walls/maze pieces not implemented (unused by the main closed loop).
Units: mm, s; forces normalised to Ct=1.
"""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp
from jax import lax


class Rod2DOwn:
    def __init__(self, n_seg=24, L=1.0, Ct=1.0, Cn=40.0, k_s=5e4, k_b=5.0, dt=1e-5, dtype=jnp.float32):
        self.n, self.L, self.ell = n_seg, L, L / n_seg
        self.Ct, self.Cn, self.k_s, self.k_b, self.dt, self.dtype = Ct, Cn, k_s, k_b, dt, dtype
        self._run = jax.jit(self._make_run())

    def initial(self, x0=(0.0, 0.0), heading=0.0):
        s = np.arange(self.n + 1) * self.ell
        return jnp.asarray(np.stack([x0[0] + s * np.cos(heading), x0[1] + s * np.sin(heading)], 1), self.dtype)

    def _forces(self, x, kappa0):
        """Analytic gradient: F = −∂E/∂x, E = E_s + E_b. Fully vectorised."""
        d = x[1:] - x[:-1]                                    # (n,2) segment vectors
        ln = jnp.linalg.norm(d, axis=1) + 1e-12               # (n,)
        # --- stretch: ∂E_s/∂d_i = k_s(1−ℓ/|d_i|) d_i ; x_i receives d_{i-1}(+) and d_i(−) ---
        gs = self.k_s * (1.0 - self.ell / ln)[:, None] * d    # (n,2) = ∂E_s/∂d_i
        # x_j enters d_{j-1} with + and d_j with − → ∂E/∂x_j = g_{j-1} − g_j
        Fs = jnp.zeros_like(x)
        Fs = Fs.at[:-1].add(-gs).at[1:].add(gs)
        # --- bending: Δφ_j = φ_{j+1}−φ_j (wrap); ∂φ_i/∂d_i = perp(d_i)/|d_i|² ---
        phi = jnp.arctan2(d[:, 1], d[:, 0])                   # (n,)
        dphi = phi[1:] - phi[:-1]
        dphi = jnp.arctan2(jnp.sin(dphi), jnp.cos(dphi))      # (n−1,)
        r = (dphi - self.ell * kappa0)                        # (n−1,) residual
        # perp(d) = (−dy, dx); ∂φ_i/∂x propagates via d_i
        perp = jnp.stack([-d[:, 1], d[:, 0]], axis=1) / (ln ** 2)[:, None]   # (n,2) = ∂φ_i/∂d_i
        gb = self.k_b * r                                     # (n−1,)
        # Δφ_j depends on x_{j}, x_{j+1}, x_{j+2}: contributions accumulated joint by joint
        Fb = jnp.zeros_like(x)
        # ∂E_b/∂d_i = Σ_j gb_j ∂Δφ_j/∂d_i = gb_{i-1}·(∂φ_i/∂d_i) − gb_i·(∂φ_i/∂d_i)  (gb out of range = 0)
        gbm = jnp.concatenate([gb[:1] * 0.0, gb])             # gb_{i-1}, i=0..n−1 (gb_{-1}=0)
        gbp = jnp.concatenate([gb, gb[-1:] * 0.0])            # gb_i,     i=0..n−1 (gb_{n-1}=0)
        q = (gbm - gbp)[:, None] * perp                       # (n,2) = ∂E_b/∂d_i
        Fb = jnp.zeros_like(x)
        Fb = Fb.at[:-1].add(-q).at[1:].add(q)                 # same as stretch: ∂E_b/∂x_j = q_{j-1} − q_j
        return -(Fs + Fb)                                     # force = negative gradient

    def _make_run(self):
        def step(x, kappa0):
            F = self._forces(x, kappa0)
            d = x[1:] - x[:-1]
            t_seg = d / (jnp.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
            t = jnp.concatenate([t_seg[:1], 0.5 * (t_seg[1:] + t_seg[:-1]), t_seg[-1:]], 0)
            t = t / (jnp.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
            Ft = jnp.sum(F * t, 1, keepdims=True) * t
            Fn = F - Ft
            v = Ft / self.Ct + Fn / self.Cn                   # RFT: tangential/normal drag split
            return x + self.dt * v, None
        def run(x, kappa0_seq):
            x, _ = lax.scan(step, x, kappa0_seq, unroll=8)
            return x
        return run

    def run(self, x, kappa0_seq):
        return self._run(x, jnp.asarray(kappa0_seq, self.dtype))

    def curvature(self, x):
        d = x[1:] - x[:-1]
        phi = jnp.arctan2(d[:, 1], d[:, 0])
        dphi = phi[1:] - phi[:-1]
        return jnp.arctan2(jnp.sin(dphi), jnp.cos(dphi)) / self.ell


def preferred_curvature_own(A_D, A_V, kappa_max):
    """Row activations (24,) → joint preferred curvatures (23,): dorsoventral difference × kmax, adjacent-row mean."""
    diff = kappa_max * (A_D - A_V)
    return 0.5 * (diff[1:] + diff[:-1])
