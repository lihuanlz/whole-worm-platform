# -*- coding: utf-8 -*-
"""自有二维粘弹性杆身体层 (2026-10-23, 补三十三) —— 替换 worm-whisperer rod2d/muscle_map 借来件。

方程来源（公开文献标准件, 无代码抄袭）:
- 能量: 拉伸 (k_s/2)(|Δx|−ℓ)² + 弯曲 (k_b/2)(Δφ − ℓκ0)²   (Boyle et al. 2012, J R Soc Interface 9:2765)
- 过阻尼低 Re 动力学: v = Ft/Ct + Fn/Cn  (阻力理论 RFT, Gray & Lissmann 1964, J Exp Biol 41:135)
- Cn/Ct ≈ 40 琼脂爬行 (Boyle 2012); k_s=5e4 不可伸长强弹簧; k_b=5 弯曲刚度 (v62 冠军栈锚)
- 曲率映射: κ0_j = kmax·mean(A_D−A_V 相邻行)  (背腹差→关节优选曲率, 线性几何)

与借来件的差异: 力用解析梯度 (手写 ∂E/∂x 封闭形式), 不用 jax.grad 自动微分;
接口逐位对齐 (initial/run/curvature + preferred_curvature), walls/迷宫件不实现 (v62 主闭环用不到)。
单位: mm, s; 力以 Ct=1 归一化。
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
        """解析梯度: F = −∂E/∂x, E = E_s + E_b。全矢量化。"""
        d = x[1:] - x[:-1]                                    # (n,2) 段向量
        ln = jnp.linalg.norm(d, axis=1) + 1e-12               # (n,)
        # --- 拉伸: ∂E_s/∂d_i = k_s(1−ℓ/|d_i|) d_i ; x_i 受 d_{i-1}(+) 与 d_i(−) ---
        gs = self.k_s * (1.0 - self.ell / ln)[:, None] * d    # (n,2) = ∂E_s/∂d_i
        # x_j 在 d_{j-1} 中取 +、在 d_j 中取 − → ∂E/∂x_j = g_{j-1} − g_j
        Fs = jnp.zeros_like(x)
        Fs = Fs.at[:-1].add(-gs).at[1:].add(gs)
        # --- 弯曲: Δφ_j = φ_{j+1}−φ_j (wrap); ∂φ_i/∂d_i = perp(d_i)/|d_i|² ---
        phi = jnp.arctan2(d[:, 1], d[:, 0])                   # (n,)
        dphi = phi[1:] - phi[:-1]
        dphi = jnp.arctan2(jnp.sin(dphi), jnp.cos(dphi))      # (n−1,)
        r = (dphi - self.ell * kappa0)                        # (n−1,) 残差
        # perp(d) = (−dy, dx); ∂φ_i/∂x 经 d_i 传递
        perp = jnp.stack([-d[:, 1], d[:, 0]], axis=1) / (ln ** 2)[:, None]   # (n,2) = ∂φ_i/∂d_i
        gb = self.k_b * r                                     # (n−1,)
        # Δφ_j 依赖 x_{j}, x_{j+1}, x_{j+2}: ∂Δφ_j/∂x_{j+1} = perp_{j+1}贡献... 逐关节累加
        Fb = jnp.zeros_like(x)
        # ∂E_b/∂d_i = Σ_j gb_j ∂Δφ_j/∂d_i = gb_{i-1}·(∂φ_i/∂d_i) − gb_i·(∂φ_i/∂d_i)  (gb 越界=0)
        gbm = jnp.concatenate([gb[:1] * 0.0, gb])             # gb_{i-1}, i=0..n−1 (gb_{-1}=0)
        gbp = jnp.concatenate([gb, gb[-1:] * 0.0])            # gb_i,     i=0..n−1 (gb_{n-1}=0)
        q = (gbm - gbp)[:, None] * perp                       # (n,2) = ∂E_b/∂d_i
        Fb = jnp.zeros_like(x)
        Fb = Fb.at[:-1].add(-q).at[1:].add(q)                 # 同拉伸: ∂E_b/∂x_j = q_{j-1} − q_j
        return -(Fs + Fb)                                     # 力 = 负梯度

    def _make_run(self):
        def step(x, kappa0):
            F = self._forces(x, kappa0)
            d = x[1:] - x[:-1]
            t_seg = d / (jnp.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
            t = jnp.concatenate([t_seg[:1], 0.5 * (t_seg[1:] + t_seg[:-1]), t_seg[-1:]], 0)
            t = t / (jnp.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
            Ft = jnp.sum(F * t, 1, keepdims=True) * t
            Fn = F - Ft
            v = Ft / self.Ct + Fn / self.Cn                   # RFT: 切/法向阻力分解
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
    """行激活 (24,) → 关节优选曲率 (23,): 背腹差×kmax, 相邻行均值。"""
    diff = kappa_max * (A_D - A_V)
    return 0.5 * (diff[1:] + diff[:-1])
