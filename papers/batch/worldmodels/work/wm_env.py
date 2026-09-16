"""Tiny pixel environment substituting for CarRacing-v0 (CPU tier-2).

"Catcher": 8x8 grid rendered to 64x64x3 RGB in [0,1].
A ball falls one row every `fall_every` steps from a random column; a paddle on
the bottom row moves left/stay/right.  Reward per step = 1 - |pad-ball_col|/(G-1)
(dense tracking) plus +1 when the ball lands on the paddle.  Batched (vectorised) over N parallel environments.

The renderer is a deterministic function of the discrete state
(ball_row, ball_col, paddle_col, phase) so the VAE encoder can be memoised over
the finite state set -- this is exact memoisation, not an approximation.
"""
from __future__ import annotations

import numpy as np

G = 8          # grid cells per side
CELL = 8       # pixels per cell -> 64x64
PAD_ROW = G - 1
N_STATES = G * G * G


def render_states(ball_r, ball_c, pad):
    """Magenta column falling from the top down to the ball row; green full-height
    column marking the paddle position.  Both are large, high-contrast features,
    which keeps the ConvVAE trainable at tier-2 compute."""
    n = len(ball_r)
    ball_r = np.asarray(ball_r); ball_c = np.asarray(ball_c); pad = np.asarray(pad)
    g = np.zeros((n, G, G, 3), dtype=np.float32)
    ar = np.arange(G)
    rowmask = ar[None, :] <= ball_r[:, None]
    colmask = ar[None, :] == ball_c[:, None]
    ball = rowmask[:, :, None] & colmask[:, None, :]
    g[..., 0][ball] = 1.0
    g[..., 2][ball] = 1.0
    padmask = (ar[None, :] == pad[:, None])
    g[..., 1][np.broadcast_to(padmask[:, None, :], (n, G, G))] = 1.0
    return np.repeat(np.repeat(g, CELL, axis=1), CELL, axis=2)


def all_state_frames():
    r, c, p = np.meshgrid(np.arange(G), np.arange(G), np.arange(G), indexing="ij")
    return render_states(r.ravel(), c.ravel(), p.ravel())


class BatchCatcher:
    def __init__(self, n: int, seed: int = 0, max_steps: int = 60, fall_every: int = 2,
                 tile_k: int = 1):
        # tile_k > 1: common random numbers -- the same tile_k-fold repeated set
        # of n//tile_k episodes is given to every candidate (variance reduction;
        # ball dynamics are action-independent so the draws can be shared).
        assert n % tile_k == 0
        self.tile_k = tile_k
        self.n = n
        self.max_steps = max_steps
        self.fall_every = fall_every
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _draw(self, k):
        m = k // self.tile_k
        return np.tile(self.rng.integers(0, G, size=m), self.tile_k)

    def reset(self):
        n = self.n
        self.ball_r = np.zeros(n, dtype=np.int64)
        self.ball_c = self._draw(n)
        self.pad = self._draw(n)
        self.t = 0
        return self.render()

    def state_index(self):
        return (self.ball_r * G + self.ball_c) * G + self.pad

    def render(self):
        return render_states(self.ball_r, self.ball_c, self.pad)

    def step(self, actions):
        """actions in {0,1,2} -> move left / stay / right."""
        a = np.asarray(actions).astype(np.int64)
        self.pad = np.clip(self.pad + (a - 1), 0, G - 1)
        self.t += 1
        # dense tracking reward (1 when the paddle is under the ball, 0 at the
        # far edge) + a catch bonus when the ball lands on the paddle
        dist = np.abs(self.pad - self.ball_c).astype(np.float32)
        reward = 1.0 - dist / (G - 1)
        if self.t % self.fall_every == 0:
            self.ball_r = self.ball_r + 1
            landed = self.ball_r >= PAD_ROW
            reward = reward + np.where(landed & (self.ball_c == self.pad), 1.0, 0.0).astype(np.float32)
            if landed.any():
                # ball_r evolves identically in every env, so all envs land together
                self.ball_r[landed] = 0
                self.ball_c[landed] = self._draw(int(landed.sum()))
        done = self.t >= self.max_steps
        return self.render(), reward, done
