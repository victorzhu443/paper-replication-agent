s = open("wm_env.py").read()
s = s.replace("""    def __init__(self, n: int, seed: int = 0, max_steps: int = 60, fall_every: int = 2):
        self.n = n""",
"""    def __init__(self, n: int, seed: int = 0, max_steps: int = 60, fall_every: int = 2,
                 tile_k: int = 1):
        # tile_k > 1: common random numbers -- the same tile_k-fold repeated set
        # of n//tile_k episodes is given to every candidate (variance reduction;
        # ball dynamics are action-independent so the draws can be shared).
        assert n % tile_k == 0
        self.tile_k = tile_k
        self.n = n""")
s = s.replace("""    def reset(self):
        n = self.n
        self.ball_r = np.zeros(n, dtype=np.int64)
        self.ball_c = self.rng.integers(0, G, size=n)
        self.pad = self.rng.integers(0, G, size=n)""",
"""    def _draw(self, k):
        m = k // self.tile_k
        return np.tile(self.rng.integers(0, G, size=m), self.tile_k)

    def reset(self):
        n = self.n
        self.ball_r = np.zeros(n, dtype=np.int64)
        self.ball_c = self._draw(n)
        self.pad = self._draw(n)""")
s = s.replace("""            if landed.any():
                k = int(landed.sum())
                self.ball_r[landed] = 0
                self.ball_c[landed] = self.rng.integers(0, G, size=k)""",
"""            if landed.any():
                # ball_r evolves identically in every env, so all envs land together
                self.ball_r[landed] = 0
                self.ball_c[landed] = self._draw(int(landed.sum()))""")
open("wm_env.py", "w").write(s)

m = open("wm_main.py").read()
m = m.replace("        env = BatchCatcher(N, seed=ep_seed, max_steps=T)\n        h = torch.zeros",
              "        env = BatchCatcher(N, seed=ep_seed, max_steps=T, tile_k=K)\n        h = torch.zeros")
m = m.replace('pop = int(cfg.get("_cma_popsize", 32))', 'pop = int(cfg.get("_cma_popsize", 64))')
open("wm_main.py", "w").write(m)
print("ok")
