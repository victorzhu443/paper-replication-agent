s = open("wm_env.py").read()
s = s.replace("""        self.t += 1
        reward = np.zeros(self.n, dtype=np.float32)""",
"""        self.t += 1
        # dense tracking reward (1 when the paddle is under the ball, 0 at the
        # far edge) + a catch bonus when the ball lands on the paddle
        dist = np.abs(self.pad - self.ball_c).astype(np.float32)
        reward = 1.0 - dist / (G - 1)""")
s = s.replace("""            reward = np.where(landed & (self.ball_c == self.pad), 1.0, 0.0).astype(np.float32)""",
"""            reward = reward + np.where(landed & (self.ball_c == self.pad), 1.0, 0.0).astype(np.float32)""")
s = s.replace("A ball falls one row every `fall_every` steps from a random column; a paddle on\nthe bottom row moves left/stay/right.  Reward +1 when the paddle is under the\nball as it lands.",
"A ball falls one row every `fall_every` steps from a random column; a paddle on\nthe bottom row moves left/stay/right.  Reward per step = 1 - |pad-ball_col|/(G-1)\n(dense tracking) plus +1 when the ball lands on the paddle.")
open("wm_env.py", "w").write(s)
print(s[1400:2600])
