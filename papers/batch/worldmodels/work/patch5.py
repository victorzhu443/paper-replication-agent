s = open("wm_env.py").read()
old = """def render_states(ball_r, ball_c, pad):
    n = len(ball_r)
    g = np.zeros((n, G, G, 3), dtype=np.float32)
    idx = np.arange(n)
    g[idx, ball_r, ball_c, :] = 1.0
    g[idx, PAD_ROW, pad, 1] = 1.0
    return np.repeat(np.repeat(g, CELL, axis=1), CELL, axis=2)"""
new = """def render_states(ball_r, ball_c, pad):
    \"\"\"White trail column from the top down to the ball, green paddle cell.\"\"\"
    n = len(ball_r)
    ball_r = np.asarray(ball_r); ball_c = np.asarray(ball_c); pad = np.asarray(pad)
    g = np.zeros((n, G, G, 3), dtype=np.float32)
    ar = np.arange(G)
    rowmask = ar[None, :] <= ball_r[:, None]
    colmask = ar[None, :] == ball_c[:, None]
    g[rowmask[:, :, None] & colmask[:, None, :]] = 1.0
    idx = np.arange(n)
    g[idx, PAD_ROW, pad, :] = 0.0
    g[idx, PAD_ROW, pad, 1] = 1.0
    return np.repeat(np.repeat(g, CELL, axis=1), CELL, axis=2)"""
assert old in s
s = s.replace(old, new)
open("wm_env.py", "w").write(s)
print("ok")
