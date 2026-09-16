s = open("wm_env.py").read()
i = s.index("def render_states"); j = s.index("def all_state_frames")
new = '''def render_states(ball_r, ball_c, pad):
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


'''
s = s[:i] + new + s[j:]
open("wm_env.py", "w").write(s)
print(open("wm_env.py").read()[:2000])
