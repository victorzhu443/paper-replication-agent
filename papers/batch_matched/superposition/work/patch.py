s = open('toy.py').read()
s = s.replace('for name, (n, m) in specs.items():\n        dens = [0.05] * restarts\n        W, b, losses = train(n, m, dens, cfg, seed + hash(name) % 1000, steps, shuffle=shuffle)',
              'for j, (name, (n, m)) in enumerate(specs.items()):\n        dens = [0.05] * restarts\n        W, b, losses = train(n, m, dens, cfg, seed + 101 * (j + 1), steps, shuffle=shuffle)')
s = s.replace('"scale": (0.1 if smoke and scale == 1.0 else scale) if smoke else scale,', '"scale": scale,')
open('toy.py', 'w').write(s)
print('ok')
