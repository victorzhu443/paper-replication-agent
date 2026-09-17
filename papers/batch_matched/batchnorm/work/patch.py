s=open("train.py").read()
s=s.replace('"scale": 1.0 if smoke else scale,', '"scale": scale,')
open("train.py","w").write(s)
