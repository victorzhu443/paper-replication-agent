import re
s = open("train_eval.py").read()
s = s.replace('"beam_check_n": 200,', '"beam_check_n": 100,')
open("train_eval.py", "w").write(s)
print('"beam_check_n": 100,' in s)
