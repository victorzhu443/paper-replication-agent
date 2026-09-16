import os
p=os.getcwd()
for i in range(5):
    print(p, os.listdir(p)[:30])
    p=os.path.dirname(p)
