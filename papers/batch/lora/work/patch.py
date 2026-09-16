s = open("train.py").read()
s = s.replace('''    "trainable_parameters_millions": lora''', '''    "glue_avg": lora["accuracy"],
    "glue_avg_full_ft": ft["accuracy"],
    "trainable_parameters_millions": lora''')
s = s.replace('''        "config_used": C,''', '''        "config_used": C,
        "glue_avg_note": "only SST-2 was run; glue_avg is the unweighted mean over the 1 task run (= SST-2 accuracy), NOT the paper's 8-task average",''')
open("train.py","w").write(s)
print("ok")
