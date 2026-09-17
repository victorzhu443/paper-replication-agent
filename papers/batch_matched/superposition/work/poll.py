import json, os, time
for _ in range(56):
    if os.path.exists("metrics_full.json"):
        d = json.load(open("metrics_full.json"))
        i = d["metrics"].pop("_intermediates")
        print(json.dumps(d, indent=1))
        print("wall", i["wall_s"])
        print(i["dstar_curve"])
        print(i["adv_ratio_curve"])
        break
    time.sleep(5)
else:
    print("not done yet", open("full.log").read()[-500:])
