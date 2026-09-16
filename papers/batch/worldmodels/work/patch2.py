s = open("wm_main.py").read()
s = s.replace("""        es.tell(fit)
        fit_hist.append(float(np.mean(fit)))""",
"""        es.tell(fit)
        fit_hist.append(float(np.mean(fit)))
        if os.environ.get("WM_VERBOSE") == "1" and g % 5 == 0:
            print(f"gen {g} mean {np.mean(fit):.3f} max {np.max(fit):.3f} "
                  f"std {np.std(fit):.3f} sigma {es.sigma:.3f} |mean| {np.abs(es.mean).mean():.3f}", flush=True)""")
open("wm_main.py", "w").write(s)
print("ok")
