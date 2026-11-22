"""Excel-friendly copies (UTF-8 with BOM so Korean headers open correctly by double-click)."""
import pandas as pd, sys, os
out = sys.argv[1]; os.makedirs(f"{out}/data/excel", exist_ok=True)
for f in ["cj_fishcake_qc", "sh_laver_qc", "sr_flavor_qc", "hc_doraji_qc", "sr_sensory_panel"]:
    pd.read_csv(f"{out}/data/{f}.csv").to_csv(f"{out}/data/excel/{f}_excel.csv", index=False, encoding="utf-8-sig")
print(sorted(os.listdir(f"{out}/data/excel")))
