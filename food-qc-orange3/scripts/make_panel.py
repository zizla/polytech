"""Synthetic sensory panel data (SR flavor theme). 60 batches x 5 inspectors.
Reference = 향강도 from sr_flavor_qc (treated as expert-panel consensus score).
Each inspector: score = reference + bias + noise; inspector 4 tires with evaluation order."""
import numpy as np, pandas as pd, sys
out = sys.argv[1]
rng = np.random.default_rng(31)
sr = pd.read_csv(f"{out}/data/sr_flavor_qc.csv")
low = sr[sr.향강도 < 6.3]
rest = sr.drop(low.index).sample(60 - min(len(low), 18), random_state=3)
s = pd.concat([low.sample(min(len(low), 18), random_state=3), rest]).sample(frac=1, random_state=5).reset_index(drop=True)
order = np.arange(1, len(s) + 1)
P = {"검사원1": (0.0, 0.30, 0.0), "검사원2": (0.6, 0.40, 0.0), "검사원3": (-0.4, 0.50, 0.0),
     "검사원4": (0.0, 0.30, 1.2), "검사원5": (0.0, 0.90, 0.0)}
df = pd.DataFrame({"배치ID": s.배치ID, "평가순서": order, "제품형태": s.제품형태, "기준향강도": s.향강도})
df["기준판정"] = np.where(df.기준향강도 >= 6.0, "Pass", "Fail")
for name, (bias, sd, fat) in P.items():
    sc = df.기준향강도 + bias + rng.normal(0, 1, len(df)) * (sd + fat * order / len(df))
    df[f"{name}_점수"] = sc.clip(1, 10).round(1)
for name in P:
    df[f"{name}_판정"] = np.where(df[f"{name}_점수"] >= 6.0, "Pass", "Fail")
df.to_csv(f"{out}/data/sr_sensory_panel.csv", index=False, encoding="utf-8")
# long format for Orange
long = []
for name in P:
    t = df[["배치ID", "평가순서", "제품형태", "기준향강도", "기준판정"]].copy()
    t["검사원"] = name; t["점수"] = df[f"{name}_점수"]; t["판정"] = df[f"{name}_판정"]
    t["오차"] = (t.점수 - t.기준향강도).round(2); t["판정일치"] = np.where(t.판정 == t.기준판정, "일치", "불일치")
    long.append(t)
long = pd.concat(long, ignore_index=True)
long.to_csv(f"{out}/data/sr_sensory_panel_long.csv", index=False, encoding="utf-8")
# summary for lecture
print("기준 Fail", (df.기준판정 == "Fail").sum(), "/", len(df))
for name in P:
    j = df[f"{name}_판정"]; acc = (j == df.기준판정).mean()
    e = df[f"{name}_점수"] - df.기준향강도
    print(name, "일치율", round(acc * 100, 1), "평균오차", round(e.mean(), 2), "오차SD 전반/후반", round(e[:30].std(), 2), round(e[30:].std(), 2))
J = df[[f"{n}_판정" for n in P]]
print("5명 만장일치 배치 비율", round((J.nunique(axis=1) == 1).mean() * 100, 1))
