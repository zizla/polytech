"""Synthetic QC datasets for Orange3 teaching (CJ / SH / SR themed).
All values are SIMULATED. Seed fixed for reproducibility."""
import numpy as np, pandas as pd, os, json, sys

out = sys.argv[1]
os.makedirs(f"{out}/data", exist_ok=True)
rng = np.random.default_rng(20261110)

def sigmoid(z): return 1 / (1 + np.exp(-z))

def add_missing(df, cols, rate=0.02):
    for c in cols:
        m = rng.random(len(df)) < rate
        df.loc[m, c] = np.nan
    return df

def shifts(n): return rng.choice(["Day", "Night"], n, p=[0.6, 0.4])

# ---------- CJ: fish cake (eomuk) ----------
n = 600
d = pd.DataFrame({
    "Batch_ID": [f"CJ-{i:04d}" for i in range(1, n + 1)],
    "Line": rng.choice(["L1", "L2", "L3"], n),
    "Shift": shifts(n),
    "Fish_Meat_Ratio": rng.normal(62, 6, n).clip(40, 80).round(1),
    "Starch_Ratio": rng.normal(10, 2, n).clip(4, 18).round(1),
    "Salt_Content": rng.normal(1.8, 0.25, n).clip(1.0, 2.8).round(2),
    "Fry_Temp": rng.normal(160, 6, n).round(1),
    "Fry_Time": rng.normal(90, 10, n).round(0),
    "Core_Temp": None,
    "Moisture": None,
})
d["Hours_Since_Cleaning"] = rng.uniform(0, 12, n).round(1)
d["Ambient_Humidity"] = rng.normal(55, 10, n).clip(25, 90).round(1)
d["Core_Temp"] = (75 + 0.25 * (d.Fry_Temp - 160) + 0.12 * (d.Fry_Time - 90) + rng.normal(0, 2, n)).round(1)
d["Moisture"] = (68 - 0.15 * (d.Fry_Time - 90) - 0.2 * (d.Starch_Ratio - 10) + rng.normal(0, 1.5, n)).round(1)
d["Gel_Strength"] = (300 + 6 * (d.Fish_Meat_Ratio - 62) - 4 * (d.Starch_Ratio - 10) + rng.normal(0, 25, n)).round(0)
night = (d.Shift == "Night").astype(int)
d["Total_Bacteria_log"] = (2.5 - 0.08 * (d.Core_Temp - 75) + 0.3 * night + 0.09 * (d.Hours_Since_Cleaning - 6) + rng.normal(0, 0.3, n)).clip(0.5).round(2)
burn = sigmoid(-3.6 + 0.40 * (d.Fry_Temp - 160) + 0.10 * (d.Fry_Time - 90))
micro = sigmoid(-3.4 + 4.0 * (d.Total_Bacteria_log - 3.0))
texture = sigmoid(-3.6 - 0.06 * (d.Gel_Strength - 240))
u = rng.random((n, 3))
d["Defect_Type"] = "None"
d.loc[u[:, 2] < texture, "Defect_Type"] = "Soft_Texture"
d.loc[u[:, 1] < micro, "Defect_Type"] = "Microbial"
d.loc[u[:, 0] < burn, "Defect_Type"] = "Burn"
d["Pass_Fail"] = np.where(d.Defect_Type == "None", "Pass", "Fail")
d = add_missing(d, ["Moisture", "Gel_Strength", "Salt_Content"])
d.loc[rng.choice(n, 3, replace=False), "Fry_Temp"] = [198.0, 201.5, 121.0]  # sensor outliers
cj = d

# ---------- SH: seasoned laver (gim) ----------
n = 600
grade = rng.choice(["A", "B", "C"], n, p=[0.35, 0.45, 0.20])
gnum = pd.Series(grade).map({"A": 0, "B": 1, "C": 2}).values
d = pd.DataFrame({
    "Batch_ID": [f"SH-{i:04d}" for i in range(1, n + 1)],
    "Line": rng.choice(["L1", "L2"], n),
    "Shift": shifts(n),
    "Raw_Laver_Grade": grade,
    "Sheet_Weight": (3.0 - 0.15 * gnum + rng.normal(0, 0.15, n)).round(2),
    "Roast_Temp": rng.normal(280, 12, n).round(1),
    "Roast_Time": rng.normal(6.0, 0.8, n).clip(3.5, 9).round(1),
    "Oil_Amount": rng.normal(12, 1.5, n).clip(7, 18).round(1),
    "Salt_Content": rng.normal(2.5, 0.35, n).clip(1.4, 4).round(2),
})
d["Hours_Since_Cleaning"] = rng.uniform(0, 12, n).round(1)
d["Ambient_Humidity"] = rng.normal(55, 10, n).clip(25, 90).round(1)
d["Moisture"] = (3.2 - 0.012 * (d.Roast_Temp - 280) - 0.25 * (d.Roast_Time - 6) + rng.normal(0, 0.35, n)).clip(1.2).round(2)
d["Water_Activity"] = (0.30 + 0.06 * (d.Moisture - 3.2) + 0.003 * (d.Ambient_Humidity - 55) + rng.normal(0, 0.015, n)).clip(0.15, 0.6).round(3)
d["Pinhole_Count"] = rng.poisson(np.exp(0.2 + 0.6 * gnum + 0.02 * (d.Roast_Temp - 280)))
tear = sigmoid(-4.2 + 1.4 * gnum - 6.0 * (d.Sheet_Weight - 2.85) + 0.09 * (d.Roast_Temp - 280))
hole = sigmoid(-3.8 + 0.8 * (d.Pinhole_Count - 2))
damp = sigmoid(-3.6 + 60 * (d.Water_Activity - 0.33))
u = rng.random((n, 3))
d["Defect_Type"] = "None"
d.loc[u[:, 2] < damp, "Defect_Type"] = "Moisture_Absorption"
d.loc[u[:, 1] < hole, "Defect_Type"] = "Pinhole"
d.loc[u[:, 0] < tear, "Defect_Type"] = "Tear"
d["Pass_Fail"] = np.where(d.Defect_Type == "None", "Pass", "Fail")
d = add_missing(d, ["Oil_Amount", "Water_Activity"])
sh = d

# ---------- SR: flavor (hyangryo) ----------
n = 500
form = rng.choice(["Liquid", "Powder"], n, p=[0.65, 0.35])
liq = (form == "Liquid").astype(int)
d = pd.DataFrame({
    "Batch_ID": [f"SR-{i:04d}" for i in range(1, n + 1)],
    "Line": rng.choice(["L1", "L2"], n),
    "Shift": shifts(n),
    "Product_Form": form,
    "Application": rng.choice(["Beverage", "Bakery", "Dairy", "Confectionery"], n),
    "Batch_Size_kg": rng.choice([200, 500, 1000], n, p=[0.4, 0.4, 0.2]),
    "Mix_Temp": rng.normal(25, 2.5, n).round(1),
    "Mix_Time": rng.normal(40, 8, n).clip(15, 70).round(0),
    "Solvent_Ratio": np.where(liq == 1, rng.normal(70, 5, n), rng.normal(20, 4, n)).round(1),
})
d["Hours_Since_Cleaning"] = rng.uniform(0, 12, n).round(1)
d["Ambient_Humidity"] = rng.normal(55, 10, n).clip(25, 90).round(1)
d["Viscosity_cP"] = np.where(liq == 1, 30 - 0.25 * (d.Solvent_Ratio - 70) - 0.8 * (d.Mix_Temp - 25) + rng.normal(0, 3, n), np.nan).round(1)
d["Specific_Gravity"] = (np.where(liq == 1, 0.98, 0.55) + 0.001 * (d.Solvent_Ratio - np.where(liq == 1, 70, 20)) + rng.normal(0, 0.01, n)).round(3)
d["Refractive_Index"] = (1.420 - 0.0004 * (d.Solvent_Ratio - 70) + rng.normal(0, 0.003, n)).round(4)
d.loc[liq == 0, "Refractive_Index"] = np.nan  # not measured for powder
d["Fragrance_Intensity"] = (7.0 - 0.06 * (d.Solvent_Ratio - np.where(liq == 1, 70, 20)) + 0.02 * (d.Mix_Time - 40) - 0.3 * np.maximum(d.Mix_Temp - 27, 0) + rng.normal(0, 0.6, n)).clip(1, 10).round(2)
d["Fill_Volume_Dev"] = (rng.normal(0, 1.0, n) + 0.002 * (d.Batch_Size_kg - 500)).round(2)
d["Cap_Torque"] = rng.normal(15, 2, n).round(1)
weak = sigmoid(-3.4 - 3.5 * (d.Fragrance_Intensity - 6.0))
fill = sigmoid(-3.8 - 3.5 * (d.Fill_Volume_Dev + 1.2))
seal = sigmoid(-4.0 + 1.6 * (np.abs(d.Cap_Torque - 15) - 1.5))
u = rng.random((n, 3))
d["Defect_Type"] = "None"
d.loc[u[:, 2] < seal, "Defect_Type"] = "Seal_Defect"
d.loc[u[:, 1] < fill, "Defect_Type"] = "Underfill"
d.loc[u[:, 0] < weak, "Defect_Type"] = "Weak_Intensity"
d["Pass_Fail"] = np.where(d.Defect_Type == "None", "Pass", "Fail")
sr = d


# ---------- HC: balloon flower (doraji) syrup / tea base / drink ----------
n = 500
prod = rng.choice(["Syrup", "TeaBase", "Drink"], n, p=[0.45, 0.30, 0.25])
target = pd.Series(prod).map({"Syrup": 65.0, "TeaBase": 45.0, "Drink": 12.0}).values
tol = pd.Series(prod).map({"Syrup": 3.0, "TeaBase": 3.0, "Drink": 1.5}).values
d = pd.DataFrame({
    "Batch_ID": [f"HC-{i:04d}" for i in range(1, n + 1)],
    "Line": rng.choice(["L1", "L2"], n),
    "Shift": shifts(n),
    "Product_Type": prod,
    "Raw_Origin": rng.choice(["Domestic_A", "Domestic_B", "Imported"], n, p=[0.45, 0.35, 0.20]),
    "Raw_Moisture": rng.normal(80, 3, n).clip(70, 88).round(1),
    "Extract_Temp": rng.normal(95, 2.5, n).round(1),
    "Extract_Time": rng.normal(120, 15, n).clip(70, 170).round(0),
    "Conc_Temp": rng.normal(85, 4, n).round(1),
    "Conc_Time": rng.normal(60, 10, n).clip(30, 95).round(0),
    "Fill_Temp": rng.normal(88, 2.5, n).round(1),
})
d["Hours_Since_Cleaning"] = rng.uniform(0, 12, n).round(1)
d["Ambient_Humidity"] = rng.normal(55, 10, n).clip(25, 90).round(1)
orig = pd.Series(d.Raw_Origin).map({"Domestic_A": 0.3, "Domestic_B": 0.0, "Imported": -0.4}).values
d["Brix"] = (target + 0.08 * (d.Conc_Time - 60) * (target / 45) - 0.15 * (d.Raw_Moisture - 80) + rng.normal(0, 1.2, n) * (tol / 3)).round(1)
d["pH"] = rng.normal(4.6, 0.15, n).round(2)
d["Saponin_mg_g"] = (2.0 + orig + 0.006 * (d.Extract_Time - 120) + 0.03 * (d.Extract_Temp - 95) + rng.normal(0, 0.25, n)).clip(0.5).round(2)
heat = (d.Conc_Temp - 85) / 4 + (d.Conc_Time - 60) / 10
d["Browning_A420"] = (0.35 + 0.05 * heat + 0.06 * np.maximum(d.Conc_Temp - 88, 0) * np.maximum(d.Conc_Time - 65, 0) / 10 + rng.normal(0, 0.04, n)).clip(0.1).round(3)
d["Total_Bacteria_log"] = (1.5 - 0.35 * (d.Fill_Temp - 88) + 0.06 * (d.Hours_Since_Cleaning - 6) + rng.normal(0, 0.3, n)).clip(0.3).round(2)
low_brix = sigmoid(-4.4 + 3.5 * (target - d.Brix) / (tol / 1.5))
brown = sigmoid(-3.4 + 30 * (d.Browning_A420 - 0.45))
micro = sigmoid(-3.8 + 5.0 * (d.Total_Bacteria_log - 2.2))
sapon = sigmoid(-3.2 - 8.0 * (d.Saponin_mg_g - 1.7))
u = rng.random((n, 4))
d["Defect_Type"] = "None"
d.loc[u[:, 3] < sapon, "Defect_Type"] = "Low_Saponin"
d.loc[u[:, 2] < micro, "Defect_Type"] = "Microbial"
d.loc[u[:, 1] < brown, "Defect_Type"] = "Browning"
d.loc[u[:, 0] < low_brix, "Defect_Type"] = "Low_Brix"
d["Pass_Fail"] = np.where(d.Defect_Type == "None", "Pass", "Fail")
d = add_missing(d, ["pH", "Saponin_mg_g"])
hc = d

# ---------- Integrated: deviation from setpoint (common schema) ----------
def integ(df, comp, temp_col, setp, temp_sd, moist_col):
    m = df[moist_col] if moist_col else pd.Series(np.nan, index=df.index)
    return pd.DataFrame({
        "Batch_ID": df.Batch_ID, "Company": comp, "Line": df.Line, "Shift": df.Shift,
        "Temp_Dev_z": ((df[temp_col] - setp) / temp_sd).round(3),
        "Moisture_Dev_z": ((m - m.mean()) / m.std()).round(3),
        "Hours_Since_Cleaning": df.Hours_Since_Cleaning, "Ambient_Humidity": df.Ambient_Humidity,
        "Pass_Fail": df.Pass_Fail, "Defect_Type": df.Defect_Type,
    })
integ_df = pd.concat([
    integ(cj, "CJ", "Fry_Temp", 160, 6, "Moisture"),
    integ(sh, "SH", "Roast_Temp", 280, 12, "Moisture"),
    integ(sr, "SR", "Mix_Temp", 25, 2.5, None),
    integ(hc, "HC", "Conc_Temp", 85, 4, None),
], ignore_index=True)


KO = {
 "Batch_ID":"배치ID","Line":"라인","Shift":"근무조","Company":"업체",
 "Fish_Meat_Ratio":"어육배합비","Starch_Ratio":"전분배합비","Salt_Content":"염도",
 "Fry_Temp":"튀김온도","Fry_Time":"튀김시간","Core_Temp":"중심온도","Moisture":"수분",
 "Hours_Since_Cleaning":"세척후경과시간","Ambient_Humidity":"작업장습도","Gel_Strength":"겔강도",
 "Total_Bacteria_log":"일반세균수_log","Raw_Laver_Grade":"원초등급","Sheet_Weight":"원초중량",
 "Roast_Temp":"구이온도","Roast_Time":"구이시간","Oil_Amount":"도유량","Water_Activity":"수분활성도",
 "Pinhole_Count":"핀홀수","Product_Form":"제품형태","Application":"적용용도","Batch_Size_kg":"배치크기_kg",
 "Mix_Temp":"배합온도","Mix_Time":"배합시간","Solvent_Ratio":"용제비율","Viscosity_cP":"점도_cP",
 "Specific_Gravity":"비중","Refractive_Index":"굴절률","Fragrance_Intensity":"향강도",
 "Fill_Volume_Dev":"충전량편차","Cap_Torque":"캡토크","Temp_Dev_z":"온도편차_z","Moisture_Dev_z":"수분편차_z",
 "Defect_Type":"불량유형","Pass_Fail":"합격여부",
 "Product_Type":"제품유형","Raw_Origin":"원료산지","Raw_Moisture":"원료수분","Extract_Temp":"추출온도",
 "Extract_Time":"추출시간","Conc_Temp":"농축온도","Conc_Time":"농축시간","Fill_Temp":"충전온도",
 "Brix":"당도_Brix","pH":"pH","Saponin_mg_g":"조사포닌_mg_g","Browning_A420":"갈변도_A420",
}
# ---------- write plain + Orange-flagged CSVs ----------
DISC = {"Line", "Shift", "Company", "Raw_Laver_Grade", "Product_Form", "Application", "Product_Type", "Raw_Origin"}
def orange_header(col):
    if col == "Pass_Fail": return "cD#Pass_Fail"
    if col in ("Batch_ID",): return "mS#" + col
    if col == "Defect_Type": return "mD#" + col
    if col in DISC: return "D#" + col
    return "C#" + col

stats = {}
for name, df in [("cj_fishcake_qc", cj), ("sh_laver_qc", sh), ("sr_flavor_qc", sr), ("hc_doraji_qc", hc), ("integrated_qc", integ_df)]:
    assert set(df.columns) <= set(KO), set(df.columns) - set(KO)
    df.rename(columns=KO).to_csv(f"{out}/data/{name}.csv", index=False, encoding="utf-8")
    df.rename(columns=lambda c: orange_header(c).split("#")[0] + "#" + KO[c]).to_csv(f"{out}/data/{name}_orange.csv", index=False, encoding="utf-8")
    stats[name] = {
        "rows": len(df), "cols": df.shape[1],
        "fail_rate": round((df.Pass_Fail == "Fail").mean() * 100, 1),
        "defects": df.Defect_Type.value_counts().to_dict(),
        "missing": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
    }
    if "Company" in df:
        stats[name]["fail_by_company"] = (df.groupby("Company").Pass_Fail.apply(lambda s: round((s == "Fail").mean() * 100, 1))).to_dict()
json.dump(stats, open(f"{out}/stats.json", "w"), indent=1, ensure_ascii=False, default=int)
print(json.dumps(stats, indent=1, default=int))
