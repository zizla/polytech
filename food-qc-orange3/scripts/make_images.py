"""Synthetic (drawn) QC inspection image set for Orange3 Image Analytics.
Output:
  images/<업체>_<OK|NG>/*.png           (folder name = class, for Import Images)
  data/qc_images.zip                    (images/ folder zipped)
  data/image_meta.tab                   (Orange 3-row header, relative image paths)
  data/image_features.csv               (Korean columns, handcrafted features, no image column)
All images are drawn by code; they are not photos of real products."""
import numpy as np, os, sys, zipfile, csv
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

out = sys.argv[1]
rng = np.random.default_rng(7)
S, N = 128, 50

def noise_bg(base, amp):
    a = np.clip(np.array(base)[None, None, :] + rng.normal(0, amp, (S, S, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(a)

def jitter(img, bg):
    img = img.rotate(float(rng.uniform(-8, 8)), resample=Image.BICUBIC, fillcolor=bg)
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.85, 1.15)))
    return img

def fishcake(defect):
    bg = (235, 235, 230); img = noise_bg(bg, 6); d = ImageDraw.Draw(img)
    x0, y0 = (int(v) for v in rng.integers(10, 22, 2)); x1, y1 = (S - int(v) for v in rng.integers(10, 22, 2))
    col = tuple(int(c) for c in rng.normal((222, 196, 150), 8))
    d.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=col)
    for _ in range(25):
        cx, cy = int(rng.integers(x0, x1)), int(rng.integers(y0, y1)); r = int(rng.integers(1, 3))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(c - 14 for c in col))
    sub = "정상"
    if defect:
        if rng.random() < 0.65:
            sub = "탄자국"
            for _ in range(int(rng.integers(1, 4))):
                cx, cy = int(rng.integers(x0 + 10, x1 - 10)), int(rng.integers(y0 + 10, y1 - 10))
                rx, ry = (int(v) for v in rng.integers(5, 15, 2)); dk = int(rng.integers(40, 100))
                d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(dk + 30, dk, max(dk - 20, 0)))
        else:
            sub = "이물질"
            for _ in range(int(rng.integers(1, 3))):
                cx, cy = int(rng.integers(x0 + 8, x1 - 8)), int(rng.integers(y0 + 8, y1 - 8))
                d.line([(cx, cy), (cx + int(rng.integers(6, 16)), cy + int(rng.integers(-8, 9)))], fill=(30, 30, 30), width=2)
    return jitter(img.filter(ImageFilter.GaussianBlur(0.8)), bg), sub

def laver(defect):
    bg = (240, 240, 240); img = noise_bg(bg, 5); d = ImageDraw.Draw(img)
    m = int(rng.integers(8, 16))
    col = tuple(int(c) for c in rng.normal((28, 45, 30), 5))
    d.rectangle([m, m, S - m, S - m], fill=col)
    for _ in range(60):
        cx, cy = (int(v) for v in rng.integers(m, S - m, 2)); d.point((cx, cy), fill=(60, 80, 55))
    sub = "정상"
    if defect:
        if rng.random() < 0.5:
            sub = "핀홀"
            for _ in range(int(rng.integers(3, 8))):
                cx, cy = (int(v) for v in rng.integers(m + 5, S - m - 5, 2)); r = int(rng.integers(1, 4))
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(235, 235, 235))
        else:
            sub = "찢김"
            y = int(rng.integers(30, S - 30)); x = S - m; pts = [(x, y - 6)]
            while x > S // 2:
                x -= int(rng.integers(4, 9)); pts.append((x, y + int(rng.integers(-5, 6))))
            pts.append((S - m, y + 6)); d.polygon(pts, fill=(238, 238, 238))
    return jitter(img.filter(ImageFilter.GaussianBlur(0.6)), bg), sub

def bottle(defect):
    bg = (225, 228, 235); img = noise_bg(bg, 5); d = ImageDraw.Draw(img)
    cx = S // 2 + int(rng.integers(-6, 7))
    d.rounded_rectangle([cx - 28, 34, cx + 28, 118], radius=10, fill=(205, 215, 225), outline=(150, 160, 170), width=2)
    liquid = tuple(int(c) for c in rng.normal((210, 160, 70), 10))
    sub = "정상"; level = int(rng.integers(48, 56)); tilt = 0
    if defect:
        if rng.random() < 0.5:
            sub = "충전미달"; level = int(rng.integers(70, 92))
        else:
            sub = "캡불량"; tilt = int(rng.integers(5, 10)) * (1 if rng.random() < 0.5 else -1)
    d.rounded_rectangle([cx - 25, level, cx + 25, 115], radius=8, fill=liquid)
    d.polygon([(cx - 16, 34), (cx + 16, 34 + tilt // 2), (cx + 16, 18 + tilt), (cx - 16, 18)], fill=(40, 40, 50))
    if sub == "캡불량" and rng.random() < 0.6:
        d.line([(cx - 8, 20), (cx - 2, 30), (cx + 4, 24)], fill=(230, 230, 230), width=2)
    return jitter(img.filter(ImageFilter.GaussianBlur(0.5)), bg), sub


def doraji(defect):
    bg = (232, 230, 224); img = noise_bg(bg, 5); d = ImageDraw.Draw(img)
    cx = S // 2 + int(rng.integers(-6, 7))
    d.rectangle([cx - 22, 14, cx + 22, 28], fill=(200, 170, 60))
    d.rounded_rectangle([cx - 34, 28, cx + 34, 118], radius=12, fill=(220, 224, 226), outline=(160, 160, 160), width=2)
    sub = "정상"; col = tuple(int(c) for c in rng.normal((205, 140, 55), 8))
    if defect:
        if rng.random() < 0.5:
            sub = "갈변"; col = tuple(int(c) for c in rng.normal((105, 58, 25), 10))
    d.rounded_rectangle([cx - 31, int(rng.integers(36, 44)), cx + 31, 115], radius=10, fill=col)
    for _ in range(int(rng.integers(4, 8))):
        x, y = int(rng.integers(cx - 24, cx + 20)), int(rng.integers(50, 100))
        d.line([(x, y), (x + int(rng.integers(5, 12)), y + int(rng.integers(-3, 4)))], fill=(238, 225, 190), width=3)
    if defect and sub == "정상":
        sub = "침전"; h = int(rng.integers(10, 20))
        d.rounded_rectangle([cx - 31, 115 - h, cx + 31, 115], radius=8, fill=(90, 70, 50))
    return jitter(img.filter(ImageFilter.GaussianBlur(0.5)), bg), sub

def features(img):
    a = np.asarray(img).astype(float); g = a.mean(axis=2)
    gx, gy = np.abs(np.diff(g, axis=1)).mean(), np.abs(np.diff(g, axis=0)).mean()
    return {"평균밝기": round(g.mean(), 2), "밝기표준편차": round(g.std(), 2),
            "어두운픽셀비율": round((g < 80).mean(), 4), "밝은픽셀비율": round((g > 225).mean(), 4),
            "에지강도": round((gx + gy) / 2, 3),
            "R평균": round(a[..., 0].mean(), 2), "G평균": round(a[..., 1].mean(), 2), "B평균": round(a[..., 2].mean(), 2)}

makers = {"CJ": ("어묵", fishcake), "SH": ("조미김", laver), "SR": ("향료병", bottle), "HC": ("도라지청병", doraji)}
rows = []
for comp, (prod, fn) in makers.items():
    for lab, defect in (("OK", False), ("NG", True)):
        folder = f"{comp}_{lab}"; os.makedirs(f"{out}/images/{folder}", exist_ok=True)
        for i in range(N):
            img, sub = fn(defect)
            name = f"{folder}_{i:03d}.png"; img.save(f"{out}/images/{folder}/{name}")
            rows.append({"이미지": f"images/{folder}/{name}", "이미지ID": name[:-4], "업체": comp, "제품": prod,
                         "판정": lab, "불량세부유형": sub, "클래스": folder, **features(img)})

feat = ["평균밝기", "밝기표준편차", "어두운픽셀비율", "밝은픽셀비율", "에지강도", "R평균", "G평균", "B평균"]
# Orange native .tab with 3-row header (image column = string meta, type=image)
cols = ["이미지", "이미지ID", "업체", "제품", "불량세부유형", "클래스"] + feat + ["판정"]
types = ["string", "string", "discrete", "discrete", "discrete", "discrete"] + ["continuous"] * len(feat) + ["discrete"]
flags = ["meta type=image", "meta", "meta", "meta", "meta", "meta"] + [""] * len(feat) + ["class"]
with open(f"{out}/data/image_meta.tab", "w", encoding="utf-8", newline="") as f:
    f.write("\t".join(cols) + "\n" + "\t".join(types) + "\n" + "\t".join(flags) + "\n")
    for r in rows:
        f.write("\t".join(str(r[c]) for c in cols) + "\n")
with open(f"{out}/data/image_features.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[c for c in cols if c != "이미지"]); w.writeheader()
    for r in rows: w.writerow({k: r[k] for k in w.fieldnames})

with zipfile.ZipFile(f"{out}/data/qc_images.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for dp, _, fs in sorted(os.walk(f"{out}/images")):
        for fn_ in sorted(fs):
            z.write(os.path.join(dp, fn_), os.path.relpath(os.path.join(dp, fn_), out))
from collections import Counter
print(len(rows), Counter((r["클래스"], r["불량세부유형"]) for r in rows))
