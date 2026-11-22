"""
안전모 착용 여부 이미지 분류 데이터셋 (가상, 교육용)
- 실제 사진이 아니라 파이썬으로 그린 합성 이미지입니다. 실제 현장 사진으로 학습한 모델을 대신하지 않습니다.
- 클래스: 안전모착용 / 안전모미착용
- 미착용에는 '헷갈리는 사례'를 일부러 넣었습니다: 야구모자·비니·후드(머리에 무언가 씀), 안전모를 손에 든 경우.
  야구모자·비니는 안전모와 같은 색 목록에서 색을 고르므로 '색'만으로는 구분되지 않습니다.
- 클래스와 무관하게 무작위인 요소: 배경(하늘·지면·구조물·라바콘), 조명, 피부색, 머리색, 옷·조끼, 보안경,
  손에 든 물건(서류판·무전기), 사람 위치·크기·기울기, 흐림.
- 출력: helmet_images/train/<클래스>/*.jpg (299×299, Inception v3 입력 크기), new/*.jpg, labels.csv
"""
import os, csv, math, numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE, SS = 299, 2                 # 2배로 그린 뒤 줄여서 계단 현상을 없앤다
W = SIZE * SS
CLASSES = ["안전모착용", "안전모미착용"]
rng = np.random.default_rng(2026)

SKIN = [(255, 219, 172), (241, 194, 125), (224, 172, 105), (198, 134, 66), (141, 85, 36), (92, 58, 30)]
HAIR = [(20, 18, 16), (55, 38, 25), (100, 70, 40), (150, 120, 80), (120, 120, 120), (200, 200, 195)]
GEAR = {"흰색": (245, 245, 240), "노랑": (250, 205, 20), "주황": (245, 120, 20),
        "파랑": (30, 90, 190), "빨강": (200, 30, 35), "초록": (40, 150, 60)}   # 안전모·모자 공통 색
CAPX = {"검정": (28, 28, 32), "남색": (30, 40, 80), "회색": (115, 115, 120)}
SHIRT = [(40, 60, 110), (70, 70, 75), (150, 150, 155), (110, 80, 60), (230, 230, 230), (30, 30, 35), (90, 120, 70)]
VEST = [(200, 240, 40), (255, 140, 20)]

def jit(c, s=10): return tuple(int(np.clip(v + rng.normal(0, s), 0, 255)) for v in c)
def shade(c, f): return tuple(int(np.clip(v * f, 0, 255)) for v in c[:3])
def A(c, a=255): return tuple(c[:3]) + (a,)

# ---------------- 배경 ----------------
def background():
    hz = rng.uniform(.38, .72) * W
    top = np.array([(110, 160, 220), (150, 170, 190), (190, 200, 210), (225, 175, 130)][rng.integers(4)], float) + rng.normal(0, 10, 3)
    bot = top + rng.uniform(20, 45)
    gnd = np.array([(150, 120, 90), (130, 130, 125), (170, 150, 120), (100, 95, 85)][rng.integers(4)], float) + rng.normal(0, 10, 3)
    y = np.arange(W)[:, None, None]
    t = np.clip(y / hz, 0, 1)
    sky = top * (1 - t) + bot * t
    g = gnd * (1 + .25 * (y - hz) / W)
    a = np.where(y < hz, sky, g) * np.ones((1, W, 1))
    a += rng.normal(0, 5, (W, W, 1))
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(rng.integers(1, 4)):                                   # 건물
        x0 = rng.uniform(-.1, .9) * W; bw = rng.uniform(.15, .4) * W; bh = rng.uniform(.15, .45) * W
        c = jit([(170, 165, 155), (140, 140, 145), (200, 190, 170), (120, 110, 100)][rng.integers(4)], 8)
        d.rectangle([x0, hz - bh, x0 + bw, hz], fill=c)
        for wy in np.arange(hz - bh + .04 * W, hz - .03 * W, .07 * W):
            for wx in np.arange(x0 + .03 * W, x0 + bw - .03 * W, .06 * W):
                d.rectangle([wx, wy, wx + .03 * W, wy + .035 * W], fill=shade(c, .65))
    for _ in range(rng.integers(0, 3)):                                   # 비계
        x0 = rng.uniform(0, W); h = rng.uniform(.2, .5) * W; c = jit((95, 95, 100), 8); lw = int(.008 * W)
        for k in range(3): d.line([(x0 + k * .05 * W, hz), (x0 + k * .05 * W, hz - h)], fill=c, width=lw)
        for yy in np.arange(hz, hz - h, -.06 * W): d.line([(x0, yy), (x0 + .1 * W, yy)], fill=c, width=lw)
    if rng.random() < .35:                                                # 크레인
        x0 = rng.uniform(.1, .9) * W; c = jit((230, 180, 30), 10)
        d.line([(x0, hz), (x0, hz - .6 * W)], fill=c, width=int(.012 * W))
        d.line([(x0 - .3 * W, hz - .58 * W), (x0 + .35 * W, hz - .58 * W)], fill=c, width=int(.01 * W))
    for _ in range(rng.integers(0, 4)):                                   # 라바콘
        x0 = rng.uniform(0, W); y0 = rng.uniform(hz, W); s = rng.uniform(.03, .07) * W
        d.polygon([(x0, y0 - 2 * s), (x0 - s * .6, y0), (x0 + s * .6, y0)], fill=jit((240, 110, 20), 8))
        d.rectangle([x0 - s * .45, y0 - 1.1 * s, x0 + s * .45, y0 - .8 * s], fill=(235, 235, 235))
    return img

# ---------------- 머리에 쓰는 것 ----------------
def draw_helmet(d, cx, cy, r, col, facing):
    dome_cy = cy - .22 * r; rx, ry = 1.13 * r, 1.05 * r
    d.pieslice([cx - rx, dome_cy - ry, cx + rx, dome_cy + ry], 180, 360, fill=A(col))
    d.rounded_rectangle([cx - .12 * r, dome_cy - ry + .02 * r, cx + .12 * r, dome_cy], radius=.1 * r, fill=A(shade(col, .86)))  # 가운데 능선
    d.arc([cx - rx * .8, dome_cy - ry * .85, cx + rx * .35, dome_cy + ry * .6], 200, 255, fill=A(shade(col, 1.25)), width=max(2, int(.11 * r)))
    off = facing * .25 * r                                                # 사방으로 둘러진 챙
    d.ellipse([cx - 1.45 * r + off, dome_cy - .1 * r, cx + 1.45 * r + off, dome_cy + .2 * r], fill=A(shade(col, .82)))
    if rng.random() < .25:                                                # 로고 스티커
        d.ellipse([cx + .3 * r, dome_cy - .55 * r, cx + .6 * r, dome_cy - .25 * r], fill=A(jit((40, 60, 120), 30)))

def draw_cap(d, cx, cy, r, col, facing):                                 # 야구모자: 둥근 머리 + 앞으로 내려온 챙
    dome_cy = cy - .3 * r; rx, ry = 1.0 * r, .78 * r
    d.pieslice([cx - rx, dome_cy - ry, cx + rx, dome_cy + ry], 180, 360, fill=A(col))
    d.line([(cx, dome_cy - ry), (cx, dome_cy)], fill=A(shade(col, .8)), width=max(1, int(.05 * r)))
    d.ellipse([cx - .08 * r, dome_cy - ry - .06 * r, cx + .08 * r, dome_cy - ry + .08 * r], fill=A(shade(col, .8)))
    if abs(facing) < .55:                                                 # 정면: 이마 위로 내려온 챙
        off = facing * .35 * r
        d.pieslice([cx - .95 * r + off, dome_cy - .3 * r, cx + .95 * r + off, dome_cy + .42 * r], 0, 180, fill=A(shade(col, .78)))
    else:                                                                 # 옆모습: 옆으로 튀어나온 챙
        s = 1 if facing > 0 else -1
        d.polygon([(cx + s * .6 * r, dome_cy - .05 * r), (cx + s * 1.65 * r, dome_cy + .12 * r), (cx + s * 1.6 * r, dome_cy + .25 * r),
                   (cx + s * .5 * r, dome_cy + .12 * r)], fill=A(shade(col, .78)))

def draw_beanie(d, cx, cy, r, col):
    top = cy - 1.15 * r; base = cy - .38 * r
    d.pieslice([cx - .99 * r, top, cx + .99 * r, base + (base - top)], 180, 360, fill=A(col))
    d.rounded_rectangle([cx - 1.0 * r, base - .28 * r, cx + 1.0 * r, base + .06 * r], radius=.12 * r, fill=A(shade(col, .82)))
    for k in np.linspace(-.8, .8, 9):
        d.line([(cx + k * r, base - .26 * r), (cx + k * r, base + .04 * r)], fill=A(shade(col, .7)), width=max(1, int(.04 * r)))
    if rng.random() < .4:
        d.ellipse([cx - .25 * r, top - .3 * r, cx + .25 * r, top + .15 * r], fill=A(shade(col, 1.1)))

def draw_hood_back(d, cx, cy, r, col):
    d.ellipse([cx - 1.38 * r, cy - 1.38 * r, cx + 1.38 * r, cy + 1.3 * r], fill=A(col))

def draw_hood_front(d, cx, cy, r, col):
    d.pieslice([cx - 1.25 * r, cy - 1.32 * r, cx + 1.25 * r, cy + .3 * r], 180, 360, fill=A(col))
    d.arc([cx - 1.05 * r, cy - 1.12 * r, cx + 1.05 * r, cy + .25 * r], 180, 360, fill=A(shade(col, .75)), width=max(2, int(.1 * r)))

def draw_hair(d, cx, cy, r, col, style):
    if style == "대머리": return
    d.pieslice([cx - .96 * r, cy - 1.08 * r, cx + .96 * r, cy + .62 * r], 180, 360, fill=A(col))
    d.rectangle([cx - .96 * r, cy - .3 * r, cx - .8 * r, cy + .15 * r], fill=A(col))
    d.rectangle([cx + .8 * r, cy - .3 * r, cx + .96 * r, cy + .15 * r], fill=A(col))

# ---------------- 손에 드는 물건 ----------------
def held_layer(kind, r, col=None):
    s = int(3.4 * r); L = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(L); c = s / 2
    if kind == "안전모":
        draw_helmet(d, c, c + .2 * r, .95 * r, col, 0)
    elif kind == "서류판":
        d.rounded_rectangle([c - .8 * r, c - 1.05 * r, c + .8 * r, c + 1.05 * r], radius=.1 * r, fill=A((120, 85, 50)))
        d.rectangle([c - .65 * r, c - .8 * r, c + .65 * r, c + .9 * r], fill=A((245, 245, 240)))
        for yy in np.linspace(c - .55 * r, c + .7 * r, 6): d.line([(c - .5 * r, yy), (c + .5 * r, yy)], fill=A((150, 150, 150)), width=max(1, int(.04 * r)))
    else:                                                                 # 무전기
        d.rounded_rectangle([c - .35 * r, c - .7 * r, c + .35 * r, c + .8 * r], radius=.12 * r, fill=A((35, 35, 38)))
        d.line([(c + .2 * r, c - .7 * r), (c + .2 * r, c - 1.4 * r)], fill=A((35, 35, 38)), width=max(2, int(.1 * r)))
        d.rectangle([c - .22 * r, c - .45 * r, c + .22 * r, c - .1 * r], fill=A((90, 140, 90)))
    return L.rotate(rng.uniform(-40, 40) if kind != "안전모" else rng.uniform(0, 360), resample=Image.BICUBIC, center=(c, c))

# ---------------- 작업자 한 명 ----------------
def make(cls, head=None, gear_col=None, hold=None):
    """head: 착용이면 '안전모', 미착용이면 '맨머리'·'대머리'·'야구모자'·'비니'·'후드' (None이면 무작위)
       hold: None(무작위) / '없음' / '안전모'(미착용 전용) / '서류판' / '무전기'"""
    img = background().convert("RGBA")
    P = Image.new("RGBA", (W, W), (0, 0, 0, 0)); d = ImageDraw.Draw(P)
    r = rng.uniform(22, 40) * SS; hx = rng.uniform(.3, .7) * W; hy = rng.uniform(.2, .46) * W
    facing = float(np.clip(rng.normal(0, .45), -.95, .95)); dx = facing * .35 * r
    skin = jit(SKIN[rng.integers(len(SKIN))], 6); hair_c = jit(HAIR[rng.integers(len(HAIR))], 6)
    shirt = jit(SHIRT[rng.integers(len(SHIRT))], 8)
    if head is None:
        head = "안전모" if cls == "안전모착용" else rng.choice(["맨머리", "대머리", "야구모자", "비니", "후드"], p=[.40, .10, .25, .15, .10])
    hair_style = "대머리" if head == "대머리" or rng.random() < .1 else "짧은머리"
    if gear_col is None:
        pool = dict(GEAR, **CAPX) if head in ("야구모자", "비니") else GEAR
        gear_col = list(pool)[rng.integers(len(pool))]
    gcol = jit(dict(GEAR, **CAPX)[gear_col], 6)
    hood_c = jit([(60, 60, 65), (40, 50, 80), (110, 110, 115), (120, 40, 40)][rng.integers(4)], 8)
    if head == "후드": draw_hood_back(d, hx, hy, r, hood_c)
    # 몸통 · 조끼 · 팔
    top = hy + 1.3 * r
    d.rounded_rectangle([hx - 2.1 * r, top, hx + 2.1 * r, W + 4 * r], radius=.8 * r, fill=A(shirt))
    if rng.random() < .7:
        vc = jit(VEST[rng.integers(2)], 6)
        d.rounded_rectangle([hx - 1.9 * r, top + .1 * r, hx + 1.9 * r, W + 4 * r], radius=.7 * r, fill=A(vc))
        d.polygon([(hx - .55 * r, top), (hx + .55 * r, top), (hx, top + 1.3 * r)], fill=A(shirt))
        for yy in (top + 1.5 * r, top + 2.3 * r):
            d.rectangle([hx - 1.9 * r, yy, hx + 1.9 * r, yy + .22 * r], fill=A((205, 205, 205)))
    for s in (-1, 1):
        d.rounded_rectangle([hx + s * 2.05 * r - .38 * r, top + .3 * r, hx + s * 2.05 * r + .38 * r, top + 3.8 * r], radius=.35 * r, fill=A(shade(shirt, .9)))
    d.rectangle([hx - .35 * r, hy + .7 * r, hx + .35 * r, top + .2 * r], fill=A(shade(skin, .92)))    # 목
    # 손에 든 물건 (팔을 앞으로 굽혀 몸 앞에 든다)
    if hold is None:
        hold = "안전모" if (cls == "안전모미착용" and rng.random() < .2) else ("없음" if rng.random() < .75 else ["서류판", "무전기"][rng.integers(2)])
    if hold != "없음":
        s = rng.choice([-1, 1]); hxh, hyh = hx + s * 1.25 * r, top + 2.6 * r
        d.line([(hx + s * 2.05 * r, top + 1.0 * r), (hxh, hyh)], fill=A(shade(shirt, .85)), width=int(.6 * r))
        L = held_layer(hold, r, jit(GEAR[gear_col] if hold == "안전모" and gear_col in GEAR else list(GEAR.values())[rng.integers(len(GEAR))], 6))
        P.alpha_composite(L, (int(hxh - L.width / 2), int(hyh - L.height / 2 - .3 * r)))
        d.ellipse([hxh - .3 * r, hyh + .3 * r, hxh + .3 * r, hyh + .85 * r], fill=A(skin))
    # 머리 · 얼굴
    for s in (-1, 1): d.ellipse([hx + s * .93 * r - .17 * r, hy - .2 * r, hx + s * .93 * r + .17 * r, hy + .3 * r], fill=A(shade(skin, .95)))
    d.ellipse([hx - .9 * r, hy - r, hx + .9 * r, hy + 1.05 * r], fill=A(skin))
    draw_hair(d, hx, hy, r, hair_c, hair_style)
    ink = (35, 28, 24)
    for s in (-1, 1):
        ex = hx + dx + s * .33 * r
        d.ellipse([ex - .08 * r, hy + .02 * r, ex + .08 * r, hy + .16 * r], fill=A(ink))
        d.line([(ex - .15 * r, hy - .12 * r), (ex + .15 * r, hy - .14 * r)], fill=A(shade(hair_c, .9)), width=max(1, int(.06 * r)))
    d.line([(hx + dx, hy + .15 * r), (hx + dx + .06 * r, hy + .4 * r)], fill=A(shade(skin, .8)), width=max(1, int(.05 * r)))
    d.arc([hx + dx - .25 * r, hy + .4 * r, hx + dx + .25 * r, hy + .65 * r], 20, 160, fill=A((120, 50, 50)), width=max(1, int(.05 * r)))
    if rng.random() < .2:                                                 # 보안경 (양쪽 클래스 공통)
        d.rounded_rectangle([hx + dx - .6 * r, hy - .02 * r, hx + dx + .6 * r, hy + .22 * r], radius=.1 * r, fill=A((60, 70, 80), 170))
    if rng.random() < .15:                                                # 수염
        d.chord([hx - .7 * r, hy + .1 * r, hx + .7 * r, hy + 1.05 * r], 0, 180, fill=A(shade(hair_c, 1.0), 200))
    # 머리에 쓰는 것: 별도 층에 그려 머리 중심으로 살짝 기울인다
    Hl = Image.new("RGBA", (W, W), (0, 0, 0, 0)); hd = ImageDraw.Draw(Hl)
    if head == "안전모":
        draw_helmet(hd, hx, hy, r, gcol, facing)
        if rng.random() < .35:                                            # 턱끈
            for s in (-1, 1): hd.line([(hx + s * .85 * r, hy - .2 * r), (hx + s * .3 * r, hy + 1.0 * r)], fill=A((40, 40, 40)), width=max(1, int(.05 * r)))
    elif head == "야구모자": draw_cap(hd, hx, hy, r, gcol, facing)
    elif head == "비니": draw_beanie(hd, hx, hy, r, gcol)
    elif head == "후드": draw_hood_front(hd, hx, hy, r, hood_c)
    Hl = Hl.rotate(rng.uniform(-12, 12), resample=Image.BICUBIC, center=(hx, hy))
    P.alpha_composite(Hl)
    P = P.rotate(rng.uniform(-6, 6), resample=Image.BICUBIC, center=(hx, hy + 2 * r))
    img.alpha_composite(P)
    out = img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    a = np.asarray(out, float) * rng.uniform(.8, 1.15) * (1 + rng.normal(0, .02, 3))[None, None, :] + rng.normal(0, 3, (SIZE, SIZE, 1))
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < .3: out = out.filter(ImageFilter.GaussianBlur(rng.uniform(.4, 1.3)))
    desc = (f"{gear_col} 안전모" if head == "안전모" else
            {"맨머리": "맨머리", "대머리": "맨머리(민머리)", "야구모자": f"{gear_col} 야구모자", "비니": f"{gear_col} 비니", "후드": "후드"}[head])
    if hold == "안전모": desc += " · 안전모를 손에 듦"
    elif hold != "없음": desc += f" · {hold} 듦"
    return out, desc

def main(root="helmet_images", n_per_class=120, n_new=15):
    rows = []
    for cls in CLASSES:
        os.makedirs(f"{root}/train/{cls}", exist_ok=True)
        for i in range(1, n_per_class + 1):
            im, desc = make(cls); fn = f"train/{cls}/{cls}_{i:03d}.jpg"
            im.save(f"{root}/{fn}", quality=90); rows.append([fn, cls, desc])
    os.makedirs(f"{root}/new", exist_ok=True); new = []
    for i in range(1, n_new + 1):
        cls = CLASSES[(i - 1) % 2]; im, desc = make(cls); fn = f"new/new_{i:02d}.jpg"
        im.save(f"{root}/{fn}", quality=90); new.append([os.path.basename(fn), cls, desc])
    with open(f"{root}/labels.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["파일", "클래스", "머리상태"]); w.writerows(rows)
    with open("정답_helmet_new.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["파일", "클래스", "머리상태"]); w.writerows(new)
    print(len(rows), "train,", len(new), "new")

if __name__ == "__main__":
    main()
