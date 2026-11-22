"""
쿠키 굽기 정도 이미지 분류 데이터셋 (가상, 교육용)
- 실제 사진이 아니라 파이썬으로 그린 합성 이미지입니다.
- 클래스: 덜구움 / 적정 / 과다구움. 굽기 정도(0~1)를 클래스별로 겹치는 분포에서 뽑아 색·가장자리 갈변·탄 자국을 정하므로
  경계 근처 이미지는 사람 눈으로도 헷갈리게 했습니다.
- 굽기와 무관한 요소(배경 쟁반, 조명, 위치, 크기, 회전, 초코칩 수, 흐림)는 모든 클래스에서 같은 분포로 무작위화했습니다.
- 출력: cookie_images/train/<클래스>/*.jpg, cookie_images/new/*.jpg(정답 숨긴 예측용), labels.csv
"""
import os, csv, math, numpy as np
from PIL import Image, ImageDraw, ImageFilter

SEED, SIZE = 2026, 299
N_PER_CLASS, N_NEW = 120, 15
CLASSES = {"덜구움": (0.18, 0.09), "적정": (0.50, 0.09), "과다구움": (0.82, 0.09)}   # 굽기 정도 평균, 표준편차
rng = np.random.default_rng(SEED)

# 굽기 정도 0→1 에 따른 반죽 색 (RGB): 크림색 → 황금 갈색 → 진한 갈색
STOPS = [(0.0, (236, 214, 170)), (0.35, (222, 176, 104)), (0.6, (184, 121, 58)), (0.85, (120, 70, 32)), (1.0, (70, 40, 20))]
def dough(b):
    b = min(max(b, 0), 1)
    for (x0, c0), (x1, c1) in zip(STOPS, STOPS[1:]):
        if b <= x1:
            t = (b - x0) / (x1 - x0)
            return np.array([c0[i] + (c1[i] - c0[i]) * t for i in range(3)])
    return np.array(STOPS[-1][1], float)

TRAYS = [(196, 199, 203), (232, 222, 200), (62, 64, 68), (214, 206, 190), (150, 155, 160)]

def make(bake):
    # 배경: 쟁반 색 + 그라데이션 + 잡음
    tray = np.array(TRAYS[rng.integers(len(TRAYS))], float) + rng.normal(0, 8, 3)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    ang = rng.uniform(0, 2 * math.pi)
    grad = ((xx * math.cos(ang) + yy * math.sin(ang)) / SIZE - .5) * rng.uniform(10, 35)
    bg = tray[None, None, :] + grad[..., None] + rng.normal(0, 4, (SIZE, SIZE, 1))
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))

    # 쿠키 윤곽: 반지름에 저주파 잡음을 더한 다각형
    cx, cy = SIZE / 2 + rng.normal(0, 18, 2)
    R = rng.uniform(78, 112)
    k = np.arange(1, 6); amp = rng.normal(0, 1, 5) * R * .025 / k; ph = rng.uniform(0, 2 * math.pi, 5)
    th = np.linspace(0, 2 * math.pi, 180, endpoint=False)
    rr = R + (amp[None, :] * np.sin(k[None, :] * th[:, None] + ph[None, :])).sum(1)
    poly = [(cx + r * math.cos(t), cy + r * math.sin(t)) for r, t in zip(rr, th)]

    # 쿠키 본체 색: 중심 → 가장자리로 갈수록 더 구워짐(가장자리 갈변은 굽기 정도에 비례)
    mask = Image.new("L", (SIZE, SIZE), 0); ImageDraw.Draw(mask).polygon(poly, fill=255)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
    local = bake + (0.10 + 0.35 * bake) * np.clip(d - .55, 0, None) / .45 + rng.normal(0, .02, (SIZE, SIZE))
    lut = np.stack([dough(v) for v in np.linspace(-.1, 1.4, 256)])
    col = lut[np.clip(((local + .1) / 1.5 * 255).astype(int), 0, 255)]
    # 표면 질감(저주파 얼룩 + 고주파 알갱이)
    blot = np.array(Image.fromarray((rng.random((24, 24)) * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC), float) / 255 - .5
    col = col * (1 + .10 * blot[..., None]) + rng.normal(0, 6, (SIZE, SIZE, 1))
    body = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(body)

    # 갈라짐: 모든 클래스에 있지만, 덜 구운 쿠키는 가늘고 옅게
    for _ in range(rng.integers(2, 7)):
        a = rng.uniform(0, 2 * math.pi); r0 = rng.uniform(.1, .6) * R; L = rng.uniform(.15, .45) * R
        pts = [(cx + r0 * math.cos(a), cy + r0 * math.sin(a))]
        for _ in range(4):
            a += rng.normal(0, .5); x, y = pts[-1]; pts.append((x + L / 4 * math.cos(a), y + L / 4 * math.sin(a)))
        c = tuple(int(v) for v in dough(bake + .25) * .8)
        dr.line(pts, fill=c, width=int(1 + 2 * bake + rng.integers(0, 2)))

    # 초코칩: 굽기와 무관한 개수(방해 요소)
    for _ in range(rng.integers(3, 12)):
        a = rng.uniform(0, 2 * math.pi); r0 = math.sqrt(rng.uniform(0, 1)) * R * .78
        x, y = cx + r0 * math.cos(a), cy + r0 * math.sin(a); s = rng.uniform(5, 11)
        dr.ellipse([x - s, y - s * .8, x + s, y + s * .8], fill=tuple(int(v) for v in (52, 30, 18) + rng.normal(0, 6, 3)))

    # 탄 자국: 굽기 0.7 이상에서 확률적으로 가장자리에 검은 점
    if bake > .7:
        for _ in range(rng.poisson(12 * (bake - .7) / .3)):
            a = rng.uniform(0, 2 * math.pi); r0 = R * rng.uniform(.7, .95)
            x, y = cx + r0 * math.cos(a), cy + r0 * math.sin(a); s = rng.uniform(3, 9)
            dr.ellipse([x - s, y - s, x + s, y + s], fill=(rng.integers(25, 45),) * 3)

    # 쿠키만 회전 → 그림자 → 배경에 합성 (배경에 회전 흔적이 남지 않게)
    ck = body.convert("RGBA"); ck.putalpha(mask)
    ck = ck.rotate(rng.uniform(0, 360), resample=Image.BICUBIC, center=(cx, cy))
    al = ck.getchannel("A")
    sh = Image.new("L", (SIZE, SIZE), 0); sh.paste(al, (6, 8)); sh = sh.point(lambda v: int(v * .43)).filter(ImageFilter.GaussianBlur(9))
    img = Image.composite(Image.new("RGB", (SIZE, SIZE), (25, 22, 20)), img, sh)
    img.paste(ck, (0, 0), ck)
    # 촬영 조건: 조명 밝기·약한 색온도 변화, 흐림
    a = np.asarray(img, float) * rng.uniform(.82, 1.15) * (1 + rng.normal(0, .015, 3))[None, None, :]
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < .35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(.5, 1.6)))
    return img

def main(root="cookie_images"):
    rows = []
    for cls, (m, s) in CLASSES.items():
        os.makedirs(f"{root}/train/{cls}", exist_ok=True)
        for i in range(1, N_PER_CLASS + 1):
            b = float(np.clip(rng.normal(m, s), 0, 1))
            fn = f"{root}/train/{cls}/{cls}_{i:03d}.jpg"
            make(b).save(fn, quality=90); rows.append([fn[len(root) + 1:], cls, round(b, 3)])
    os.makedirs(f"{root}/new", exist_ok=True)
    keys = list(CLASSES)
    for i in range(1, N_NEW + 1):
        cls = keys[(i - 1) % 3]; m, s = CLASSES[cls]; b = float(np.clip(rng.normal(m, s), 0, 1))
        fn = f"{root}/new/new_{i:02d}.jpg"; make(b).save(fn, quality=90); rows.append([fn[len(root) + 1:], cls, round(b, 3)])
    with open(f"{root}/labels.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["파일", "클래스", "굽기정도_가상값"]); w.writerows(rows)
    print(len(rows), "images")

if __name__ == "__main__":
    main()
