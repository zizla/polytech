"""안전모 테스트 이미지 10장: 학습·new와 다른 난수 씨앗(3030), 사례를 지정해 만든다."""
import os, csv, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("m", "make_helmet_images.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.rng = np.random.default_rng(3030)
CASES = [  # (클래스, 머리, 색, 손에 든 것, 수업에서 볼 점)
    ("안전모착용", "안전모", "노랑", "없음", "기본 사례"),
    ("안전모착용", "안전모", "흰색", "서류판", "흰 안전모 + 서류판"),
    ("안전모착용", "안전모", "파랑", "없음", "어두운 색 안전모"),
    ("안전모착용", "안전모", "빨강", "무전기", "빨강 안전모 + 무전기"),
    ("안전모착용", "안전모", "초록", "없음", "학습에 적은 색일 수 있음"),
    ("안전모미착용", "맨머리", None, "없음", "기본 사례"),
    ("안전모미착용", "야구모자", "노랑", "없음", "헷갈림: 안전모와 같은 색 모자"),
    ("안전모미착용", "비니", "주황", "없음", "헷갈림: 둥근 모자"),
    ("안전모미착용", "맨머리", "노랑", "안전모", "헷갈림: 안전모를 손에 듦"),
    ("안전모미착용", "후드", None, "없음", "헷갈림: 머리를 덮은 후드"),
]
order = m.rng.permutation(len(CASES)); os.makedirs("test10", exist_ok=True); rows = []
for i, k in enumerate(order, 1):
    cls, head, col, hold, note = CASES[k]
    im, desc = m.make(cls, head=head, gear_col=col, hold=hold)
    fn = f"test_{i:02d}.jpg"; im.save(f"test10/{fn}", quality=90); rows.append([fn, cls, desc, note])
with open("정답_helmet_test10.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["파일", "클래스", "머리상태", "수업에서 볼 점"]); w.writerows(rows)
for r in rows: print(r)
