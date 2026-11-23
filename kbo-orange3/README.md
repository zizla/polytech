# KBO 2025 타자 · 투수 데이터 (Orange3 군집분석)

2025 시즌 KBO 리그 기록을 Orange3 실습에 맞춰 정리한 CSV 2종입니다. 모든 열 이름은 한글·영문 약어 그대로이며, 첫 줄이 헤더입니다.

## 파일
- `kbo_2025_hitters_orange3.csv` — 타자 155명 (PA ≥ 100), 16개 열
- `kbo_2025_pitchers_orange3.csv` — 투수 139명 (IP ≥ 30), 19개 열
- `index.html` — URL 복사 버튼·열 설명·군집분석 안내 페이지

인코딩 UTF-8(헤더), 쉼표 구분, 빈 칸 없음.

## Orange3에서 열기
File 위젯 → URL 탭 → 아래 주소 붙여넣기 → Apply

- https://zizla.github.io/polytech/kbo-orange3/kbo_2025_hitters_orange3.csv
- https://zizla.github.io/polytech/kbo-orange3/kbo_2025_pitchers_orange3.csv

## 워크플로 요약
`File → Select Columns → Preprocess(Normalize) → k-Means → Scatter Plot / Data Table / Box Plot`

- `선수명`·`팀명`은 메타 열로, `year`(전부 2025)는 특징에서 제외합니다.
- 타율과 타점·루타수처럼 단위가 다른 열이 섞여 있으므로 k-Means 전에 Normalize를 적용합니다.

## 열
- 타자: `선수명, 팀명, year, G, PA, AB, AVG, R, H, 2B, 3B, HR, TB, RBI, SAC, SF`
- 투수: `선수명, 팀명, year, ERA, G, W, L, SV, HLD, WPCT, IP, H, HR, BB, HBP, SO, R, ER, WHIP`

## 배포
Settings → Pages → Deploy from a branch → `main` / `(root)`
