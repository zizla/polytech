# 엑셀로 막히고, 데이터로 풀린다 — 식품 QC 기계학습 실습 (Orange3)

어묵 · 조미김 · 식품향료 · 도라지 가공품 공정을 주제로 만든 **교육용 가상(시뮬레이션) 데이터**와 강의록입니다.
실제 기업의 공정 조건·규격·불량률과 무관합니다. 모든 CSV의 열 이름은 한글입니다.

## 강의 구성
각 모듈은 기존 방법(엑셀 기술통계·t-검정, 통제 실험, 관능검사, 피벗, 육안 검사)으로 먼저 분석 → 한계 확인 → Orange3로 재분석 → 비교 순서로 진행합니다.

## 구성
- `index.html` — 강의록 (실험 시뮬레이터, 육안 검사 체험 포함)
- `data/*_orange.csv` — Orange3 헤더 플래그 포함 (합격여부 자동 타깃), UTF-8
- `data/*.csv` — 일반 CSV, UTF-8
- `data/excel/*_excel.csv` — 엑셀 더블클릭용 (UTF-8 BOM)
- `data/sr_sensory_panel*.csv` — 관능 패널 60배치 × 검사원 5명 (가로형/세로형)
- `images/` — 합성 외관 검사 이미지 400장 (8클래스 × 50장)
- `data/image_meta.tab`, `data/image_features.csv`, `data/qc_images.zip` — 이미지 실습용
- `scripts/` — 재생성 스크립트 (make_data → make_panel → make_images → make_excel)

## 배포
Settings → Pages → Deploy from a branch → `main` / `(root)`

## Orange3에서 열기
File 위젯 → URL → `https://<사용자명>.github.io/<저장소명>/data/cj_fishcake_qc_orange.csv`
