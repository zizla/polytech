# 식품 분류 데이터셋 5종 (Orange3 분류 실습)

Kaggle 원본 CSV 9개를 Orange3 분류 실습에 바로 쓸 수 있게 정리했습니다. 인코딩 UTF-8(BOM 없음), 쉼표 구분, 첫 줄이 헤더이며, 빈 칸은 결측치입니다(임의 대체 없음).

## 파일
| 데이터셋 | 파일 | 인스턴스 | 열 | 타깃 | 크기 |
|---|---|---|---|---|---|
| ① 수산물 신선도 (e-nose) | `fish_freshness_cod.csv` | 78,238 | 11 | `label` | 5.00 MB |
| | `fish_freshness_crab.csv` | 68,491 | 11 | `label` | 5.02 MB |
| | `fish_freshness_salmon.csv` | 72,671 | 11 | `label` | 4.85 MB |
| | `fish_freshness_shrimp.csv` | 49,608 | 11 | `label` | 3.46 MB |
| | `fish_freshness_tuna.csv` | 67,969 | 11 | `label` | 4.40 MB |
| ② 식품 부패 IoT (합성) | `food_spoilage_iot.csv` | 150,830 | 16 | `Target_Spoilage_Class` | 14.89 MB |
| ③ 우유 품질 | `milk_quality.csv` | 1,059 | 8 | `Grade` | 0.03 MB |
| ④ 레드 와인 품질 | `red_wine_quality.csv` | 1,599 | 12 | `quality` | 0.10 MB |
| ⑤ 커피 품질 CQI | `coffee_cqi_arabica.csv` | 207 | 41 | `Quality_Class`(파생) | 0.11 MB |

`index.html`은 URL 복사 버튼·열 설명·분류 워크플로 안내 페이지입니다.

## Orange3에서 열기
File 위젯 → URL 탭 → 아래 주소 붙여넣기 → Apply

- https://zizla.github.io/polytech/food-orange3/fish_freshness_cod.csv
- https://zizla.github.io/polytech/food-orange3/fish_freshness_crab.csv
- https://zizla.github.io/polytech/food-orange3/fish_freshness_salmon.csv
- https://zizla.github.io/polytech/food-orange3/fish_freshness_shrimp.csv
- https://zizla.github.io/polytech/food-orange3/fish_freshness_tuna.csv
- https://zizla.github.io/polytech/food-orange3/food_spoilage_iot.csv
- https://zizla.github.io/polytech/food-orange3/milk_quality.csv
- https://zizla.github.io/polytech/food-orange3/red_wine_quality.csv
- https://zizla.github.io/polytech/food-orange3/coffee_cqi_arabica.csv

## 워크플로 요약
`File → Select Columns(타깃 지정) → Test and Score → Tree / Random Forest / Logistic Regression → Confusion Matrix`

- 클래스: ① `1` 신선 / `2` 비신선 · ② `Fresh` / `Acceptable` / `Spoiled` · ③ `low` / `medium` / `high` · ④ `3`~`8` · ⑤ `A`(≥86) / `B`(84~86) / `C`(82~84) / `D`(<82)
- **누출 주의**: ① `TVC`, ② `Target_TVC_log10`, ⑤ `Total Cup Points`는 타깃을 만든 근거 수치입니다. 특징에서 빼십시오. ①의 `Time`도 넣으면 정확도가 거의 100%가 됩니다.
- ②는 같은 `Item_ID`(10,000개 팩)를 반복 측정한 데이터라 무작위 교차검증 점수가 낙관적입니다. Test and Score에는 그룹 지정 칸이 없으므로 이 한계를 함께 설명하십시오.
- ② `Meat_Type`·`Packaging`은 `Beef / BEEF / "beef "`, `Vacuum / vac / vacuum-sealed`처럼 표기가 섞여 있고, 열마다 3~4% 결측이 있습니다. 센서 ADC가 65535(포화)인 값 703건도 원본 그대로입니다.
- ④는 5·6등급이 82%를 차지하고, `quality`는 값이 6가지인 숫자 열이라 Orange3가 수치형으로 읽습니다. 분류로 쓰려면 **Discretize**로 구간을 나누십시오(임계값 6.5 하나면 7 이상 217 vs 그 외 1,382로 이진 분류).
- ⑤의 `Quality_Class`는 원본 `Total Cup Points`에 구간 규칙만 적용해 맨 뒤에 붙인 열입니다. `ID`·농장·로트·주소·날짜 열은 메타로 보내고, 상수 열(`Status`·`Clean Cup`·`Sweetness`·`Defects`)은 무시하십시오.

## 원본
| 데이터셋 | Kaggle 원본 | 라이선스 |
|---|---|---|
| ① | https://www.kaggle.com/datasets/rafimahfuz/e-nose-based-fish-freshness-evaluation | CC0 |
| ② | https://www.kaggle.com/datasets/razanihababdellatif/food-spoilage-iot-dataset | CC0 (제작자 공개 시뮬레이션 데이터) |
| ③ | https://www.kaggle.com/datasets/cpluzshrijayan/milkquality | Unknown |
| ④ | https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009 | DbCL (UCI, Cortez et al. 2009) |
| ⑤ | https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi | Other (CQI 스크랩) |

행·값은 변형하지 않았습니다. 예외는 ⑤의 `Quality_Class` 추가 하나이며, ③의 헤더 `Fat `(끝 공백)만 `Fat`으로 다듬었습니다. ③의 `Temprature`는 원본 철자 그대로입니다.

## 배포
Settings → Pages → Deploy from a branch → `main` / `(root)`
