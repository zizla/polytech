# 안전모 착용 여부 이미지 분류 (Orange3)

폴리텍 「식품 품질관리(QC) 자동화 및 AI 실무 융합 과정」 이미지 분류 실습용 데이터셋입니다.
작업자 그림을 **안전모착용 / 안전모미착용** 두 클래스로 나눠 학습·평가합니다.
현장 안전 점검(안전모 착용 확인)을 사람 눈 대신 이미지 분류로 처리하는 사례로 씁니다.

## 출처
- 원본: Google Drive `폴리텍 AI QC자동화 강의 > 과제 > data` 폴더 (`ryuki.kim@gmail.com`, 폴더 ID `14IvD53ZmR7_TRSIdgEapD9wBKhG0wXLG`)
- 받은 파일
  - `helmet_images.zip` (5,168,083 bytes, 파일 ID `1oRl5y4EsneZNEDQtd9_P5MOO3y3pySPY`)
  - `helmet_test10.zip` (204,403 bytes, 파일 ID `1ZSsjZBnRp5nweya8e0bCctkeW3Eq9M3m`)
- zip을 풀어 **이미지 내용을 바꾸지 않고** 폴더만 `images/` 아래로 정리했습니다. 파일명·라벨은 원본 그대로입니다.

## 구성
```
images/
  train/안전모착용/     120장
  train/안전모미착용/   120장
  test10/                10장 (정답 있음 — 강사용 정답표는 answers/)
  new/                   15장 (정답 숨김 — 새 이미지 예측용)
data/
  labels.csv          train 240행: 파일, 클래스, 머리상태
  README_원본.txt     원본 zip에 들어 있던 제작 안내문
answers/
  정답_helmet_new.csv      new 15장 정답 (강사용)
  정답_helmet_test10.csv   test10 10장 정답 (강사용)
scripts/
  make_helmet_images.py  합성 이미지 생성 스크립트 (난수 씨앗 2026)
  make_test10.py         test10 생성 스크립트 (난수 씨앗 3030)
```

| 구분 | 안전모착용 | 안전모미착용 | 합계 |
| --- | ---: | ---: | ---: |
| train | 120 | 120 | 240 |
| test10 | 5 | 5 | 10 |
| new | 8 | 7 | 15 |
| **합계** | **133** | **132** | **265** |

## 이미지 사양
- 265장 모두 **299 × 299 px, RGB JPEG** (Inception v3 입력 크기), 전체 5,380,859 bytes.
- 파이썬(`scripts/make_helmet_images.py`)으로 그린 **합성 이미지**입니다. 실제 현장 사진이 아닙니다.
- 100 MB를 넘는 파일은 없습니다(최대 27,108 bytes = `images/train/안전모착용/안전모착용_105.jpg`).

## 데이터 설계 (수업에서 짚을 점)
- `안전모미착용` 120장 구성: 맨머리 약 50%, 야구모자 약 25%, 비니 약 15%, 후드 약 10%
  - 그중 25장은 **안전모를 손에 들고** 있습니다(쓰지 않았으므로 미착용).
  - 야구모자·비니는 안전모와 **같은 색 목록**에서 색을 골라, 색만으로는 구분되지 않게 했습니다.
- 클래스와 무관하게 무작위인 요소: 배경(하늘·지면·건물·비계·크레인·라바콘), 조명, 피부색, 머리색, 옷·형광조끼, 보안경, 서류판·무전기, 사람 위치·크기·기울기, 흐림.
- 제작자가 확인한 점검 결과(train 240장, 5겹 교차검증)
  - 배경 모서리 색만으로 분류 → 47% (무작위 50% 수준): 배경이 정답 단서가 아님
  - 이미지 전체 색 분포로 분류 → 58%: 색만으로는 거의 못 맞힘 → 모양을 보는 특징(임베딩)이 필요한 과제
- `test10`에는 일부러 헷갈리는 사례가 들어 있습니다(안전모와 같은 색 야구모자, 둥근 비니, 안전모를 손에 든 맨머리, 후드).

## 정답표에 관하여
- `answers/정답_helmet_new.csv`, `answers/정답_helmet_test10.csv`는 **강사용**입니다. 수강생에게 배포하기 전에 이 폴더를 빼고 전달하세요(원본 안내문의 '수강생 배포 전 분리 보관' 지침).
- 원본 Drive 폴더에는 `new/` 정답 파일이 없었습니다. 두 정답표는 `scripts/`의 생성 스크립트를 같은 난수 씨앗으로 다시 돌려 만들었고, **다시 만든 265장이 배포본 이미지와 바이트 단위로 모두 일치**하는 것을 확인한 뒤 그대로 옮겼습니다. 즉 배포본 이미지에 대한 정답이 맞습니다.

## Orange3에서 쓰기
1. `Image Analytics → Import Images`에서 `images/train/안전모착용`·`images/train/안전모미착용` 폴더를 차례로 불러옵니다(하위 폴더 이름이 클래스가 됩니다).
2. `Image Embedding` — Embedder = **Inception v3** (이미지를 외부 서버로 보내 계산하므로 인터넷 필요, 오프라인이면 SqueezeNet).
3. `Test and Score` → `Confusion Matrix` → `Image Viewer`로 틀린 이미지를 직접 확인합니다.
4. `images/test10` 또는 `images/new`를 `Import Images → Image Embedding → Predictions`로 연결해 새 이미지를 예측하고, `answers/`의 정답표와 맞춰 봅니다.
