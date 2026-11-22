# 무는 강아지 입장 불가! — 강아지 얼굴 이미지 분류 (Orange3)

『나는 오렌지로 데이터 분석한다』 자료실 **활동 10 「무는 강아지 입장 불가!」** 의 이미지 데이터셋입니다.
강아지 얼굴 그림을 **무는 강아지(biting) / 안 무는 강아지(nonbiting)** 두 클래스로 나눠 학습·평가하는 이미지 분류 실습에 씁니다.

## 출처
- 원본: CMASS 자료실 Google Drive — 활동 10 전용 폴더
  `https://drive.google.com/drive/folders/1belpJlIIGr8yMYEj8t9HQdLs66fniWV1`
- 받은 파일: `Dog Dataset.zip` (24,668,245 bytes, Drive 파일 ID `116U5JOWQ0foBvVfoaFB7JM4gAu1vibTo`, 2022-01-18 수정)
- 같은 폴더의 Orange3 워크플로: `활동10 무는 강아지 입장 불가!.ows` → `workflows/activity10-biting-dog.ows` 로 함께 올림
- 이 저장소에는 원본 zip을 풀어 **파일 내용을 바꾸지 않고** 폴더 구조 그대로 담았습니다. 이미지 파일명·라벨은 원본 그대로입니다.

## 구성
```
images/
  Training/biting/      13장
  Training/nonbiting/   15장
  Test/biting/           4장
  Test/nonbiting/        8장
  new/                   2장 (라벨 없음 — 새 강아지 예측용)
data/
  image_meta.csv        42행: path, class, split, width, height, bytes, sha256
  image_meta.tab        같은 내용의 Orange3용 .tab (class 열이 타깃)
workflows/
  activity10-biting-dog.ows
```

| 구분 | biting | nonbiting | 합계 |
| --- | ---: | ---: | ---: |
| Training | 13 | 15 | 28 |
| Test | 4 | 8 | 12 |
| new (라벨 없음) | — | — | 2 |
| **합계** | **17** | **23** | **42** |

## 이미지 사양
- 42장 모두 **827 × 827 px JPEG, 흰 배경**이며 한 장당 약 0.8 MB(전체 34.6 MB)입니다.
- 모두 **같은 갈색 강아지 얼굴 캐릭터**에 눈·입 모양, 소품(안대·리본·안경), 털 모양을 바꾼 **그림(일러스트)** 입니다. 실제 강아지 사진이 아닙니다.
- JPEG 색 공간이 **CMYK**입니다. 보기·변환 도구에 따라 색이 다르게 보일 수 있고, 일부 라이브러리는 RGB 변환을 요구합니다. 필요하면 `PIL.Image.open(p).convert('RGB')` 로 변환해 쓰세요.
- 100 MB를 넘는 파일은 없습니다(최대 900,786 bytes = `new/newDog2.jpg`).

## 라벨에 관한 주의
원본 라벨을 그대로 옮겼고, 두 클래스가 눈으로 봐도 완전히 일관되지는 않습니다.
- `biting` 인데 이빨이 보이지 않는 얼굴: `Training/biting/d5.jpg`, `d9.jpg`, `d28.jpg`
- `nonbiting` 인데 이빨을 드러낸 얼굴: `Test/nonbiting/d3.jpg`, `Training/nonbiting/d4.jpg`, `d23.jpg`
- `Test/biting/d6.jpg` 는 윙크하며 웃는 얼굴입니다.

그래서 이 데이터는 "정답률 100%"를 목표로 하기보다, **라벨 노이즈가 있는 실제 데이터에서 혼동행렬·오분류 사례를 읽는 실습**에 적합합니다.

## Orange3에서 쓰기
1. `Image Analytics → Import Images` 위젯에서 `images/Training/biting` 과 `images/nonbiting` 폴더를 차례로 불러옵니다(폴더명이 클래스 이름이 됩니다).
2. `Image Embedding` (예: `squeezenet`) → `Logistic Regression` / `kNN` / `Naive Bayes` → `Test and Score`, `Confusion Matrix`.
3. `Test/` 폴더를 별도로 불러 평가에 쓰고, `new/` 폴더는 `Predictions` 로 새 강아지 예측에 씁니다.
4. `workflows/activity10-biting-dog.ows` 는 원본 워크플로의 위젯 구성만 담고 있습니다(파일 경로가 저장되어 있지 않음). 열어서 `Import Images` 위젯의 폴더를 위 경로로 다시 지정하면 됩니다.

## 배포
Settings → Pages → Deploy from a branch → `main` / `(root)`
