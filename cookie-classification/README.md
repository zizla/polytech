# 쿠키 굽기 정도 이미지 분류 (Orange3)

폴리텍 「식품 품질관리(QC) 자동화 및 AI 실무 융합 과정」 이미지 분류 실습용 데이터셋입니다.
쿠키 외관 사진을 **덜구움 / 적정 / 과다구움** 세 클래스로 나눠 학습·평가합니다.
제과 공정의 육안 품질 검사(굽기 정도 판정)를 이미지 분류로 옮긴 사례로 씁니다.

## 출처
- 원본: Google Drive `폴리텍 AI QC자동화 강의 > 과제 > data` 폴더 (`ryuki.kim@gmail.com`, 폴더 ID `14IvD53ZmR7_TRSIdgEapD9wBKhG0wXLG`)
- 받은 파일
  - `cookie_images.zip` (7,314,150 bytes, 파일 ID `1Bj21-wSwmOjUNjcbpHz2tahWE_CMp4y4`)
  - `정답_new_폴더.csv` (524 bytes, 파일 ID `1wVqfnq6xBNhASrqKMQzIYvL6m2I5jWJD`)
- zip을 풀어 **이미지 내용을 바꾸지 않고** 폴더만 `images/` 아래로 정리했습니다. 파일명·라벨은 원본 그대로입니다.

## 구성
```
images/
  train/덜구움/     120장
  train/적정/       120장
  train/과다구움/   120장
  new/               15장 (정답 숨김 — 새 이미지 예측용)
data/
  labels.csv          train 360행: 파일, 클래스, 굽기정도_가상값
  README_원본.txt     원본 zip에 들어 있던 제작 안내문
answers/
  정답_new_폴더.csv   new 15장 정답 (강사용): 파일, 클래스, 굽기정도_가상값
scripts/
  make_cookie_images.py  합성 이미지 생성 스크립트 (난수 씨앗 2026)
```

| 구분 | 덜구움 | 적정 | 과다구움 | 합계 |
| --- | ---: | ---: | ---: | ---: |
| train | 120 | 120 | 120 | 360 |
| new | 5 | 5 | 5 | 15 |
| **합계** | **125** | **125** | **125** | **375** |

## 이미지 사양
- 375장 모두 **299 × 299 px, RGB JPEG** (Inception v3 입력 크기), 전체 7,334,099 bytes.
- 파이썬(`scripts/make_cookie_images.py`)으로 그린 **합성 이미지**입니다. 실제 사진이 아닙니다.
- 100 MB를 넘는 파일은 없습니다(최대 27,509 bytes = `images/train/적정/적정_041.jpg`).

## 데이터 설계 (수업에서 짚을 점)
- `labels.csv`의 `굽기정도_가상값`은 이미지를 그릴 때 쓴 0~1 값입니다(클래스별 분포: 덜구움 낮음, 과다구움 높음).
- **클래스 경계 근처(굽기 정도가 겹치는 구간)는 일부러 헷갈리게** 만들었습니다. 그래서 사람이 봐도 애매한 이미지가 섞여 있고, 혼동행렬에서 경계 클래스끼리 오분류가 나오는 것이 정상입니다.
- 배경 쟁반 색, 조명, 위치, 크기, 회전, 초코칩 개수, 흐림은 **클래스와 무관하게 무작위**입니다. 이런 요소로는 정답을 맞힐 수 없게 했습니다.

## 정답표에 관하여
`answers/정답_new_폴더.csv`는 **강사용**입니다. 수강생에게 배포하기 전에 이 폴더를 빼고 전달하세요(원본 안내문의 '수강생 배포 전 분리 보관' 지침).
행 수 15 = `images/new` 15장과 정확히 대응합니다(덜구움 5 · 적정 5 · 과다구움 5).

## Orange3에서 쓰기
1. `Image Analytics → Import Images`에서 `images/train/덜구움`·`images/train/적정`·`images/train/과다구움` 폴더를 차례로 불러옵니다(하위 폴더 이름이 클래스가 됩니다).
2. `Image Embedding` — Embedder = **Inception v3** (이미지를 외부 서버로 보내 계산하므로 인터넷 필요, 오프라인이면 SqueezeNet).
3. `Test and Score`(교차검증) → `Confusion Matrix` → `Image Viewer`로 틀린 이미지를 직접 확인합니다.
4. `images/new`를 `Import Images → Image Embedding → Predictions`로 연결해 새 쿠키를 예측하고, `answers/정답_new_폴더.csv`와 맞춰 봅니다. `굽기정도_가상값`을 타깃으로 두면 분류가 아니라 회귀(굽기 정도 예측) 실습으로도 쓸 수 있습니다.
