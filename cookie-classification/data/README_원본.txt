쿠키 굽기 정도 이미지 데이터셋 (가상, 교육용)
=================================================
- 파이썬(make_cookie_images.py, 난수 씨앗 2026)으로 그린 합성 이미지입니다. 실제 사진이 아닙니다.
- train/덜구움, train/적정, train/과다구움 : 클래스별 120장, 299×299 JPEG (Inception v3 입력 크기)
- new/ : 정답을 숨긴 예측 실습용 15장 (정답은 별도 파일 '정답_new_폴더.csv' — 수강생 배포 전 분리 보관)
- labels.csv : train 이미지의 클래스와 그리는 데 쓴 '굽기정도_가상값'(0~1)
- 클래스 경계 근처(굽기 정도가 겹치는 구간)는 일부러 헷갈리게 만들었습니다.
- 배경 쟁반 색, 조명, 위치, 크기, 회전, 초코칩 개수, 흐림은 클래스와 무관하게 무작위입니다.

Orange 워크플로 (Image Analytics 애드온 필요: Options → Add-ons)
1) Import Images : train 폴더 선택 → 하위 폴더 이름이 클래스(타깃)가 됨
2) Image Embedding : Embedder = Inception v3
3) Test and Score : Logistic Regression 등 연결, 교차검증 → Confusion Matrix
4) Image Viewer : 틀린 이미지를 골라 직접 확인
5) 새 이미지 예측 : new 폴더를 Import Images → Image Embedding → Predictions (3번에서 학습한 모델 연결)
참고: Orange의 Inception v3 임베딩은 이미지를 외부 서버로 보내 계산합니다(인터넷 필요). 오프라인이면 SqueezeNet(로컬)을 쓰세요.
