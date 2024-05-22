# 실행 방법
1. 필요한 패키지 설치:
```shell
pip install -r requirements.txt
```
2. 데이터 전처리:
```shell
python scripts/data_preprocessing.py
```
3. 모델 훈련:
```shell
python scripts/train_model.py
```
4. 서버 실행:
```shell
python app.py
```

# 파일 구조
```bash
medical-chatbot/
│
├── data/
│   ├── eng_external_medical_knowledge.json
│   ├── eng_test_multi.json
│   ├── eng_test_single.json
│   ├── eng_test_unseen.json
│   └── eng_train.json
│
├── models/
│   └── kogpt2-base-v2/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.json
│
├── results/
│   ├── checkpoint-10000/
│   ├── checkpoint-20000/
│   └── checkpoint-30000/
│
├── logs/
│   ├── events.out.tfevents.xxxxxxxxxxxxx
│   └── ...
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── run_server.py
│
├── app.py
└── requirements.txt
```

## 각 파일과 디렉토리 설명
1.	data/: 원본 데이터셋이 저장되는 디렉토리입니다.
	•	eng_external_medical_knowledge.json: 외부 의학 지식 데이터.
	•	eng_test_multi.json, eng_test_single.json, eng_test_unseen.json: 테스트 데이터셋.
	•	eng_train.json: 훈련 데이터셋.
2.	models/: 사전 훈련된 모델 파일들이 저장되는 디렉토리입니다.
	•	kogpt2-base-v2/: KoGPT2 모델 파일들이 저장된 디렉토리.
	•	config.json, pytorch_model.bin, vocab.json: 모델 구성 파일들.
3.	results/: 모델 훈련 결과와 체크포인트가 저장되는 디렉토리입니다.
	•	checkpoint-10000/, checkpoint-20000/, checkpoint-30000/: 각 체크포인트마다 생성된 디렉토리.
4.	logs/: 훈련 중 생성된 로그 파일들이 저장되는 디렉토리입니다.
5.	scripts/: 데이터 전처리, 모델 훈련, 서버 실행 스크립트가 포함된 디렉토리입니다.
	•	data_preprocessing.py: 데이터 전처리 스크립트.
	•	train_model.py: 모델 훈련 스크립트.
	•	run_server.py: Flask 서버 실행 스크립트.
6.	app.py: Flask 서버 메인 파일. 챗봇 인터페이스를 제공.
7.	requirements.txt: 필요한 Python 패키지 목록.