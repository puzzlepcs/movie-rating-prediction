# Predicting Movie Ratings
## Tested Environment
- language: python 3.5
- os: Ubuntu 16.04 LTS

## Description of the dataset 
- [Naver sentiment movie corpus](https://github.com/e9t/nsmc/)
- 1~10 점까지의 평점 중에서 중립적인 평점(5~8점)은 제외하고 1~4점을 부정, 9~10점을 긍정으로 동일한 비율로 데이터에 포함 시킴.
- id, document, label 3개의 열로 구성되어 있음.
  - id: 리뷰의 고유한 key값
  - document: 리뷰의 내용
  - label: 부정(0), 긍정(1)
- 총 200,000 개의 리뷰 
  - `ratings_test.txt` : 50만 reviews for testing
  - `ratings_test.txt` : 15만 reviews for training

## Objective
- 네이버 영화 리뷰 데이터 셋의 긍정/부정 분류하기
- `ratings_train.txt`으로 학습하여 `ratings_test.txt`의 label 예측
- 임의의 덧글 내용으로 긍정/부정 평가

## Naive Bayes Classification, `naiveBayes.py`
### Test results
두가지 방식으로 Model을 학습시켜 보았다. 단순 띄워쓰기 기준으로 파싱 하여 학습, `konlpy` 패키지를 사용하여 형태소 기준으로 파싱하여 학습을 진행하였는데, 형태소 기준으로 파싱 하였을 때 더 좋은 성능을 보이는 것을 확인하였다.  

임의의 덧글 내용으로 테스트 시, 다음과 같은 결과를 얻었다.

comment | pos/neg
--------|--------
올해 최고의 영화! 세 번 넘게 봐도 질리지가 않네요. | 1
배경 음악이 영화의 분위기랑 너무 안 맞앗습니다. 몰입에 방해가 됩니다. |  0 
주연 배우가 신인인데 연기를 진짜 잘 하네요. 몰입감 ㅎㄷㄷ | 1
주연배우 때문에 봤어요 |  1
진짜 너무 너무 |  0


## Sentiment Analysis Using Keras, `keras.py`
### Data Preprocessing
KoNLpy를 사용하여 형태소 분석을 통해 품사를 태깅 해준 후, 이를 json 파일로 저장.  
`nltk`라이브러리를 통해 데이터에서 가장 자주 사용되는 단어 5000개를 가져와 CountVectorization을 생성. (문서 집합에서 단어 토큰을 생성, 각 단어 수를 세어 BOW 인코딩한 벡터를 만든다. 즉, size 가 5000인 벡터가 생성된다.)  
### 모델 정의 및 학습
모델은 3 layer 로 구성하였다.
- layer 1: input vector size = 5000, units = 64, ReLU
- layer 2: units = 64, ReLu
- layer 3: units = 1, sigmoid  


긍정의 리뷰일 확률을 출력해야 하므로 마지막 레이어는 unit을 1개로 구성하고, 활성함수는 sigmoid를 사용하였다. output이 50%를 넘으면 긍정의 리뷰라고 판단하였다.  

손실 함수로 `binary_crossentropy`를 사용했으며, `RMSProp` 옵티마이저를 사용하였다.  
batch size 는 512, learning rate는 0.001, epoch는 10번으로 학습시켰다.


### Test Results
Test 데이터로 확인해본 결과 85%의 성능을 보여주었다.

임의의 덧글 내용으로 테스트 시, 다음과 같은 결과를 얻었다.   

comment |확률|pos/neg
--------|---|------
올해 최고의 영화! 세 번 넘게 봐도 질리지가 않네요.| 99.12% | 1
배경 음악이 영화의 분위기랑 너무 안 맞앗습니다. 몰입에 방해가 됩니다. | 2.72% | 0 
주연 배우가 신인인데 연기를 진짜 잘 하네요. 몰입감 ㅎㄷㄷ | 98.02%| 1
주연배우 때문에 봤어요 | 24.33% | 1
진짜 너무 너무 | 33.73% | 0