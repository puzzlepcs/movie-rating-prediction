# Predicting Movie Ratings
## Environment
- language: python 3.5
- os: Ubuntu 16.04 LTS

## Naive Bayes Classification 
### Objective
- 네이버 영화 리뷰 데이터 셋의 긍정/부정 분류하기
- `ratings_train.txt`으로 나이브-베이즈 모델을 학습하여 `ratings_test.txt`의 label 예측, 그 결과를 `ratings_result.txt`에 저장.

### Description of the dataset (`ratings_train.txt`)
- document(Raw Sentence) / 0 = 부정, 1 = 긍정 레이블

### Test results
두가지 방식으로 Model을 학습시켜 보았다. 단순 띄워쓰기 기준으로 파싱 하여 학습, `konlpy` 패키지를 사용하여 형태소 기준으로 파싱하여 학습을 진행하였는데, 형태소 기준으로 파싱 하였을 때 더 좋은 성능을 보이는 것을 확인하였다.