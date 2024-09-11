# 필수 라이브러리 설치
# pip install streamlit pandas scikit-learn seaborn

import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# 타이틀 설정
st.title("Iris 데이터셋을 사용한 간단한 머신러닝 애플리케이션")


# 데이터 불러오기
iris = sns.load_dataset('iris')
st.write("Iris 데이터셋:", iris.head())


# 사용자 입력을 위한 사이드바 생성
st.sidebar.header('모델 매개변수')
test_size = st.sidebar.slider('테스트 데이터 비율', 0.1, 0.5, 0.2)
# n_estimators = st.sidebar.slider('랜덤 포레스트의 트리 개수', 10, 100, 50)
tree_num = st.sidebar.slider('모델의 K-num 개수', 1, 20, 2)


# 데이터 분할
X = iris.drop(columns=['species'])
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                         test_size=test_size, 
                                         random_state=77)


# 모델 훈련
# model = RandomForestClassifier(n_estimators=n_estimators, 
#                                random_state=77)
model = KNeighborsClassifier(n_neighbors=tree_num)
model.fit(X_train, y_train)


# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)


# 결과 표시
st.write(f"모델 정확도: {accuracy:.2f}")
st.write("분류 리포트:", pd.DataFrame(report).transpose())

# # 특성 중요도 시각화
# feature_importances = pd.DataFrame(model.feature_importances_,
#                     index = X.columns,
#                     columns=['importance']).sort_values('importance', 
#                                               ascending=False)
# st.write("특성 중요도:", feature_importances)

# 사용자 입력 데이터로 예측
st.sidebar.header('새 데이터 예측')
sepal_length = st.sidebar.number_input('Sepal Length', min_value=4.0, max_value=8.0, value=5.0)
sepal_width = st.sidebar.number_input('Sepal Width', min_value=2.0, max_value=4.5, value=3.0)
petal_length = st.sidebar.number_input('Petal Length', min_value=1.0, max_value=7.0, value=4.0)
petal_width = st.sidebar.number_input('Petal Width', min_value=0.1, max_value=2.5, value=1.0)

new_data = pd.DataFrame({
    'sepal_length': [sepal_length],
    'sepal_width': [sepal_width],
    'petal_length': [petal_length],
    'petal_width': [petal_width]
})

if st.sidebar.button('예측 실행'):
    prediction = model.predict(new_data)
    st.sidebar.write(f"예측된 종: {prediction[0]}")
