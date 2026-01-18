import streamlit as st
import pandas as pd
import joblib

# モデルとカラム名の読み込み
model = joblib.load('titanic_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("タイタニック生存予測アプリ")

# ユーザー入力（スライダーやセレクトボックス）
age = st.slider("年齢", 0, 100, 29)
fare = st.number_input("運賃", value=150.0)
sex = st.selectbox("性別", ["male", "female"])
pclass = st.selectbox("客室クラス", [1, 2, 3])

# 入力データを学習時と同じ形式に変換
input_df = pd.DataFrame([{
    'Age': age, 'Fare': fare, 'Sex': sex, 'Pclass': pclass
}])
input_df = pd.get_dummies(input_df).reindex(columns=model_columns, fill_value=0)

# 予測実行
if st.button("生存率を判定する"):
    prob = model.predict_proba(input_df)[0][1] # 生存確率
    st.write(f"あなたの生存確率は {prob*100:.2f} % です")