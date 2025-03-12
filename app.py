import os
import pandas as pd
import streamlit as st
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
import joblib

import warnings
warnings.filterwarnings('ignore')

# CSVファイルをPandasで読み込む関数
def read_csv_data_as_pandas(file_name):
    pass

# Train, Testデータの前処理をする関数
def preprocess_train_test_set(train, test):
    pass

# 文字列型の特徴量をカテゴリカルエンコードする関数
def categorical_encode_string_type_features(train, test):
    pass

# 欠損値の補完を行う関数
def complete_missing_features_value(train, test):
    pass

# 特徴量と目的変数を指定する関数
def define_feature_target_col(train):
    pass

# モデルを学習する関数
def fit_model_into_train(train, feature_cols, target_col):
    pass

# 学習したモデルをTestデータに適用し、予測確率を返す関数
def predict_default_proba_in_test(test, feature_cols, model):
    pass

# Streamlit 部分
st.title('Default Probability Prediction')

# セッションステートに予測結果を保存する変数
if 'predicted_result' not in st.session_state:
    st.session_state.predicted_result = None

# モデルのロード（保存されているモデルを使用）
def load_model(model_path):
    pass

model = load_model('dataset/model.pkl')

# バッチ予測の処理
st.subheader('Batch Prediction')

if st.button('バッチ予測を実行'):
    pass

if st.session_state.predicted_result is not None:
    pass

# バッチ予測結果リセットボタン
if st.button('バッチ予測結果リセット'):
    pass

# Single Prediction（個別予測）機能
st.subheader('Single Prediction')

# ID入力
input_id = st.text_input('IDを入力してください')

if input_id:
    pass
