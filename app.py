import os
import pandas as pd
import streamlit as st
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
import joblib
import altair as alt

import warnings
warnings.filterwarnings('ignore')

# CSVデータを読み込む関数
def read_csv_data_as_pandas(file_name):
    pass

# トレーニングとテストセットの前処理
def preprocess_train_test_set(train, test):
    pass

# カテゴリデータのエンコード
def categorical_encode_string_type_features(train, test):
    pass

# 欠損値を中央値で埋める
def complete_missing_features_value(train, test):
    pass

# 特徴量とターゲットの列を定義
def define_feature_target_col(train):
    pass

# モデルをロードする
def load_model(model_path):
    pass

# テストデータでデフォルト確率を予測
def predict_default_proba_in_test(test, feature_cols, model):
    pass

# Streamlit 部分
st.title('Default Probability Prediction')

# セッション状態の初期化
if 'predicted_result' not in st.session_state:
    st.session_state.predicted_result = None

# モデルのロード
model = load_model('dataset/model.pkl')

# タブの作成
tab1, tab2, tab3 = st.tabs(["バッチ予測", "個別予測", "特徴量の重要度分析"])

# バッチ予測タブ
with tab1:
    pass

# 個別予測タブ
with tab2:
    pass

# 特徴量の重要度分析タブ
with tab3:
    pass
