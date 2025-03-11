import os
import pandas as pd
import streamlit as st
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

# CSVファイルをPandasで読み込む関数を実装してください
def read_csv_data_as_pandas(file_name):
    pass

# Train, Testデータの前処理をする関数を実装してください
def preprocess_train_test_set(train, test):
    pass

# 前処理において、文字列型の特徴量をカテゴリカルエンコードする関数を実装してください
def categorical_encode_string_type_features(train, test):
    pass

# 前処理において、欠損値の存在する特徴量の値を補完する関数を実装してください
def complete_missing_features_value(train, test):
    pass

# 特徴量と目的変数を指定する関数を実装してください
def define_feature_target_col(train):
    pass

# Trainデータでモデルを学習させる関数を実装してください
def fit_model_into_train(train, feature_cols, target_col):
    pass

# 学習したモデルをTestデータに適用し、予測確率を返却する関数を実装してください
def predict_default_proba_in_test(test, feature_cols, model):
    pass

# Streamlitアプリのタイトルを設定してください
pass

# セッションステートに予測結果を保存する変数を作成してください
pass

# 実行ボタンが押されたときに処理を実行する部分を実装してください
pass

# 予測結果を表示する部分を実装してください
pass

# リセットボタンを実装し、予測結果を初期化する処理を追加してください
pass
