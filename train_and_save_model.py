import os
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

import warnings
warnings.filterwarnings('ignore')

# CSVファイルを読み込む関数
def read_csv_data_as_pandas(file_name):
    return pd.read_csv(file_name)

# トレーニングデータとテストデータを前処理する関数
def preprocess_train_test_set(train, test):
    train_preprocessed, test_preprocessed = categorical_encode_string_type_features(train, test)
    train_preprocessed, test_preprocessed = complete_missing_features_value(train_preprocessed, test_preprocessed)
    return train_preprocessed, test_preprocessed

# カテゴリカル特徴量をエンコードする関数
def categorical_encode_string_type_features(train, test):
    categorical_feature_cols = list(train.select_dtypes(include=['object']).columns)
    ce_oe = ce.OrdinalEncoder(cols=categorical_feature_cols)
    train_encoded = pd.concat([ce_oe.fit_transform(train.drop(columns='y')), train['y']], axis=1)
    test_encoded = ce_oe.transform(test)
    return train_encoded, test_encoded

# 欠損値を埋める関数
def complete_missing_features_value(train, test):
    for col in train.columns[train.isnull().sum() > 0]:
        median_value = train[col].median()
        train[col].fillna(median_value, inplace=True)
        test[col].fillna(median_value, inplace=True)
    return train, test

# 特徴量とターゲット列を定義する関数
def define_feature_target_col(train):
    feature_cols = [col for col in train.columns if col not in ['y', 'id']]
    return feature_cols, 'y'

# モデルをトレーニングする関数
def fit_model_into_train(train, feature_cols, target_col):
    X_train, X_valid, y_train, y_valid = train_test_split(train[feature_cols], train[target_col], test_size=0.2, random_state=0)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# 保存する関数
def save_model(model, filename='model.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# 保存したモデルを読み込む関数
def load_model(filename='model.pkl'):
    return joblib.load(filename)

# テストデータで予測を行う関数
def predict_default_proba_in_test(test, feature_cols, model):
    pred_proba = model.predict_proba(test[feature_cols])[:, 1]
    result_df = pd.DataFrame({'id': test['id'], 'predicted_proba': pred_proba})
    result_df = result_df.sort_values(by='predicted_proba', ascending=False)
    result_df['predicted_proba'] = result_df['predicted_proba'].apply(lambda x: f'{x*100:.2f}%') 
    return result_df

# 実行部分
if __name__ == "__main__":
    train = read_csv_data_as_pandas('dataset/train.csv')
    test = read_csv_data_as_pandas('dataset/test.csv')
    train_preprocessed, test_preprocessed = preprocess_train_test_set(train, test)
    feature_cols, target_col = define_feature_target_col(train_preprocessed)


    model = fit_model_into_train(train_preprocessed, feature_cols, target_col)
    save_model(model)

