import os
import pandas as pd
import streamlit as st
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

def read_csv_data_as_pandas(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    return pd.read_csv(file_path)

def preprocess_train_test_set(train, test):
    train_preprocessed, test_preprocessed = categorical_encode_string_type_features(train, test)
    train_preprocessed, test_preprocessed = complete_missing_features_value(train_preprocessed, test_preprocessed)
    return train_preprocessed, test_preprocessed

def categorical_encode_string_type_features(train, test):
    categorical_feature_cols = list(train.select_dtypes(include=['object']).columns)
    ce_oe = ce.OrdinalEncoder(cols=categorical_feature_cols)
    train_encoded = pd.concat([ce_oe.fit_transform(train.drop(columns='y')), train['y']], axis=1)
    test_encoded = ce_oe.transform(test)
    return train_encoded, test_encoded

def complete_missing_features_value(train, test):
    for col in train.columns[train.isnull().sum() > 0]:
        median_value = train[col].median()
        train[col].fillna(median_value, inplace=True)
        test[col].fillna(median_value, inplace=True)
    return train, test

def define_feature_target_col(train):
    feature_cols = [col for col in train.columns if col not in ['y', 'id']]
    return feature_cols, 'y'

def fit_model_into_train(train, feature_cols, target_col):
    X_train, X_valid, y_train, y_valid = train_test_split(train[feature_cols], train[target_col], test_size=0.2, random_state=0)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def predict_default_proba_in_test(test, feature_cols, model):
    pred_proba = model.predict_proba(test[feature_cols])[:, 1]
    result_df = pd.DataFrame({'id': test['id'], 'predicted_proba': pred_proba})
    result_df = result_df.sort_values(by='predicted_proba', ascending=False)
    result_df['predicted_proba'] = result_df['predicted_proba'].apply(lambda x: f'{x*100:.2f}%') 
    return result_df


st.title('Default Probability Prediction')

if 'predicted_result' not in st.session_state:
    st.session_state.predicted_result = None

if st.button('実行'):
    train = read_csv_data_as_pandas('dataset/train.csv')
    test = read_csv_data_as_pandas('dataset/test.csv')
    train_preprocessed, test_preprocessed = preprocess_train_test_set(train, test)
    feature_cols, target_col = define_feature_target_col(train_preprocessed)
    model = fit_model_into_train(train_preprocessed, feature_cols, target_col)
    st.session_state.predicted_result = predict_default_proba_in_test(test_preprocessed, feature_cols, model)
    st.success('予測が完了しました！')

if st.session_state.predicted_result is not None:
    st.dataframe(st.session_state.predicted_result)

if st.button('リセット'):
    st.session_state.predicted_result = None
    st.rerun()
