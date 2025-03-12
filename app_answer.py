import os
import pandas as pd
import streamlit as st
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
import joblib

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

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def predict_default_proba_in_test(test, feature_cols, model):
    pred_proba = model.predict_proba(test[feature_cols])[:, 1]
    result_df = pd.DataFrame({'id': test['id'], 'predicted_proba': pred_proba})
    result_df = result_df.sort_values(by='predicted_proba', ascending=False)
    result_df['predicted_proba'] = result_df['predicted_proba'].apply(lambda x: f'{x*100:.2f}%') 
    return result_df

# Streamlit 部分
st.title('Default Probability Prediction')

if 'predicted_result' not in st.session_state:
    st.session_state.predicted_result = None

model = load_model('dataset/model.pkl')

st.subheader('Batch')

if st.button('バッチ予測を実行'):
    train = read_csv_data_as_pandas('dataset/train.csv')
    test = read_csv_data_as_pandas('dataset/test.csv')
    train_preprocessed, test_preprocessed = preprocess_train_test_set(train, test)
    feature_cols, target_col = define_feature_target_col(train_preprocessed)
    st.session_state.predicted_result = predict_default_proba_in_test(test_preprocessed, feature_cols, model)
    st.success('予測が完了しました！')

if st.session_state.predicted_result is not None:
    st.session_state.predicted_result['id'] = st.session_state.predicted_result['id'].astype(str)
    st.dataframe(st.session_state.predicted_result[['id', 'predicted_proba']])

# バッチ予測結果リセットボタン
if st.button('バッチ予測結果リセット'):
    st.session_state.predicted_result = None
    st.rerun()  
    
# Single Prediction（個別予測）機能
st.subheader('Single Prediction')

# ID入力
input_id = st.text_input('IDを入力してください')

if input_id:
    # predicted_resultがNoneでない場合のみ処理を進める
    if st.session_state.predicted_result is not None:
        # IDに対応する行をバッチ予測結果から取得
        input_data = st.session_state.predicted_result[st.session_state.predicted_result['id'] == input_id]
        
        if not input_data.empty:
            result = input_data['predicted_proba'].values[0]
            st.write(f'ID {input_id} の予測結果: {result} の確率でデフォルト')
        else:
            st.warning('IDが見つかりませんでした')
    else:
        st.warning('バッチ予測結果がまだありません。')

