import os
import pandas as pd
import streamlit as st
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import joblib
import altair as alt
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# CSVデータを読み込む関数
def read_csv_data_as_pandas(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    return pd.read_csv(file_path)

# トレーニングとテストセットの前処理
def preprocess_train_test_set(train, test):
    train_preprocessed, test_preprocessed = categorical_encode_string_type_features(train, test)
    train_preprocessed, test_preprocessed = complete_missing_features_value(train_preprocessed, test_preprocessed)
    return train_preprocessed, test_preprocessed

# カテゴリデータのエンコード
def categorical_encode_string_type_features(train, test):
    categorical_feature_cols = list(train.select_dtypes(include=['object']).columns)
    ce_oe = ce.OrdinalEncoder(cols=categorical_feature_cols)
    train_encoded = pd.concat([ce_oe.fit_transform(train.drop(columns='y')), train['y']], axis=1)
    test_encoded = ce_oe.transform(test)
    return train_encoded, test_encoded

# 欠損値を中央値で埋める
def complete_missing_features_value(train, test):
    for col in train.columns[train.isnull().sum() > 0]:
        median_value = train[col].median()
        train[col].fillna(median_value, inplace=True)
        test[col].fillna(median_value, inplace=True)
    return train, test

# 特徴量とターゲットの列を定義
def define_feature_target_col(train):
    feature_cols = [col for col in train.columns if col not in ['y', 'id']]
    return feature_cols, 'y'

# モデルをロードする
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# テストデータでデフォルト確率を予測
def predict_default_proba_in_test(test, feature_cols, model):
    pred_proba = model.predict_proba(test[feature_cols])[:, 1]
    result_df = pd.DataFrame({'id': test['id'], 'predicted_proba': pred_proba})
    result_df = result_df.sort_values(by='predicted_proba', ascending=False)
    result_df['predicted_proba'] = result_df['predicted_proba'].apply(lambda x: f'{x*100:.2f}%') 
    return result_df

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
    st.subheader('バッチ予測')

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

# 日本語ラベル辞書
feature_labels = {
    'contract_type': '契約の種類',
    'gender': '性別',
    'has_own_car': '車を所有しているか',
    'has_own_reality': '不動産を所有しているか',
    'cnt_children': '子供の数',
    'amt_income_total': '総収入額',
    'amt_credit': 'クレジット額',
    'amt_annuity': '年金額',
    'amt_goods_price': '商品の価格',
    'income_type': '収入の種類',
    'education_type': '教育の種類',
    'falimy_status': '家族の状態',
    'days_birth': '年齢（誕生日からの日数）',
    'days_employed': '就業日数',
    'flag_mobile': 'モバイルを持っているか',
    'flag_email': 'Eメールを持っているか',
    'occupation_type': '職業タイプ',
    'region_population': '地域の人口',
    'region_rating': '地域の評価',
    'organization_type': '組織の種類',
    'weekday_apply_start': '申請開始曜日',
    'hour_apply_start': '申請開始時間'
}

with tab2:
    st.subheader('個別予測')

    # trainデータを読み込む
    train = read_csv_data_as_pandas('dataset/train.csv')  # データ読み込み

    # 特徴量の列を取得
    feature_cols, target_col = define_feature_target_col(train)

    # 特徴量入力のためのフォーム作成
    input_data = {}
    for feature in feature_cols:
        label = feature_labels.get(feature, feature)  # 日本語ラベルがあればそれを使用

        # カテゴリーデータかどうかを確認し、選択肢を表示
        unique_values = train[feature].dropna().unique()  # trainデータからユニークな値を取得
        
        if len(unique_values) < 10:  # 値の種類が少ない場合、選択肢として表示
            input_data[feature] = st.selectbox(f'{label}を選択してください', options=unique_values)
        else:  # 数値的な特徴量の場合、テキスト入力を使用
            # 数値型特徴量のデフォルト値として、中央値を設定
            median_value = train[feature].median() if train[feature].dtype in ['int64', 'float64'] else ''
            input_data[feature] = st.text_input(f'{label}の値を入力してください', value=median_value)

    # 最頻値と平均値で埋める処理
    for feature in feature_cols:
        if isinstance(input_data[feature], str) and input_data[feature] == '':  # カテゴリカル変数で入力がない場合
            most_frequent_value = train[feature].mode()[0]  # 最頻値を取得
            input_data[feature] = most_frequent_value  # 最頻値で埋める
        elif isinstance(input_data[feature], (int, float)) and input_data[feature] == '':  # 数値型の特徴量
            median = train[feature].median()  # 平均値を取得
            input_data[feature] = median  # 平均値で埋める

    # 予測を実行するボタンを追加
    if st.button('予測を実行'):
        # 入力された値をDataFrame形式に変換
        input_df = pd.DataFrame([input_data])

        # 特徴量をエンコード（事前に学習したエンコーダを使用）
        categorical_feature_cols = list(train.select_dtypes(include=['object']).columns)
        ce_oe = ce.OrdinalEncoder(cols=categorical_feature_cols)
        input_df_encoded = ce_oe.fit_transform(input_df)  # 特徴量をエンコード

        # 欠損値を平均値で補完する前に、数値型として変換
        for col in input_df_encoded.select_dtypes(include=['object']).columns:
            input_df_encoded[col] = pd.to_numeric(input_df_encoded[col], errors='coerce')

        # 数値型に変換した後、欠損値を補完
        input_df_encoded.fillna(input_df_encoded.mean(), inplace=True)

        # 学習済みモデルで予測
        model = load_model('dataset/model.pkl')  # モデルをロード
        pred_proba = model.predict_proba(input_df_encoded[feature_cols])[:, 1]

        result = f'{pred_proba[0] * 100:.2f}%'
        st.write(f'入力された特徴量に基づくデフォルト確率: {result}')

    else:
        st.warning('特徴量を入力してください。')

with tab3: 
    st.subheader('特徴量の重要度分析とカテゴリカル変数の解釈')

    # セクション選択のラジオボタン
    section = st.radio(
        '表示したいセクションを選択してください',
        ('特徴量の重要度分析', 'カテゴリカル変数の解釈')
    )
    # モデルを使って特徴量の重要度を取得
    feature_importances = model.coef_[0]  # LogisticRegression の係数
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importances
    })

    # 特徴量の重要度を標準化
    scaler = StandardScaler()
    importance_df['Standardized_Importance'] = scaler.fit_transform(importance_df[['Importance']])

    # 重要度を降順に並べ替え
    importance_df = importance_df.sort_values(by='Standardized_Importance', ascending=False)

    # 日本語ラベル辞書を使って日本語ラベルを付ける
    importance_df['Feature_JP'] = importance_df['Feature'].map(feature_labels)

    # 「特徴量の重要度分析」セクション
    if section == '特徴量の重要度分析':
        st.write('### 特徴量の重要度')

        # Altairを使って降順の棒グラフを作成
        chart = alt.Chart(importance_df).mark_bar().encode(
            x='Standardized_Importance',
            y=alt.Y('Feature_JP', sort='-x', title=None),  
        ).properties(
            width=600,
            height=400
        ).configure_axis(
            labelFontSize=7,  # ラベルの文字サイズを小さく設定(すべて表示させるため)
        )
        # グラフを表示
        st.altair_chart(chart, use_container_width=True)

        # 重要度のテーブル表示
        st.dataframe(importance_df[['Feature_JP', 'Standardized_Importance']]) 

        # 解釈の説明
        st.markdown(""" 
        **解釈方法**： 
        - **正の係数**: 特徴量の値が増加すると、デフォルト確率が増加します。 
        - **負の係数**: 特徴量の値が増加すると、デフォルト確率が減少します。 
        - **カテゴリカル変数**: 基準カテゴリと比較して、他のカテゴリがターゲットに与える影響を示します。 
        """)

    # 「カテゴリカル変数の解釈」セクション
    elif section == 'カテゴリカル変数の解釈':
        st.write('### カテゴリカル特徴量の解釈')

        # カテゴリカル特徴量を抽出
        categorical_features = importance_df[importance_df['Feature'].isin(train.select_dtypes(include=['object']).columns)]

        # カテゴリカル特徴量を選択できるようにする
        if len(categorical_features) > 0:
            selected_feature = st.selectbox(
                'カテゴリカル変数の特徴量を選んでください',
                categorical_features['Feature_JP'].unique()  
            )

            # 日本語ラベルを英語に変換
            selected_feature_english = [key for key, value in feature_labels.items() if value == selected_feature][0]

            # 選択した特徴量に基づいてそのカテゴリの値を取得
            selected_column_values = train[selected_feature_english].dropna().unique()

            # 基準カテゴリを選択するインターフェース
            base_category = st.selectbox(
                f'{selected_feature}の基準カテゴリを選んでください',
                selected_column_values
            )

            # 対象カテゴリを選択するインターフェース
            target_category = st.selectbox(
                f'{selected_feature}の対象カテゴリを選んでください',
                [cat for cat in selected_column_values if cat != base_category]  # 基準カテゴリ以外を選択肢にする
            )

            # 特徴量のダミー変数の係数を取得
            # ここで基準カテゴリを0として他のカテゴリとの相対的な係数を取得
            # モデルの coef_ に基づき、基準カテゴリは除外されることを前提とする

            base_category_index = list(selected_column_values).index(base_category)  # 基準カテゴリのインデックス
            target_category_index = list(selected_column_values).index(target_category)  # 対象カテゴリのインデックス

            # 特徴量の係数を取得（model.coef_ から基準カテゴリを除いた係数）
            base_category_coeff = model.coef_[0][base_category_index]  # 基準カテゴリの係数
            target_category_coeff = model.coef_[0][target_category_index]  # 対象カテゴリの係数

            # 解釈の表示
            interpretation = ''
            if target_category_coeff > base_category_coeff:
                interpretation = f"{base_category} と比較して、{target_category} が選ばれると、デフォルト確率が増加します。"
            elif target_category_coeff < base_category_coeff:
                interpretation = f"{base_category} と比較して、{target_category} が選ばれると、デフォルト確率が減少します。"
            else:
                interpretation = f"{base_category} と {target_category} の間にターゲット変数への影響はありません。"

            st.write(f"分析する特徴量: {selected_feature}")
            st.write(f"基準カテゴリ: {base_category}")
            st.write(f"対象カテゴリ: {target_category}")
            st.write(f"⇒ {interpretation}")
