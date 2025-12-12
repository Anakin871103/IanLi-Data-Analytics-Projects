# 載入必要函式庫
import pandas as pd
## 導入函式 [train_test_split] 用於將資料集分割成訓練集和測試集
from sklearn.model_selection import train_test_split

def prepare_data_for_modeling(
    cleaned_data_path, 
    features_to_drop, 
    target_column, 
    test_size=0.2, 
    random_state=42):
    """
    從清理後的數據載入資料，進行特徵編碼和資料分割，為模型訓練做準備。

    Args:
        cleaned_data_path (str): train_cleaned.csv 檔案的路徑。
        features_to_drop (list): 需要從特徵集中移除的欄位列表。
        target_column (str): 目標變數的欄位名稱。
        test_size (float): 分割給測試集的資料比例。
        random_state (int): 亂數種子，確保每次分割結果一致。

    Returns:
        tuple: 回傳 (X_train, X_test, y_train, y_test)
        X_train 訓練用的特徵
        X_test  測試用的特徵
        y_train 訓練用的答案
        y_test  測試用的答案


    """
    # 載入清理後的資料
    df = pd.read_csv(cleaned_data_path)

    # --- 特徵編碼 ---
    # 對指定的類別欄位進行 One-Hot Encoding
    # drop_first=True 可以避免共線性問題
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Age_Group']
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # --- 定義特徵(X)與目標(y) ---
    # 定義目標變數 y
    y = df_encoded[target_column]

    # 從編碼後的 df 中移除目標變數和指定的其他欄位，得到特徵集 X
    X = df_encoded.drop(columns=[target_column] + features_to_drop, axis=1)
    
    # 確保所有特徵都是數值型態
    X = X.apply(pd.to_numeric)

    # --- 資料分割 ---
    # 將資料分割為訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
