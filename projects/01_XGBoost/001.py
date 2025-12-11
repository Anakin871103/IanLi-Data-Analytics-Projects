## -------------------------------------------------------------------------------------------
# 載入必要套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # 新增 joblib 以便儲存 GridSearch 的最佳模型
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
## -------------------------------------------------------------------------------------------
# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 全域變數初始化 (取代 Class 內的 self 變數)
# 這些變數將在各個步驟中被賦值和使用
df = None
X_train_scaled = None
X_test_scaled = None
y_train = None
y_test = None
scaler = StandardScaler()
best_model = None

# ==============================================================================
# 步驟 1: 載入資料集
# ==============================================================================
def load_data():
    """載入資料集 - 使用乳癌資料集作為範例"""
    global df
    print("步驟 1: 載入資料集")
    print("=" * 50)
    
    # 載入乳癌資料集 (Wisconsin Breast Cancer Dataset)
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    print(f"資料集形狀: {df.shape}")
    print(f"特徵數量: {len(data.feature_names)}")
    print(f"類別分布:")
    print(df['target'].value_counts())
    print("\n前5筆資料:")
    print(df.head())
    print(df)
    return df

# ==============================================================================
# 步驟 2: 資料探索與視覺化
# ==============================================================================
def explore_data(df):
    """資料探索與視覺化"""
    print("\n步驟 2: 資料探索與視覺化")
    print("=" * 50)
    
    # 基本統計資訊
    print("基本統計資訊:")
    print(df.describe())
    
    # 檢查缺失值
    print(f"\n缺失值數量: {df.isnull().sum().sum()}")
    
    # 視覺化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 目標變數分布
    df['target'].value_counts().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('目標變數分布')
    axes[0,0].set_xlabel('類別 (0: 惡性, 1: 良性)')
    
    # 特徵相關性熱力圖 (選擇前10個特徵)
    # 
    corr_matrix = df.iloc[:, :10].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[0,1], fmt=".2f")
    axes[0,1].set_title('特徵相關性熱力圖 (前10個特徵)')
    
    # 特徵分布 (選擇幾個重要特徵)
    important_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
    for i, feature in enumerate(important_features[:2]):
        axes[1,i].hist(df[feature], bins=30, alpha=0.7)
        axes[1,i].set_title(f'{feature} 分布')
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==============================================================================
# 步驟 3: 資料預處理
# ==============================================================================
def preprocess_data(df):
    """資料預處理"""
    global X_train_scaled, X_test_scaled, y_train, y_test, scaler
    print("\n步驟 3: 資料預處理")
    print("=" * 50)
    
    # 分離特徵和目標變數
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 分割訓練集和測試集
    # X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 標準化特徵
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"訓練集大小: {X_train.shape}")
    print(f"測試集大小: {X_test.shape}")
    print(f"訓練集目標分布: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"測試集目標分布: {pd.Series(y_test).value_counts().to_dict()}")
    
    # 函數回傳所有必要的變數
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ==============================================================================
# 步驟 4: 訓練XGBoost模型
# ==============================================================================
def train_model(X_train_scaled, y_train, X_test_scaled, y_test):
    """訓練XGBoost模型並進行超參數調優"""
    global best_model
    print("\n步驟 4: 訓練XGBoost模型")
    print("=" * 50)
    
    # 初始模型 (用於比較)
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    basic_accuracy = accuracy_score(y_test, y_pred)
    print(f"基本模型準確率: {basic_accuracy:.4f}")
    
    # 超參數調優
    print("\n進行超參數調優...")
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    grid_search = GridSearchCV(
        xgb.XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='logloss'),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # 使用最佳參數的模型
    best_model = grid_search.best_estimator_
    print(f"\n最佳參數: {grid_search.best_params_}")
    print(f"最佳交叉驗證分數: {grid_search.best_score_:.4f}")
    
    return best_model

# ==============================================================================
# 步驟 5: 模型評估
# ==============================================================================
def evaluate_model(best_model, X_train_scaled, y_train, X_test_scaled, y_test, df):
    """模型評估與視覺化"""
    print("\n步驟 5: 模型評估")
    print("=" * 50)
    
    # 預測
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # 準確率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"測試集準確率: {accuracy:.4f}")
    
    # 分類報告
    print("\n分類報告:")
    print(classification_report(y_test, y_pred))
    
    # 交叉驗證分數 (在訓練集上進行)
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
    print(f"\n5折交叉驗證分數: {cv_scores}")
    print(f"平均CV分數: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 視覺化結果
    visualize_results(best_model, X_test_scaled, y_test, y_pred, y_pred_proba, df)

# ==============================================================================
# 步驟 5.1: 視覺化結果
# ==============================================================================
def visualize_results(best_model, X_test_scaled, y_test, y_pred, y_pred_proba, df):
    """視覺化結果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('混淆矩陣')
    axes[0,0].set_xlabel('預測值')
    axes[0,0].set_ylabel('實際值')
    # 
    
    # 特徵重要性
    feature_importance = best_model.feature_importances_
    feature_names = df.columns[:-1] # 排除target欄位
    
    # 選擇前15個最重要的特徵
    top_indices = np.argsort(feature_importance)[-15:]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = feature_importance[top_indices]
    
    axes[0,1].barh(range(len(top_features)), top_importance)
    axes[0,1].set_yticks(range(len(top_features)))
    axes[0,1].set_yticklabels(top_features)
    axes[0,1].set_title('前15個重要特徵')
    axes[0,1].set_xlabel('重要性分數')
    
    # 預測機率分布
    axes[1,0].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='惡性 (0)', color='red')
    axes[1,0].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='良性 (1)', color='blue')
    axes[1,0].set_title('預測機率分布')
    axes[1,0].set_xlabel('預測機率')
    axes[1,0].set_ylabel('頻率')
    axes[1,0].legend()
    
    # ROC曲線
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[1,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲線 (AUC = {roc_auc:.2f})')
    axes[1,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1,1].set_xlim([0.0, 1.0])
    axes[1,1].set_ylim([0.0, 1.05])
    axes[1,1].set_xlabel('偽陽性率')
    axes[1,1].set_ylabel('真陽性率')
    axes[1,1].set_title('ROC曲線')
    axes[1,1].legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==============================================================================
# 步驟 6: 儲存模型
# ==============================================================================
def save_model(best_model, scaler):
    """儲存模型 (使用 joblib 儲存最佳估計器)"""
    print("\n步驟 6: 儲存模型")
    print("=" * 50)
    
    # 儲存 GridSearchCV 得到的最佳模型 (使用 joblib 標準方法)
    # 檔案名稱改為 .joblib 以符合慣例
    model_filename = 'xgboost_grid_search_model.joblib'
    joblib.dump(best_model, model_filename)
    print(f"✅ 訓練好的模型已儲存為 '{model_filename}' (使用 joblib)。")
    
    # 儲存預處理器
    scaler_filename = 'scaler.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f"✅ 標準化器已儲存為 '{scaler_filename}'。")

# ==============================================================================
# 執行主要流程
# ==============================================================================
if __name__ == "__main__":
    
    print("🚀 開始XGBoost機器學習專案")
    print("=" * 60)
    
    # 1. 載入資料
    df_data = load_data()
    
    # 2. 資料探索
    explore_data(df_data)
    
    # 3. 資料預處理 (回傳分割和標準化後的資料)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df_data)
    
    # 4. 訓練模型
    best_model = train_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 5. 模型評估
    # 注意：這裡使用 df_data 是為了在 visualize_results 內取得特徵名稱
    evaluate_model(best_model, X_train_scaled, y_train, X_test_scaled, y_test, df_data)
    
    # 6. 儲存模型
    save_model(best_model, scaler)
    
    print("\n✅ XGBoost專案完成！")
    print("=" * 60)
    print("生成的檔案:")
    print("- data_exploration.png: 資料探索視覺化")
    print("- model_evaluation.png: 模型評估結果")
    print("- xgboost_grid_search_model.joblib: 訓練好的XGBoost最佳模型 (Joblib格式)")
    print("- scaler.pkl: 資料標準化器")