# XGBoost 專案 - 完整機器學習流程
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class XGBoostProject:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """載入資料集 - 使用乳癌資料集作為範例"""
        print("步驟 1: 載入資料集")
        print("=" * 50)
        
        # 載入乳癌資料集
        data = load_breast_cancer()
        self.df = pd.DataFrame(data.data, columns=data.feature_names)
        self.df['target'] = data.target
        
        print(f"資料集形狀: {self.df.shape}")
        print(f"特徵數量: {len(data.feature_names)}")
        print(f"類別分布:")
        print(self.df['target'].value_counts())
        print("\n前5筆資料:")
        print(self.df.head())
        
        return self.df
    
    def explore_data(self):
        """資料探索與視覺化"""
        print("\n步驟 2: 資料探索與視覺化")
        print("=" * 50)
        
        # 基本統計資訊
        print("基本統計資訊:")
        print(self.df.describe())
        
        # 檢查缺失值
        print(f"\n缺失值數量: {self.df.isnull().sum().sum()}")
        
        # 視覺化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 目標變數分布
        self.df['target'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('目標變數分布')
        axes[0,0].set_xlabel('類別 (0: 惡性, 1: 良性)')
        
        # 特徵相關性熱力圖 (選擇前10個特徵)
        corr_matrix = self.df.iloc[:, :10].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[0,1])
        axes[0,1].set_title('特徵相關性熱力圖 (前10個特徵)')
        
        # 特徵分布 (選擇幾個重要特徵)
        important_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
        for i, feature in enumerate(important_features[:2]):
            axes[1,i].hist(self.df[feature], bins=30, alpha=0.7)
            axes[1,i].set_title(f'{feature} 分布')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """資料預處理"""
        print("\n步驟 3: 資料預處理")
        print("=" * 50)
        
        # 分離特徵和目標變數
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        
        # 分割訓練集和測試集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 標準化特徵
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"訓練集大小: {self.X_train.shape}")
        print(f"測試集大小: {self.X_test.shape}")
        print(f"訓練集目標分布: {pd.Series(self.y_train).value_counts().to_dict()}")
        print(f"測試集目標分布: {pd.Series(self.y_test).value_counts().to_dict()}")
        
    def train_model(self):
        """訓練XGBoost模型"""
        print("\n步驟 4: 訓練XGBoost模型")
        print("=" * 50)
        
        # 基本XGBoost模型
        print("訓練基本XGBoost模型...")
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # 基本預測
        y_pred = self.model.predict(self.X_test_scaled)
        basic_accuracy = accuracy_score(self.y_test, y_pred)
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
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # 使用最佳參數的模型
        self.best_model = grid_search.best_estimator_
        print(f"\n最佳參數: {grid_search.best_params_}")
        print(f"最佳交叉驗證分數: {grid_search.best_score_:.4f}")
        
    def evaluate_model(self):
        """模型評估"""
        print("\n步驟 5: 模型評估")
        print("=" * 50)
        
        # 預測
        y_pred = self.best_model.predict(self.X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # 準確率
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"測試集準確率: {accuracy:.4f}")
        
        # 分類報告
        print("\n分類報告:")
        print(classification_report(self.y_test, y_pred))
        
        # 交叉驗證分數
        cv_scores = cross_val_score(self.best_model, self.X_train_scaled, self.y_train, cv=5)
        print(f"\n5折交叉驗證分數: {cv_scores}")
        print(f"平均CV分數: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 視覺化結果
        self.visualize_results(y_pred, y_pred_proba)
        
    def visualize_results(self, y_pred, y_pred_proba):
        """視覺化結果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 混淆矩陣
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('混淆矩陣')
        axes[0,0].set_xlabel('預測值')
        axes[0,0].set_ylabel('實際值')
        
        # 特徵重要性
        feature_importance = self.best_model.feature_importances_
        feature_names = self.df.columns[:-1]  # 排除target欄位
        
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
        axes[1,0].hist(y_pred_proba[self.y_test == 0], bins=20, alpha=0.7, label='惡性 (0)', color='red')
        axes[1,0].hist(y_pred_proba[self.y_test == 1], bins=20, alpha=0.7, label='良性 (1)', color='blue')
        axes[1,0].set_title('預測機率分布')
        axes[1,0].set_xlabel('預測機率')
        axes[1,0].set_ylabel('頻率')
        axes[1,0].legend()
        
        # ROC曲線
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
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
        
    def save_model(self):
        """儲存模型"""
        print("\n步驟 6: 儲存模型")
        print("=" * 50)
        
        # 儲存XGBoost模型
        self.best_model.save_model('xgboost_model.json')
        print("模型已儲存為 'xgboost_model.json'")
        
        # 儲存預處理器
        import joblib
        joblib.dump(self.scaler, 'scaler.pkl')
        print("標準化器已儲存為 'scaler.pkl'")
        
    def run_complete_pipeline(self):
        """執行完整的機器學習流程"""
        print("🚀 開始XGBoost機器學習專案")
        print("=" * 60)
        
        self.load_data()
        self.explore_data()
        self.preprocess_data()
        self.train_model()
        self.evaluate_model()
        self.save_model()
        
        print("\n✅ XGBoost專案完成！")
        print("=" * 60)
        print("生成的檔案:")
        print("- data_exploration.png: 資料探索視覺化")
        print("- model_evaluation.png: 模型評估結果")
        print("- xgboost_model.json: 訓練好的XGBoost模型")
        print("- scaler.pkl: 資料標準化器")

if __name__ == "__main__":
    # 建立並執行XGBoost專案
    project = XGBoostProject()
    project.run_complete_pipeline()