# XGBoost 機器學習專案

這是一個完整的XGBoost機器學習專案，包含從資料載入到模型部署的完整流程。

## 專案結構

```
xgboost-project/
├── requirements.txt          # 依賴套件
├── xgboost_project.py       # 主要程式碼
├── model_inference.py       # 模型推論範例
├── README.md               # 專案說明
└── 生成的檔案/
    ├── data_exploration.png    # 資料探索視覺化
    ├── model_evaluation.png    # 模型評估結果
    ├── xgboost_model.json     # 訓練好的模型
    └── scaler.pkl             # 資料標準化器
```

## 安裝依賴

```bash
pip install -r requirements.txt
```

## 執行專案

### 1. 完整流程執行
```bash
python xgboost_project.py
```

### 2. 模型推論
```bash
python model_inference.py
```

## 專案流程

### 步驟 1: 資料載入
- 使用sklearn的乳癌資料集作為範例
- 包含569個樣本，30個特徵
- 二元分類問題（惡性/良性）

### 步驟 2: 資料探索
- 基本統計資訊分析
- 缺失值檢查
- 特徵分布視覺化
- 相關性分析

### 步驟 3: 資料預處理
- 特徵標準化
- 訓練/測試集分割 (80/20)
- 保持類別平衡

### 步驟 4: 模型訓練
- 基本XGBoost模型訓練
- 超參數網格搜尋調優
- 5折交叉驗證

### 步驟 5: 模型評估
- 準確率計算
- 分類報告
- 混淆矩陣
- ROC曲線和AUC
- 特徵重要性分析

### 步驟 6: 模型儲存
- 儲存訓練好的模型
- 儲存預處理器

## 主要特色

- **完整的ML流程**: 從資料載入到模型部署
- **視覺化分析**: 豐富的圖表和統計分析
- **超參數調優**: 自動尋找最佳參數組合
- **模型評估**: 多種評估指標和視覺化
- **可重複使用**: 模組化設計，易於擴展

## 結果解讀

### 模型性能指標
- **準確率**: 模型預測正確的比例
- **精確率**: 預測為正類中實際為正類的比例
- **召回率**: 實際正類中被正確預測的比例
- **F1分數**: 精確率和召回率的調和平均

### 視覺化結果
- **混淆矩陣**: 顯示預測結果的詳細分布
- **ROC曲線**: 評估分類器在不同閾值下的性能
- **特徵重要性**: 識別對預測最有影響的特徵

## 自定義使用

要使用自己的資料集，請修改 `load_data()` 方法：

```python
def load_data(self):
    # 載入你的資料
    self.df = pd.read_csv('your_data.csv')
    # 確保目標變數名為 'target'
```

## 技術細節

- **XGBoost版本**: 2.0.3
- **Python版本**: 3.8+
- **主要依賴**: pandas, numpy, scikit-learn, matplotlib, seaborn

## 常見問題

**Q: 如何使用不同的資料集？**
A: 修改 `load_data()` 方法，載入你的資料並確保目標變數名為 'target'。

**Q: 如何調整超參數範圍？**
A: 修改 `train_model()` 方法中的 `param_grid` 字典。

**Q: 如何添加更多評估指標？**
A: 在 `evaluate_model()` 方法中添加所需的評估指標。