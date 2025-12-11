# XGBoost 模型推論範例
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.datasets import load_breast_cancer

def load_trained_model():
    """載入訓練好的模型和預處理器"""
    try:
        # 載入XGBoost模型
        model = xgb.XGBClassifier()
        model.load_model('xgboost_model.json')
        
        # 載入標準化器
        scaler = joblib.load('scaler.pkl')
        
        print("✅ 模型和預處理器載入成功")
        return model, scaler
    
    except FileNotFoundError as e:
        print(f"❌ 檔案未找到: {e}")
        print("請先執行 xgboost_project.py 來訓練模型")
        return None, None

def predict_single_sample(model, scaler, sample_data):
    """對單一樣本進行預測"""
    # 標準化輸入資料
    sample_scaled = scaler.transform([sample_data])
    
    # 預測類別
    prediction = model.predict(sample_scaled)[0]
    
    # 預測機率
    probability = model.predict_proba(sample_scaled)[0]
    
    return prediction, probability

def predict_batch(model, scaler, batch_data):
    """對批次資料進行預測"""
    # 標準化輸入資料
    batch_scaled = scaler.transform(batch_data)
    
    # 預測類別
    predictions = model.predict(batch_scaled)
    
    # 預測機率
    probabilities = model.predict_proba(batch_scaled)
    
    return predictions, probabilities

def demo_inference():
    """示範模型推論"""
    print("🔮 XGBoost 模型推論示範")
    print("=" * 50)
    
    # 載入模型
    model, scaler = load_trained_model()
    if model is None:
        return
    
    # 載入測試資料
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    print(f"載入測試資料: {X.shape[0]} 個樣本, {X.shape[1]} 個特徵")
    
    # 示範1: 單一樣本預測
    print("\n📍 示範1: 單一樣本預測")
    print("-" * 30)
    
    sample_idx = 0
    sample = X[sample_idx]
    actual_label = y[sample_idx]
    
    prediction, probability = predict_single_sample(model, scaler, sample)
    
    print(f"樣本索引: {sample_idx}")
    print(f"實際標籤: {actual_label} ({'良性' if actual_label == 1 else '惡性'})")
    print(f"預測標籤: {prediction} ({'良性' if prediction == 1 else '惡性'})")
    print(f"預測機率: 惡性={probability[0]:.4f}, 良性={probability[1]:.4f}")
    print(f"預測{'正確' if prediction == actual_label else '錯誤'} ✅" if prediction == actual_label else "預測錯誤 ❌")
    
    # 示範2: 批次預測
    print("\n📍 示範2: 批次預測 (前10個樣本)")
    print("-" * 40)
    
    batch_size = 10
    batch_X = X[:batch_size]
    batch_y = y[:batch_size]
    
    predictions, probabilities = predict_batch(model, scaler, batch_X)
    
    print(f"{'索引':<4} {'實際':<4} {'預測':<4} {'惡性機率':<8} {'良性機率':<8} {'結果':<4}")
    print("-" * 50)
    
    correct_count = 0
    for i in range(batch_size):
        actual = batch_y[i]
        pred = predictions[i]
        prob_malignant = probabilities[i][0]
        prob_benign = probabilities[i][1]
        is_correct = "✅" if pred == actual else "❌"
        
        if pred == actual:
            correct_count += 1
            
        print(f"{i:<4} {actual:<4} {pred:<4} {prob_malignant:<8.4f} {prob_benign:<8.4f} {is_correct:<4}")
    
    accuracy = correct_count / batch_size
    print(f"\n批次準確率: {accuracy:.2%} ({correct_count}/{batch_size})")
    
    # 示範3: 特徵重要性分析
    print("\n📍 示範3: 模型特徵重要性 (前10名)")
    print("-" * 40)
    
    feature_importance = model.feature_importances_
    
    # 取得前10個最重要的特徵
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    
    print(f"{'排名':<4} {'特徵名稱':<25} {'重要性分數':<10}")
    print("-" * 45)
    
    for rank, idx in enumerate(top_indices, 1):
        feature_name = feature_names[idx]
        importance = feature_importance[idx]
        print(f"{rank:<4} {feature_name:<25} {importance:<10.4f}")
    
    # 示範4: 自定義預測函數
    print("\n📍 示範4: 自定義預測函數")
    print("-" * 30)
    
    def predict_with_explanation(sample_data, threshold=0.5):
        """帶解釋的預測函數"""
        prediction, probability = predict_single_sample(model, scaler, sample_data)
        
        confidence = max(probability)
        predicted_class = "良性" if prediction == 1 else "惡性"
        
        if confidence > 0.9:
            confidence_level = "非常高"
        elif confidence > 0.8:
            confidence_level = "高"
        elif confidence > 0.7:
            confidence_level = "中等"
        else:
            confidence_level = "低"
        
        return {
            'prediction': prediction,
            'predicted_class': predicted_class,
            'probability': probability,
            'confidence': confidence,
            'confidence_level': confidence_level
        }
    
    # 測試自定義函數
    sample_idx = 5
    result = predict_with_explanation(X[sample_idx])
    
    print(f"樣本 {sample_idx} 預測結果:")
    print(f"- 預測類別: {result['predicted_class']}")
    print(f"- 信心度: {result['confidence']:.4f} ({result['confidence_level']})")
    print(f"- 詳細機率: 惡性={result['probability'][0]:.4f}, 良性={result['probability'][1]:.4f}")

if __name__ == "__main__":
    demo_inference()