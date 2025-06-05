from flask import Flask, request, jsonify
import joblib
import numpy as np
from feature_extractor import extract_features_from_url  # ← 自訂的特徵擷取程式

app = Flask(__name__)

# 載入模型與標準化器
model = joblib.load('phishing_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url')

    if url is None:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        features = extract_features_from_url(url)  # 應該是 2D list，例如 [[1, 0, 0, ...]]
        
        # 新增部分：轉成 DataFrame 並加上欄位名稱
        import pandas as pd
        feature_names = [
            "URL length", "Number of dots", "Number of slashes", "Dangerous char ratio", "Numerical char ratio", 
            "Dangerous TLD", "Entropy", "IP Address", "Domain name length", "Full domain length", 
            "Subdomain count", "Suspicious keyword ratio", "Repetitions", "Redirections", "Brand Spoof Score", 
            "Whitelisted"
        ]
        features_df = pd.DataFrame(features, columns=feature_names)

        # 使用 scaler 與 model
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

@app.route('/')
def home():
    return '這是釣魚網站偵測 API，請使用 POST /predict 傳送網址。'
