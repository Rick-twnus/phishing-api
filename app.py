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
    url = data.get('url')  # 是網址字串

    if url is None:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        features = extract_features_from_url(url)  # ← 回傳 2 維 list
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
