from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 載入模型與標準化器
model = joblib.load('phishing_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url_features = data.get('url')

    if url_features is None:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        features = np.array(url_features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return jsonify({'result': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 這段是重點！！！
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
