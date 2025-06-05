from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('phishing_model.pkl')
scaler = joblib.load('scaler.pkl')

feature_names = [
    'url_length', 'num_dots', 'num_slashes', 'dangerous_char_ratio',
    'numerical_char_ratio', 'dangerous_tld', 'entropy', 'is_ip',
    'domain_length', 'full_domain_length', 'subdomain_count',
    'suspicious_keyword_ratio', 'repetitions', 'redirections',
    'brand_spoof_score', 'whitelisted'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url_features = data.get('url')

    if url_features is None:
        return jsonify({'error': 'No URL features provided'}), 400

    try:
        features_df = pd.DataFrame(url_features, columns=feature_names)
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
