from flask import Flask, request, jsonify
import joblib
from extract_features import extract_features_from_url

app = Flask(__name__)
model = joblib.load("phishing_model.pkl")
scaler = joblib.load("scaler.pkl")  # 加入這一行

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "Missing URL"}), 400

    features = extract_features_from_url(url)
    features_scaled = scaler.transform([features])  # 標準化處理
    prediction = model.predict(features_scaled)[0]
    return jsonify({"result": int(prediction)})
