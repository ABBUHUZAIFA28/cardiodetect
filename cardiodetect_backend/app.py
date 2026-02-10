from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model, scaler, features
model = joblib.load("models/final_lightgbm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/brfss_selected_features.pkl")

TRAINING_ACCURACY = 0.9058  # 90.58%

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])[features]
        df_scaled = scaler.transform(df)

        # Predict class 0/1
        prediction = int(model.predict(df_scaled)[0])

        # Get confidence (max probability)
        probs = model.predict_proba(df_scaled)[0]
        confidence = float(max(probs))

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "training_accuracy": float(TRAINING_ACCURACY)
        })

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
