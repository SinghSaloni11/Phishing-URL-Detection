from flask import Flask, request, render_template
import numpy as np
import pickle
from feature_extraction import FeatureExtraction

# Load the pre-trained model
file = open("pickle/model.pkl", "rb")
gbc = pickle.load(file)
file.close()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        obj = FeatureExtraction(url)
        features = np.array(obj.getFeaturesList()).reshape(1, 30)

        # Model predictions
        prediction = gbc.predict(features)[0]  # 1 for safe, -1 for phishing
        confidence_safe = gbc.predict_proba(features)[0, 1]  # Probability of being safe
        confidence_phishing = gbc.predict_proba(features)[0, 0]  # Probability of being phishing

        # Interpretation
        if prediction == 1:
            result = "Safe"
            confidence = f"{confidence_safe * 100:.2f}% confident this URL is safe."
        else:
            result = "Phishing"
            confidence = f"{confidence_phishing * 100:.2f}% confident this URL is phishing."

        return render_template(
            "index.html",
            url=url,
            result=result,
            confidence=confidence
        )
    return render_template("index.html", result=None, confidence=None)


if __name__ == "__main__":
    app.run(debug=True)
