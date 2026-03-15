from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction_text="",
        risk_percentage="Heart Disease Risk: 0%"
    )

@app.route("/predict", methods=["POST"])
def predict():

    values = [float(x) for x in request.form.values()]
    final = np.array(values).reshape(1, -1)

    prediction = model.predict(final)
    probability = model.predict_proba(final)[0][1]

    risk = round(probability * 100, 2)

    if prediction[0] == 1:
        result = "High Risk of Heart Disease"
    else:
        result = "Low Risk of Heart Disease"

    return render_template(
        "index.html",
        prediction_text=result,
        risk_percentage=f"Heart Disease Risk: {risk}%"
    )

if __name__ == "__main__":
    app.run(debug=True)