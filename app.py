
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_detection_model.pkl")

# Load dataset
data = pd.read_csv("dataset/creditcard.csv")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        row_number = int(request.form["row_number"])

        # Validate transaction row
        if row_number < 0 or row_number >= len(data):
            return """
            <h2>Invalid transaction row number!</h2>
            <a href="/">Go Back</a>
            """

        # Select transaction from dataset
        transaction = data.iloc[[row_number]].copy()

        # Match model's expected features
        expected_features = list(model.feature_names_in_)
        transaction = transaction[expected_features]

        # Predict transaction
        prediction = model.predict(transaction)[0]

        # Calculate fraud probability if supported
        probability_text = ""

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transaction)[0]
            classes = list(model.classes_)

            if 1 in classes:
                fraud_probability = probabilities[classes.index(1)] * 100
                probability_text = (
                    f"<p>Estimated Fraud Probability: "
                    f"{fraud_probability:.2f}%</p>"
                )

        # Set result
        if prediction == 0:
            result = "Normal Transaction"
        else:
            result = "Fraud Transaction"

        # Get amount, if available
        amount_text = ""

        if "Amount" in data.columns:
            amount = data.iloc[row_number]["Amount"]
            amount_text = f"<p>Transaction Amount: {amount:.2f}</p>"

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fraud Detection Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    text-align: center;
                    padding: 60px 15px;
                }}
                .card {{
                    background: white;
                    max-width: 500px;
                    margin: auto;
                    padding: 30px;
                    border-radius: 18px;
                    box-shadow: 0 5px 20px #33333333;
                }}
                h1 {{ color: #4c1d95; }}
                h2 {{ color: {"#dc2626" if prediction == 1 else "#15803d"}; }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    padding: 12px 20px;
                    background: #6d28d9;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Credit Card Fraud Detection</h1>
                <p>Transaction Row: {row_number}</p>
                {amount_text}
                <h2>{result}</h2>
                {probability_text}
                <a href="/">Check Another Transaction</a>
            </div>
        </body>
        </html>
        """

    except Exception as e:
        return f"""
        <h2>Prediction Error</h2>
        <p>{str(e)}</p>
        <a href="/">Go Back</a>
        """


if __name__ == "__main__":
    app.run(debug=True)
