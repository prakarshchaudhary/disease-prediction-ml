from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load trained model and columns
# -----------------------------
trained_diabetes_model = joblib.load("diabetes_model.pkl")
training_column_order = joblib.load("columns.pkl")

# -----------------------------
# Home route - input form
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # -----------------------------
        # Get user input from form
        # -----------------------------
        pregnancies = float(request.form["Pregnancies"])
        glucose = float(request.form["Glucose"])
        blood_pressure = float(request.form["BloodPressure"])
        skin_thickness = float(request.form["SkinThickness"])
        insulin = float(request.form["Insulin"])
        bmi = float(request.form["BMI"])
        diabetes_pedigree_function = float(request.form["DiabetesPedigreeFunction"])
        age = float(request.form["Age"])

        # -----------------------------
        # Prepare input DataFrame
        # -----------------------------
        user_input_values = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                       insulin, bmi, diabetes_pedigree_function, age]])
        user_input_dataframe = pd.DataFrame(user_input_values, columns=training_column_order)

        # -----------------------------
        # Make prediction and probability
        # -----------------------------
        predicted_class_label = trained_diabetes_model.predict(user_input_dataframe)[0]
        predicted_diabetes_probability = trained_diabetes_model.predict_proba(user_input_dataframe)[0][1]

        # -----------------------------
        # Generate explanation based on input features
        # -----------------------------
        feature_based_explanations = []

        if user_input_dataframe["Glucose"].values[0] > 140:
            feature_based_explanations.append("High glucose level → significant risk factor for diabetes")
        if user_input_dataframe["BMI"].values[0] > 30:
            feature_based_explanations.append("Body Mass Index above healthy range → increases risk")
        if user_input_dataframe["Age"].values[0] > 45:
            feature_based_explanations.append("Age greater than 45 → higher risk of diabetes")
        if user_input_dataframe["Insulin"].values[0] == 0:
            feature_based_explanations.append("No insulin reading → can indicate higher risk in dataset patterns")
        if user_input_dataframe["SkinThickness"].values[0] > 30:
            feature_based_explanations.append("Skin thickness in typical high-risk range")

        # -----------------------------
        # Construct result message
        # -----------------------------
        if predicted_class_label == 1:
            risk_message = f"High Risk of Diabetes ⚠️ (Predicted probability: {predicted_diabetes_probability*100:.1f}%)"
        else:
            risk_message = f"Low Risk of Diabetes ✅ (Predicted probability: {predicted_diabetes_probability*100:.1f}%)"

        full_result_message = risk_message + "\n\nReason based on input features:\n- " + "\n- ".join(feature_based_explanations)

        return render_template("index.html", result_message=full_result_message)

    # GET request
    return render_template("index.html", result_message="")

# -----------------------------
# Run the Flask app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
