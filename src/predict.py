import pandas as pd
import joblib


# Load Trained Model

model = joblib.load("models/titanic_model.pkl")


# Passenger Information

passenger = pd.DataFrame({

    "Pclass": [1],

    "Sex": ["female"],

    "Age": [25],

    "SibSp": [0],

    "Parch": [1],

    "Fare": [80],

    "Embarked": ["C"]

})


# Generate Prediction

prediction = model.predict(passenger)[0]


# Generate Probability

probability = model.predict_proba(passenger)[0][1]


# Display Result

if prediction == 1:

    print("Passenger is predicted to survive.")

else:

    print("Passenger is predicted not to survive.")


print("Survival Probability:", round(probability * 100, 2), "%")