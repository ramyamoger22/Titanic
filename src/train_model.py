import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier


# Load Dataset

train = pd.read_csv("data/train.csv")


# Select Features

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = train[features]

y = train["Survived"]


# Numerical Features

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]


# Categorical Features

categorical_features = [
    "Sex",
    "Embarked"
]


# Numerical Preprocessing

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical Preprocessing

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine Preprocessing

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# Create ML Pipeline

model = Pipeline([
    ("preprocessor", preprocessor),

    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    ))
])


# Train Model

model.fit(X, y)


# Save Model

joblib.dump(model, "models/titanic_model.pkl")


print("Titanic Model Training Completed Successfully!")

print("Model saved inside models folder.")