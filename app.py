import streamlit as st
import pandas as pd
import joblib


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Titanic Survival AI",
    page_icon="🚢",
    layout="wide"
)


# -------------------------------
# LOAD MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/titanic_model.pkl")


model = load_model()


# -------------------------------
# CUSTOM CSS
# -------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    font-size: 38px;
    font-weight: bold;
    color: #17365D;
}

.subtitle {
    color: #64748b;
    font-size: 17px;
}

div.stButton > button {
    background-color: #17365D;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 17px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #24558c;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# HEADER
# -------------------------------

st.markdown(
    '<div class="title">🚢 Titanic Survival Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered passenger survival prediction dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.title("About Project")

st.sidebar.write(
    """
    This application uses Machine Learning
    to predict Titanic passenger survival.

    **Algorithms:**
    - Logistic Regression
    - Random Forest

    **Technology:**
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    """
)

st.sidebar.divider()

st.sidebar.info(
    "This is an educational prediction project. "
    "Predictions are estimates, not historical verification."
)


# -------------------------------
# DASHBOARD
# -------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Project", "Titanic ML")

with col2:
    st.metric("Model", "Random Forest")

with col3:
    st.metric("Task", "Classification")


st.divider()


# -------------------------------
# PASSENGER INFORMATION
# -------------------------------

st.subheader("👤 Passenger Information")

col1, col2 = st.columns(2)

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=25
    )

    sibsp = st.number_input(
        "Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0
    )
with col2:

    parch = st.number_input(
        "Parents / Children",
        min_value=0,
        max_value=10,
        value=0
    )

    fare = st.number_input(
        "Ticket Fare",
        min_value=0.0,
        value=50.0
    )

    embarked = st.selectbox(
        "Embarkation Port",
        ["S", "C", "Q"]
    )


st.divider()


# -------------------------------
# PREDICTION
# -------------------------------

if st.button("🔍 Predict Passenger Survival"):

    input_data = pd.DataFrame({

        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]

    })

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    survival_probability = probabilities[1]

    st.divider()

    st.subheader("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        if prediction == 1:

            st.success("Passenger is predicted to survive!")

        else:

            st.error("Passenger is predicted not to survive.")

    with col2:

        st.metric(
            "Survival Probability",
            f"{survival_probability * 100:.2f}%"
        )


    # Probability visualization

    st.write("### Survival Probability")

    probability_data = pd.DataFrame({

        "Outcome": ["Not Survived", "Survived"],

        "Probability": [
            probabilities[0],
            probabilities[1]
        ]

    })

    st.bar_chart(
        probability_data.set_index("Outcome")
    )


    # Passenger details

    st.write("### Passenger Details")

    st.dataframe(
        input_data,
        use_container_width=True
    )


    # Store prediction history

    if "history" not in st.session_state:

        st.session_state.history = []

    st.session_state.history.append({

        "Passenger Class": pclass,

        "Gender": sex,

        "Age": age,

        "Prediction": (
            "Survived" if prediction == 1
            else "Not Survived"
        ),

        "Probability": round(
            survival_probability * 100, 2
        )

    })


# -------------------------------
# PREDICTION HISTORY
# -------------------------------

st.divider()

st.subheader("📋 Prediction History")

if "history" in st.session_state and st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

    if st.button("Clear Prediction History"):

        st.session_state.history = []

        st.rerun()

else:

    st.info("No predictions generated yet.")


# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.caption(
    "Titanic Survival Prediction | "
    "Machine Learning Project | Python & Streamlit"
)