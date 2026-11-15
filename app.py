import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Keep these files in the same folder as app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rf_model.pkl"
ENCODER_PATH = BASE_DIR / "encoder.pkl"

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🛵",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        vehicle_encoder = pickle.load(f)
    return model, vehicle_encoder

model, vehicle_encoder = load_artifacts()

# Category mappings for the categorical columns.
# The model's saved feature names show these original column names.
# Weather, Traffic_Level and Time_of_Day use alphabetical integer encoding.
WEATHER_MAP = {
    "Clear": 0,
    "Foggy": 1,
    "Rainy": 2,
    "Snowy": 3,
    "Windy": 4,
}
TRAFFIC_MAP = {"High": 0, "Low": 1, "Medium": 2}
TIME_MAP = {"Afternoon": 0, "Evening": 1, "Morning": 2, "Night": 3}

st.title("🛵 Food Delivery Time Predictor")
st.write(
    "Enter the delivery details below to estimate the delivery time. "
    "The prediction is returned in minutes."
)

with st.form("delivery_form"):
    st.subheader("Delivery details")

    distance = st.number_input(
        "Distance (km)", min_value=0.1, max_value=100.0,
        value=7.5, step=0.1
    )

    col1, col2 = st.columns(2)
    with col1:
        weather = st.selectbox(
            "Weather",
            ["Clear", "Foggy", "Rainy", "Snowy", "Windy"]
        )
        time_of_day = st.selectbox(
            "Time of Day",
            ["Morning", "Afternoon", "Evening", "Night"]
        )
        preparation_time = st.number_input(
            "Preparation Time (minutes)",
            min_value=0, max_value=180, value=15, step=1
        )

    with col2:
        traffic = st.selectbox(
            "Traffic Level", ["Low", "Medium", "High"]
        )
        vehicle = st.selectbox(
            "Vehicle Type", list(vehicle_encoder.classes_)
        )
        courier_experience = st.number_input(
            "Courier Experience (years)",
            min_value=0.0, max_value=50.0, value=2.0, step=0.5
        )

    submitted = st.form_submit_button(
        "Predict Delivery Time",
        type="primary",
        use_container_width=True
    )

if submitted:
    # The order of columns must match model.feature_names_in_.
    input_data = pd.DataFrame([{
        "Distance_km": distance,
        "Weather": WEATHER_MAP[weather],
        "Traffic_Level": TRAFFIC_MAP[traffic],
        "Time_of_Day": TIME_MAP[time_of_day],
        "Vehicle_Type": int(vehicle_encoder.transform([vehicle])[0]),
        "Preparation_Time_min": preparation_time,
        "Courier_Experience_yrs": courier_experience,
    }])

    # Reorder defensively to match the model's training feature order.
    input_data = input_data[model.feature_names_in_]
    prediction = float(model.predict(input_data)[0])

    st.success("Prediction completed")
    st.metric("Estimated Delivery Time", f"{prediction:.1f} minutes")

    if prediction < 30:
        st.info("This delivery is estimated to be relatively quick.")
    elif prediction < 60:
        st.info("This delivery is estimated to take around half an hour to one hour.")
    else:
        st.warning("This delivery may take more than one hour.")

st.caption(
    "Order_ID is not used as an input. Delivery_Time_min is the value "
    "predicted by the Random Forest model."
)
