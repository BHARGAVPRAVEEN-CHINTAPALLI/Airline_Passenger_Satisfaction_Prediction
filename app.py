import streamlit as st
import pickle
import pandas as pd

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(
    page_title="Airline Passenger Satisfaction Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Airline Passenger Satisfaction Prediction App")

st.write("Enter passenger details below:")

# ── Categorical Features ──────────────────────────────────────────────────────
type_of_travel = st.selectbox(
    "Type of Travel",
    options=["Business travel", "Personal Travel"]
)

travel_class = st.selectbox(
    "Class",
    options=["Business", "Eco", "Eco Plus"]
)

customer_type = st.selectbox(
    "Customer Type",
    options=["Loyal Customer", "disloyal Customer"]
)

# ── Service Ratings ───────────────────────────────────────────────────────────
st.write("---")
st.write("**Rate the following services (0 = Not Applicable, 1–5 scale):**")

online_boarding = st.slider(
    "Online Boarding",
    min_value=0, max_value=5, value=3
)

inflight_wifi = st.slider(
    "In-flight Wifi Service",
    min_value=0, max_value=5, value=3
)

inflight_entertainment = st.slider(
    "In-flight Entertainment",
    min_value=0, max_value=5, value=3
)

leg_room_service = st.slider(
    "Leg Room Service",
    min_value=0, max_value=5, value=3
)

checkin_service = st.slider(
    "Check-in Service",
    min_value=0, max_value=5, value=3
)

cleanliness = st.slider(
    "Cleanliness",
    min_value=0, max_value=5, value=3
)

onboard_service = st.slider(
    "On-board Service",
    min_value=0, max_value=5, value=3
)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict"):

    input_data = pd.DataFrame({
        "Online Boarding":          [online_boarding],
        "In-flight Wifi Service":   [inflight_wifi],
        "Type of Travel":           [type_of_travel],
        "Class":                    [travel_class],
        "In-flight Entertainment":  [inflight_entertainment],
        "Customer Type":            [customer_type],
        "Leg Room Service":         [leg_room_service],
        "Check-in Service":         [checkin_service],
        "Cleanliness":              [cleanliness],
        "On-board Service":         [onboard_service],
    })

    prediction = model.predict(input_data)[0]

    if prediction == "Satisfied":
        st.success(f"Predicted Satisfaction: {prediction} 😊")
    else:
        st.warning(f"Predicted Satisfaction: {prediction} 😐")
