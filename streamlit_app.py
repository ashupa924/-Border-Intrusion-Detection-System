# streamlit_app_binary_intrusion.py
# =============================================
# Binary Border Intrusion Detection (LIVE COUNTERS)
# =============================================

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------
# Streamlit Config
# ---------------------------------------------
st.set_page_config(page_title="Border Intrusion Detection", layout="wide")
st.title("🚨 Border Intrusion Detection System (Binary)")
st.markdown("Detect **Intrusion vs Normal** using trained ML model.")

# ---------------------------------------------
# Session State (LIVE COUNTERS & Prediction Table)
# ---------------------------------------------
if 'total_events' not in st.session_state:
    st.session_state.total_events = 0
    st.session_state.intrusions = 0
    st.session_state.normal = 0
if 'pred_table' not in st.session_state:
    st.session_state.pred_table = pd.DataFrame(columns=[
        'sensor_id','latitude','longitude','motion_detected',
        'sound_level_db','thermal_level','vibration_level',
        'visibility','prediction','probability'
    ])

# ---------------------------------------------
# Load Artifacts
# ---------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("intrusion_model.pkl")
    scaler = joblib.load("scaler.pkl")
    visibility_encoder = joblib.load("visibility_encoder.pkl")
    return model, scaler, visibility_encoder

model, scaler, visibility_encoder = load_artifacts()

# ---------------------------------------------
# Sidebar Inputs
# ---------------------------------------------
st.sidebar.header("🔧 Sensor Inputs")

sensor_id = st.sidebar.number_input("Sensor ID", 1, 100, 10)
latitude = st.sidebar.slider("Latitude", 20.0, 35.0, 28.0)
longitude = st.sidebar.slider("Longitude", 70.0, 90.0, 78.0)
motion_detected = int(st.sidebar.checkbox("Motion Detected"))
sound_level = st.sidebar.slider("Sound Level (dB)", 20.0, 130.0, 60.0)
thermal_level = st.sidebar.slider("Thermal Level", 5.0, 100.0, 35.0)
vibration = st.sidebar.slider("Vibration Level", 0.0, 15.0, 3.0)
visibility = st.sidebar.selectbox("Visibility", ["clear", "fog", "rain", "night"])

# ---------------------------------------------
# Encode & Scale Input
# ---------------------------------------------
visibility_encoded = visibility_encoder.transform([visibility])[0]

input_df = pd.DataFrame([[sensor_id, latitude, longitude, motion_detected,
                          sound_level, thermal_level, vibration, visibility_encoded]],
                        columns=['sensor_id', 'latitude', 'longitude', 'motion_detected',
                                 'sound_level_db', 'thermal_level', 'vibration_level',
                                 'visibility_encoded'])
input_scaled = scaler.transform(input_df)

# ---------------------------------------------
# Live Single Prediction
# ---------------------------------------------
st.subheader("🔍 Live Single Prediction")

if st.button("Predict Intrusion", key="predict_button"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.session_state.total_events += 1
    if pred == 1:
        st.session_state.intrusions += 1
        label = 'Intrusion'
        st.error(f"⚠️ Intrusion Detected (Confidence: {prob:.2f})")
    else:
        st.session_state.normal += 1
        label = 'Normal'
        st.success(f"✅ No Intrusion Detected (Confidence: {1 - prob:.2f})")

    new_row = {
        'sensor_id': sensor_id,
        'latitude': latitude,
        'longitude': longitude,
        'motion_detected': motion_detected,
        'sound_level_db': sound_level,
        'thermal_level': thermal_level,
        'vibration_level': vibration,
        'visibility': visibility,
        'prediction': label,
        'probability': round(prob, 3)
    }

    st.session_state.pred_table = pd.concat(
        [st.session_state.pred_table, pd.DataFrame([new_row])],
        ignore_index=True
    )

# ---------------------------------------------
# Live Metrics
# ---------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Live Events", st.session_state.total_events)
c2.metric("Live Intrusions", st.session_state.intrusions)
c3.metric("Live Normal", st.session_state.normal)

# Reset button
if st.button("🔄 Reset Live Counters", key="reset_button"):
    st.session_state.total_events = 0
    st.session_state.intrusions = 0
    st.session_state.normal = 0
    st.session_state.pred_table = pd.DataFrame(columns=[
        'sensor_id','latitude','longitude','motion_detected',
        'sound_level_db','thermal_level','vibration_level',
        'visibility','prediction','probability'
    ])

# ---------------------------------------------
# Batch Prediction (CSV Upload)
# ---------------------------------------------
st.markdown("---")
st.subheader("📂 Batch Prediction (CSV Upload)")
st.info("📊 Metrics below apply to CSV batch predictions only")

uploaded_file = st.file_uploader("Upload sensor CSV", type="csv")

def highlight_severity(row):
    if row['alert_severity'] == 'High':
        return ['background-color: #ffcccc'] * len(row)
    elif row['alert_severity'] == 'Medium':
        return ['background-color: #fff2cc'] * len(row)
    else:
        return ['background-color: #e6ffe6'] * len(row)

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        if 'visibility' not in data.columns:
            st.error("CSV must contain 'visibility' column")
            st.stop()

        data['visibility_encoded'] = visibility_encoder.transform(data['visibility'])

        required_cols = ['sensor_id', 'latitude', 'longitude', 'motion_detected',
                         'sound_level_db', 'thermal_level', 'vibration_level',
                         'visibility_encoded']
        for col in required_cols:
            if col not in data.columns:
                st.error(f"Missing required column: {col}")
                st.stop()

        X = data[required_cols]
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

        data['intrusion_detected'] = preds
        data['intrusion_probability'] = probs
        data['alert_severity'] = pd.cut(probs,
                                        bins=[0.0, 0.5, 0.75, 1.0],
                                        labels=['Low', 'Medium', 'High'],
                                        include_lowest=True)

        total = len(data)
        intrusions = (preds == 1).sum()
        normal = (preds == 0).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Events", total)
        c2.metric("Intrusions", intrusions)
        c3.metric("Normal", normal)

        st.subheader("🗺 Intrusion Map View")
        st.map(data.rename(columns={'latitude': 'lat', 'longitude': 'lon'})[['lat', 'lon']])

        st.subheader("📊 Prediction Results")
        st.dataframe(data.style.apply(highlight_severity, axis=1), use_container_width=True)

    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown("---")
st.caption("Binary ML Border Intrusion Detection | Live Monitoring Dashboard")
