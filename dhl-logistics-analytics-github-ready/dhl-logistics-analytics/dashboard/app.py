import streamlit as st
import requests

st.set_page_config(page_title="DHL SLA Risk Dashboard", layout="wide")
st.title("DHL Express — SLA Breach Early-Warning Dashboard")
st.caption("Human-in-the-loop decision support")

with st.form("prediction"):
    c1, c2 = st.columns(2)
    with c1:
        hub = st.selectbox("Hub", ["HUB_US", "HUB_EU", "HUB_APAC", "HUB_LATAM"])
        service = st.selectbox("Service", ["Express_Worldwide", "Same_Day", "Economy_Select"])
        transit = st.number_input("Transit duration", min_value=0.1, value=18.5)
        distance = st.number_input("Route distance", min_value=1.0, value=850.0)
    with c2:
        weather = st.number_input("Weather severity", min_value=0.0, max_value=10.0, value=3.0)
        traffic = st.number_input("Traffic density", min_value=0.0, max_value=20.0, value=7.0)
        weight = st.number_input("Package weight", min_value=0.1, value=12.4)

    submitted = st.form_submit_button("Assess shipment risk")

if submitted:
    payload = {
        "hub_location": hub,
        "service_level": service,
        "transit_duration": transit,
        "weather_severity": weather,
        "traffic_density": traffic,
        "route_distance": distance,
        "package_weight": weight,
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        st.metric("Delay probability", f"{result['delay_probability']:.1%}")
        st.subheader(result["risk_category"])
        st.write(result["recommended_action"])
        st.info("Final operational action must be validated by a dispatcher.")
    except requests.RequestException as exc:
        st.error(f"API unavailable: {exc}")
