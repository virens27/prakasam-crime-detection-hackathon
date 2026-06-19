"""
app.py

Simple Streamlit dashboard to display alerts from the detection pipeline.

TODO (Team):
1. Connect this to wherever main.py is logging alerts (file/database).
2. Add a video upload widget so judges can upload a clip live during the demo.
3. Display flagged snapshots alongside each alert for visual evidence.
4. Add basic filtering (by alert type, by time).

Run with: streamlit run src/dashboard/app.py
"""

import streamlit as st

st.set_page_config(page_title="Crime Detection Dashboard", layout="wide")

st.title("AI-Powered Crime Detection — Live Alert Dashboard")
st.caption("Mission Y4 – Prakasam Police Hackathon 2026")

st.divider()

# TODO: replace this placeholder section with real alert data once
# main.py is logging to a file/database the dashboard can read from.

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Weapon Alerts", 0)

with col2:
    st.metric("Fight Alerts", 0)

with col3:
    st.metric("Abandoned Object Alerts", 0)

st.divider()
st.subheader("Upload a video clip to analyze")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.info("TODO: wire this upload into main.py's run_pipeline() and display results below.")

st.divider()
st.subheader("Recent Alerts")
st.write("TODO: display alert log here (timestamp, type, confidence, snapshot).")
