import streamlit as st

st.set_page_config(
    page_title="Decision Tree Visualizer",
    page_icon="🌳",
    layout="wide"
)

st.title(
    "🌳 Decision Tree Model Comparison Dashboard"
)

st.markdown(
"""
Compare multiple Decision Tree models
using the Iris Dataset.
"""
)

st.success(
"""
Navigate using the sidebar:

• Dashboard

• Model Comparison

• Prediction
"""
)