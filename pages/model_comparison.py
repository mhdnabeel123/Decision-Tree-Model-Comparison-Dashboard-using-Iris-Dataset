import streamlit as st

from models.decision_tree_models import train_models
from utils.plots import accuracy_chart

st.title("📈 Model Comparison")

results = train_models()

st.dataframe(
    results,
    use_container_width=True
)

fig = accuracy_chart(results)

st.plotly_chart(
    fig,
    use_container_width=True
)