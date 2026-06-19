import streamlit as st
import pandas as pd

st.title("📊 Dataset Dashboard")

df = pd.read_csv("data/iris.csv")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Samples",
    len(df)
)

col2.metric(
    "Features",
    4
)

col3.metric(
    "Classes",
    3
)

st.subheader("Dataset")

st.dataframe(
    df,
    use_container_width=True
)