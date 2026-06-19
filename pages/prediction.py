import streamlit as st
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data/iris.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = DecisionTreeClassifier()

model.fit(X, y)

st.title("🔮 Iris Prediction")

sl = st.number_input(
    "Sepal Length",
    0.0,
    10.0,
    5.1
)

sw = st.number_input(
    "Sepal Width",
    0.0,
    10.0,
    3.5
)

pl = st.number_input(
    "Petal Length",
    0.0,
    10.0,
    1.4
)

pw = st.number_input(
    "Petal Width",
    0.0,
    10.0,
    0.2
)

if st.button("Predict"):

    pred = model.predict(
        [[sl, sw, pl, pw]]
    )[0]

    names = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    st.success(
        f"Prediction: {names[pred]}"
    )