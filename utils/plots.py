import plotly.express as px


def accuracy_chart(df):

    fig = px.bar(
        df,
        x="Model",
        y="Accuracy",
        title="Decision Tree Model Comparison"
    )

    return fig