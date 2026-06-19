import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def train_models():

    df = pd.read_csv("data/iris.csv")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Default": DecisionTreeClassifier(),
        "Gini": DecisionTreeClassifier(
            criterion="gini"
        ),
        "Entropy": DecisionTreeClassifier(
            criterion="entropy"
        ),
        "MaxDepth3": DecisionTreeClassifier(
            max_depth=3
        ),
        "MaxDepth5": DecisionTreeClassifier(
            max_depth=5
        ),
        "MinSplit4": DecisionTreeClassifier(
            min_samples_split=4
        ),
        "MinLeaf2": DecisionTreeClassifier(
            min_samples_leaf=2
        )
    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(
            y_test,
            pred
        )

        results.append(
            {
                "Model": name,
                "Accuracy": round(acc * 100, 2)
            }
        )

    return pd.DataFrame(results)