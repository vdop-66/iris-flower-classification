"""Flask app exposing a simple Iris flower prediction API and demo UI."""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


def build_model() -> Tuple[Any, Any]:
    """Train a small KNN pipeline on the Iris dataset."""
    iris = load_iris()
    pipeline = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=3),
    )
    pipeline.fit(iris.data, iris.target)
    return iris, pipeline


iris_dataset, knn_model = build_model()


@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        feature_names=iris_dataset.feature_names,
        target_names=iris_dataset.target_names,
    )


@app.post("/predict")
def predict() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    expected_fields: List[Tuple[str, str]] = [
        ("sepal_length", "Sepal length (cm)"),
        ("sepal_width", "Sepal width (cm)"),
        ("petal_length", "Petal length (cm)"),
        ("petal_width", "Petal width (cm)"),
    ]

    values: List[float] = []
    missing: List[str] = []
    for key, label in expected_fields:
        raw_value = payload.get(key)
        if raw_value is None:
            missing.append(label)
            continue
        try:
            values.append(float(raw_value))
        except (TypeError, ValueError):
            return (
                jsonify({"error": f"{label} must be a number."}),
                400,
            )

    if missing:
        return (
            jsonify({"error": f"Missing fields: {', '.join(missing)}."}),
            400,
        )

    features = np.array([values])
    prediction = knn_model.predict(features)[0]
    probabilities = knn_model.predict_proba(features)[0]

    return jsonify(
        {
            "species": iris_dataset.target_names[prediction],
            "probabilities": {
                iris_dataset.target_names[idx]: round(float(score), 3)
                for idx, score in enumerate(probabilities)
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
