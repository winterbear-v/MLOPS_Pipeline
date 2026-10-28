import io

import os

import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validator import detect_and_drop_pii

from pipeline import build_feature_pipeline, compute_baseline_statistics

from sklearn.linear_model import LogisticRegression


def test_pii_detection_drops_sensitive_columns():

    df = pd.DataFrame({

        "customer_id": [1, 2, 3],

        "email": ["a@b.com", "c@d.com", "e@f.com"],

        "age": [25, 35, 45],

        "salary": [50000, 60000, 70000],

        "purchased": [0, 1, 0],

    })

    cleaned_df, dropped = detect_and_drop_pii(df)

    assert "customer_id" in dropped

    assert "email" in dropped

    assert "age" in cleaned_df.columns

    assert "salary" in cleaned_df.columns


def test_sklearn_feature_pipeline_handles_missing_and_categoricals():

    X = pd.DataFrame({

        "age": [25.0, None, 40.0, 30.0],

        "city": ["Mumbai", "Delhi", None, "Mumbai"],

    })

    y = [0, 1, 1, 0]

    pipe = build_feature_pipeline(X, LogisticRegression())

    pipe.fit(X, y)

    preds = pipe.predict(pd.DataFrame({"age": [28.0], "city": ["Pune"]}))

    assert len(preds) == 1


def test_baseline_statistics_calculation():

    X = pd.DataFrame({"age": [20.0, 30.0, 40.0], "dept": ["IT", "HR", "IT"]})

    stats = compute_baseline_statistics(X)

    assert "age" in stats

    assert stats["age"]["mean"] == 30.0
