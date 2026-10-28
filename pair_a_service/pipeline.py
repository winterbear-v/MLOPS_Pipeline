import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_feature_pipeline(X: pd.DataFrame, estimator) -> Pipeline:

    """

    Builds an end-to-end Scikit-Learn Pipeline combining imputation,

    scaling, encoding, and the final ML estimator (Unit-III).

    """

    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_transformer = Pipeline(

        steps=[

            ("imputer", SimpleImputer(strategy="median")),

            ("scaler", StandardScaler()),

        ]

    )

    cat_transformer = Pipeline(

        steps=[

            ("imputer", SimpleImputer(strategy="most_frequent")),

            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),

        ]

    )

    preprocessor = ColumnTransformer(

        transformers=[

            ("num", num_transformer, numerical_cols),

            ("cat", cat_transformer, categorical_cols),

        ],

        remainder="drop",

    )

    full_pipeline = Pipeline(

        steps=[

            ("preprocessor", preprocessor),

            ("model", estimator),

        ]

    )

    return full_pipeline


def compute_baseline_statistics(X: pd.DataFrame) -> dict:

    """

    Computes training feature baseline statistics (mean, std, min, max)

    so Pair B can calculate real-time Data Drift in Prometheus (Unit-VI).

    """

    baselines = {}

    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    for col in numerical_cols:

        col_series = X[col].dropna()

        if len(col_series) > 0:

            baselines[col] = {

                "mean": float(col_series.mean()),

                "std": float(col_series.std()) if float(col_series.std()) > 1e-6 else 1.0,

                "min": float(col_series.min()),

                "max": float(col_series.max()),

            }

    return baselines
