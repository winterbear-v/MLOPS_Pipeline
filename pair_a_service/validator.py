import os

import re

import uuid

import json

import hashlib

import subprocess

import pandas as pd

import numpy as np

PII_COLUMN_PATTERNS = re.compile(

    r"(^id$|_id$|email|e-mail|phone|mobile|ssn|social_security|passport|credit_card|name|first_name|last_name|address|zip_code)",

    re.IGNORECASE,

)

EMAIL_VALUE_REGEX = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")

PHONE_VALUE_REGEX = re.compile(r"^(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}$")


def compute_sha256(filepath: str) -> str:

    """Computes SHA-256 hash of the dataset for strict data versioning (Unit-II)."""

    sha256_hash = hashlib.sha256()

    with open(filepath, "rb") as f:

        for byte_block in iter(lambda: f.read(4096), b""):

            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def track_with_dvc_or_manifest(filepath: str, dataset_id: str, data_hash: str) -> str:

    """Attempts DVC tracking; always writes a .dvc.json lineage file for reproducibility."""

    dvc_meta_path = f"{filepath}.dvc.json"

    lineage_data = {

        "dataset_id": dataset_id,

        "file_path": os.path.basename(filepath),

        "sha256_version": data_hash,

        "size_bytes": os.path.getsize(filepath),

    }

    try:

        subprocess.run(

            ["dvc", "add", filepath],

            check=True,

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

        )

        lineage_data["tracked_by"] = "dvc_cli"

    except Exception:

        lineage_data["tracked_by"] = "sha256_dvc_manifest"

    with open(dvc_meta_path, "w", encoding="utf-8") as f:

        json.dump(lineage_data, f, indent=2)

    return dvc_meta_path


def detect_and_drop_pii(df: pd.DataFrame):

    """Detects and drops PII columns by header name and cell value regex (Unit-VIII)."""

    dropped_columns = []

    for col in list(df.columns):

        if PII_COLUMN_PATTERNS.search(str(col)):

            dropped_columns.append(col)

            continue

        if df[col].dtype == object:

            sample_vals = df[col].dropna().astype(str).head(25)

            if len(sample_vals) > 0:

                email_matches = sample_vals.apply(lambda x: bool(EMAIL_VALUE_REGEX.match(x))).mean()

                phone_matches = sample_vals.apply(lambda x: bool(PHONE_VALUE_REGEX.match(x))).mean()

                if email_matches > 0.5 or phone_matches > 0.5:

                    dropped_columns.append(col)

    cleaned_df = df.drop(columns=dropped_columns, errors="ignore")

    return cleaned_df, dropped_columns


def validate_and_version_csv(uploaded_file, datasets_dir: str) -> dict:

    """Validates CSV, removes PII, saves cleaned dataset, and records data version."""

    os.makedirs(datasets_dir, exist_ok=True)

    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"

    raw_path = os.path.join(datasets_dir, f"{dataset_id}_raw.csv")

    clean_path = os.path.join(datasets_dir, f"{dataset_id}.csv")

    uploaded_file.save(raw_path)

    try:

        df = pd.read_csv(raw_path)

    except Exception as e:

        if os.path.exists(raw_path):

            os.remove(raw_path)

        raise ValueError(f"Uploaded file is not a valid CSV: {str(e)}")

    os.remove(raw_path)

    if df.empty or len(df) < 15:

        raise ValueError("Dataset must contain at least 15 rows for reliable training and validation.")

    # Drop completely empty columns

    df = df.dropna(axis=1, how="all")

    # Privacy Gate: Detect and drop PII columns (Unit-VIII)

    cleaned_df, dropped_pii = detect_and_drop_pii(df)

    if cleaned_df.shape[1] < 2:

        raise ValueError("Dataset must have at least 1 feature column and 1 target column after PII removal.")

    cleaned_df.to_csv(clean_path, index=False)

    # Version control the dataset (Unit-II)

    data_hash = compute_sha256(clean_path)

    dvc_manifest = track_with_dvc_or_manifest(clean_path, dataset_id, data_hash)

    numerical_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()

    categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns.tolist()

    missing_summary = cleaned_df.isnull().sum().to_dict()

    return {

        "dataset_id": dataset_id,

        "dataset_version_hash": data_hash,

        "dvc_manifest": os.path.basename(dvc_manifest),

        "rows": int(len(cleaned_df)),

        "columns": cleaned_df.columns.tolist(),

        "numerical_columns": numerical_cols,

        "categorical_columns": categorical_cols,

        "dropped_pii_columns": dropped_pii,

        "missing_values": {k: int(v) for k, v in missing_summary.items()},

    }
